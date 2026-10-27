#!/bin/bash
# Score generated images for every model in config.sh.
#
#   longtext     X-Omni TextBench, Qwen2.5-VL OCR             -> TextScore
#   oneig        OneIG-Bench text, Qwen2.5-VL OCR             -> TextScore / ED / CR / WAC
#   cvtg         CVTG-2K, PaddleOCR 2.10 (EN only)            -> WordAcc / NED
#   gentexteval  GenTextEval, TextPecker vLLM server          -> Qua / Sem
#   polytext     PolyText-Bench, TextDART vLLM server (CPU)   -> CS / CG / OG
#
# A (bench, base, model) cell is skipped when its images are incomplete or its
# scores already exist; the remaining cells run in parallel, one GPU per job.
#
# Usage:
#   bash scripts/eval_metrics.sh
#   EVAL_BENCH=longtext,oneig GPUS="0 1" bash scripts/eval_metrics.sh
#
# Env:
#   EVAL_BENCH            comma list (default: longtext,oneig,cvtg,gentexteval,polytext)
#   EVAL_LANG             optional language filter: EN or ZH
#   GPUS                  space-separated GPU ids (default: 0 1 2 3 4 5 6 7)
#   CVTG_PYTHON           python of the PaddleOCR 2.10 env used for cvtg (default: python)
#   QWEN25VL_PATH         OneIG Qwen2.5-VL weights (default: HF id)
#   HF_HOME               Hugging Face / OpenCLIP cache for the upstream CVTG evaluator
#   TEXTPECKER_VLLM_HOST  TextPecker vLLM host(s), comma separated (required for gentexteval)
#   TEXTPECKER_PORT       TextPecker vLLM port (default 1925)
#   TEXTDART_SERVER_URLS  TextDART vLLM URLs, comma separated (default http://127.0.0.1:1925)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_BASE="${EVAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${EVAL_BASE}/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib_model_name.sh"

IFS=',' read -r -a BENCHES <<< "${EVAL_BENCH:-longtext,oneig,cvtg,gentexteval,polytext}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
CVTG_PYTHON="${CVTG_PYTHON:-python}"
TEXTPECKER_PORT="${TEXTPECKER_PORT:-1925}"
POLYTEXT_REPEATS="${POLYTEXT_REPEATS:-4}"

if [[ -n "${EVAL_LANG:-}" && "${EVAL_LANG}" != "EN" && "${EVAL_LANG}" != "ZH" ]]; then
    echo "Error: EVAL_LANG must be EN or ZH." >&2
    exit 1
fi

for bench in "${BENCHES[@]}"; do
    case "$bench" in
        longtext|oneig|cvtg|polytext) ;;
        gentexteval)
            if [[ -z "${TEXTPECKER_VLLM_HOST:-}" ]]; then
                echo "Error: gentexteval needs TEXTPECKER_VLLM_HOST (TextPecker vLLM server)." >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: unsupported bench \"$bench\"; expected longtext|oneig|cvtg|gentexteval|polytext." >&2
            exit 1
            ;;
    esac
done

for selected_bench in "${BENCHES[@]}"; do
    case "$selected_bench" in
        longtext) repository=xomni ;;
        oneig) repository=oneig ;;
        cvtg|CVTG2K) repository=textcrafter ;;
        *) continue ;;
    esac
    EVAL_ROOT="${EVAL_BASE}" bash "${SCRIPT_DIR}/../setup_third_party.sh" "$repository" --check || exit 1
done

LOG_DIR="${SCRIPT_DIR}/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_metrics_${RUN_TS}.log"
JOB_LOG_DIR="${LOG_DIR}/eval_metrics_${RUN_TS}_jobs"
mkdir -p "$JOB_LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

polytext_args() {
    local base="$1" model="$2"
    printf '%s\n' --prompts "${EVAL_BASE}/PolyText-Bench/data" \
        --image-dir "${EVAL_BASE}/PolyText-Bench/eval_results/${base}/${model}" \
        --mode "${model##*_}" --repeats "$POLYTEXT_REPEATS" --width 1024 --height 1024
}

# Returns 0 when the generated images for this cell are complete and readable.
generation_complete() {
    local bench="$1" base="$2" model="$3"
    if [[ "$bench" == "cvtg" && "$model" == *_ZH ]]; then
        return 1
    fi
    python3 "${SCRIPT_DIR}/check_generation.py" \
        --bench "$bench" --base "$base" --model "$model" --eval_root "$EVAL_BASE" \
        --polytext-repeats "$POLYTEXT_REPEATS" >> "$LOG_FILE" 2>&1
}

# Returns 0 when this cell already has a complete score file.
eval_already_complete() {
    python3 "${SCRIPT_DIR}/collect_metrics.py" --eval_root "$EVAL_BASE" check "$1" "$2" "$3"
}

# One scoring job pinned to one GPU; all output goes to its own log.
run_job() {
    local gpu="$1" bench="$2" base="$3" lang="$4" model="$5"
    local jlog="${JOB_LOG_DIR}/${bench}__${base}__${model}.log"
    local start_time rc
    start_time=$(date +%s)
    log INFO "[GPU $gpu] START $bench/$base/$model -> $jlog"
    (
        set -e
        echo "===== JOB $bench/$base/$model ($lang) on GPU $gpu @ $(date) ====="
        case "$bench" in
            longtext)
                export CUDA_VISIBLE_DEVICES="$gpu"
                cd "${EVAL_BASE}/X-Omni/textbench"
                local lmode out port
                lmode="$(echo "$lang" | tr '[:upper:]' '[:lower:]')"
                out="eval_metrics/${base}/${model}"
                mkdir -p "$out"
                port=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
                torchrun --nnodes=1 --node-rank=0 --nproc_per_node=1 --master-port="$port" \
                    evaluate_text_reward.py \
                    --sample_dir "${EVAL_BASE}/X-Omni/eval_results/${base}/${model}" \
                    --output_dir "$out" \
                    --mode "$lmode"
                cp "$out/results_chunk0.jsonl" "$out/results.jsonl"
                python3 summary_scores.py "$out/results.jsonl" --mode "$lmode"
                ;;
            oneig)
                local out_dir="${EVAL_BASE}/OneIG-Benchmark/eval_metrics/${base}/${model}"
                mkdir -p "$out_dir"
                # --gpu pins CUDA before the Qwen2.5-VL inference module initializes it.
                python3 "${EVAL_BASE}/integrations/oneig/text_score_from_jsonl.py" \
                    --mode "$lang" \
                    --eval_root "$EVAL_BASE" \
                    --base "$base" \
                    --model_name "$model" \
                    --output_csv "${out_dir}/text_score.csv" \
                    --gpu "$gpu"
                ;;
            cvtg)
                export CUDA_VISIBLE_DEVICES="$gpu"
                cd "${EVAL_BASE}/TextCrafter/TextCrafter_Eval"
                local out="eval_metrics/${base}/${model}"
                mkdir -p "$out"
                "$CVTG_PYTHON" unified_metrics_eval.py \
                    --benchmark_dir ../CVTG-2K \
                    --result_dir "${EVAL_BASE}/TextCrafter/CVTG-2K/eval_results/${base}/${model}" \
                    --output_file "${out}/results.json" \
                    --device auto
                ;;
            gentexteval)
                python3 "${EVAL_BASE}/GenTextEval/textpecker_eval.py" \
                    --sample_dir "${EVAL_BASE}/GenTextEval/eval_results/${base}/${model}" \
                    --output_dir "${EVAL_BASE}/GenTextEval/eval_metrics/${base}/${model}" \
                    --lang "$lang" \
                    --vllm_host "$TEXTPECKER_VLLM_HOST" \
                    --port "$TEXTPECKER_PORT"
                ;;
            polytext)
                # TextDART runs behind vLLM; the client side is CPU only.
                export CUDA_VISIBLE_DEVICES=""
                local args servers=() out_dir
                mapfile -t args < <(polytext_args "$base" "$model")
                out_dir="${EVAL_BASE}/PolyText-Bench/eval_metrics/${base}/${model}"
                if [[ -n "${TEXTDART_SERVER_URLS:-}" ]]; then
                    read -r -a servers <<< "${TEXTDART_SERVER_URLS//,/ }"
                    args+=(--server-urls "${servers[@]}")
                fi
                PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
                python3 "${EVAL_BASE}/PolyText-Bench/evaluate.py" "${args[@]}" \
                    --output-dir "$out_dir" \
                    --concurrency "${POLYTEXT_CONCURRENCY:-8}"
                python3 "${EVAL_BASE}/PolyText-Bench/summarize.py" \
                    --prompts "${EVAL_BASE}/PolyText-Bench/data" \
                    --scores-dir "$out_dir" \
                    --mode "$lang" \
                    --repeats "$POLYTEXT_REPEATS"
                ;;
        esac
    ) > "$jlog" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        log OK "[GPU $gpu] $bench/$base/$model ($(( $(date +%s) - start_time ))s)"
    else
        log ERROR "[GPU $gpu] $bench/$base/$model failed (rc=$rc), see $jlog"
        touch "${JOB_LOG_DIR}/${bench}__${base}__${model}.failed"
    fi
    return "$rc"
}

log INFO "eval root: $EVAL_BASE"
log INFO "benches: ${BENCHES[*]} | GPUs: ${GPUS[*]} | log: $LOG_FILE"

JOBS=()
declare -A SEEN_JOBS=()
for i in "${!BASE_MODELS[@]}"; do
    BASE_MODEL="${BASE_MODELS[$i]}"
    require_base_config "$i" "$BASE_MODEL"
    declare -n LANGS="LANGS_GROUP_${i}"
    declare -n REWARDS="REWARD_GROUP_${i}"
    for LANG in "${LANGS[@]}"; do
        [[ -n "${EVAL_LANG:-}" && "$LANG" != "$EVAL_LANG" ]] && continue
        for ENTRY in "${REWARDS[@]}"; do
            MODEL_NAME="$(build_model_name "$BASE_MODEL" "$ENTRY" "$LANG")"
            for BENCH in "${BENCHES[@]}"; do
                key="${BENCH}|${BASE_MODEL}|${MODEL_NAME}"
                [[ -n "${SEEN_JOBS[$key]:-}" ]] && continue
                SEEN_JOBS[$key]=1
                if ! generation_complete "$BENCH" "$BASE_MODEL" "$MODEL_NAME"; then
                    log WARN "images incomplete or bench unsupported for this language, skip: $BENCH/$BASE_MODEL/$MODEL_NAME"
                    continue
                fi
                if eval_already_complete "$BENCH" "$BASE_MODEL" "$MODEL_NAME"; then
                    log INFO "already scored, skip: $BENCH/$BASE_MODEL/$MODEL_NAME"
                    continue
                fi
                JOBS+=("${BENCH}|${BASE_MODEL}|${LANG}|${MODEL_NAME}")
            done
        done
    done
    unset -n LANGS REWARDS
done

if [ "${#JOBS[@]}" -eq 0 ]; then
    log WARN "nothing to score."
    exit 0
fi
log INFO "${#JOBS[@]} jobs, up to ${#GPUS[@]} in parallel."

# GPU pool: each job takes one free GPU and returns it when it exits.
declare -A PID2GPU=()
FREE_GPUS=("${GPUS[@]}")
reclaim_one() {
    local finished_pid=""
    wait -n -p finished_pid 2>/dev/null || true
    if [ -n "$finished_pid" ] && [ -n "${PID2GPU[$finished_pid]:-}" ]; then
        FREE_GPUS+=("${PID2GPU[$finished_pid]}")
        unset "PID2GPU[$finished_pid]"
        return 0
    fi
    [ -n "$finished_pid" ]
}

for job in "${JOBS[@]}"; do
    while [ "${#FREE_GPUS[@]}" -eq 0 ]; do
        reclaim_one || { [ "${#PID2GPU[@]}" -eq 0 ] && break; }
    done
    IFS='|' read -r bench base lang model <<< "$job"
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    run_job "$gpu" "$bench" "$base" "$lang" "$model" &
    PID2GPU[$!]="$gpu"
done
while [ "${#PID2GPU[@]}" -gt 0 ]; do
    reclaim_one || break
done
wait || true

if compgen -G "${JOB_LOG_DIR}/*.failed" > /dev/null; then
    log ERROR "some jobs failed: $(cd "$JOB_LOG_DIR" && ls ./*.failed | tr '\n' ' ')"
    exit 1
fi
log OK "all jobs finished. Summarize with: bash scripts/cal_metrics.sh"
