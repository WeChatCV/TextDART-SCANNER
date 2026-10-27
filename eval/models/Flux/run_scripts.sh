#!/bin/bash
set -uo pipefail

# Run from the eval root (this script lives in eval/models/<family>/).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Force single-node, 8-GPU execution regardless of cluster-injected multi-node
# env vars. The PyTorch operator (k8s) sets these for a 4-node job, which makes
# torchrun try to rendezvous across nodes and crash when only one node is run.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK NODE_RANK
unset MASTER_ADDR MASTER_PORT
unset PET_NNODES PET_NODE_RANK PET_NPROC_PER_NODE PET_MASTER_ADDR PET_MASTER_PORT
unset PET_RDZV_ID PET_RDZV_BACKEND PET_RDZV_ENDPOINT

BENCH="oneig"
NPROC_PER_NODE=8
MODEL_PATH=""
MODE="EN"
MODEL_NAME=""
LORA_PATH=""
BASE_MODEL_NAME="FLUX.1-dev"
EVAL_ROOT="$(pwd)"
LOG_FILE=""
NEG_PROMPT=""
WIDTH=1024
HEIGHT=1024

log_msg() {
    local message="$1"
    local line="[$(date '+%Y-%m-%d %H:%M:%S')] $message"
    echo "$line"
    if [[ -n "$LOG_FILE" ]]; then
        echo "$line" >> "$LOG_FILE"
    fi
}

validate_lora_path() {
    local lora_path="$1"

    if [[ "$lora_path" == "None" || -z "$lora_path" ]]; then
        return 0
    fi

    if [[ ! -e "$lora_path" ]]; then
        log_msg "Skip: LoRA path does not exist: $lora_path"
        return 1
    fi

    if [[ ! -f "$lora_path" ]]; then
        log_msg "Skip: LoRA path is not a regular file: $lora_path"
        return 1
    fi

    if [[ ! -s "$lora_path" ]]; then
        log_msg "Skip: LoRA file is empty or broken: $lora_path"
        return 1
    fi

    local validation_error
    if ! validation_error=$(python3 - "$lora_path" 2>&1 <<'PY'
import sys
from safetensors import safe_open

path = sys.argv[1]
try:
    with safe_open(path, framework="pt", device="cpu") as f:
        _ = f.keys()
except Exception as exc:
    print(exc)
    sys.exit(1)
PY
); then
        log_msg "Skip: LoRA safetensors validation failed: $lora_path | error: $validation_error"
        return 1
    fi

    return 0
}

usage() {
    cat <<'EOF'
Usage:
  bash models/Flux/run_scripts.sh \
    --bench oneig \
    --nproc_per_node 8 \
    --model_path /path/to/FLUX.1-dev \
    --mode EN,ZH \
    --model_name model_a,model_b \
    --lora_path /path/a.safetensors,/path/b.safetensors \
    --base_model_name FLUX.1-dev \
    --eval_root /path/to/eval

Options:
  --bench            Comma-separated benches: polytext,longtext,oneig,CVTG2K,gentexteval
  --nproc_per_node   Number of GPU processes per torchrun task. Default: 8
  --model_path       Base model path. Required.
  --mode             Language modes separated by comma. Default: EN
  --model_name       Model names separated by comma. Required.
  --lora_path        LoRA paths separated by comma. Use None for no LoRA. Required.
  --base_model_name  Base model directory name for generated images. Default: FLUX.1-dev.
  --eval_root        Path to the eval/ directory. Default: the current eval/ root.
  --log_file         Optional log file path. Invalid LoRA and skipped tasks will be recorded here.
  --negative_prompt  Optional negative prompts separated by comma, index-aligned with --model_name.
                     An empty or "None" entry falls back to the infer script's built-in default.
  --width            Width of the generated image. Default: 1024
  --height           Height of the generated image. Default: 1024
  -h, --help         Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)
            BENCH="$2"
            shift 2
            ;;
        --nproc_per_node|--nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --model_path|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model_name|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --lora_path|--lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --base_model_name|--base-model-name)
            BASE_MODEL_NAME="$2"
            shift 2
            ;;
        --eval_root|--eval-root)
            EVAL_ROOT="$2"
            shift 2
            ;;
        --log_file|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --negative_prompt|--negative-prompt)
            NEG_PROMPT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$MODEL_NAME" || -z "$LORA_PATH" ]]; then
    echo "Error: --model_path, --model_name and --lora_path are required."
    usage
    exit 1
fi

if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    log_msg "Flux run started."
    log_msg "Bench: $BENCH"
    log_msg "Modes: $MODE"
    log_msg "Model path: $MODEL_PATH"
    log_msg "Base model name: $BASE_MODEL_NAME"
    log_msg "Eval root: $EVAL_ROOT"
    log_msg "NPROC_PER_NODE: $NPROC_PER_NODE"
fi

IFS=',' read -r -a BENCHES <<< "$BENCH"
IFS=',' read -r -a MODES <<< "$MODE"
IFS=',' read -r -a MODEL_NAMES <<< "$MODEL_NAME"
IFS=',' read -r -a LORA_PATHS <<< "$LORA_PATH"

if [[ "${#MODEL_NAMES[@]}" -ne "${#LORA_PATHS[@]}" ]]; then
    echo "Error: Number of model names doesn't match number of LoRA paths."
    echo "MODEL_NAME count: ${#MODEL_NAMES[@]}"
    echo "LORA_PATH count: ${#LORA_PATHS[@]}"
    exit 1
fi

# Negative prompts are optional and index-aligned with model names.
# An empty / "None" entry means: fall back to the infer script's built-in default.
NEG_PROMPTS=()
if [[ -n "$NEG_PROMPT" ]]; then
    IFS=',' read -r -a NEG_PROMPTS <<< "$NEG_PROMPT"
    if [[ "${#NEG_PROMPTS[@]}" -ne "${#MODEL_NAMES[@]}" ]]; then
        echo "Error: Number of negative prompts doesn't match number of model names."
        echo "MODEL_NAME count: ${#MODEL_NAMES[@]}"
        echo "NEGATIVE_PROMPT count: ${#NEG_PROMPTS[@]}"
        exit 1
    fi
fi

for ((i=0; i<"${#MODEL_NAMES[@]}"; i++)); do
    log_msg "Configured model: ${MODEL_NAMES[$i]} | LoRA: ${LORA_PATHS[$i]}"
done

get_infer_script() {
    local bench="$1"
    case "$bench" in
        polytext|PolyText-Bench)
            echo "PolyText-Bench/generate.py"
            ;;
        oneig)
            echo "models/Flux/infer_flux_dev_oneig.py"
            ;;
        CVTG2K|cvtg2k)
            echo "models/Flux/infer_flux_dev_CVTG2K.py"
            ;;
        longtext)
            echo "models/Flux/infer_flux_dev_longtext.py"
            ;;
        gentexteval|GenTextEval)
            echo "models/Flux/infer_flux_dev_gentexteval.py"
            ;;
        *)
            echo "Error: Unsupported bench: $bench" >&2
            exit 1
            ;;
    esac
}

bench_needs_mode() {
    local bench="$1"
    case "$bench" in
        oneig|longtext|gentexteval|GenTextEval|polytext|PolyText-Bench)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

bench_supports_language() {
    local bench="$1"
    local lang="$2"
    case "$bench" in
        oneig|longtext|gentexteval|GenTextEval|polytext|PolyText-Bench)
            [[ "$lang" == "EN" || "$lang" == "ZH" ]]
            ;;
        CVTG2K|cvtg2k)
            [[ "$lang" == "EN" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_language() {
    local lang="$1"
    lang="$(echo "$lang" | tr '[:lower:]' '[:upper:]')"
    case "$lang" in
        EN|ZH)
            echo "$lang"
            ;;
        *)
            echo "Error: Unsupported mode: $lang. Only EN and ZH are supported." >&2
            exit 1
            ;;
    esac
}

overall_status=0
for bench in "${BENCHES[@]}"; do
    infer_script=$(get_infer_script "$bench")

    active_modes=()
    for lang in "${MODES[@]}"; do
        normalized_lang=$(normalize_language "$lang")
        if bench_supports_language "$bench" "$normalized_lang"; then
            active_modes+=("$normalized_lang")
        else
            log_msg "Skip task once at preflight: bench=$bench mode=$normalized_lang reason=unsupported_mode"
        fi
    done

    if [[ "${#active_modes[@]}" -eq 0 ]]; then
        log_msg "Skip bench once at preflight: bench=$bench requested_modes=$MODE reason=no_supported_mode"
        continue
    fi

    for lang in "${active_modes[@]}"; do
        for ((i=0; i<"${#MODEL_NAMES[@]}"; i++)); do
            model_name="${MODEL_NAMES[$i]}"
            lora_path="${LORA_PATHS[$i]}"

            # A model name containing "None" denotes the no-reward (base model)
            # result, so we never load a LoRA for it regardless of lora_path.
            if [[ "$lora_path" == "None" || -z "$lora_path" || "$model_name" == *None* ]]; then
                lora_arg=()
                log_msg "No LoRA is used for bench=$bench lang=$lang model=$model_name."
            else
                if ! validate_lora_path "$lora_path"; then
                    overall_status=1
                    log_msg "Skip task because LoRA is invalid. bench=$bench lang=$lang model=$model_name lora=$lora_path"
                    continue
                fi
                lora_arg=(--ckpt_path "$lora_path")
            fi

            # Per-model negative prompt (optional). If provided and non-empty,
            # pass it through; otherwise the infer script uses its built-in default.
            neg_arg=()
            if [[ "${#NEG_PROMPTS[@]}" -gt 0 ]]; then
                neg_prompt="${NEG_PROMPTS[$i]}"
                if [[ -n "$neg_prompt" && "$neg_prompt" != "None" ]]; then
                    neg_arg=(--negative_prompt "$neg_prompt")
                    log_msg "Using passed negative prompt for bench=$bench lang=$lang model=$model_name."
                fi
            fi

            FREE_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

            echo "=========================================================="
            echo "Running bench: $bench"
            echo "Running inference script: $infer_script"
            echo "Running model: $model_name"
            echo "Using LoRA: $lora_path"
            echo "Using port: $FREE_PORT"
            echo "Using nproc_per_node: $NPROC_PER_NODE"
            echo "Current language: $lang"
            echo "=========================================================="

            case "$bench" in
                polytext|PolyText-Bench)
                    gallery_output_dir="$EVAL_ROOT/PolyText-Bench/eval_results/$BASE_MODEL_NAME"
                    ;;
                oneig)
                    gallery_output_dir="$EVAL_ROOT/OneIG-Benchmark/images/text/$BASE_MODEL_NAME"
                    ;;
                CVTG2K|cvtg2k)
                    gallery_output_dir="$EVAL_ROOT/TextCrafter/CVTG-2K/eval_results/$BASE_MODEL_NAME"
                    ;;
                longtext)
                    gallery_output_dir="$EVAL_ROOT/X-Omni/eval_results/$BASE_MODEL_NAME"
                    ;;
                gentexteval|GenTextEval)
                    gallery_output_dir="$EVAL_ROOT/GenTextEval/eval_results/$BASE_MODEL_NAME"
                    ;;
                *)
                    echo "Error: Unsupported bench: $bench" >&2
                    exit 1
                    ;;
            esac

            cmd=(torchrun --nnodes=1 --node-rank=0 --nproc_per_node="$NPROC_PER_NODE" --master-addr 127.0.0.1 --master-port "$FREE_PORT" "$infer_script")
            if bench_needs_mode "$bench"; then
                cmd+=(--mode "$lang")
            fi
            cmd+=(--model_name "$model_name" --model_path "$MODEL_PATH" --gallery_output_dir "$gallery_output_dir" "${lora_arg[@]}" "${neg_arg[@]}" --width "$WIDTH" --height "$HEIGHT")
            if [[ "$bench" == "polytext" || "$bench" == "PolyText-Bench" ]]; then
                export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
                cmd+=(--model-family flux --prompts "${POLYTEXT_PROMPTS:-$EVAL_ROOT/PolyText-Bench/data}" --repeats "${POLYTEXT_REPEATS:-4}")
                if [[ "${POLYTEXT_DRY_RUN:-0}" == "1" ]]; then cmd+=(--dry-run); fi
            fi

            log_msg "Start task. bench=$bench lang=$lang base_model=$BASE_MODEL_NAME model=$model_name output=$gallery_output_dir lora=$lora_path port=$FREE_PORT script=$infer_script"
            # Do not `|| true` before reading PIPESTATUS: that overwrites
            # PIPESTATUS[0] with `true`'s 0 and reports a crashed torchrun as
            # "Finish task successfully".
            task_status=0
            if [[ -n "$LOG_FILE" ]]; then
                "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
                task_status=${PIPESTATUS[0]}
            else
                "${cmd[@]}"
                task_status=$?
            fi

            if [[ "$task_status" -eq 0 ]]; then
                log_msg "Finish task successfully. bench=$bench lang=$lang model=$model_name"
            else
                log_msg "ERROR: Task exited with code $task_status. bench=$bench lang=$lang model=$model_name lora=$lora_path"
                log_msg "Continuing to next task..."
                overall_status=1
            fi
        done

        if ! bench_needs_mode "$bench"; then
            break
        fi
    done
done

if [[ "$overall_status" -ne 0 ]]; then
    log_msg "Flux run finished with one or more failed tasks."
    exit "$overall_status"
fi
log_msg "Flux run finished successfully."
