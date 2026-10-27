#!/bin/bash
# Generate benchmark images for every model in config.sh.
#
# Usage:
#   bash scripts/gen.sh
#   GEN_BENCH=polytext,oneig GEN_BASE_MODELS=FLUX.1-dev bash scripts/gen.sh
#
# Env:
#   GEN_BENCH        comma list of: polytext, longtext, oneig, CVTG2K, gentexteval
#   GEN_BASE_MODELS  comma list restricting BASE_MODELS (default: all)
#   GEN_NPROC        GPUs per torchrun job (default 8)
#   POLYTEXT_REPEATS images per PolyText prompt (default 4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${EVAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib_model_name.sh"

IFS=',' read -r -a BENCH <<< "${GEN_BENCH:-polytext,longtext,oneig,CVTG2K,gentexteval}"
NPROC_PER_NODE="${GEN_NPROC:-8}"
WIDTH=1024
HEIGHT=1024

for selected_bench in "${BENCH[@]}"; do
    case "$selected_bench" in
        longtext) repository=xomni ;;
        oneig) repository=oneig ;;
        cvtg|CVTG2K) repository=textcrafter ;;
        *) continue ;;
    esac
    EVAL_ROOT="${EVAL_ROOT}" bash "${SCRIPT_DIR}/../setup_third_party.sh" "$repository" --check || exit 1
done

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gen_$(date '+%Y%m%d_%H%M%S').log"
echo "Log file: $LOG_FILE"

join_by_comma() {
    local IFS=','
    echo "$*"
}

base_selected() {
    local base="$1" wanted
    [[ -z "${GEN_BASE_MODELS:-}" ]] && return 0
    IFS=',' read -r -a wanted <<< "$GEN_BASE_MODELS"
    for w in "${wanted[@]}"; do
        [[ "$w" == "$base" ]] && return 0
    done
    return 1
}

run_scripts_for() {
    local base="$1"
    case "$base" in
        FLUX*|flux*) echo "${EVAL_ROOT}/models/Flux/run_scripts.sh" ;;
        Qwen*|qwen*) echo "${EVAL_ROOT}/models/Qwenimage/run_scripts.sh" ;;
        *)
            echo "Error: no generation launcher for base model \"$base\"; expected a FLUX* or Qwen* name." >&2
            exit 1
            ;;
    esac
}

for i in "${!BASE_MODELS[@]}"; do
    BASE_MODEL="${BASE_MODELS[$i]}"
    base_selected "$BASE_MODEL" || continue
    require_base_config "$i" "$BASE_MODEL"
    declare -n MODEL_PATH="MODEL_PATH_${i}"
    declare -n LANGS="LANGS_GROUP_${i}"
    declare -n REWARDS="REWARD_GROUP_${i}"
    NEG_VAR="NEGATIVE_PROMPT_${i}"
    NEG_PROMPT="${!NEG_VAR:-}"
    if [[ "$NEG_PROMPT" == *,* ]]; then
        echo "Error: ${NEG_VAR} must not contain ASCII commas (run_scripts.sh splits on ','): $NEG_PROMPT" >&2
        exit 1
    fi

    MODEL_NAMES=()
    LORA_FILES=()
    NEG_PROMPTS=()
    for entry in "${REWARDS[@]}"; do
        name="$(reward_entry_name "$entry")"
        path="$(reward_entry_path "$entry")"
        if [[ "$name" == "None" ]]; then
            MODEL_NAMES+=("$BASE_MODEL")
            LORA_FILES+=("None")
        else
            if [[ ! -d "$path" ]]; then
                echo "Error: LoRA checkpoint directory does not exist for \"$name\": $path" >&2
                exit 1
            fi
            MODEL_NAMES+=("$name")
            LORA_FILES+=("${path%/}/adapter_model.safetensors")
        fi
        NEG_PROMPTS+=("${NEG_PROMPT:-None}")
    done

    echo "== $BASE_MODEL: models=${MODEL_NAMES[*]} langs=${LANGS[*]} bench=${BENCH[*]}"
    bash "$(run_scripts_for "$BASE_MODEL")" \
        --bench "$(join_by_comma "${BENCH[@]}")" \
        --nproc_per_node "$NPROC_PER_NODE" \
        --model_path "$MODEL_PATH" \
        --mode "$(join_by_comma "${LANGS[@]}")" \
        --model_name "$(join_by_comma "${MODEL_NAMES[@]}")" \
        --lora_path "$(join_by_comma "${LORA_FILES[@]}")" \
        --negative_prompt "$(join_by_comma "${NEG_PROMPTS[@]}")" \
        --base_model_name "$BASE_MODEL" \
        --eval_root "$EVAL_ROOT" \
        --log_file "$LOG_FILE" \
        --width "$WIDTH" \
        --height "$HEIGHT"
    unset -n MODEL_PATH LANGS REWARDS
done
