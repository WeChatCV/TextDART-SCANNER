#!/bin/bash
# Resolve REWARD_GROUP entries (see config.sh) to result names and LoRA paths.
#
#   "None" / ""            -> name "None", path "None"
#   "NAME=/path/to/ckpt"   -> name NAME, path /path/to/ckpt
#   "/path/to/ckpt"        -> name derived from the path, path unchanged
#
# Derived name: ".../<run_dir>/checkpoints/checkpoint-<N>" -> "<run_dir>_<N>";
# any other dir -> its basename. Spaces/parentheses are dropped and commas
# become "x" because names are passed as comma-separated lists.

sanitize_model_token() {
    printf '%s' "$1" | tr -d ' ()' | tr ',' 'x'
}

model_name_from_lora_dir() {
    local lora_dir="${1%/}"
    if [[ -z "$lora_dir" ]]; then
        echo "Error: cannot derive model name from an empty LoRA path." >&2
        exit 1
    fi
    case "$lora_dir" in
        *.safetensors|*.bin) lora_dir="$(dirname "$lora_dir")" ;;
    esac
    local leaf parent
    leaf="$(basename "$lora_dir")"
    parent="$(basename "$(dirname "$lora_dir")")"
    if [[ "$leaf" =~ ^checkpoint-([0-9]+)$ && "$parent" == "checkpoints" ]]; then
        sanitize_model_token "$(basename "$(dirname "$(dirname "$lora_dir")")")_${BASH_REMATCH[1]}"
    else
        sanitize_model_token "$leaf"
    fi
}

reward_entry_name() {
    local entry="$1"
    if [[ -z "$entry" || "$entry" == "None" ]]; then
        printf 'None'
    elif [[ "$entry" == *=* ]]; then
        sanitize_model_token "${entry%%=*}"
    else
        model_name_from_lora_dir "$entry"
    fi
}

reward_entry_path() {
    local entry="$1"
    if [[ -z "$entry" || "$entry" == "None" ]]; then
        printf 'None'
    elif [[ "$entry" == *=* ]]; then
        printf '%s' "${entry#*=}"
    else
        printf '%s' "$entry"
    fi
}

# Result directory name for one (base model, reward entry, language) cell.
build_model_name() {
    local base_model="$1" entry="$2" lang="$3"
    local name
    name="$(reward_entry_name "$entry")"
    if [[ "$name" == "None" ]]; then
        printf '%s_%s' "$base_model" "$lang"
    else
        printf '%s_%s' "$name" "$lang"
    fi
}

# Fail fast when BASE_MODELS[i] has no matching *_<i> config arrays.
require_base_config() {
    local idx="$1" base_model="$2" var
    for var in MODEL_PATH LANGS_GROUP REWARD_GROUP; do
        if ! declare -p "${var}_${idx}" >/dev/null 2>&1; then
            echo "Error: config.sh is missing ${var}_${idx} for BASE_MODELS[${idx}]=\"${base_model}\"." >&2
            exit 1
        fi
    done
}
