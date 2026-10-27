#!/bin/bash
# Summarize the scores of every model in config.sh into one Markdown table.
#
# Usage:
#   bash scripts/cal_metrics.sh                    # writes eval/metric_summary.md
#   OUTPUT=/tmp/table.md bash scripts/cal_metrics.sh
#
# Env:
#   EVAL_BENCH  comma list (default: longtext,oneig,cvtg,gentexteval,polytext)
#   OUTPUT      Markdown output path (default: <eval>/metric_summary.md)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_BASE="${EVAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/lib_model_name.sh"

PAIRS=()
for i in "${!BASE_MODELS[@]}"; do
    BASE_MODEL="${BASE_MODELS[$i]}"
    require_base_config "$i" "$BASE_MODEL"
    declare -n LANGS="LANGS_GROUP_${i}"
    declare -n REWARDS="REWARD_GROUP_${i}"
    for LANG in "${LANGS[@]}"; do
        for ENTRY in "${REWARDS[@]}"; do
            PAIRS+=("${BASE_MODEL}/$(build_model_name "$BASE_MODEL" "$ENTRY" "$LANG")")
        done
    done
    unset -n LANGS REWARDS
done

python3 "${SCRIPT_DIR}/collect_metrics.py" --eval_root "$EVAL_BASE" table "${PAIRS[@]}" \
    --benches "${EVAL_BENCH:-longtext,oneig,cvtg,gentexteval,polytext}" \
    --output "${OUTPUT:-${EVAL_BASE}/metric_summary.md}"
