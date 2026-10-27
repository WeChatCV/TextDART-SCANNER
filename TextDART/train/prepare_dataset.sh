#!/bin/bash
# Build separate cached train/test sets from a local copy of SujieHu/TextDART-123K.
# CPU only; run once before train_textdart_qwen3vl.sh.
#
# Single machine (all shards here, then merge):
#   bash prepare_dataset.sh
#
# Several machines sharing one filesystem (e.g. 4 x 8 workers): on machine k run
#   NUM_SHARDS=32 NUM_WORKERS=8 SHARD_START=$((8 * k)) SKIP_MERGE=1 bash prepare_dataset.sh
# then, once every machine has finished, on any one of them:
#   NUM_SHARDS=32 MERGE_ONLY=1 bash prepare_dataset.sh
#
# Environment:
#   DATA_DIR      local dataset dir   (default: <TextDART>/data/TextDART-123K, see scripts/download_dataset.sh)
#   CACHED_ROOT   output root         (default: <TextDART>/data/cached/TextDART-123K_qwen3vl); training reads train/ and validation reads test/
#   BASE_MODEL    processor source    (default: Qwen/Qwen3-VL-8B-Instruct)
#   NUM_SHARDS    total shards        (default: NUM_WORKERS)
#   NUM_WORKERS   shards run in parallel on this machine (default: 8)
#   SHARD_START   first shard index handled by this machine (default: 0)
#   BATCH_SIZE    rows per map batch  (default: 32)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXTDART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${TEXTDART_ROOT}/data/TextDART-123K}"
CACHED_ROOT="${CACHED_ROOT:-${TEXTDART_ROOT}/data/cached/TextDART-123K_qwen3vl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SHARDS="${NUM_SHARDS:-${NUM_WORKERS}}"
SHARD_START="${SHARD_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SKIP_MERGE="${SKIP_MERGE:-0}"
MERGE_ONLY="${MERGE_ONLY:-0}"

# Must equal IMAGE_MAX_TOKEN_NUM in train_textdart_qwen3vl.sh: the cached lengths
# are only valid for the image token budget they were computed with.
export IMAGE_MAX_TOKEN_NUM=2304

LOG_DIR="${CACHED_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

for SPLIT in train test; do
    if [ "${MERGE_ONLY}" != "1" ]; then
        if [ ! -d "${DATA_DIR}" ]; then
            echo "ERROR: dataset dir not found: ${DATA_DIR}. Run scripts/download_dataset.sh first or set DATA_DIR." >&2
            exit 1
        fi
        SHARD_END=$(( SHARD_START + NUM_WORKERS ))
        if [ "${SHARD_END}" -gt "${NUM_SHARDS}" ]; then
            SHARD_END="${NUM_SHARDS}"
        fi
        echo "Preparing ${SPLIT} shards [${SHARD_START}, ${SHARD_END}) of ${NUM_SHARDS}: ${DATA_DIR} -> ${CACHED_ROOT}"
        mkdir -p "${LOG_DIR}/${SPLIT}"
        PIDS=()
        for (( i = SHARD_START; i < SHARD_END; i++ )); do
            python "${SCRIPT_DIR}/prepare_dataset.py" \
                --data_dir "${DATA_DIR}" \
                --split "${SPLIT}" \
                --output_dir "${CACHED_ROOT}" \
                --model "${BASE_MODEL}" \
                --num_shards "${NUM_SHARDS}" \
                --shard_index "${i}" \
                --batch_size "${BATCH_SIZE}" \
                > "${LOG_DIR}/${SPLIT}/shard_$(printf '%03d' "${i}").log" 2>&1 &
            PIDS+=($!)
        done
        FAILED=0
        for pid in "${PIDS[@]}"; do
            wait "${pid}" || FAILED=$(( FAILED + 1 ))
        done
        if [ "${FAILED}" -gt 0 ]; then
            echo "ERROR: ${FAILED} shard(s) failed; see ${LOG_DIR}/${SPLIT}/shard_*.log. Re-running skips finished shards." >&2
            exit 1
        fi
    fi

    if [ "${SKIP_MERGE}" != "1" ]; then
        python "${SCRIPT_DIR}/prepare_dataset.py" \
            --output_dir "${CACHED_ROOT}" \
            --split "${SPLIT}" \
            --num_shards "${NUM_SHARDS}" \
            --merge
        echo "Prepared ${CACHED_ROOT}/${SPLIT}"
    fi
done
