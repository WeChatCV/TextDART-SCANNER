#!/bin/bash
# Full-parameter SFT of Qwen3-VL-8B-Instruct into TextDART on TextDART-123K.
# TextDART-8B-Qwen3VL training (4 nodes x 8 GPUs);
# fewer GPUs keep the same global batch through gradient accumulation.
#
# Prerequisite: bash prepare_dataset.sh   (builds the train and test caches)
#
# Single node (extra arguments are forwarded to `swift sft`, e.g. --max_steps 10):
#   bash train_textdart_qwen3vl.sh
# Multi node: run on every node with the same MASTER_ADDR / MASTER_PORT
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node0 ip> MASTER_PORT=29500 bash train_textdart_qwen3vl.sh
#
# Environment:
#   BASE_MODEL      default: Qwen/Qwen3-VL-8B-Instruct
#   CACHED_DATASET  default: <TextDART>/data/cached/TextDART-123K_qwen3vl/train
#   CACHED_VAL_DATASET default: <TextDART>/data/cached/TextDART-123K_qwen3vl/test
#   OUTPUT_DIR      default: <TextDART>/output/TextDART-8B-Qwen3VL
#   NNODES / NODE_RANK / NPROC_PER_NODE / MASTER_ADDR / MASTER_PORT
#   REPORT_TO       default: tensorboard (e.g. "tensorboard wandb"; wandb reads WANDB_API_KEY)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXTDART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
CACHED_DATASET="${CACHED_DATASET:-${TEXTDART_ROOT}/data/cached/TextDART-123K_qwen3vl/train}"
CACHED_VAL_DATASET="${CACHED_VAL_DATASET:-${TEXTDART_ROOT}/data/cached/TextDART-123K_qwen3vl/test}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEXTDART_ROOT}/output/TextDART-8B-Qwen3VL}"
REPORT_TO="${REPORT_TO:-tensorboard}"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

MICRO_BS=2
GLOBAL_BS=128
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
if [ $(( GLOBAL_BS % (MICRO_BS * WORLD_SIZE) )) -ne 0 ]; then
    echo "ERROR: global batch ${GLOBAL_BS} is not divisible by micro batch ${MICRO_BS} x ${WORLD_SIZE} GPUs" >&2
    exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BS / (MICRO_BS * WORLD_SIZE) ))

if [ ! -d "${CACHED_DATASET}" ]; then
    echo "ERROR: cached dataset not found: ${CACHED_DATASET}. Run prepare_dataset.sh first or set CACHED_DATASET." >&2
    exit 1
fi

python "${SCRIPT_DIR}/validate_cached_splits.py" "${CACHED_DATASET}" "${CACHED_VAL_DATASET}"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
echo "nodes=${NNODES} rank=${NODE_RANK} gpus/node=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} " \
     "train=${CACHED_DATASET} validation=${CACHED_VAL_DATASET} output=${OUTPUT_DIR} log=${LOG_FILE}"

# IMAGE_MAX_TOKEN_NUM must match prepare_dataset.sh (cached lengths) and the
# deployment. Quads in the answers are 0-1000 normalized, so it does not affect labels.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=2304 \
swift sft \
    --model "${BASE_MODEL}" \
    --use_hf true \
    --cached_dataset "${CACHED_DATASET}" \
    --cached_val_dataset "${CACHED_VAL_DATASET}" \
    --split_dataset_ratio 0 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size "${MICRO_BS}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size "${MICRO_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --max_length 8192 \
    --truncation_strategy delete \
    --lazy_tokenize true \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --save_only_model true \
    --logging_steps 10 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to ${REPORT_TO} \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
