#!/bin/bash
# Serve TextDART behind an OpenAI-compatible endpoint (ms-swift + vLLM).
# The 8B model fits on one GPU, so every listed GPU runs a full replica
# (data parallel) for maximum throughput.
#
# Usage:
#   bash deploy_server.sh [MODEL] [PORT] [GPUS]
#     MODEL  HF repo id or local path   (default: SujieHu/TextDART-8B-Qwen3VL)
#     PORT   listening port             (default: 1925)
#     GPUS   CUDA_VISIBLE_DEVICES list  (default: 0,1,2,3,4,5,6,7)
#
# The served model name is "TextDART"; check with
#   curl http://127.0.0.1:1925/v1/models
set -euo pipefail

MODEL="${1:-SujieHu/TextDART-8B-Qwen3VL}"
PORT="${2:-1925}"
GPUS="${3:-0,1,2,3,4,5,6,7}"
GPU_MEM_UTILIZATION="${GPU_MEM_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
DP_SIZE="${#GPU_LIST[@]}"

echo "Deploying ${MODEL} on port ${PORT} | GPUs ${GPUS} | data parallel ${DP_SIZE}"

# IMAGE_MAX_TOKEN_NUM matches training. Clients may send smaller images
# (the Flow-Factory reward resizes rollouts to its configured token budget).
exec env CUDA_VISIBLE_DEVICES="${GPUS}" \
IMAGE_MAX_TOKEN_NUM=2304 \
python -m swift.cli.deploy \
    --model "${MODEL}" \
    --use_hf true \
    --model_type qwen3_vl \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization "${GPU_MEM_UTILIZATION}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_data_parallel_size "${DP_SIZE}" \
    --vllm_max_num_seqs "${MAX_NUM_SEQS}" \
    --vllm_max_model_len 32768 \
    --vllm_limit_mm_per_prompt '{"image": 1}' \
    --max_new_tokens 8192 \
    --served_model_name TextDART \
    --host 0.0.0.0 \
    --port "${PORT}"
