#!/bin/bash
# Download a model from the Hugging Face Hub.
#
# Usage:
#   bash scripts/download_model.sh [REPO_ID] [LOCAL_DIR]
#
#   REPO_ID    default: SujieHu/TextDART-8B-Qwen3VL.
#              Use Qwen/Qwen3-VL-8B-Instruct to fetch the base model for training.
#   LOCAL_DIR  default: <TextDART>/models/<repo name>
#
# Environment:
#   HF_TOKEN      Hugging Face access token (only needed for gated/private repos)
#   HF_ENDPOINT   Hub endpoint, e.g. https://hf-mirror.com (default: https://huggingface.co)
#   HFD_TOOL      aria2c (default, faster) or wget
#   HFD_JOBS      parallel download threads per file (default: 8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXTDART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID="${1:-SujieHu/TextDART-8B-Qwen3VL}"
LOCAL_DIR="${2:-${TEXTDART_ROOT}/models/${REPO_ID##*/}}"
HFD_TOOL="${HFD_TOOL:-aria2c}"
HFD_JOBS="${HFD_JOBS:-8}"

TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN_ARGS=(--hf_token "${HF_TOKEN}")
fi

echo "Downloading model ${REPO_ID} -> ${LOCAL_DIR}"
bash "${SCRIPT_DIR}/hfd.sh" "${REPO_ID}" \
    --local-dir "${LOCAL_DIR}" \
    --tool "${HFD_TOOL}" -x "${HFD_JOBS}" \
    "${TOKEN_ARGS[@]}"
