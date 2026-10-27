#!/bin/bash
# Download the TextDART-123K train and test splits from the Hugging Face Hub.
#
# Usage:
#   bash scripts/download_dataset.sh [LOCAL_DIR]
#
# Environment:
#   HF_TOKEN      Hugging Face access token (only needed for gated/private repos)
#   HF_ENDPOINT   Hub endpoint, e.g. https://hf-mirror.com (default: https://huggingface.co)
#   HFD_TOOL      aria2c (default, faster) or wget
#   HFD_JOBS      parallel download threads per file (default: 8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXTDART_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ID="SujieHu/TextDART-123K"
LOCAL_DIR="${1:-${TEXTDART_ROOT}/data/TextDART-123K}"
HFD_TOOL="${HFD_TOOL:-aria2c}"
HFD_JOBS="${HFD_JOBS:-8}"

TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
    TOKEN_ARGS=(--hf_token "${HF_TOKEN}")
fi

echo "Downloading dataset ${REPO_ID} -> ${LOCAL_DIR}"
bash "${SCRIPT_DIR}/hfd.sh" "${REPO_ID}" --dataset \
    --local-dir "${LOCAL_DIR}" \
    --tool "${HFD_TOOL}" -x "${HFD_JOBS}" \
    "${TOKEN_ARGS[@]}"
