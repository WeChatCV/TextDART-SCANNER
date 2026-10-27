#!/bin/bash
# Environment for TextDART training, data preparation, deployment and the demo.
# Requires Python 3.10/3.11 and CUDA 12.x.
#
#   conda create -n TextDART python=3.11 -y
#   conda activate TextDART
#   bash install_all.sh
set -euo pipefail

# ms-swift commit used for training (4.2.0.dev0).
MS_SWIFT_COMMIT="201fce88718ea97bbfe2291ccea18a2ace7a7868"
MS_SWIFT_PACKAGE="${MS_SWIFT_PACKAGE:-ms-swift @ git+https://github.com/modelscope/ms-swift.git@${MS_SWIFT_COMMIT}}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

# Resolve together: installing ms-swift after vLLM can otherwise leave an
# incompatible Starlette/prometheus-fastapi-instrumentator combination.
python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 vllm==0.11.0 \
    "transformers==4.57.6" "trl==0.24.0" "peft==0.18.1" \
    "accelerate==1.13.0" "datasets==3.6.0" "deepspeed==0.17.6" \
    "qwen_vl_utils==0.0.14" "${MS_SWIFT_PACKAGE}" \
    wandb tensorboard requests pillow

# Serving works without the optional upstream FlashAttention extension.
# The SFT launcher uses flash_attn and requires this opt-in plus a CUDA toolkit.
# On older GLIBC systems, set FLASH_ATTENTION_FORCE_BUILD=TRUE to build locally.
if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    python -m pip install flash_attn==2.8.3 --no-build-isolation
    python -c 'import flash_attn'
elif [[ "${INSTALL_FLASH_ATTN}" != "0" ]]; then
    echo "INSTALL_FLASH_ATTN must be 0 or 1" >&2
    exit 2
fi
python -m pip check
