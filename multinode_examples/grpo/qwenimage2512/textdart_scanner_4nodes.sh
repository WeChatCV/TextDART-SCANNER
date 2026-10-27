#!/usr/bin/env bash
# Qwen-Image-2512 + TextDART, SCANNER, 4 nodes x 8 GPUs.
#
# Run the same command on every node. The cluster scheduler must export the
# variables listed in multinode_examples/launch_multinode.sh (MASTER_ADDR,
# MASTER_PORT, NODE_RANK, ...). NUM_MACHINES / GPUS_PER_NODE default to 4 / 8.
#
# Usage:
#   bash multinode_examples/grpo/qwenimage2512/textdart_scanner_4nodes.sh [extra ff-train args...]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TextDART_SCANNER_ROOT="${TextDART_SCANNER_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
TRAIN_CONFIG="${TRAIN_CONFIG:-examples/grpo/lora/qwen_image_2512/textdart_scanner.yaml}"

export NUM_MACHINES="${NUM_MACHINES:-4}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

cd "${TextDART_SCANNER_ROOT}"
bash multinode_examples/launch_multinode.sh "${TRAIN_CONFIG}" "$@"
