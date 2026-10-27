#!/bin/bash
# multinode_examples/launch_multinode.sh
#
# Multi-node launch script for Flow-Factory.
# Now simply delegates to `ff-train`, which auto-detects multi-node env vars.
#
# Required environment variables (injected by cluster scheduler):
#   MASTER_IP / MASTER_ADDR / CHIEF_IP   - Master node IP
#   MASTER_PORT                          - Master node port
#   MACHINE_RANK / NODE_RANK / INDEX     - Current node rank
#   NUM_MACHINES / NUM_NODES / HOST_NUM  - Total number of nodes
#   GPUS_PER_NODE / HOST_GPU_NUM         - GPUs per node
#
# Usage:
#   bash launch_multinode.sh <train_config.yaml> [extra_args...]
#
# Example:
#   bash launch_multinode.sh examples/grpo/lora/flux1/textdart_scanner.yaml
#
# If your cluster uses non-standard variable names, map them before calling:
#   export MASTER_IP=${MY_CUSTOM_MASTER_IP}
#   export NUM_MACHINES=${MY_CUSTOM_NODE_COUNT}

set -euo pipefail

TRAIN_CONFIG=${1:?"Usage: bash launch_multinode.sh <train_config.yaml> [extra_args...]"}
shift  # Remove first arg so "$@" contains only extra args

SCHEDULER_MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK:-${INDEX:-${RANK:-}}}}"
SCHEDULER_NUM_MACHINES="${NUM_MACHINES:-${NUM_NODES:-${HOST_NUM:-${WORLD_SIZE:-}}}}"
if [[ -z "${SCHEDULER_MACHINE_RANK}" ]]; then
    echo "Missing node rank. Set MACHINE_RANK, NODE_RANK, INDEX, or scheduler RANK." >&2
    exit 2
fi
if [[ -z "${SCHEDULER_NUM_MACHINES}" ]]; then
    echo "Missing node count. Set NUM_MACHINES, NUM_NODES, HOST_NUM, or scheduler WORLD_SIZE." >&2
    exit 2
fi

export MACHINE_RANK="${SCHEDULER_MACHINE_RANK}"
export NUM_MACHINES="${SCHEDULER_NUM_MACHINES}"

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK

echo "=== Flow-Factory Multi-Node Launch ==="
echo "Master:         ${MASTER_IP:-${MASTER_ADDR:-${CHIEF_IP:-unknown}}}:${MASTER_PORT:-unknown}"
echo "Num nodes:      ${NUM_MACHINES}"
echo "GPUs per node:  ${GPUS_PER_NODE:-${HOST_GPU_NUM:-unknown}}"
echo "Node rank:      ${MACHINE_RANK}"
echo "Train config:   ${TRAIN_CONFIG}"
echo ""

# ff-train auto-detects multi-node env vars and builds the accelerate launch command
ff-train "${TRAIN_CONFIG}" "$@" 2>&1 | tee "train_node_${MACHINE_RANK}.log"
