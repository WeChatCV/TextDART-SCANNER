# Multi-Node Training with Flow-Factory

This directory contains the launchers for the two 4-node, 32-GPU TextDART
SCANNER recipes. First install the [SCANNER environment](../README.md#part-ii-generative-model-rl-scanner),
deploy the [TextDART reward server](../TextDART/train/README.md#2-quick-start-run-the-model),
and configure its address in the [training YAML](../examples/README.md).
Run the commands below from the `TextDART_SCANNER` project root.

## Required scheduler variables

Set a master address and port, a node rank, the number of nodes, and the number
of GPUs per node. The launcher recognizes these aliases:

| Value | Accepted variables |
|---|---|
| Master address | `MASTER_IP`, `MASTER_ADDR`, `CHIEF_IP` |
| Master port | `MASTER_PORT` |
| Node rank | `MACHINE_RANK`, `NODE_RANK`, `INDEX`, scheduler-level `RANK` |
| Number of nodes | `NUM_MACHINES`, `NUM_NODES`, `HOST_NUM`, scheduler-level `WORLD_SIZE` |
| GPUs per node | `GPUS_PER_NODE`, `HOST_GPU_NUM` |

`launch_multinode.sh` converts scheduler-level `RANK/WORLD_SIZE` to
`MACHINE_RANK/NUM_MACHINES` and clears the former before calling `ff-train`.
This distinction is required on systems where the login shell describes pods
rather than individual GPU workers.

## Launch

Activate the SCANNER environment on every node and set its rendezvous variables.
Use the same master address and port on all nodes, and a unique node rank from
0 to 3:

```bash
conda activate scanner
export MASTER_ADDR=<node-0-address>
export MASTER_PORT=29500
export NODE_RANK=<0-to-3>
```

Choose the launcher for your backbone and run it on all four nodes:

```bash
# FLUX.1-dev
bash multinode_examples/grpo/flux1/textdart_scanner_4nodes.sh

# Qwen-Image-2512
bash multinode_examples/grpo/qwenimage2512/textdart_scanner_4nodes.sh
```

Both wrappers default to 4 nodes and 8 GPUs per node. They call the shared
`launch_multinode.sh`, which starts Accelerate and writes one log per node.
Use `TRAIN_CONFIG` or `TextDART_SCANNER_ROOT` only when the default repository
layout does not apply.
