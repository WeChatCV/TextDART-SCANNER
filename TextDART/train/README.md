# TextDART: Training, Deployment and Demo

TextDART is a Qwen3-VL-8B reward model that locates every piece of rendered text in a generated image as a quadrilateral and transcribes it, marking structurally broken characters with `<#>`/`<###>`/`<#####>` tokens. Training and serving are built on [ms-swift](https://github.com/modelscope/ms-swift).

| Resource | Hugging Face |
| --- | --- |
| Model | [SujieHu/TextDART-8B-Qwen3VL](https://huggingface.co/SujieHu/TextDART-8B-Qwen3VL) |
| Training and test data | [SujieHu/TextDART-123K](https://huggingface.co/datasets/SujieHu/TextDART-123K) |
| Base model | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |

```
TextDART/
├── assets/                              # test images for the demo
├── system_prompt.txt                    # system prompt used for training and inference
├── scripts/
│   ├── download_dataset.sh              # SujieHu/TextDART-123K -> data/TextDART-123K
│   ├── download_model.sh                # any HF model -> models/<name>
│   └── hfd.sh                           # Hugging Face downloader (aria2c / wget)
└── train/
    ├── install_all.sh
    ├── deploy_server.sh                 # OpenAI-compatible vLLM server
    ├── infer_demo.py                    # query the server and draw the result
    ├── prepare_dataset.sh / .py         # parquet -> ms-swift cached dataset
    └── train_textdart_qwen3vl.sh        # full-parameter SFT
```

## 1. Environment

Start from the `TextDART_SCANNER` project root. The commands enter
`TextDART/train`; run subsequent commands from that directory.

```bash
conda create -n TextDART python=3.11 -y
conda activate TextDART
cd TextDART/train
bash install_all.sh
```

The default environment supports serving and the image demo without the optional
FlashAttention extension. For the SFT launcher, install it explicitly:

```bash
INSTALL_FLASH_ATTN=1 bash install_all.sh
```

## 2. Quick Start: Run the Model

Start the server on GPUs with enough memory to hold the model and inference
cache. Each listed GPU runs a full replica:

```bash
# bash deploy_server.sh [MODEL] [PORT] [GPUS]
bash deploy_server.sh SujieHu/TextDART-8B-Qwen3VL 1925 0,1,2,3,4,5,6,7
```

`MODEL` can also be a local directory (e.g. one fetched with `bash ../scripts/download_model.sh`). Once `curl http://127.0.0.1:1925/v1/models` lists `TextDART`, run the demo on the images in `assets/`:

```bash
python infer_demo.py --server-url http://127.0.0.1:1925
```

It prints every detected region and writes `TextDART/demo_results/results.json` plus `<image>_vis.png` with the quadrilaterals and transcriptions drawn on the original image.

Quads are returned on a 0-1000 grid normalized to the image size. `quad_orig` in `results.json` maps them back to pixels of the original image.

Configure this server address in the SCANNER reward settings and
[PolyText-Bench](https://huggingface.co/datasets/SujieHu/PolyText-Bench) evaluation
commands. See the [training configurations](../../examples/README.md) and
[evaluation guide](../../eval/README.md).

## 3. Training

### 3.1 Download

```bash
bash ../scripts/download_dataset.sh                              # -> TextDART/data/TextDART-123K
bash ../scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct      # optional: local copy of the base model
```

Set `HF_TOKEN` for gated repos and `HF_ENDPOINT` to use a mirror.

The dataset contains `textdart_train_*.parquet` files for training and
`textdart_test_*.parquet` files for evaluation.

### 3.2 Build the cached dataset

```bash
bash prepare_dataset.sh
```

Preparation writes separate `train/` and `test/` caches under
`TextDART/data/cached/TextDART-123K_qwen3vl/`, each with a split manifest.
Set `DATA_DIR` to use a local dataset directory. Training requires both caches
and their manifests; run preparation before launching training.

To spread the work over several machines that share a filesystem, give each machine its own slice of shards and merge once:

```bash
# on machine k = 0..3
NUM_SHARDS=32 NUM_WORKERS=8 SHARD_START=$((8 * k)) SKIP_MERGE=1 bash prepare_dataset.sh
# after all machines finish, on any one of them
NUM_SHARDS=32 MERGE_ONLY=1 bash prepare_dataset.sh
```

### 3.3 Train

```bash
# single node, 8 GPUs
bash train_textdart_qwen3vl.sh

# 4 nodes x 8 GPUs; run on every node
NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node0 ip> MASTER_PORT=29500 bash train_textdart_qwen3vl.sh
```
Logs go to `TextDART/output/TextDART-8B-Qwen3VL/logs/`. Set `REPORT_TO="tensorboard wandb"` (with `WANDB_API_KEY`) to log to Weights & Biases.

Only the `train` cache is used for optimization. The `test` cache is used for
validation every 500 optimizer steps, reporting evaluation loss. No random
validation split is taken from training data. Set `CACHED_DATASET` and
`CACHED_VAL_DATASET` to override the train and test cache directories together.
Samples exceeding the 8192-token limit are excluded from both splits.
