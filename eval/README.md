# Evaluation

Evaluate FLUX.1-dev and Qwen-Image-2512 checkpoints on five visual text
generation benchmarks. This guide covers environment setup, benchmark
downloads, image generation, scoring and result summaries.

| Bench | Languages | Scorer | Metrics |
|---|---|---|---|
| [LongText](https://huggingface.co/datasets/X-Omni/LongText-Bench) (X-Omni TextBench) | EN, ZH | Qwen2.5-VL-7B OCR | TextScore |
| [OneIG-Bench](https://oneig-bench.github.io/) (text subset) | EN, ZH | Qwen2.5-VL-7B OCR | TextScore, ED, CR, WAC |
| [CVTG-2K](https://github.com/NJU-PCALab/TextCrafter) | EN | PaddleOCR 2.10 | WordAcc, NED |
| [GenTextEval](https://github.com/CIawevy/TextPecker) | EN, ZH | TextPecker | Qua, Sem |
| [PolyText-Bench](https://huggingface.co/datasets/SujieHu/PolyText-Bench) | EN, ZH | TextDART | CS, CG, OG |

Metric definitions are in [`scripts/METRICS.md`](scripts/METRICS.md).

## Repository files

The following directories are included in this repository:

```text
eval/
├── scripts/           # configure models, generate images, score and summarize
├── models/            # FLUX and Qwen-Image generation entry points
├── GenTextEval/       # prompts and TextPecker scoring
├── PolyText-Bench/    # prompts, generation and TextDART scoring
└── integrations/      # connect benchmark tools to generated image outputs
```

LongText, OneIG-Bench and CVTG-2K tools are downloaded during setup, as described
below. Their directories appear under `eval/` after the download completes.

## 1. Install the evaluation environment

Start from the `TextDART_SCANNER` project root. Conda, Git and network access
to GitHub and Hugging Face are required. Image generation and OCR scoring
require CUDA GPUs.

```bash
cd eval
conda create -n textdart_eval python=3.10 -y
conda activate textdart_eval
bash install_all.sh
```

The remaining commands in this guide run from `TextDART_SCANNER/eval`, unless
stated otherwise.

## 2. Download benchmark tools and data

```bash
bash setup_third_party.sh all
```

This downloads the official benchmark tools at fixed commits so repeated
installations use the same source code:

```text
eval/
├── X-Omni/            # LongText tools and prompts
├── OneIG-Benchmark/   # OneIG evaluation tools
└── TextCrafter/       # CVTG-2K evaluation tools
```

| Benchmark | Official repository |
|---|---|
| LongText | [X-Omni](https://github.com/X-Omni-Team/X-Omni) |
| OneIG-Bench | [OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark) |
| CVTG-2K | [TextCrafter](https://github.com/NJU-PCALab/TextCrafter) |

To download one benchmark, run `bash setup_third_party.sh xomni`,
`bash setup_third_party.sh oneig`, or `bash setup_third_party.sh textcrafter`.
The launchers verify the downloaded source before starting GPU jobs. To check
it manually without downloading, run `bash setup_third_party.sh all --check`.
See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for component terms.

LongText prompts are included in `X-Omni/textbench/`. The environment installed
in step 1 provides the dependencies for LongText scoring.

The OneIG-Bench and CVTG-2K datasets require separate downloads. From `eval/`,
download the official prompt files before generation:

```bash
hf download OneIG-Bench/OneIG-Bench --repo-type dataset \
  --include 'OneIG-Bench*.json' --local-dir OneIG-Benchmark/OneIG-Bench
hf download dnkdnk/CVTG-2K CVTG-2K.zip --repo-type dataset --local-dir TextCrafter
unzip -n TextCrafter/CVTG-2K.zip -d TextCrafter
```

The expected paths are `OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json`,
`OneIG-Benchmark/OneIG-Bench/OneIG-Bench-ZH.json`, and
`TextCrafter/CVTG-2K/{CVTG,CVTG-Style}/{2,3,4,5}.json`.

### CVTG-2K scoring environment

Install PaddleOCR 2.10 in a separate environment for CVTG-2K scoring:

```bash
conda create -n textcrafter_eval python=3.10 -y
conda activate textcrafter_eval
pip install -r TextCrafter/TextCrafter_Eval/requirements.txt
pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==2.10.0 "opencv-python<5" "albumentations<2" "albucore<0.0.17" Levenshtein openai-clip
```

The CVTG evaluator downloads PaddleOCR and CLIP weights on first use,
using their library-specific caches. Set `HF_HOME` for its Hugging Face and
OpenCLIP cache. The optional `scripts/download_eval_models.py` creates a separate
archive of weights for manual use.

Qwen2.5-VL-7B-Instruct is pulled from Hugging Face on first use. LongText uses
the official model ID and the Hugging Face cache (`HF_HOME`). For OneIG,
`QWEN25VL_PATH` can select a local copy.

## 3. Start the scoring servers

GenTextEval and PolyText-Bench query OpenAI-compatible vLLM servers.
Follow the [TextDART deployment guide](../TextDART/train/README.md#2-quick-start-run-the-model)
to start TextDART. Start TextPecker in a separate environment with vLLM installed.
Use separate ports if both servers run on the same host.

| Resource | Hugging Face |
| --- | --- |
| TextDART | [SujieHu/TextDART-8B-Qwen3VL](https://huggingface.co/SujieHu/TextDART-8B-Qwen3VL) |
| TextPecker | [CIawevy/TextPecker-8B-Qwen3VL](https://huggingface.co/CIawevy/TextPecker-8B-Qwen3VL) |

Download and deploy the released TextDART checkpoint:

```bash
hf download SujieHu/TextDART-8B-Qwen3VL --local-dir TextDART-8B-Qwen3VL
bash ../TextDART/train/deploy_server.sh ./TextDART-8B-Qwen3VL 1925 0
```

```bash
# TextPecker, used by GenTextEval
vllm serve <path/to/TextPecker> --served-model-name TextPecker --port 1925
```

Check that each server responds:

```bash
# TextPecker
python GenTextEval/demo_server.py --host <host> --port 1925 --image <any_image_with_text.png>
```

## 4. Configure models

Edit [`scripts/config.sh`](scripts/config.sh): each base model has a weights path, a list of languages and a list of rows to evaluate. A row is `"None"` (base model), a LoRA checkpoint directory, or `"NAME=/path/to/checkpoint"`. Results are stored under `<bench>/.../{base}/{name}_{LANG}`.

## 5. Generate images, score and summarize

Return to the `textdart_eval` environment and run from `TextDART_SCANNER/eval`:

```bash
conda activate textdart_eval
bash scripts/gen.sh                       # generate images for all benches

TEXTPECKER_VLLM_HOST=<host> \
TEXTDART_SERVER_URLS=http://<textdart-host>:<textdart-port> \
CVTG_PYTHON=/path/to/textcrafter_eval/bin/python \
bash scripts/eval_metrics.sh              # score every complete, not-yet-scored cell

bash scripts/cal_metrics.sh               # Markdown table -> eval/metric_summary.md
```

PolyText-Bench is self-contained in `PolyText-Bench/`: `generate.py` generates
the benchmark images, `evaluate.py` performs resumable TextDART scoring, and
`summarize.py` validates the score set and writes `summary.json`. The commands
above invoke these three entry points automatically. To run only PolyText-Bench:

```bash
torchrun --nproc_per_node=8 PolyText-Bench/generate.py \
  --model-family qwen \
  --mode EN \
  --model_name qwen_image \
  --model_path Qwen/Qwen-Image-2512 \
  --gallery_output_dir PolyText-Bench/eval_results

PYTHONPATH="../src${PYTHONPATH:+:$PYTHONPATH}" \
python PolyText-Bench/evaluate.py \
  --mode EN \
  --image-dir PolyText-Bench/eval_results/qwen_image_EN \
  --output-dir PolyText-Bench/eval_metrics/qwen_image_EN \
  --server-urls http://127.0.0.1:1925

python PolyText-Bench/summarize.py \
  --mode EN \
  --scores-dir PolyText-Bench/eval_metrics/qwen_image_EN
```

Useful variables:

| Variable | Used by | Meaning |
|---|---|---|
| `GEN_BENCH` | `gen.sh` | subset of `polytext,longtext,oneig,CVTG2K,gentexteval` |
| `GEN_BASE_MODELS` | `gen.sh` | subset of `BASE_MODELS` in `config.sh` |
| `GEN_NPROC` | `gen.sh` | GPUs per generation job (default 8) |
| `POLYTEXT_PROMPTS` | `gen.sh` | PolyText prompt directory (default `PolyText-Bench/data`) |
| `EVAL_BENCH` | `eval_metrics.sh`, `cal_metrics.sh` | subset of `longtext,oneig,cvtg,gentexteval,polytext` |
| `EVAL_LANG` | `eval_metrics.sh` | optional language filter (`EN` or `ZH`) |
| `GPUS` | `eval_metrics.sh` | GPU ids for parallel scoring jobs (default `0 1 2 3 4 5 6 7`) |
| `TEXTPECKER_PORT` | `eval_metrics.sh` | TextPecker server port (default 1925) |

Generation skips completed images, and scoring skips complete score files.
An interrupted LongText scoring job reruns the entire model/language cell;
its official evaluator does not resume individual OCR records. Each job uses
one GPU and summarizes the newly written `results_chunk0.jsonl`.
Use a fresh model output name for an independent evaluation; existing complete
scores under the same name are reused. Logs go to `scripts/logs/`.
