# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OneIG text score computed on recorded 2x2 tiles.

Same methodology as ``scripts.text.text_score`` (Qwen2.5-VL OCR on each tile,
then ED / CR / WAC / "text score" with identical preprocessing and
aggregation), but tiles come from ``oneig_tiles.ensure_oneig_tiles`` so they
are split once and reused. Output is a one-row CSV (index = model name,
columns ``ED, CR, WAC, "text score"``).
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve --gpu BEFORE importing the torch-backed inference module: that module
# initializes CUDA at import time (``torch.cuda.empty_cache()`` runs before its
# own hard-coded ``CUDA_VISIBLE_DEVICES="0"``), so the device must be pinned here
# for the model to land on the GPU assigned by the caller.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import pandas as pd
from tqdm import tqdm

_INTEGRATION_DIR = Path(__file__).resolve().parent
_EVAL_ROOT = _INTEGRATION_DIR.parents[1]
_ONEIG_ROOT = _EVAL_ROOT / "OneIG-Benchmark"
if not _ONEIG_ROOT.is_dir():
    raise FileNotFoundError(
        f"OneIG-Benchmark is not installed at {_ONEIG_ROOT}. "
        "Run `bash eval/setup_third_party.sh oneig` from the repository root."
    )
sys.path.insert(0, str(_ONEIG_ROOT))
sys.path.insert(0, str(_INTEGRATION_DIR))
from oneig_tiles import ensure_oneig_tiles  # noqa: E402
from scripts.text.text_utils import (
    calculate_char_match_ratio,
    clean_and_remove_hallucinations,
    levenshtein_distance,
    preprocess_string,
)
from scripts.utils.inference import Qwen2_5VLBatchInferencer

# Ground-truth word count above which the OCR generation budget is raised, and
# the two budgets themselves -- mirror text_score.py exactly.
_LONG_TEXT_WORDS = 60
_SHORT_BUDGET = 128
_LONG_BUDGET = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="OneIG text score from recorded 2x2 tiles (split on first use)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["EN", "ZH"],
        help="Language mode; sets MAX_EDIT_DISTANCE (EN=100, ZH=50).",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        required=True,
        help="eval/ root used to resolve relative 'image' paths.",
    )
    parser.add_argument(
        "--base", type=str, required=True, help="Base model directory name (e.g. FLUX.1-dev)."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Row index written to the output CSV."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Destination CSV (created with the same schema as text_score.py).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Physical GPU id to pin via CUDA_VISIBLE_DEVICES "
        "(applied at import time, before CUDA initializes).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("QWEN25VL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
        help="Qwen2.5-VL checkpoint used for OCR.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of tiles per OCR batch.")
    return parser.parse_args()


def main():
    args = parse_args()

    records = ensure_oneig_tiles(args.eval_root, args.base, args.model_name, args.mode)
    if not records:
        raise ValueError(
            f"no usable OneIG tiles for {args.base}/{args.model_name} "
            f"(eval_root={args.eval_root!r})"
        )

    max_edit_distance = 100 if args.mode == "EN" else 50

    image_paths = []
    gts = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"expected dict tile record [{idx}], got {type(rec).__name__}")
        target = rec.get("target")
        if not isinstance(target, str) or target == "":
            raise ValueError(f"missing/invalid 'target' at tile [{idx}]: {target!r}")
        path = rec.get("image")
        if not isinstance(path, str) or not path:
            raise ValueError(f"missing/invalid 'image' at tile [{idx}]: {path!r}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OneIG tile vanished after ensure_oneig_tiles: {path}")
        image_paths.append(path)
        gts.append(target)

    print(f"Scoring {len(image_paths)} tiles for {args.model_name} [mode={args.mode}]")

    inferencer = Qwen2_5VLBatchInferencer(args.model_path)

    edit_distances = []
    completion_ratios = []
    match_word_counts = []
    gt_word_counts = []

    for start in tqdm(
        range(0, len(image_paths), args.batch_size),
        desc=f"OCR {args.model_name}",
        unit="batch",
    ):
        batch_imgs = image_paths[start : start + args.batch_size]
        batch_gts = gts[start : start + args.batch_size]
        # Match text_score.py: raise the token budget when any GT in the batch is long.
        max_new_tokens = (
            _LONG_BUDGET
            if any(len(g.split()) > _LONG_TEXT_WORDS for g in batch_gts)
            else _SHORT_BUDGET
        )
        ocr_results = inferencer.infer_ocr(batch_imgs, max_new_tokens)
        ocr_results = clean_and_remove_hallucinations(ocr_results)

        for ocr_text, gt in zip(ocr_results, batch_gts):
            ocr_pre = preprocess_string(ocr_text)
            gt_pre = preprocess_string(gt)
            edit_distance = levenshtein_distance(ocr_pre, gt_pre)
            completion_ratio = 1 if edit_distance == 0 else 0
            match_word_count, _, gt_word_count = calculate_char_match_ratio(gt_pre, ocr_pre)
            edit_distances.append(edit_distance)
            completion_ratios.append(completion_ratio)
            match_word_counts.append(match_word_count)
            gt_word_counts.append(gt_word_count)

    total_gt = sum(gt_word_counts)
    ed = sum(edit_distances) / len(edit_distances)
    cr = sum(completion_ratios) / len(completion_ratios)
    wac = sum(match_word_counts) / total_gt if total_gt > 0 else 0.0
    text_score = 1 - min(max_edit_distance, ed) * (1 - cr) * (1 - wac) / max_edit_distance

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(index=[args.model_name], columns=["ED", "CR", "WAC", "text score"])
    df.loc[args.model_name] = [ed, cr, wac, text_score]
    df.to_csv(args.output_csv)

    print(
        f"✅ {args.model_name}: ED={ed:.4f} CR={cr:.4f} WAC={wac:.4f} "
        f"text_score={text_score:.4f} -> {args.output_csv}"
    )


if __name__ == "__main__":
    main()
