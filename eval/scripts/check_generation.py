#!/usr/bin/env python3
"""Check that generated images for one (bench, base model, model) cell are complete.

Every image must be non-empty and decodable. Expected counts:

    longtext     160 prompts x 4 images (``{id}_{1..4}.png``), EN and ZH
    oneig        200 prompts x one 2x2 grid (``{id}.webp``), EN and ZH
    cvtg         EN only; CVTG/{2,3,4,5} and CVTG-Style/{2,3,4,5} sub-folders
    gentexteval  EN 316 / ZH 417 prompts x 4 images (``{id}_{1..4}.png``)
    polytext     critical + optional prompts x 4 images
                 (``{pair_id}_{task_type}_{1..4}.png``), EN and ZH

Exit code 0 means complete, 1 means incomplete.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".png", ".webp", ".jpg", ".jpeg"}

IMAGE_DIRS = {
    "longtext": "X-Omni/eval_results/{base}/{model}",
    "oneig": "OneIG-Benchmark/images/text/{base}/{model}",
    "cvtg": "TextCrafter/CVTG-2K/eval_results/{base}/{model}",
    "gentexteval": "GenTextEval/eval_results/{base}/{model}",
    "polytext": "PolyText-Bench/eval_results/{base}/{model}",
}

EXPECTED_PROMPTS = {
    "longtext": {"EN": 160, "ZH": 160},
    "oneig": {"EN": 200, "ZH": 200},
    "gentexteval": {"EN": 316, "ZH": 417},
}
# Units per prompt: separate images for longtext / gentexteval, 2x2 tiles for oneig.
UNITS_PER_PROMPT = {"longtext": 4, "oneig": 4, "gentexteval": 4}
EXPECTED_CVTG = {
    "CVTG": {"2": 200, "3": 301, "4": 300, "5": 201},
    "CVTG-Style": {"2": 200, "3": 299, "4": 302, "5": 197},
}


def lang_of(model: str) -> str:
    upper = model.upper()
    for lang in ("EN", "ZH"):
        if upper.endswith(f"_{lang}"):
            return lang
    raise ValueError(f"model name {model!r} must end with _EN or _ZH")


def image_issue(path: Path) -> Optional[str]:
    if path.stat().st_size == 0:
        return "empty file"
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def valid_images(folder: Path) -> tuple[List[Path], List[str]]:
    if not folder.is_dir():
        return [], [f"missing dir {folder}"]
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    with ThreadPoolExecutor(max_workers=32) as pool:
        issues = list(pool.map(image_issue, files))
    bad = [f"{p.name}: {msg}" for p, msg in zip(files, issues) if msg]
    good = [p for p, msg in zip(files, issues) if not msg]
    return good, bad


def check_flat(bench: str, folder: Path, lang: str) -> List[str]:
    good, problems = valid_images(folder)
    units: Counter = Counter()
    for p in good:
        head, _, tail = p.stem.rpartition("_")
        if head and tail.isdigit():
            units[head] += 1
        else:
            units[p.stem] += 4 if bench == "oneig" else 1
    expected = EXPECTED_PROMPTS[bench][lang]
    need = UNITS_PER_PROMPT[bench]
    partial = [pid for pid, n in units.items() if n < need]
    if len(units) != expected:
        problems.append(f"prompts {len(units)}/{expected}")
    if partial:
        problems.append(f"{len(partial)} prompts with < {need} images, e.g. {sorted(partial)[:5]}")
    return problems


def check_cvtg(folder: Path) -> List[str]:
    problems = []
    for split, areas in EXPECTED_CVTG.items():
        for area, expected in areas.items():
            good, bad = valid_images(folder / split / area)
            problems.extend(bad)
            if len(good) != expected:
                problems.append(f"{split}/{area}: {len(good)}/{expected}")
    return problems


def check_polytext(eval_root: Path, folder: Path, lang: str, repeats: int) -> List[str]:
    prompt_dir = eval_root / "PolyText-Bench" / "data"
    rows = []
    for task in ("critical", "optional"):
        path = prompt_dir / f"{task}_{lang.lower()}.jsonl"
        if not path.is_file():
            return [f"missing prompt file {path}"]
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    keys = [f"{row['pair_id']}_{row['task_type']}" for row in rows]
    if len(keys) != len(set(keys)):
        return ["duplicate pair_id/task_type keys in PolyText prompt data"]
    expected = [folder / f"{key}_{repeat}.png" for key in keys for repeat in range(1, repeats + 1)]
    if not folder.is_dir():
        return [f"missing dir {folder}"]

    def issue(path: Path) -> Optional[str]:
        problem = image_issue(path) if path.is_file() else "missing file"
        if problem:
            return f"{path.name}: {problem}"
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                return f"{path.name}: size {image.size}, expected (1024, 1024)"
        return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        issues = list(pool.map(issue, expected))
    return [problem for problem in issues if problem]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", required=True, choices=sorted(IMAGE_DIRS))
    parser.add_argument("--base", required=True, help="Base model directory name, e.g. FLUX.1-dev.")
    parser.add_argument(
        "--model", required=True, help="Result directory name ending in _EN or _ZH."
    )
    parser.add_argument("--eval_root", required=True, help="Path to the eval/ directory.")
    parser.add_argument("--polytext-repeats", type=int, default=4)
    args = parser.parse_args()
    if args.polytext_repeats < 1:
        parser.error("--polytext-repeats must be positive")

    lang = lang_of(args.model)
    if args.bench == "cvtg" and lang != "EN":
        print(f"[{args.bench}] {args.model}: skipped (CVTG-2K is English only)")
        return 1
    eval_root = Path(args.eval_root)
    folder = eval_root / IMAGE_DIRS[args.bench].format(base=args.base, model=args.model)
    if args.bench == "cvtg":
        problems = check_cvtg(folder)
    elif args.bench == "polytext":
        problems = check_polytext(eval_root, folder, lang, args.polytext_repeats)
    else:
        problems = check_flat(args.bench, folder, lang)
    if problems:
        print(f"[{args.bench}] {args.base}/{args.model}: INCOMPLETE ({folder})")
        for msg in problems[:20]:
            print(f"    {msg}")
        return 1
    print(f"[{args.bench}] {args.base}/{args.model}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
