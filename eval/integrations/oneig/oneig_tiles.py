#!/usr/bin/env python3
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

"""Split OneIG 2x2 grids into tiles and record them for reuse.

OneIG generations are 2x2 grids. ``ensure_oneig_tiles`` reuses an existing
``tiles.jsonl`` when its tile files are still on disk; otherwise it splits the
grids, writes ``tiles.jsonl`` and returns the tile records.

Record layout (under ``--eval_root``)::

    OneIG-Benchmark/eval_metrics/{base}/{model}/tiles.jsonl
    OneIG-Benchmark/eval_metrics/{base}/{model}/tiles/
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

IMAGE_DIR_REL = "OneIG-Benchmark/images/text/{base}/{model}"
TILES_JSONL_REL = "OneIG-Benchmark/eval_metrics/{base}/{model}/tiles.jsonl"
TILES_DIR_REL = "OneIG-Benchmark/eval_metrics/{base}/{model}/tiles"

META_REL = {
    "en": "OneIG-Benchmark/scripts/text/text_content.csv",
    "zh": "OneIG-Benchmark/scripts/text/text_content_zh.csv",
}


def eval_path(eval_root: str, rel: str) -> str:
    """Join ``eval_root`` with a repo-relative path."""
    return os.path.join(eval_root, rel)


def lang_mode(lang: str) -> str:
    """Normalize EN/ZH (or en/zh) to the lowercase mode used by prompt CSVs."""
    if not isinstance(lang, str) or not lang:
        raise ValueError(f"expected non-empty lang, got {type(lang).__name__}: {lang!r}")
    upper = lang.upper()
    if upper == "ZH":
        return "zh"
    if upper == "EN":
        return "en"
    raise ValueError(f"unknown lang {lang!r}; expected 'EN' or 'ZH'")


def tiles_jsonl_path(eval_root: str, base: str, model: str) -> str:
    """Return the OneIG tile index for this model."""
    return eval_path(eval_root, TILES_JSONL_REL.format(base=base, model=model))


def tiles_dir_path(eval_root: str, base: str, model: str) -> str:
    """Return the directory that holds cropped OneIG tiles."""
    return eval_path(eval_root, TILES_DIR_REL.format(base=base, model=model))


def parse_quoted_text(prompt: str) -> List[str]:
    """Extract quoted spans from a Chinese-style OneIG prompt."""
    quote_map = {
        "“": "”",
        "‘": "’",
        '"': '"',
        "'": "'",
        "《": "》",
    }
    results: List[str] = []
    in_quote = False
    current_closing = None
    start_pos = 0
    for i, char in enumerate(prompt):
        if not in_quote:
            if char in quote_map:
                in_quote = True
                current_closing = quote_map[char]
                start_pos = i + 1
        elif char == current_closing:
            content = prompt[start_pos:i].strip()
            if content:
                results.append(content)
            in_quote = False
            current_closing = None
    return results


def parse_text_atlas(prompt: str) -> List[str]:
    """Extract quoted spans from an English-style OneIG prompt."""

    def recursive_extract(text: str) -> List[str]:
        results: List[str] = []
        remaining = text
        for quote in ("''", '""', "'", '"'):
            pattern = (
                r"(?<!\w)"
                r"(?:the\s+text\s*:\s*)?"
                r"(?<!\\)(?:\\\\)*" + re.escape(quote) + r"(.*?)"
                r"(?<!\\)(?:\\\\)*" + re.escape(quote) + r"(?!\w)"
            )
            while True:
                match = re.search(pattern, remaining, re.DOTALL)
                if not match:
                    break
                start, end = match.span()
                content = match.group(1).strip()
                nested = recursive_extract(content)
                if nested:
                    results.extend(nested)
                elif content:
                    results.append(content)
                remaining = remaining[:start] + remaining[end:]
        return results

    seen = set()
    final: List[str] = []
    for text in recursive_extract(prompt):
        if text not in seen:
            seen.add(text)
            final.append(text)
    return final


def parse_prompt(prompt: str, mode: str) -> str:
    """Parse OneIG ground-truth text from a prompt for ``en`` or ``zh``."""
    if mode in ("cn", "zh"):
        return " ".join(parse_quoted_text(prompt))
    if mode == "en":
        filtered = [
            text.replace('"', "").replace("'", "").replace("“", "").replace("”", "")
            for text in parse_text_atlas(prompt)
            if text
        ]
        return " ".join(filtered)
    raise ValueError(f"unknown parse_prompt mode {mode!r}; expected 'en' or 'zh'")


def split_2x2_grid(image_path: str, cache_dir: str) -> List[str]:
    """Crop a 2x2 grid into four jpg tiles with a path-unique prefix."""
    os.makedirs(cache_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    path_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]
    prefix = f"{basename}_{path_hash}_"
    with open(image_path, "rb") as handle:
        grid = Image.open(handle)
        width, height = grid.size
        tile_w = width // 2
        tile_h = height // 2
        tiles = []
        for row in range(2):
            for col in range(2):
                box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
                tiles.append(grid.crop(box))
    paths: List[str] = []
    for idx, tile in enumerate(tiles):
        out = os.path.join(cache_dir, f"{prefix}{idx}.jpg")
        if not os.path.isfile(out):
            tile.save(out)
        paths.append(out)
    return paths


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_no}: {line[:160]!r}") from exc
            if not isinstance(rec, dict):
                raise TypeError(f"expected dict at {path}:{line_no}, got {type(rec).__name__}")
            records.append(rec)
    return records


def resolve_image(image_field: Any, eval_root: str, context: str) -> str:
    """Resolve a JSONL ``image`` path against ``eval_root`` if it is relative."""
    if not isinstance(image_field, str) or not image_field:
        raise ValueError(f"missing/invalid 'image' field ({context}): {image_field!r}")
    if os.path.isabs(image_field):
        return image_field
    return os.path.join(eval_root, image_field)


def find_oneig_grid(sample_dir: str, img_id: str) -> Optional[str]:
    """Find the generated 2x2 grid file for a OneIG prompt id."""
    for suffix in IMAGE_SUFFIXES:
        candidate = os.path.join(sample_dir, f"{img_id}{suffix}")
        if os.path.isfile(candidate):
            return candidate
    matches = glob.glob(os.path.join(sample_dir, f"{img_id}.*"))
    return matches[0] if matches else None


def load_oneig_from_jsonl(jsonl_path: str, eval_root: str) -> List[Dict[str, Any]]:
    """Load tile records from a jsonl, skipping tiles whose files are gone."""
    records = load_jsonl(jsonl_path)
    samples: List[Dict[str, Any]] = []
    missing = 0
    for idx, rec in enumerate(records):
        target = rec.get("target", rec.get("ori_target", ""))
        if not isinstance(target, str) or target == "":
            raise ValueError(f"missing/invalid 'target' at {jsonl_path}[{idx}]: {target!r}")
        path = resolve_image(rec.get("image"), eval_root, f"{jsonl_path}[{idx}]")
        if not os.path.isfile(path):
            missing += 1
            print(f"  ⚠️  OneIG tile missing, skip: {path}")
            continue
        samples.append(
            {
                "image": path,
                "prompt": rec.get("prompt", ""),
                "target": target,
                "ori_target": rec.get("ori_target", target),
                "id": rec.get("id", f"oneig_{idx}"),
            }
        )
    print(f"  OneIG tiles from {jsonl_path}: {len(samples)} usable, {missing} missing")
    return samples


def write_tiles_jsonl(samples: List[Dict[str, Any]], dump_jsonl: str) -> None:
    """Write tile records so later runs can reuse them."""
    parent = os.path.dirname(dump_jsonl)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dump_jsonl, "w", encoding="utf-8") as handle:
        for rec in samples:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  OneIG recorded {len(samples)} tiles -> {dump_jsonl}")


def load_oneig_by_split(
    eval_root: str, sample_dir: str, lang: str, cache_dir: str, dump_jsonl: str
) -> List[Dict[str, Any]]:
    """Split OneIG 2x2 grids, write ``dump_jsonl``, and return tile records."""
    mode = lang_mode(lang)
    csv_rel = META_REL[mode]
    csv_path = eval_path(eval_root, csv_rel)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"OneIG prompt CSV not found: {csv_path}")
    samples: List[Dict[str, Any]] = []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_id = row.get("id")
            if not img_id:
                raise ValueError(f"OneIG CSV {csv_path} has a row without 'id': {row!r}")
            prompt_key = "prompt_cn" if mode == "zh" else "prompt_en"
            prompt = row.get(prompt_key, "") or ""
            text_str = row.get("text_content", "[]")
            text_list = ast.literal_eval(text_str)
            if not isinstance(text_list, list):
                raise TypeError(
                    f"OneIG text_content must be a list literal, got "
                    f"{type(text_list).__name__} for id={img_id!r}"
                )
            target_text = " ".join(str(t) for t in text_list)
            target = parse_prompt(prompt, mode)
            if not target:
                target = target_text
            elif target_text and (len(target_text) / max(len(target), 1)) >= 2:
                target = target_text
            grid_path = find_oneig_grid(sample_dir, img_id)
            if grid_path is None:
                print(f"  ⚠️  OneIG grid image not found for id={img_id!r} under {sample_dir}")
                continue
            for tile_path in split_2x2_grid(grid_path, cache_dir):
                samples.append(
                    {
                        "image": tile_path,
                        "prompt": prompt,
                        "target": target,
                        "ori_target": target_text,
                        "id": f"oneig_{len(samples)}",
                    }
                )
    if not samples:
        raise ValueError(
            f"OneIG split produced 0 tiles from {sample_dir} " f"(csv={csv_path}, lang={lang!r})"
        )
    write_tiles_jsonl(samples, dump_jsonl)
    return samples


def ensure_oneig_tiles(eval_root: str, base: str, model: str, lang: str) -> List[Dict[str, Any]]:
    """Reuse recorded OneIG tiles, or split the 2x2 grids and record them.

    Args:
        eval_root: Path to the ``eval/`` directory.
        base: Base model directory name.
        model: Language-suffixed model directory name.
        lang: ``EN`` or ``ZH``.

    Returns:
        Tile records with ``image``, ``prompt``, ``target``, ``ori_target``,
        and ``id``.
    """
    sample_dir = eval_path(eval_root, IMAGE_DIR_REL.format(base=base, model=model))
    dump_jsonl = tiles_jsonl_path(eval_root, base, model)
    tile_cache = tiles_dir_path(eval_root, base, model)

    if os.path.isfile(dump_jsonl):
        existing = load_oneig_from_jsonl(dump_jsonl, eval_root)
        if existing:
            print(f"  OneIG reuse tiles: {dump_jsonl}")
            return existing
        print(f"  OneIG tiles.jsonl has no tiles on disk ({dump_jsonl}); re-splitting")

    print(f"  OneIG splitting 2x2 grids under {sample_dir}")
    return load_oneig_by_split(eval_root, sample_dir, lang, tile_cache, dump_jsonl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure OneIG 2x2 tiles exist (reuse tiles.jsonl, otherwise split)."
    )
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", required=True, choices=("EN", "ZH", "en", "zh"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = ensure_oneig_tiles(args.eval_root, args.base, args.model, args.lang)
    print(
        f"OneIG tiles ready: count={len(samples)} tiles_jsonl={tiles_jsonl_path(args.eval_root, args.base, args.model)}"
    )


if __name__ == "__main__":
    main()
