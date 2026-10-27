# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
"""Validate and summarize a completed PolyText-Bench score journal."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data"
SPLIT_FILES = (
    "critical_en.jsonl",
    "optional_en.jsonl",
    "critical_zh.jsonl",
    "optional_zh.jsonl",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read nonempty JSONL objects."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def prompt_key(row: Dict[str, Any]) -> str:
    """Return the collision-free pair/task key."""
    pair = row.get("pair_id")
    task = row.get("task_type")
    if not isinstance(pair, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pair):
        raise ValueError(f"Invalid pair_id: {pair!r}")
    if task not in ("critical", "optional"):
        raise ValueError(f"Invalid task_type: {task!r}")
    return f"{pair}_{task}"


def load_prompts(path: Path, mode: str) -> List[Dict[str, Any]]:
    """Load and validate one language of benchmark prompts."""
    if path.is_dir():
        missing = [name for name in SPLIT_FILES if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Prompt directory {path} is missing {missing}")
        rows = [row for name in SPLIT_FILES for row in read_jsonl(path / name)]
    else:
        rows = read_jsonl(path)
    selected = []
    seen = set()
    for row in rows:
        key = prompt_key(row)
        if key in seen:
            raise ValueError(f"Duplicate prompt ID: {key}")
        seen.add(key)
        if row.get("language") == mode.lower():
            selected.append(row)
    if not selected:
        raise ValueError(f"No prompts for mode={mode} in {path}")
    return selected


def mean(values: List[float], name: str) -> float:
    """Return a finite arithmetic mean."""
    if not values:
        raise ValueError(f"No values available for metric {name}")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite value in metric {name}")
    return sum(values) / len(values)


def summarize(
    rows: List[Dict[str, Any]], records: List[Dict[str, Any]], repeats: int
) -> Dict[str, Any]:
    """Validate the exact score set and compute benchmark metrics."""
    expected = {f"{prompt_key(row)}_{repeat}" for row in rows for repeat in range(1, repeats + 1)}
    observed = [record.get("key") for record in records]
    if len(observed) != len(set(observed)):
        raise ValueError("Score journal contains duplicate keys")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"Incomplete score set: missing={missing[:5]}, extra={extra[:5]}")
    critical = [record for record in records if record["task_type"] == "critical"]
    optional = [record for record in records if record["task_type"] == "optional"]
    if len(critical) + len(optional) != len(records):
        raise ValueError("Unknown task type in score journal")
    cs = mean([float(record["CS"]) for record in critical], "CS")
    cg_valid = [record for record in critical if record["CG_valid"]]
    og_valid = [record for record in optional if record["OG_valid"]]
    return dict(
        complete=True,
        language=rows[0]["language"],
        images=len(records),
        score_scale="0-1",
        metrics=dict(
            CS=cs,
            CG=(mean([float(record["CG"]) for record in cg_valid], "CG") if cg_valid else None),
            CG_valid=f"{len(cg_valid)}/{len(critical)}",
            OG=(mean([float(record["OG"]) for record in og_valid], "OG") if og_valid else None),
            OG_valid=f"{len(og_valid)}/{len(optional)}",
        ),
    )


def main() -> None:
    """Parse options, validate scores, and write the final summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["EN", "ZH"], required=True)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    rows = load_prompts(args.prompts, args.mode)
    records = read_jsonl(args.scores_dir / "eval_results.jsonl")
    summary = summarize(rows, records, args.repeats)
    output = args.scores_dir / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Summary written to {output}")


if __name__ == "__main__":
    main()
