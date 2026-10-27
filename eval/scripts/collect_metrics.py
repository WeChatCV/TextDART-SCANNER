#!/usr/bin/env python3
"""Read per-cell scores of the five benchmarks and render the main table.

Subcommands:
    check  BENCH BASE MODEL   exit 0 iff the cell already has complete scores
    table  BASE/MODEL ...     write a Markdown table (one row per BASE/MODEL)

Score files (under --eval_root), all keyed by ``{base}/{model}``:

    longtext     X-Omni/textbench/eval_metrics/.../scores.txt (+ results.jsonl)
    oneig        OneIG-Benchmark/eval_metrics/.../text_score.csv
    cvtg         TextCrafter/TextCrafter_Eval/eval_metrics/.../results.json
    gentexteval  GenTextEval/eval_metrics/.../eval_results.jsonl
    polytext     PolyText-Bench/eval_metrics/.../summary.json

Incomplete or missing cells are rendered as NA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

BENCHES = ("longtext", "oneig", "cvtg", "gentexteval", "polytext")

RESULT_FILES = {
    "longtext": "X-Omni/textbench/eval_metrics/{base}/{model}/scores.txt",
    "oneig": "OneIG-Benchmark/eval_metrics/{base}/{model}/text_score.csv",
    "cvtg": "TextCrafter/TextCrafter_Eval/eval_metrics/{base}/{model}/results.json",
    "gentexteval": "GenTextEval/eval_metrics/{base}/{model}/eval_results.jsonl",
    "polytext": "PolyText-Bench/eval_metrics/{base}/{model}/summary.json",
}

# (metric key, column name, higher is better)
COLUMNS = {
    "longtext": [("text_score", "LongText", True)],
    "oneig": [
        ("text_score", "OneIG", True),
        ("ED", "ED", False),
        ("CR", "CR", True),
        ("WAC", "WAC", True),
    ],
    "cvtg": [("word_accuracy", "CVTG WordAcc", True), ("ned", "CVTG NED", True)],
    "gentexteval": [("qua", "GenTextEval Qua", True), ("sem", "GenTextEval Sem", True)],
    "polytext": [
        ("CS", "PolyText CS", True),
        ("CG", "PolyText CG", True),
        ("OG", "PolyText OG", True),
    ],
}

LONGTEXT_RECORDS = {"EN": 160 * 4, "ZH": 160 * 4}
GENTEXTEVAL_RECORDS = {"EN": 316 * 4, "ZH": 417 * 4}
CVTG_IMAGES = 2000

_TEXT_SCORE_RE = re.compile(r"Text\s*Score\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def lang_of(model: str) -> str:
    upper = model.upper()
    for lang in ("EN", "ZH"):
        if upper.endswith(f"_{lang}"):
            return lang
    raise ValueError(f"model name {model!r} must end with _EN or _ZH")


def count_jsonl(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_longtext(path: str, lang: str) -> Optional[Dict[str, float]]:
    records = os.path.join(os.path.dirname(path), "results.jsonl")
    if not os.path.isfile(records) or count_jsonl(records) != LONGTEXT_RECORDS[lang]:
        return None
    with open(path, encoding="utf-8") as f:
        match = _TEXT_SCORE_RE.search(f.read())
    if match is None:
        raise ValueError(f"no 'Text Score: <float>' line in {path}")
    return {"text_score": float(match.group(1))}


def read_oneig(path: str, lang: str) -> Optional[Dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one data row in {path}, got {len(rows)}")
    row = rows[0]
    return {
        "text_score": float(row["text score"]),
        "ED": float(row["ED"]),
        "CR": float(row["CR"]),
        "WAC": float(row["WAC"]),
    }


def read_cvtg(path: str, lang: str) -> Optional[Dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        overall = json.load(f)["overall_results"]
    if overall["total_images"] != CVTG_IMAGES or overall["total_words"] <= 0:
        return None
    return {"word_accuracy": overall["word_accuracy"], "ned": overall["ned"]}


def read_gentexteval(path: str, lang: str) -> Optional[Dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) != GENTEXTEVAL_RECORDS[lang]:
        return None
    qua = [r["pecker_qua"] for r in records if r.get("pecker_qua") is not None]
    sem = [r["pecker_gned"] for r in records if r.get("pecker_gned") is not None]
    return {"qua": sum(qua) / len(qua) if qua else 0.0, "sem": sum(sem) / len(sem) if sem else 0.0}


def read_polytext(path: str, lang: str) -> Optional[Dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    if not summary.get("complete") or summary.get("language") != lang.lower():
        return None
    scores = summary["metrics"]
    return {key: scores[key] for key in ("CS", "CG", "OG")}


READERS = {
    "longtext": read_longtext,
    "oneig": read_oneig,
    "cvtg": read_cvtg,
    "gentexteval": read_gentexteval,
    "polytext": read_polytext,
}


def read_cell(eval_root: str, bench: str, base: str, model: str) -> Optional[Dict[str, float]]:
    lang = lang_of(model)
    if bench == "cvtg" and lang != "EN":
        return None
    path = os.path.join(eval_root, RESULT_FILES[bench].format(base=base, model=model))
    if not os.path.isfile(path):
        return None
    return READERS[bench](path, lang)


def fmt(value: Optional[float], best: Optional[float]) -> str:
    if value is None:
        return "NA"
    text = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return f"**{text}**" if best is not None and abs(value - best) < 1e-12 else text


def render_table(eval_root: str, pairs: List[str], benches: List[str]) -> str:
    rows = []
    for pair in pairs:
        if "/" not in pair:
            raise ValueError(f"expected BASE/MODEL, got {pair!r}")
        base, model = pair.split("/", 1)
        rows.append((base, model, {b: read_cell(eval_root, b, base, model) for b in benches}))

    columns = [(b, key, name, hib) for b in benches for key, name, hib in COLUMNS[b]]
    lines = ["Bold marks the best value within each base model and language."]
    for lang in ("EN", "ZH"):
        group = [r for r in rows if lang_of(r[1]) == lang]
        if not group:
            continue
        lines += ["", f"### {lang}", ""]
        lines.append("| Base | Model | " + " | ".join(c[2] for c in columns) + " |")
        lines.append("|:---|:---|" + "---:|" * len(columns))
        for base, model, cells in group:
            same_base = [g for g in group if g[0] == base]
            out = []
            for bench, key, _name, hib in columns:
                values = [
                    g[2][bench][key]
                    for g in same_base
                    if g[2][bench] and g[2][bench].get(key) is not None
                ]
                best = (max(values) if hib else min(values)) if len(values) > 1 else None
                cell = cells[bench]
                out.append(fmt(cell.get(key) if cell else None, best))
            lines.append(f"| {base} | {model} | " + " | ".join(out) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval_root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("bench", choices=BENCHES)
    check.add_argument("base")
    check.add_argument("model")
    table = sub.add_parser("table")
    table.add_argument("pairs", nargs="+", help="BASE/MODEL entries, MODEL ends with _EN or _ZH.")
    table.add_argument("--benches", default=",".join(BENCHES))
    table.add_argument(
        "--output", default=None, help="Markdown output path (default: stdout only)."
    )
    args = parser.parse_args()

    if args.cmd == "check":
        return 0 if read_cell(args.eval_root, args.bench, args.base, args.model) is not None else 1

    benches = [b for b in args.benches.split(",") if b]
    unknown = sorted(set(benches) - set(BENCHES))
    if unknown:
        raise ValueError(f"unknown benches {unknown}; expected a subset of {list(BENCHES)}")
    text = render_table(args.eval_root, args.pairs, benches)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
