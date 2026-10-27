# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
"""Score a complete PolyText-Bench image directory with a TextDART server."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from accelerate import Accelerator
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards.textdart import (
    TextDARTRewardModel,
    get_scaled_quality_score,
    process_raw_text,
    score_regions_by_target,
    split_target_component_rewards,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data"
DEFAULT_SERVERS = ["http://127.0.0.1:1925"]
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


def recover_scores(path: Path) -> List[Dict[str, Any]]:
    """Recover only a torn final journal record and retain its bytes."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        offset = raw.rfind(b"\n") + 1
        tail = raw[offset:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            backup = path.with_name(f"{path.name}.partial")
            if backup.exists():
                raise FileExistsError(
                    f"Cannot recover {path}: retained tail already exists at {backup}"
                )
            backup.write_bytes(tail)
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            print(f"Recovered torn score record; retained bytes at {backup}")
            raw = raw[:offset]
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            raw += b"\n"
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise TypeError(f"Expected JSON objects in score journal {path}")
    return records


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
        targets = row.get("target_list")
        if not isinstance(targets, list) or any(
            not isinstance(target, str) or not target.strip() for target in targets
        ):
            raise ValueError(f"Invalid target_list for {key}: {targets!r}")
        if bool(targets) != (row["task_type"] == "critical"):
            raise ValueError(f"Target/task mismatch for {key}")
        if row.get("language") == mode.lower():
            selected.append(row)
    if not selected:
        raise ValueError(f"No prompts for mode={mode} in {path}")
    return selected


def file_digest(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_score(value: float) -> float:
    """Return a finite normalized score."""
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Expected a normalized score, got {score}")
    return score


def image_scores(
    task: str,
    targets: List[str],
    semantic: Dict[str, float],
    glyph: Dict[str, float],
    region_glyph: List[float],
) -> Dict[str, Any]:
    """Compute per-image critical or optional scores."""
    if task == "critical":
        if not targets:
            raise ValueError("Critical images require at least one target")
        allowed = {str(index) for index in range(len(targets))}
        if set(semantic) - allowed - {"extra"} or set(glyph) - allowed - {"extra"}:
            raise ValueError("TextDART returned an unknown target index")
        if (set(semantic) & allowed) != (set(glyph) & allowed):
            raise ValueError("Semantic and glyph target matches differ")
        target_scores = [
            dict(
                target=target,
                matched=str(index) in semantic,
                semantic=checked_score(semantic.get(str(index), 0)),
                glyph=checked_score(glyph.get(str(index), 0)),
            )
            for index, target in enumerate(targets)
        ]
        return dict(
            CS=sum(item["semantic"] for item in target_scores) / len(target_scores),
            CG=sum(item["glyph"] for item in target_scores) / len(target_scores),
            OG=None,
            CG_valid=any(item["matched"] for item in target_scores),
            OG_valid=False,
            target_scores=target_scores,
            detected_regions=len(region_glyph),
        )
    if task != "optional" or targets:
        raise ValueError("Optional images must have an empty target list")
    scores = [checked_score(score) for score in region_glyph]
    return dict(
        CS=None,
        CG=None,
        OG=sum(scores) / len(scores) if scores else None,
        CG_valid=False,
        OG_valid=bool(scores),
        target_scores=[],
        detected_regions=len(scores),
    )


class TextDARTScorer:
    """Query TextDART and apply the official local target matcher."""

    def __init__(
        self,
        servers: List[str],
        timeout: float,
        retries: int,
        quality_amplification: float,
        count_markers_by_weight: bool,
    ) -> None:
        options = dict(
            server_urls=servers,
            emit_quad_2d=True,
            group_union_target_merge=False,
            qua_amplify_factor=quality_amplification,
            count_markers_by_weight=count_markers_by_weight,
            max_empty_retries=0,
            max_tokens=4096,
            timeout=timeout,
            retry_attempts=retries,
        )
        self.model = TextDARTRewardModel(
            RewardArguments.from_dict(
                dict(name="textdart", reward_model="textdart", device="cpu", **options)
            ),
            Accelerator(cpu=True),
        )

    async def score(self, row: Dict[str, Any], path: Path) -> Dict[str, Any]:
        """Score one image with bounded malformed-response retries."""
        with Image.open(path) as source:
            image = source.convert("RGB")
        regions = []
        for attempt in range(10):
            regions, failed = await self.model._collect_regions_one(image)
            if not failed:
                break
            if attempt == 9:
                raise ValueError(f"TextDART returned malformed OCR 10 times for {path.name}")
            await asyncio.sleep(min(2**attempt, 8))
        region_glyph = [
            float(
                get_scaled_quality_score(
                    process_raw_text(region["text"]),
                    self.model.qua_amplify_factor,
                    marker_weights=self.model.marker_weights,
                    count_markers_by_weight=self.model.count_markers_by_weight,
                )
            )
            for region in regions
        ]
        semantic, glyph, assigned = {}, {}, []
        if row["task_type"] == "critical":
            mixed, assigned, count = score_regions_by_target(
                regions, row["target_list"], **self.model._score_kwargs()
            )
            if count != len(row["target_list"]):
                raise ValueError(
                    f"Expected {len(row['target_list'])} targets for "
                    f"{prompt_key(row)}, got {count}"
                )
            semantic, glyph = split_target_component_rewards(mixed, assigned)
        return dict(
            **image_scores(
                row["task_type"],
                row["target_list"],
                semantic,
                glyph,
                region_glyph,
            ),
            ocr_regions=regions,
            assigned_regions=assigned,
            region_glyph=region_glyph,
        )


class ScoreJournal:
    """Append completed scores durably and safely across asynchronous tasks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = asyncio.Lock()

    async def append(self, record: Dict[str, Any]) -> None:
        """Append one JSON record and sync it before returning."""
        content = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        async with self.lock:
            await asyncio.to_thread(self._write, content)

    def _write(self, content: bytes) -> None:
        with self.path.open("ab", buffering=0) as handle:
            remaining = memoryview(content)
            while remaining:
                written = handle.write(remaining)
                if not written:
                    raise OSError(f"Score journal append made no progress: {self.path}")
                remaining = remaining[written:]
            os.fsync(handle.fileno())


def expected_jobs(
    rows: List[Dict[str, Any]],
    image_dir: Path,
    repeats: int,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """Build jobs and require every image to decode at the expected size."""
    jobs = []
    incomplete = []
    for row in rows:
        for repeat in range(1, repeats + 1):
            key = f"{prompt_key(row)}_{repeat}"
            path = image_dir / f"{key}.png"
            try:
                with Image.open(path) as image:
                    image.load()
                    valid = image.size == (width, height)
            except (OSError, ValueError):
                valid = False
            if not valid:
                incomplete.append(str(path))
            jobs.append(dict(key=key, row=row, repeat=repeat, path=path))
    if incomplete:
        raise ValueError(
            f"Expected {len(jobs)} complete {width}x{height} images; "
            f"{len(incomplete)} are missing or invalid: {incomplete[:5]}"
        )
    return jobs


async def evaluate(args: argparse.Namespace) -> None:
    """Score all missing jobs and preserve completed records for resumption."""
    rows = load_prompts(args.prompts, args.mode)
    jobs = expected_jobs(rows, args.image_dir, args.repeats, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "eval_results.jsonl"
    records = recover_scores(cache)
    completed = {}
    expected = {job["key"]: job for job in jobs}
    for record in records:
        key = record.get("key")
        if key in completed or key not in expected:
            raise ValueError(f"Invalid or duplicate cached score key: {key!r}")
        if record.get("image_sha256") != file_digest(expected[key]["path"]):
            raise ValueError(f"Image changed after scoring: {key}")
        completed[key] = record
    scorer = TextDARTScorer(
        args.server_urls,
        args.timeout,
        args.retries,
        args.quality_amplification,
        args.count_markers_by_weight,
    )
    journal = ScoreJournal(cache)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def score_one(job: Dict[str, Any]) -> None:
        async with semaphore:
            image_hash = await asyncio.to_thread(file_digest, job["path"])
            scores = await scorer.score(job["row"], job["path"])
            if image_hash != await asyncio.to_thread(file_digest, job["path"]):
                raise ValueError(f"Image changed during scoring: {job['key']}")
            record = dict(
                key=job["key"],
                pair_id=job["row"]["pair_id"],
                repeat=job["repeat"],
                language=job["row"]["language"],
                task_type=job["row"]["task_type"],
                target_list=job["row"]["target_list"],
                image_file=job["path"].name,
                image_sha256=image_hash,
                **scores,
            )
            await journal.append(record)
            completed[job["key"]] = record
            print(f"Scored {len(completed)}/{len(jobs)} images", flush=True)

    await asyncio.gather(*(score_one(job) for job in jobs if job["key"] not in completed))
    ordered = [completed[job["key"]] for job in jobs]
    temporary = cache.with_suffix(".complete.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in ordered
        )
    )
    temporary.replace(cache)
    print(f"Complete: {len(ordered)} scores written to {cache}")


def main() -> None:
    """Parse options, lock the output directory, and run scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["EN", "ZH"], required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--server-urls", nargs="+", default=DEFAULT_SERVERS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--quality-amplification", type=float, default=5.0)
    parser.add_argument("--count-markers-by-weight", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.repeats,
            args.width,
            args.height,
            args.concurrency,
            args.retries,
        )
        < 1
        or args.timeout <= 0
        or args.quality_amplification <= 0
    ):
        parser.error("Counts, dimensions, timeout, and amplification must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".evaluate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
