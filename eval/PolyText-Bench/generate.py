# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
"""Generate independent PolyText PNGs with Qwen-Image or FLUX and optional PEFT LoRA."""

import argparse
import fcntl
import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel
from PIL import Image

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data"
SPLIT_FILES = (
    "critical_en.jsonl",
    "optional_en.jsonl",
    "critical_zh.jsonl",
    "optional_zh.jsonl",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read nonempty JSONL records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        if row.get("language") not in ("en", "zh"):
            raise ValueError(f"Invalid language for {key}: {row.get('language')!r}")
        targets = row.get("target_list")
        if not isinstance(targets, list) or any(
            not isinstance(target, str) or not target.strip() for target in targets
        ):
            raise ValueError(f"Invalid target_list for {key}: {targets!r}")
        if bool(targets) != (row["task_type"] == "critical"):
            raise ValueError(f"Target/task mismatch for {key}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Empty prompt for {key}")
        if row["language"] == mode.lower():
            selected.append(row)
    if not selected:
        raise ValueError(f"No prompts for mode={mode} in {path}")
    return selected


def make_seed(row: Dict[str, Any], repeat: int) -> int:
    """Return the deterministic paired seed for one repeat."""
    return zlib.crc32(f"{row['pair_id']}_{repeat}".encode()) & 0xFFFFFFFF


def image_complete(path: Path, width: int, height: int) -> bool:
    """Return whether an image decodes at the required resolution."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (width, height)
    except (OSError, ValueError):
        return False


def image_jobs(rows: List[Dict[str, Any]], image_dir: Path, repeats: int) -> List[Dict[str, Any]]:
    """Enumerate expected output images."""
    return [
        dict(
            row=row,
            repeat=repeat,
            key=f"{prompt_key(row)}_{repeat}",
            path=image_dir / f"{prompt_key(row)}_{repeat}.png",
        )
        for row in rows
        for repeat in range(1, repeats + 1)
    ]


def shard_jobs(
    jobs: List[Dict[str, Any]],
    task_type: str,
    node_index: int,
    num_nodes: int,
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    """Partition jobs across nodes and local workers without padding."""
    if num_nodes < 1 or not 0 <= node_index < num_nodes:
        raise ValueError(f"Expected 0 <= node_index({node_index}) < num_nodes({num_nodes})")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Expected 0 <= rank({rank}) < world_size({world_size})")
    selected = [job for job in jobs if task_type == "all" or job["row"]["task_type"] == task_type]
    return selected[node_index::num_nodes][rank::world_size]


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Run disjoint torchrun workers without padding or inter-rank collectives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=["qwen", "flux"], required=True)
    parser.add_argument("--mode", choices=["EN", "ZH"], default="EN")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt_path")
    parser.add_argument("--gallery_output_dir", type=Path, required=True)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-type", choices=["all", "critical", "optional"], default="all")
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--progress-dir", type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.steps, args.repeats) <= 0:
        parser.error("Dimensions, steps and repeats must be positive")
    if args.limit_prompts is not None and args.limit_prompts <= 0:
        parser.error("limit-prompts must be positive")
    rows = load_prompts(args.prompts, args.mode)
    if args.limit_prompts:
        rows = rows[: args.limit_prompts]
    output = args.gallery_output_dir / f"{args.model_name}_{args.mode}"
    jobs = image_jobs(rows, output, args.repeats)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    owned = shard_jobs(jobs, args.task_type, args.node_index, args.num_nodes, rank, world)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    output=str(output),
                    expected=len(jobs),
                    rank=rank,
                    owned=len(owned),
                    first=str(owned[0]["path"]) if owned else None,
                )
            )
        )
        return
    adapter = None
    if args.ckpt_path and args.ckpt_path != "None":
        adapter = Path(args.ckpt_path)
        if adapter.is_file():
            adapter = adapter.parent
        if (
            not (adapter / "adapter_config.json").is_file()
            or not (adapter / "adapter_model.safetensors").is_file()
        ):
            raise FileNotFoundError(f"Incomplete PEFT adapter: {adapter}")
    negative = args.negative_prompt
    if negative in (None, "", "None"):
        negative = " " if args.model_family == "qwen" else None
    protocol = dict(
        version=1,
        prompts=rows,
        model_path=args.model_path,
        adapter=str(adapter.resolve()) if adapter else None,
        adapter_stat=(
            (adapter / "adapter_model.safetensors").stat().st_mtime_ns if adapter else None
        ),
        model_family=args.model_family,
        width=args.width,
        height=args.height,
        steps=args.steps,
        repeats=args.repeats,
        negative_prompt=negative,
        guidance=4.0 if args.model_family == "qwen" else 3.5,
        seeds=(
            "explicit_prompt_seeds_or_crc32(pair_id_repeat)"
            if any("seeds" in row for row in rows)
            else "crc32(pair_id_repeat)"
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output / "generation.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != protocol:
                raise ValueError("Generation settings changed; choose a new model/output directory")
        elif any(output.glob("*.png")):
            raise ValueError("Existing images lack generation provenance; choose a fresh directory")
        else:
            atomic_json(manifest, protocol)
    pending = [j for j in owned if not image_complete(j["path"], args.width, args.height)]
    initial_done = len(owned) - len(pending)
    durations = []
    started_at = time.time()

    def report(state: str, completed: int) -> None:
        if args.progress_dir:
            remaining = len(owned) - completed
            seconds_per_image = sum(durations[-8:]) / len(durations[-8:]) if durations else None
            atomic_json(
                args.progress_dir / f"rank-{rank}.json",
                dict(
                    rank=rank,
                    pid=os.getpid(),
                    observed_at=time.time(),
                    started_at=started_at,
                    state=state,
                    done=completed,
                    total=len(owned),
                    initial_done=initial_done,
                    eta_sec=(
                        remaining * seconds_per_image if seconds_per_image is not None else None
                    ),
                    seconds_per_image=seconds_per_image,
                    task_type=args.task_type,
                    node_index=args.node_index,
                    num_nodes=args.num_nodes,
                ),
            )

    report("loading", initial_done)
    if not pending:
        report("done", len(owned))
        print(f"Rank {rank}: all assigned images complete")
        return
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    if adapter:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(adapter))
        pipe.transformer.eval()
    kwargs = dict(width=args.width, height=args.height, num_inference_steps=args.steps)
    if args.model_family == "qwen":
        kwargs.update(negative_prompt=negative, true_cfg_scale=4.0)
    else:
        kwargs.update(guidance_scale=3.5)
        if negative is not None:
            kwargs.update(negative_prompt=negative, true_cfg_scale=4.0)
    for index, job in enumerate(pending):
        image_started = time.monotonic()
        with torch.inference_mode():
            image = pipe(
                prompt=job["row"]["prompt"],
                generator=torch.Generator(device=device).manual_seed(
                    make_seed(job["row"], job["repeat"])
                ),
                **kwargs,
            ).images[0]
        tmp = job["path"].with_name(f".{job['key']}.{os.getpid()}.tmp")
        image.save(tmp, format="PNG")
        tmp.replace(job["path"])
        durations.append(time.monotonic() - image_started)
        report("generating", initial_done + index + 1)
        print(f"Rank {rank}: {index + 1}/{len(pending)} {job['path']}", flush=True)
    report("done", len(owned))


if __name__ == "__main__":
    main()
