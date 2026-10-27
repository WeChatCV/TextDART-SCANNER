#!/usr/bin/env python3
"""Prepare separate TextDART-123K train/test caches for ms-swift.

Each row is converted to ms-swift's ``messages``/``images`` layout and gets a
``lengths`` column computed with the Qwen3-VL template under the same
``IMAGE_MAX_TOKEN_NUM`` as training, so ``--truncation_strategy delete`` can drop
over-long samples without encoding them on the fly.

Why not ``swift sft --dataset <parquet>`` directly: ms-swift preprocesses Hugging
Face datasets in batches of 1000 rows, and datasets combines each batch into a
single Arrow table before writing it. TextDART rows embed ~1.2MB of image bytes,
so a 1000-row batch can pass Arrow's ~2GB int32-offset limit and the map dies
with "offset overflow while concatenating arrays". This script runs the same
ms-swift preprocessors with a small batch size instead, and it can be split into
shards that run in parallel.

  map:    python prepare_dataset.py --data_dir D --output_dir O --split train --num_shards N --shard_index i
  merge:  python prepare_dataset.py --output_dir O --split train --num_shards N --merge

``prepare_dataset.sh`` launches all shards and the merge for you.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

# Pure CPU work: keep this process off the GPUs and out of any torchrun job
# whose RANK/WORLD_SIZE would make ms-swift wait on a process group.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    os.environ.pop(_key, None)

import torch  # noqa: E402
from datasets import concatenate_datasets, load_dataset, load_from_disk  # noqa: E402
from swift.arguments import ExportArguments  # noqa: E402
from swift.dataset import AddLengthPreprocessor, MessagesPreprocessor  # noqa: E402
from swift.utils import disable_safe_ddp_context_use_barrier  # noqa: E402
from swift.utils.utils import _patch_get_type_hints  # noqa: E402
from transformers import HfArgumentParser  # noqa: E402


def list_parquet_files(data_dir: Path, split: str) -> list[str]:
    files = []
    for path in sorted(data_dir.rglob("*.parquet")):
        relative = path.relative_to(data_dir)
        labels = set(relative.parts[:-1]) & {"train", "test"}
        labels.update(re.findall(r"(?:^|[-_])(train|test)(?=[-_]|$)", path.stem))
        if data_dir.name in {"train", "test"}:
            labels.add(data_dir.name)
        if len(labels) != 1:
            raise ValueError(f"cannot determine one train/test split for {path}")
        if split in labels:
            files.append(str(path.resolve()))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under --data_dir {data_dir}")
    return files


def split_manifest(args: argparse.Namespace, files: list[str]) -> dict:
    return {
        "split": args.split,
        "source_files": files,
        "source_stats": [[Path(f).stat().st_size, Path(f).stat().st_mtime_ns] for f in files],
        "model": args.model,
        "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
    }


def build_template(model: str, scratch_dir: Path):
    """Qwen3-VL template in train mode, with the processor only (no weights)."""
    with _patch_get_type_hints():
        parser = HfArgumentParser([ExportArguments])
    args = parser.parse_dict(
        {
            "model": model,
            "template": "qwen3_vl",
            "use_hf": True,
            "to_cached_dataset": True,
            "exist_ok": True,
            "output_dir": str(scratch_dir),
        },
        allow_extra_keys=True,
    )[0]
    template_cls = args.template_meta.template_cls
    kwargs = (
        {"return_dummy_model": True}
        if template_cls is not None and template_cls.use_model
        else {"load_model": False}
    )
    with torch.device("meta"):
        model_obj, processor = args.get_model_processor(**kwargs)
    template = args.get_template(processor)
    template.set_mode("train")
    if template.use_model:
        template.model = model_obj
    return template


def map_shard(args: argparse.Namespace) -> None:
    shards_dir = args.output_dir / "_shards" / args.split
    shard_dir = shards_dir / f"shard_{args.shard_index:03d}"
    done_path = shards_dir / f"shard_{args.shard_index:03d}.done"
    files = list_parquet_files(args.data_dir, args.split)
    manifest = split_manifest(args, files)
    if done_path.is_file():
        saved = json.loads(done_path.read_text(encoding="utf-8"))
        if saved["manifest"] != manifest:
            raise ValueError(f"shard metadata differs: {done_path}; use a separate output directory")
    if done_path.is_file() and (saved["rows"] == 0 or shard_dir.is_dir()):
        print(f"[prepare] shard {args.shard_index} already done: {shard_dir}", flush=True)
        return

    t0 = time.time()
    raw = load_dataset("parquet", data_files={args.split: files}, split=args.split)
    if len(raw) == 0:
        raise ValueError(f"{args.split} dataset is empty")
    start = len(raw) * args.shard_index // args.num_shards
    end = len(raw) * (args.shard_index + 1) // args.num_shards
    raw = raw.select(range(start, end))
    if len(raw) == 0:
        done_path.write_text(json.dumps({"rows": 0, "manifest": manifest}), encoding="utf-8")
        return
    print(
        f"[prepare] shard {args.shard_index}/{args.num_shards}: {len(raw)} rows, "
        f"IMAGE_MAX_TOKEN_NUM={os.environ.get('IMAGE_MAX_TOKEN_NUM', '<unset>')}",
        flush=True,
    )

    # MessagesPreprocessor (not AutoPreprocessor) because only RowPreprocessor
    # subclasses forward batch_size to datasets.map.
    with disable_safe_ddp_context_use_barrier():
        messages = MessagesPreprocessor()(
            raw, num_proc=1, load_from_cache_file=False, strict=False, batch_size=args.batch_size
        )
        template = build_template(args.model, shards_dir / f"_processor_{args.shard_index:03d}")
        mapped = AddLengthPreprocessor(template=template)(
            messages,
            num_proc=1,
            load_from_cache_file=False,
            strict=False,
            batch_size=args.batch_size,
        )
    for column in ("messages", "images", "lengths"):
        if column not in mapped.column_names:
            raise ValueError(
                f"shard {args.shard_index}: expected column {column!r}, got {mapped.column_names}"
            )

    tmp_dir = shard_dir.with_name(f"{shard_dir.name}.writing")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    mapped.save_to_disk(str(tmp_dir), num_proc=1)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp_dir.rename(shard_dir)
    done_path.write_text(
        json.dumps({"rows": len(mapped), "manifest": manifest}), encoding="utf-8"
    )
    print(
        f"[prepare] shard {args.shard_index}: saved {len(mapped)} rows in {time.time() - t0:.0f}s",
        flush=True,
    )


def merge_shards(args: argparse.Namespace) -> None:
    shards_dir = args.output_dir / "_shards" / args.split
    done_files = sorted(shards_dir.glob("shard_*.done"))
    if not done_files:
        raise FileNotFoundError(f"no shard_*.done under {shards_dir}; run the map step first")
    indices = [int(path.name[len("shard_") : -len(".done")]) for path in done_files]
    if indices != list(range(args.num_shards)):
        missing = sorted(set(range(args.num_shards)) - set(indices))
        raise FileNotFoundError(
            f"expected shards 0..{args.num_shards - 1} under {shards_dir}, have {indices}, missing {missing}"
        )

    records = [json.loads(path.read_text(encoding="utf-8")) for path in done_files]
    manifest = records[0]["manifest"]
    if manifest["split"] != args.split or manifest["num_shards"] != args.num_shards:
        raise ValueError("shard split/count does not match the requested merge")
    if any(record["manifest"] != manifest for record in records):
        raise ValueError("shards have inconsistent source files or preprocessing settings")
    datasets = [
        load_from_disk(str(shards_dir / f"shard_{i:03d}"))
        for i, record in zip(indices, records)
        if record["rows"] > 0
    ]
    if not datasets:
        raise ValueError(f"no usable rows in {args.split}")
    full = concatenate_datasets(datasets)
    lengths = [max(x) if isinstance(x, list) else int(x) for x in full["lengths"]]
    over = sum(1 for value in lengths if value > args.max_length)

    split_dir = args.output_dir / args.split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    full.save_to_disk(str(split_dir))
    (split_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[prepare] merged {len(indices)} shards -> {split_dir} ({len(full)} rows; "
        f"lengths min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.0f}; "
        f"{over} rows > {args.max_length} will be dropped at train time)",
        flush=True,
    )
    if not args.keep_shards:
        shutil.rmtree(shards_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="local copy of SujieHu/TextDART-123K (searched recursively for *.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="cached dataset root; writes <output_dir>/<split>",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="base model id or path (processor only)",
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="rows per datasets.map batch; keep small so a batch stays well under 2GB of image bytes",
    )
    parser.add_argument(
        "--merge", action="store_true", help="concatenate finished shards into <output_dir>/<split>"
    )
    parser.add_argument(
        "--max_length", type=int, default=8192, help="only used to report how many rows exceed it"
    )
    parser.add_argument(
        "--keep_shards", action="store_true", help="keep <output_dir>/_shards/<split> after merging"
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge_shards(args)
        return

    if args.data_dir is None:
        raise ValueError("--data_dir is required for the map step")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"expected 0 <= shard_index < num_shards, got shard_index={args.shard_index}, num_shards={args.num_shards}"
        )
    if args.batch_size < 1:
        raise ValueError(f"expected batch_size >= 1, got {args.batch_size}")
    (args.output_dir / "_shards" / args.split).mkdir(parents=True, exist_ok=True)
    map_shard(args)


if __name__ == "__main__":
    main()
