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
#
# Portions derived from CIawevy/TextPecker, licensed under Apache-2.0.
"""Score GenTextEval images with a TextPecker vLLM server (Qua / Sem).

Images are read from ``<sample_dir>/<prompt_id>_<k>.png`` (or ``<prompt_id>.png``)
and matched to ``GenTextEval_{en,zh}.jsonl`` in this directory. Each image gets
one record in ``<output_dir>/eval_results.jsonl`` with ``pecker_qua`` (text
quality) and ``pecker_gned`` (semantic accuracy against the target text).

The run is resumable: images whose id is already in ``eval_results.jsonl`` are
skipped, so an interrupted run can simply be relaunched.

Usage:
    python GenTextEval/textpecker_eval.py \
        --sample_dir GenTextEval/eval_results/<base>/<model>_EN \
        --output_dir GenTextEval/eval_metrics/<base>/<model>_EN \
        --vllm_host 127.0.0.1 --port 1925
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import glob
import json
import os
import sys
from io import BytesIO
from typing import Dict, List

import openai
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BENCH_DIR, "..", ".."))
from eval.GenTextEval.scoring import get_score_v2  # noqa: E402

PROMPT_FILES = {
    "EN": os.path.join(BENCH_DIR, "GenTextEval_en.jsonl"),
    "ZH": os.path.join(BENCH_DIR, "GenTextEval_zh.jsonl"),
}
SERVED_MODEL_NAME = "TextPecker"
# Scores are reported without amplification at evaluation time.
QUA_AMPLIFY_FACTOR = 1.0

REC_PROMPT = """
        This is a text-generated image. Please recognize all visible text in the entire image.
        Marking rules:
        1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
        2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
        Output in the following JSON format:
        {
        "recognized_text": "All text in the image (including structural error markers)"
        }
        """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample_dir", required=True, help="Directory of generated images for one model."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory that receives eval_results.jsonl."
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "EN", "ZH"],
        default="auto",
        help="Prompt language; 'auto' reads the _EN/_ZH suffix of --sample_dir.",
    )
    parser.add_argument(
        "--vllm_host", required=True, help="TextPecker vLLM host, or a comma-separated list."
    )
    parser.add_argument("--port", type=int, default=1925, help="TextPecker vLLM port.")
    parser.add_argument("--batch_size", type=int, default=32, help="Concurrent requests per batch.")
    parser.add_argument(
        "--max_retries", type=int, default=20, help="Retries per request on API errors."
    )
    return parser.parse_args()


def resolve_lang(lang: str, sample_dir: str) -> str:
    if lang != "auto":
        return lang
    name = os.path.basename(os.path.normpath(sample_dir)).upper()
    if name.endswith("_EN"):
        return "EN"
    if name.endswith("_ZH"):
        return "ZH"
    raise ValueError(
        f"cannot infer language from sample_dir {sample_dir!r}; expected a _EN/_ZH suffix or pass --lang"
    )


def load_samples(sample_dir: str, lang: str) -> List[Dict]:
    with open(PROMPT_FILES[lang], encoding="utf-8") as f:
        prompt_map = {}
        for line in f:
            if line.strip():
                rec = json.loads(line)
                prompt_map[int(rec["prompt_id"])] = rec

    samples = []
    for image_path in sorted(glob.glob(os.path.join(sample_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        head = stem.split("_")[0]
        if not head.isdigit():
            raise ValueError(f"expected '<prompt_id>[_<k>].png', got {image_path}")
        prompt_id = int(head)
        if prompt_id not in prompt_map:
            raise KeyError(
                f"prompt_id {prompt_id} from {image_path} not found in {PROMPT_FILES[lang]}"
            )
        info = prompt_map[prompt_id]
        samples.append(
            {
                "id": f"gentexteval_{stem}",
                "image": image_path,
                "prompt": info["prompt"],
                "target": info["target"],
            }
        )
    return samples


def load_done_ids(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def append_jsonl(path: str, records: List[Dict]) -> None:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    with open(path, "ab") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def image_to_data_url(path: str) -> str:
    buffered = BytesIO()
    with Image.open(path) as img:
        img.convert("RGB").save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_clients(hosts: str, port: int) -> List[AsyncOpenAI]:
    clients = []
    for host in (h.strip() for h in hosts.split(",")):
        if not host:
            continue
        netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host
        clients.append(AsyncOpenAI(base_url=f"http://{netloc}:{port}/v1", api_key="EMPTY"))
    if not clients:
        raise ValueError(f"no valid host in --vllm_host {hosts!r}")
    return clients


async def recognize(client: AsyncOpenAI, data_url: str, max_retries: int) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=SERVED_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": REC_PROMPT},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=2048,
                extra_body={"repetition_penalty": 1.2},
            )
            return response.choices[0].message.content
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
            if attempt == max_retries:
                raise RuntimeError(f"TextPecker request failed after {max_retries} attempts") from e
            await asyncio.sleep(min(5 * 2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


async def score_batch(
    clients: List[AsyncOpenAI], batch: List[Dict], offset: int, max_retries: int
) -> List[Dict]:
    tasks = [
        recognize(clients[(offset + i) % len(clients)], image_to_data_url(s["image"]), max_retries)
        for i, s in enumerate(batch)
    ]
    responses = await asyncio.gather(*tasks)
    results = []
    for sample, response in zip(batch, responses):
        qua, gned, recs = get_score_v2(response, sample["target"], QUA_AMPLIFY_FACTOR, True)
        results.append({**sample, "pecker_qua": qua, "pecker_gned": gned, "pecker_recs": recs})
    return results


async def run(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.sample_dir):
        raise FileNotFoundError(f"sample_dir does not exist: {args.sample_dir}")
    lang = resolve_lang(args.lang, args.sample_dir)
    samples = load_samples(args.sample_dir, lang)
    if not samples:
        raise ValueError(f"no *.png images found under {args.sample_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_jsonl = os.path.join(args.output_dir, "eval_results.jsonl")
    done = load_done_ids(result_jsonl)
    pending = [s for s in samples if s["id"] not in done]
    print(
        f"[GenTextEval/{lang}] total={len(samples)} done={len(done)} pending={len(pending)} -> {result_jsonl}"
    )

    clients = build_clients(args.vllm_host, args.port)
    for start in tqdm(range(0, len(pending), args.batch_size), desc="TextPecker", unit="batch"):
        batch = pending[start : start + args.batch_size]
        append_jsonl(result_jsonl, await score_batch(clients, batch, start, args.max_retries))


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
