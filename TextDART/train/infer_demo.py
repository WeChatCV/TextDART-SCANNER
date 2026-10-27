#!/usr/bin/env python3
"""Query a deployed TextDART server on a few images and visualize the answer.

Sends each image with the same system / user prompts TextDART was trained on,
parses the ``regions`` JSON, and writes ``results.json`` plus an annotated copy of
every image (quadrilaterals and transcriptions drawn at the original resolution).

TextDART answers quads on a 0-1000 normalized grid of the image, so
``quad_orig = quad * (width, height) / 1000``.

Usage:
    bash deploy_server.sh                       # in another shell
    python infer_demo.py                        # defaults to ../assets
    python infer_demo.py --images a.png b.jpg --server-url http://127.0.0.1:1925
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

TEXTDART_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
NORM_SIDE = 1000
# IMAGE_MAX_TOKEN_NUM (2304) x Qwen3-VL patch 16 x spatial merge 2, squared.
MAX_UPLOAD_PIXELS = 2304 * 32 * 32
PALETTE = (
    (255, 59, 48),
    (0, 122, 255),
    (52, 199, 89),
    (255, 149, 0),
    (175, 82, 222),
    (255, 45, 85),
    (0, 199, 190),
)

# Same wording as the training data, minus the leading "<image>" line: the
# OpenAI-compatible server inserts the image placeholder from the image_url part.
USER_TEMPLATE_PLAIN = """This is a text-generated image.
Locate and recognize all visible text in the entire image, and report every text region as a quadrilateral, following the marking and grounding rules.
Output in the following JSON format:
{
"regions": [{"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "recognized_text": "text inside the quadrilateral (including structural error markers)"}]
}"""


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        paths = [Path(p).expanduser().resolve() for p in args.images]
    else:
        if not args.image_dir.is_dir():
            raise NotADirectoryError(f"--image-dir is not a directory: {args.image_dir}")
        paths = sorted(
            p.resolve() for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no {'/'.join(IMAGE_SUFFIXES)} file in {args.image_dir}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
    return paths


def image_to_data_url(path: Path) -> tuple[str, int, int]:
    """Return (data URL, original width, original height).

    Images above the server's pixel budget are downscaled before upload; the
    server would shrink them anyway and quads are resolution independent.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width * height > MAX_UPLOAD_PIXELS:
            scale = (MAX_UPLOAD_PIXELS / (width * height)) ** 0.5
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.Resampling.BICUBIC
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}", width, height


def query(
    args: argparse.Namespace,
    session: requests.Session,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
) -> str:
    payload = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    response = session.post(
        f"{args.server_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=args.timeout
    )
    response.raise_for_status()
    choices = response.json().get("choices")
    if not choices:
        raise RuntimeError(f"server returned no choices: {response.text[:500]}")
    return choices[0]["message"]["content"] or ""


def parse_regions(raw_response: str) -> list[dict[str, Any]]:
    match = JSON_BLOCK_RE.search(raw_response)
    body = match.group(1) if match else raw_response.strip()
    parsed = json.loads(body)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("regions"), list):
        raise ValueError(f"expected an object with a 'regions' list, got: {body[:300]!r}")
    return parsed["regions"]


def to_original(quad: list[float], width: int, height: int) -> list[int]:
    if len(quad) != 8:
        raise ValueError(f"expected 8 numbers per quad, got {len(quad)}: {quad!r}")
    return [round(v * (width if i % 2 == 0 else height) / NORM_SIDE) for i, v in enumerate(quad)]


def load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path is None:
        # Pillow's bundled font has no CJK glyphs; pass --font-path for Chinese labels.
        return ImageFont.load_default(size=size)
    if not Path(font_path).is_file():
        raise FileNotFoundError(f"--font-path not found: {font_path}")
    return ImageFont.truetype(font_path, size)


def annotate(record: dict[str, Any], output_dir: Path, font_path: str | None) -> str:
    with Image.open(record["image"]) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_path, max(14, min(canvas.size) // 36))
    line_width = max(2, min(canvas.size) // 400)
    for index, region in enumerate(record["regions"]):
        points = list(zip(region["quad_orig"][0::2], region["quad_orig"][1::2]))
        color = PALETTE[index % len(PALETTE)]
        draw.polygon(points, outline=color, width=line_width)
        x, y = min(p[0] for p in points), min(p[1] for p in points)
        label = f"{index}:{region['recognized_text']}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        y = max(0, y - (bottom - top) - 4)
        draw.rectangle((x, y, x + right - left + 4, y + bottom - top + 4), fill=color)
        draw.text((x + 2 - left, y + 2 - top), label, font=font, fill=(255, 255, 255))
    out_path = output_dir / f"{Path(record['image']).stem}_vis.png"
    canvas.save(out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images", nargs="+", help="image paths (default: every image in --image-dir)"
    )
    parser.add_argument("--image-dir", type=Path, default=TEXTDART_ROOT / "assets")
    parser.add_argument("--server-url", default="http://127.0.0.1:1925")
    parser.add_argument(
        "--model-name", default="TextDART", help="--served_model_name of deploy_server.sh"
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=TEXTDART_ROOT / "demo_results")
    parser.add_argument("--font-path", help="CJK-capable .ttf/.otf for labels, e.g. NotoSansCJK")
    args = parser.parse_args()

    system_prompt = (TEXTDART_ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip()

    images = collect_images(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    records = []
    for path in images:
        image_url, width, height = image_to_data_url(path)
        started = time.time()
        raw_response = query(args, session, system_prompt, USER_TEMPLATE_PLAIN, image_url)
        record: dict[str, Any] = {
            "image": str(path),
            "width": width,
            "height": height,
            "latency_s": round(time.time() - started, 2),
            "raw_response": raw_response,
            "regions": [],
        }
        # A malformed answer is a model-quality signal, not a script bug: keep the
        # raw text in the report and continue with the next image.
        try:
            regions = parse_regions(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
            regions = []
        for region in regions:
            quad = [float(v) for v in region["quad"]]
            record["regions"].append(
                {
                    "quad": quad,
                    "quad_orig": to_original(quad, width, height),
                    "recognized_text": region.get("recognized_text", ""),
                }
            )
        if record["regions"]:
            record["vis_path"] = annotate(record, args.output_dir, args.font_path)
        records.append(record)
        status = record.get("parse_error") or f"{len(record['regions'])} regions"
        print(f"{path.name:<40} {record['latency_s']:6.2f}s  {status}", flush=True)
        for region in record["regions"]:
            print(f"    {region['recognized_text']!r}  quad={region['quad']}", flush=True)

    out_file = args.output_dir / "results.json"
    out_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_file} and annotated images to {args.output_dir}")


if __name__ == "__main__":
    main()
