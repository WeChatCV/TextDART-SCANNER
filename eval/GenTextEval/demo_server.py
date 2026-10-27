"""Sanity check that a TextPecker vLLM server responds.

Sends the TextPecker recognition prompt with one image and prints the
recognized text, quality score and (if ``--target`` is given) semantic score.

Example:
    python demo_server.py --image path/to/image.png --target "Hello World"
"""

import argparse
import base64
import os
import sys
from io import BytesIO

from openai import OpenAI
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from eval.GenTextEval.scoring import get_score_v2, get_template  # noqa: E402

MODEL_NAME = "TextPecker"


def image_to_data_url(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    buffered = BytesIO()
    with Image.open(image_path) as image:
        image.convert("RGB").save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1", help="vLLM server host")
    parser.add_argument("--port", type=int, default=1925, help="vLLM server port")
    parser.add_argument("--image", required=True, help="path to an image containing text")
    parser.add_argument(
        "--target", default=None, help="optional reference text for the semantic score"
    )
    args = parser.parse_args()

    host = f"[{args.host}]" if ":" in args.host else args.host  # IPv6 needs brackets
    client = OpenAI(base_url=f"http://{host}:{args.port}/v1", api_key="EMPTY")

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(args.image)}},
                    {"type": "text", "text": get_template()},
                ],
            },
        ],
        temperature=0.0,
        max_tokens=2048,
        extra_body={"repetition_penalty": 1.2},
    )
    response = completion.choices[0].message.content
    if response is None:
        raise ValueError(f"server returned empty content for model {MODEL_NAME!r}: {completion!r}")

    quality_score, semantic_score, info = get_score_v2(
        response, args.target, qua_amplify_factor=1.0, vis_cls=True
    )
    print(f"Raw response:     {response}")
    print(f"Recognized text:  {info['recognized_text']}")
    print(f"Quality score:    {quality_score}")
    print(f"Semantic score:   {semantic_score}")


if __name__ == "__main__":
    main()
