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

# src/flow_factory/rewards/ocr.py
"""
OCR Reward Model using PP-OCRv5.
Optional dependencies:
    pip install -c constraints-scanner.txt -e ".[ocr]"
    pip install -c constraints-scanner.txt paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
For CPU inference, install paddlepaddle==3.3.0 instead of paddlepaddle-gpu and
set the reward device to "cpu". The included RL configs do not enable this reward.
When ocr_det_model_dir and ocr_rec_model_dir are unset, PaddleOCR downloads the
PP-OCRv5_server_det and PP-OCRv5_server_rec weights during initialization. To use
local weights, set those two paths in the reward config. Verify CUDA/cuDNN runtime
compatibility when combining GPU PaddlePaddle with PyTorch; pip check does not
validate shared-library compatibility.
"""

import os

# Must be set before paddle is imported: disable oneDNN / PIR (PP-OCRv5
# compatibility on paddle 3.x), skip the model-source connectivity check and
# run headless.
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["PADDLE_DISABLE_PIR"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "false"

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger
from .abc import PointwiseRewardModel, RewardModelOutput

logger = setup_logger(__name__)

# Local PP-OCRv5 model directories. ``None`` lets PaddleOCR download the
# official ``PP-OCRv5_server_det`` / ``PP-OCRv5_server_rec`` weights; override
# via ``ocr_det_model_dir`` / ``ocr_rec_model_dir`` in the reward config.
DEFAULT_PADDLEOCR_DET_MODEL_DIR: Optional[str] = None
DEFAULT_PADDLEOCR_REC_MODEL_DIR: Optional[str] = None

# Keys searched (in order) inside each sample's metadata dict to find the
# ground-truth string the rendered image is supposed to contain.
_DEFAULT_TARGET_KEYS: tuple[str, ...] = (
    "target",
    "targets",
    "target_text",
    "target_texts",
    "text",
    "texts",
    "ocr_text",
    "ocr_texts",
    "gt",
)

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

try:
    from Levenshtein import distance
except ImportError:
    raise ImportError(
        "python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein"
    )


def _build_paddleocr(
    device: str = "cpu",
    det_model_dir: Optional[str] = DEFAULT_PADDLEOCR_DET_MODEL_DIR,
    rec_model_dir: Optional[str] = DEFAULT_PADDLEOCR_REC_MODEL_DIR,
) -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir=det_model_dir,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir=rec_model_dir,
        device=device,
        # PaddleOCR's predictor setting overrides the process-level flags above.
        # Disable oneDNN explicitly to avoid unsupported PP-OCRv5 PIR attributes.
        enable_mkldnn=False,
        # Match the detector's training-time input scale (long side <= 960).
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.5,
        text_det_box_thresh=0.6,
        lang="ch",
    )


def _extract_target_text(prompt: str) -> str:
    """Extract all quoted target texts from prompt."""
    quoted_texts = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
    return "".join(quoted_texts) if quoted_texts else prompt


def _resolve_targets(
    prompts: List[str],
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    target: Optional[Union[str, List[str]]] = None,
    targets: Optional[Union[str, List[str]]] = None,
    target_keys: Sequence[str] = _DEFAULT_TARGET_KEYS,
    joiner: str = "",
) -> List[str]:
    """Resolve one ground-truth string per prompt.

    Resolution order (first source that yields a non-None value wins):
        1. Explicit ``targets`` / ``target`` arg (broadcast scalar to list).
        2. Per-sample ``metadata[i][key]`` for ``key in target_keys``.
        3. Quoted substrings inside the prompt (e.g. ``... saying "Hello"``).
        4. The full prompt as a conservative fallback.

    A list value found in metadata is joined with ``joiner`` (default ""):
    each character/word block in OCR targets is typically meant to be
    concatenated, matching the existing ``_extract_target_text`` behavior.
    """
    explicit = targets if targets is not None else target
    if explicit is not None:
        if isinstance(explicit, str):
            return [explicit] * len(prompts)
        if len(explicit) != len(prompts):
            raise ValueError(
                f"explicit target list length {len(explicit)} does not match "
                f"prompt batch size {len(prompts)}"
            )
        return [str(item) for item in explicit]

    metadata = metadata or [None] * len(prompts)
    if len(metadata) != len(prompts):
        raise ValueError(
            f"metadata length {len(metadata)} does not match prompt batch " f"size {len(prompts)}"
        )

    resolved: List[str] = []
    for prompt, meta in zip(prompts, metadata):
        value = None
        if isinstance(meta, dict):
            for key in target_keys:
                if key in meta and meta[key] is not None:
                    value = meta[key]
                    break
        if isinstance(value, (list, tuple)):
            value = joiner.join(str(item) for item in value)
        if value is None:
            value = _extract_target_text(prompt)
        resolved.append(str(value))
    return resolved


def _recognize_text(ocr: PaddleOCR, image: Image.Image | np.ndarray) -> str:
    """Run PP-OCRv5 and concatenate recognized texts."""
    if isinstance(image, Image.Image):
        image = np.array(image)
        # PaddleOCR expects BGR input.
        image = image[:, :, ::-1]

    result = ocr.predict(image)
    recognized_text = ""
    for res in result:
        if isinstance(res, dict):
            recognized_text += "".join(res.get("rec_texts", []))
        elif hasattr(res, "json"):
            rec_texts = res.json.get("res", {}).get("rec_texts", [])
            recognized_text += "".join(rec_texts)
    return recognized_text


def _compute_ocr_reward(recognized_text: str, target_text: str) -> float:
    """Compute OCR reward according to normalized edit distance."""
    recognized_text = recognized_text.replace(" ", "").lower()
    target_text = target_text.replace(" ", "").lower()
    if not target_text:
        return 0.0

    if target_text in recognized_text:
        dist = 0
    else:
        dist = distance(recognized_text, target_text)
    if dist > len(target_text):
        dist = len(target_text)

    return 1 - dist / len(target_text)


class OCRRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image", "video")

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        device_index = self.accelerator.local_process_index

        det_model_dir = getattr(self.config, "ocr_det_model_dir", DEFAULT_PADDLEOCR_DET_MODEL_DIR)
        rec_model_dir = getattr(self.config, "ocr_rec_model_dir", DEFAULT_PADDLEOCR_REC_MODEL_DIR)

        # Initialize the PP-OCRv5 server det/rec reader
        self.model = _build_paddleocr(
            device=f"gpu:{device_index}" if "cuda" in str(self.device) else "cpu",
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
        )

    def _compute_scores_batch(
        self,
        prompt: list[str],
        image: list[Image.Image],
        target_texts: list[str],
    ) -> torch.Tensor:
        """Compute OCR reward for a batch of image-prompt pairs.

        ``target_texts`` must already be aligned with ``prompt``/``image``;
        callers are responsible for resolving them via :func:`_resolve_targets`.
        """
        if len(target_texts) != len(prompt):
            raise ValueError(
                f"target_texts length {len(target_texts)} does not match "
                f"prompt batch size {len(prompt)}"
            )
        rewards = []
        for img, target_text in zip(image, target_texts):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                recognized_text = _recognize_text(self.model, img)
                reward = _compute_ocr_reward(recognized_text, target_text)

            except Exception as e:  # pylint: disable=broad-except
                # Preserve the per-sample fallback for OCR backend failures.
                # PaddleOCR can raise on degenerate inputs (empty crops, etc.).
                # Treat as a 0-reward sample rather than killing the rollout.
                logger.error(f"OCR processing failed: {str(e)}")
                reward = 0.0

            rewards.append(reward)

        return torch.tensor(rewards, device=self.device, dtype=torch.float32)

    def _compute_video_scores(
        self,
        prompt: list[str],
        video: list[list[Image.Image]],
        target_texts: list[str],
        batch_size: int,
    ) -> torch.Tensor:
        """Mean OCR reward across all frames of each video.

        Uses flat-reconstruct strategy to handle variable frame counts
        while maintaining efficient batched computation. Each video shares
        a single target text replicated across its frames.
        """
        if len(target_texts) != len(prompt):
            raise ValueError(
                f"target_texts length {len(target_texts)} does not match "
                f"video batch size {len(prompt)}"
            )
        frame_counts = [len(clip) for clip in video]
        flat_images = [frame for clip in video for frame in clip]
        flat_prompts = [p for p, n in zip(prompt, frame_counts) for _ in range(n)]
        flat_targets = [t for t, n in zip(target_texts, frame_counts) for _ in range(n)]

        all_scores = []
        for i in range(0, len(flat_images), batch_size):
            batch_scores = self._compute_scores_batch(
                flat_prompts[i : i + batch_size],
                flat_images[i : i + batch_size],
                flat_targets[i : i + batch_size],
            )
            all_scores.append(batch_scores)
        flat_scores = torch.cat(all_scores, dim=0)

        scores = flat_scores.split(frame_counts)
        scores = torch.stack([s.mean() for s in scores])
        return scores

    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        target: Optional[Union[str, List[str]]] = None,
        targets: Optional[Union[str, List[str]]] = None,
    ) -> RewardModelOutput:
        """Compute OCR reward for a batch.

        Target-string resolution order (see :func:`_resolve_targets`):
            ``targets`` / ``target`` kwarg -> per-sample ``metadata`` dict (keys
            ``target``, ``targets``, ``target_text``, ``text``, ``ocr_text``,
            ``gt``, ...) -> quoted substrings in ``prompt`` -> full ``prompt``.

        This lets datasets that ship a separate ground-truth column (e.g.
        ``dataset/ocr_v2`` with ``targets``) bypass the brittle quote-extraction
        heuristic. Prompts without separate targets use quote extraction.
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is not None and video is not None:
            raise ValueError("Only one of image or video can be provided.")

        target_texts = _resolve_targets(
            prompts=prompt,
            metadata=metadata,
            target=target,
            targets=targets,
        )

        batch_size = getattr(self.config, "batch_size", len(prompt))

        if video is not None:
            scores = self._compute_video_scores(prompt, video, target_texts, batch_size)
        else:
            scores = self._compute_scores_batch(prompt, image, target_texts)

        return RewardModelOutput(
            rewards=scores,
            extra_info={"target_texts": target_texts},
        )


def download_model():
    ocr = _build_paddleocr()
    logger.info("PaddleOCR initialized successfully")
    return ocr


if __name__ == "__main__":
    download_model()
