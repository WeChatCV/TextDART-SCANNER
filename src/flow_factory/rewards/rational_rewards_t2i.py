# Copyright 2026 Jayce-Ping, Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rational Rewards for text-to-image: VLM rubric scoring over OpenAI-compatible HTTP.

Parses structured aspect scores from the model reply, averages selected aspects,
and maps the mean from rubric scale [1, 4] to reward [0, 1] via (mean - 1) / 3.
See ``guidance/rewards.md`` (VLM-as-Judge, Example: Rational Rewards) for
configuration and alignment notes with the reference rubric.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from ._rational_client import initialize_rational_client
from ._rational_scoring import (
    _extract_score_from_block as _extract_score_from_block,
    extract_numeric_score as extract_numeric_score,
    parse_aspect_scores,
)
from .abc import PointwiseRewardModel, RewardModelOutput

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

SUPPORTED_ASPECTS: Tuple[str, ...] = (
    "text_faithfulness",
    "physical_quality",
    "text_rendering",
)

RATIONAL_T2I_SYSTEM_PROMPT = (
    "You are an expert image generation evaluator. Your task is to evaluate "
    "the quality of a generated image based on a user instruction. Afterwards, "
    "you need to suggest how to refine the original user request to produce "
    "better image generation (if any)."
)


def parse_scores_from_detailed_judgement(
    detailed_judgement: str,
) -> Dict[str, Optional[Union[float, str]]]:
    """
    Parse aspect scores from the ``# Detailed Judgement`` section.

    Returns keys ``text_faithfulness``, ``physical_quality``, ``text_rendering``
    with float, the string ``N/A``, or ``None`` if missing.
    """
    return parse_aspect_scores(detailed_judgement, (
        ('text_faithfulness', 'Text Faithfulness'),
        ('physical_quality', 'Physical and Visual Quality'),
        ('text_rendering', 'Text Rendering'),
    ))


def aggregate_aspect_scores(
    parsed: Dict[str, Optional[Union[float, str]]],
    aspects: Sequence[str],
    *,
    supported_aspects: Sequence[str] = SUPPORTED_ASPECTS,
) -> float:
    """
    Average numeric scores for ``aspects`` (skip ``None`` and ``N/A``), clamp each
    to [1, 4], then map mean m to reward (m - 1) / 3 in [0, 1].

    ``supported_aspects`` defines the allowed aspect names (e.g. T2I vs edit rubrics).
    """
    if not aspects:
        raise ValueError("expected non-empty aspects sequence")

    allowed = tuple(supported_aspects)
    unknown = [a for a in aspects if a not in allowed]
    if unknown:
        raise ValueError(f"unknown aspect(s) {unknown!r}; supported: {list(allowed)}")

    scores: List[float] = []
    for aspect in aspects:
        score = parsed.get(aspect)
        if score is None or score == "N/A":
            continue
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"expected float or int for aspect {aspect!r} after parse, "
                f"got {type(score).__name__}: {score!r}"
            )
        score_value = float(score)
        if not math.isfinite(score_value):
            raise ValueError(f"non-finite score for aspect {aspect!r}: {score!r}")
        score_value = max(1.0, min(4.0, score_value))
        scores.append(score_value)

    if not scores:
        raise ValueError(
            f"no usable numeric scores for aspects {list(aspects)!r}; parsed={parsed!r}"
        )

    overall = sum(scores) / len(scores)
    normalized = (overall - 1.0) / 3.0
    return max(0.0, min(1.0, float(normalized)))


def _clip_vlm_text_for_log(text: str, max_len: int = 400) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return f"{collapsed[:max_len]}..."


# Rubric and output instructions after the generated image. Kept as one literal
# (no user-controlled ``<image>`` substring) so prompts may contain that text.
T2I_SCORING_PROMPT_SUFFIX = """


To do this, you must first assess the image on three critical aspects, provide justifications and absolute scores in 1-4 scale.

### Critical Aspects & Scoring Rubric
**1. Text Faithfulness** (How accurately does the output follow the instruction?)
- **4 (Full match):** All key elements (objects, colors, actions) are represented exactly as described. No hallucinations or unrequested changes.
- **3 (Minor mismatch):** Most key elements are present, but minor details are missing, incorrect, or slightly inaccurate.
- **2 (Some mismatch):** Some key elements are missing, altered, or interpreted incorrectly.
- **1 (Major deviations):** Key elements are completely missing, altered, or contradicted. Instruction is ignored.

**2. Physical and Visual Quality** (Technical errors, composition, realism, and physics)
- **4 (No noticeable flaws):** The image is physically plausible (correct lighting, shadows, geometry, anatomy). No visible artifacts (seams, blurring, noise).
- **3 (Minor flaws):** Small inaccuracies that are noticeable but not strongly disruptive (e.g., slight lighting mismatch, minor texture issues).
- **2 (Some flaws):** Clear physical or visual errors that disrupt the image (e.g., incorrect perspective, "floating" objects, wrong shadow direction, obvious seams).
- **1 (Severe flaws):** Major physical/visual errors (e.g., impossible geometry, distorted anatomy, garbled objects, severe artifacts).

**3. Text Rendering** (Only if the instruction involves generating text)
- **4 (Full match):** Text is correct, legible, and integrated well.
- **3 (Mostly match):** Minor misspellings or inconsistent capitalization.
- **2 (Partial match):** Major misspellings or distorted text.
- **1 (Major deviations):** Text is unreadable, severely distorted, or missing. (Use N/A if no text generation is required).

### Scoring Methodology (CRITICAL)
During assessment for each aspect, recall the initial user request and the scoring rubrics of the aspect, provide scores with detailed justifications for the generated image and reflect fine-grained preferences.
1. **Anchor:** Have a global inspection based on the user request and the resulting generation. Determine the rough integer score level (1, 2, 3, or 4) according to the definitions provided.
2. **Justify and Adjust:** Do careful visual analysis and identify specific flaws in generation. Justify the score with concrete evidence and scoring logic. Fine-tune this anchor score into a float value. Add small increments for exceptional execution or deduct points for specific flaws.
   - *Example:* deduct points from 4.0 for slight flaws if the assessed dimension is close to satisfaction. add increments from 1.0 or 2.0 based on severity of flaws.

Afterwards, try to construct a refined user request that helps the visual generation model to produce better image generation.
Think of the weaknesses identified in the judgement, then map them to instruction details and apply specific fixes.
Provide a final new user request that enrich the initial user request.

Output your evaluation in the following format:
# User Request Analysis
[ understanding the user request, try to analyze or decompose the user request deeper. Think of what the request might imply or what needs to be inferred to successfully execute the request. ]
# Detailed Judgement
1. Text Faithfulness:
## Justification: [ Analysis of the user request and the assessment of the resulting generation.\
 How it comes to a final score. ]
## Score: [ float score ]
2. Physical and Visual Quality:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score ]
3. Text Rendering:
## Justification: [ Similar to above. Analysis and assessment. ]
## Score: [ float score or N/A ]
# Summary: [ Summary of the evaluation ]

# User Request Refinement:
## Refinement Comments: [Specific suggestions for improving the user request]
## Refined Request: [The improved, more specific user request for generation like a standard user instruction]"""


def build_scoring_user_content(prompt: str, image_data_url: str) -> List[dict]:
    text_before = f"""User Instruction: {prompt}
You are provided with one image:
1. Generated Image """
    return [
        {"type": "text", "text": text_before},
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": T2I_SCORING_PROMPT_SUFFIX},
    ]


def build_scoring_messages(prompt: str, image_data_url: str) -> List[dict]:
    return [
        {"role": "system", "content": RATIONAL_T2I_SYSTEM_PROMPT},
        {"role": "user", "content": build_scoring_user_content(prompt, image_data_url)},
    ]


class RationalRewardsT2IRewardModel(PointwiseRewardModel):
    """
    Pointwise T2I reward via remote VLM (OpenAI-compatible chat completions).

    ``extra_kwargs`` (YAML keys alongside standard reward fields):

    - ``api_base_url`` (str): e.g. ``http://localhost:8000/v1``
    - ``api_key`` (str): default ``EMPTY``
    - ``vlm_model`` (str): OpenAI ``model`` id (default ``RationalRewards-8B-T2I``, must match vLLM
        ``--served-model-name``)
    - ``max_concurrent`` (int): default ``8``
    - ``max_retries`` (int): default ``5``
    - ``timeout`` (float | int): per request, default ``180``
    - ``temperature`` (float): default ``0.1``
    - ``max_tokens`` (int): default ``2048``
    - ``aspects`` (list[str]): subset of supported aspects; default all three
    """

    required_fields = ("prompt", "image")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        initialize_rational_client(
            self, config, "RationalRewards-8B-T2I", SUPPORTED_ASPECTS,
            "RationalRewardsT2IRewardModel",
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if image is None and video is not None:
            image = [frames[0] for frames in video]
        if image is None:
            raise ValueError("Either 'image' or 'video' must be provided for RationalRewardsT2I")
        if len(prompt) != len(image):
            raise ValueError(
                f"expected len(prompt)==len(image), got {len(prompt)} prompts and {len(image)} images"
            )

        scores = asyncio.run(self._async_score_batch(prompt, image))
        rewards = torch.tensor(scores, dtype=torch.float32, device=self.device)
        return RewardModelOutput(rewards=rewards, extra_info={})

    async def _async_score_batch(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> List[float]:
        tasks = [self._score_single(p, img) for p, img in zip(prompts, images)]
        return list(await asyncio.gather(*tasks))

    async def _score_single(self, prompt: str, image: Image.Image) -> float:
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        data_url = pil_image_to_base64(image, format="PNG")
        messages = build_scoring_messages(prompt, data_url)

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.vlm_model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                    )
            except (APIConnectionError, APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                last_err = e
                logger.warning(
                    "RationalRewardsT2I API transport error (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt + 1 >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
                continue

            content = completion.choices[0].message.content
            if content is None or not str(content).strip():
                logger.warning(
                    "RationalRewardsT2I VLM returned empty assistant content; using reward 0.0"
                )
                return 0.0
            text = str(content)
            try:
                parsed = parse_scores_from_detailed_judgement(text)
                return aggregate_aspect_scores(parsed, self.aspects)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "RationalRewardsT2I failed to parse or aggregate VLM response; using reward 0.0: %s. "
                    "Response (truncated): %s",
                    e,
                    _clip_vlm_text_for_log(text),
                )
                return 0.0

        logger.warning(
            "RationalRewardsT2I HTTP request failed after %s attempt(s); using reward 0.0. Last error: %s",
            self.max_retries,
            last_err,
        )
        return 0.0


__all__ = [
    'RationalRewardsT2IRewardModel',
    '_extract_score_from_block',
    'aggregate_aspect_scores',
    'build_scoring_messages',
    'build_scoring_user_content',
    'extract_numeric_score',
    'parse_scores_from_detailed_judgement',
]
