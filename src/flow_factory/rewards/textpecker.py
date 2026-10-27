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

# src/flow_factory/rewards/textpecker.py
"""Remote TextPecker reward model.

This module implements the TextPecker scoring interface rather than binding the
reward to a specific VLM backend. The remote server can be backed by any
TextPecker-compatible VLM, such as a custom fine-tuned Qwen3-VL, Intern3-VL, or
another compatible model. It sends generated images, prompts, and target texts
to a remote server, then aggregates TextPecker's two component scores:

    final = semantic_weight * semantic_score + quality_weight * quality_score

If the server only returns a scalar ``rewards`` list, the wrapper falls back to
that list so it remains compatible with the generic Flow-Factory reward server.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests
import torch
from accelerate import Accelerator
from PIL import Image

from eval.GenTextEval.scoring import get_score_v2

from flow_factory.hparams import RewardArguments
from flow_factory.rewards._http_clients import (
    EndpointRotation,
    ServerPool,
    make_requests_session,
    normalize_openai_base_url,
)
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput
from flow_factory.rewards.json_lenient import normalize_ocr_json_answer

logger = logging.getLogger(__name__)

_IMAGE_KEYS = ("image", "images")
_SEMANTIC_KEYS = ("semantic_scores", "sem_scores", "semantic", "semantics")
_QUALITY_KEYS = (
    "quality_scores",
    "qua_scores",
    "scaled_quality_scores",
    "quality",
    "qualities",
)
_REWARD_KEYS = ("rewards", "scores", "outputs", "avg")
_DEFAULT_TARGET_KEYS = (
    "target",
    "targets",
    "target_text",
    "target_texts",
    "text",
    "texts",
    "ocr_text",
    "ocr_texts",
)
_TEXTPECKER_REC_PROMPT = """
This is a text-generated image. Please recognize all visible text in the entire image.
Marking rules:
1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
Output in the following JSON format:
{
    "recognized_text": "All text in the image (including structural error markers)"
}
"""
_thread_local = threading.local()


def _image_to_b64(img: Union[Image.Image, torch.Tensor], image_format: str = "PNG") -> str:
    """Convert a PIL image or ``(C,H,W)`` tensor in ``[0,1]`` to base64."""
    if isinstance(img, torch.Tensor):
        arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=image_format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_to_data_url(img: Union[Image.Image, torch.Tensor]) -> str:
    return f"data:image/jpeg;base64,{_image_to_b64(img, image_format='JPEG')}"


def _video_to_b64(video: Union[List[Image.Image], torch.Tensor]) -> List[str]:
    if isinstance(video, torch.Tensor):
        return [_image_to_b64(frame) for frame in video]
    return [_image_to_b64(frame) for frame in video]


def _resolve_server_urls(
    server_url: Optional[str],
    server_urls: Union[str, Sequence[str], None],
) -> List[str]:
    """Coerce ``server_url`` / ``server_urls`` into a non-empty list of URLs."""
    if server_urls:
        if isinstance(server_urls, str):
            urls = [u.strip() for u in server_urls.split(",") if u.strip()]
        else:
            urls = [str(u).strip() for u in server_urls if str(u).strip()]
    elif server_url:
        urls = [server_url.strip()]
    else:
        raise ValueError(
            "TextPecker reward requires `server_url` or `server_urls` " "(list or comma-separated)"
        )
    urls = [u.rstrip("/") for u in urls if u]
    if not urls:
        raise ValueError("`server_urls` resolved to an empty URL list")
    return urls


def _run_async_task(coro: Any) -> Any:
    if not hasattr(_thread_local, "loop") or _thread_local.loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    loop = _thread_local.loop
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    return loop.run_until_complete(coro)


def _load_get_score_v2(
    parse_utils_path: Union[str, Path, None],
) -> Callable[..., Tuple[float, float, Dict[str, Any]]]:
    """Load ``get_score_v2`` and wrap it with lenient OCR JSON repair.

    ``eval.GenTextEval.scoring`` scores the model
    answer after strict JSON normalization so unescaped quotes / truncated braces do not
    zero out a sample.
    """

    def _wrap(get_score_v2: Callable[..., Tuple[float, float, Dict[str, Any]]]):
        def _safe_get_score_v2(string, *args, **kwargs):
            return get_score_v2(normalize_ocr_json_answer(string), *args, **kwargs)

        return _safe_get_score_v2

    if parse_utils_path is None:
        return _wrap(get_score_v2)

    path = Path(parse_utils_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scoring module not found at {path}. Set parse_utils_path in reward config."
        )
    module_name = "flow_factory_textpecker_parse_utils"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scoring module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return _wrap(module.get_score_v2)


def _extract_quoted_target(prompt: str, joiner: str = " ") -> str:
    """Extract target text from quoted substrings in a prompt.

    The OCR dataset uses prompts such as ``... displaying "Spring Collection"``.
    If no quoted text exists, the full prompt is used as a conservative fallback.
    """
    quoted_texts = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
    return joiner.join(quoted_texts) if quoted_texts else prompt


def _as_float_list(values: Any, name: str) -> Optional[List[float]]:
    """Convert a sequence/tensor/scalar-list field to ``List[float]``."""
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    if isinstance(values, (int, float)):
        values = [values]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a list-like value, got {type(values).__name__}")
    return [float(v) for v in values]


def _is_nested_score_triplet(values: Any) -> bool:
    """Return whether values look like ``[final, quality, semantic]`` score lists."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    if len(values) != 3:
        return False
    return all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in values)


def _find_first(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _normalize_component_weights(weights: Any) -> Dict[str, float]:
    if weights is None:
        return {"semantic": 0.5, "quality": 0.5}
    if not isinstance(weights, dict):
        raise TypeError("component_weights must be a mapping, e.g. {semantic: 0.5, quality: 0.5}")
    return {
        "semantic": float(weights.get("semantic", weights.get("sem", 0.0))),
        "quality": float(weights.get("quality", weights.get("qua", 0.0))),
    }


def _extract_targets_from_metadata(
    prompts: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    target: Optional[Union[List[str], str]] = None,
    targets: Optional[Union[List[str], str]] = None,
    target_keys: Sequence[str] = _DEFAULT_TARGET_KEYS,
    prompt_target_joiner: str = " ",
) -> List[str]:
    """Resolve target texts from explicit target fields, metadata, or prompt quotes."""
    explicit = targets if targets is not None else target
    if explicit is not None:
        if isinstance(explicit, str):
            return [explicit] * len(prompts)
        return [str(item) for item in explicit]

    resolved: List[str] = []
    metadata = metadata or [{} for _ in prompts]
    for prompt, meta in zip(prompts, metadata):
        value = None
        if isinstance(meta, dict):
            for key in target_keys:
                if key in meta and meta[key] is not None:
                    value = meta[key]
                    break
        if isinstance(value, (list, tuple)):
            value = prompt_target_joiner.join(str(item) for item in value)
        if value is None:
            value = _extract_quoted_target(prompt, prompt_target_joiner)
        resolved.append(str(value))
    return resolved


def aggregate_textpecker_scores(
    semantic_scores: List[float],
    quality_scores: List[float],
    component_weights: Dict[str, float],
    normalize_component_weights: bool = False,
) -> List[float]:
    """Aggregate TextPecker semantic and quality scores into one scalar per sample."""
    semantic_weight = component_weights["semantic"]
    quality_weight = component_weights["quality"]
    denominator = semantic_weight + quality_weight if normalize_component_weights else 1.0
    if denominator == 0:
        raise ValueError("semantic and quality component weights cannot both be 0")
    return [
        (semantic_weight * sem + quality_weight * qua) / denominator
        for sem, qua in zip(semantic_scores, quality_scores)
    ]


class TextPeckerRemoteClient:
    """HTTP client for a pool of TextPecker-compatible vLLM or reward servers.

    Requests round-robin over the pool starting at ``start_index`` (callers
    pass the process rank so ranks spread out). A server that errors is put in
    a short cooldown and the request immediately moves to the next one.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 3,
        protocol: str = "openai",
        model_name: str = "TextPecker",
        api_key: str = "EMPTY",
        max_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        use_env_proxies: bool = False,
        server_urls: Union[str, Sequence[str], None] = None,
        start_index: int = 0,
        failure_cooldown: float = 60.0,
        probe_health: bool = False,
    ):
        urls = _resolve_server_urls(server_url, server_urls)
        self.timeout = timeout
        self.retries = retries
        self.protocol = protocol
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        # Default to ignoring HTTP(S)_PROXY env vars: TextPecker / vLLM
        # endpoints are LAN-local services configured by URL and should
        # not be routed through a corporate outbound proxy (see
        # ``flow_factory.rewards._http_clients``).
        self._session = make_requests_session(use_env_proxies=use_env_proxies)
        self.pool: Optional[ServerPool] = None

        if self.protocol == "openai":
            self.pool = ServerPool(
                urls=urls,
                api_key=api_key,
                timeout=timeout,
                use_env_proxies=use_env_proxies,
                model_name=model_name,
                session=self._session,
                probe_health=probe_health,
                failure_cooldown=failure_cooldown,
                start_index=start_index,
                pool_name="TextPecker server",
            )
            self.urls = self.pool.base_urls
        else:
            if probe_health:
                urls = self._keep_reachable(urls)
            self.urls = urls
            self._rotation = EndpointRotation(
                len(urls),
                failure_cooldown=failure_cooldown,
                start_index=start_index,
            )
        # Primary endpoint for this rank; kept for logs and error messages.
        self.base_url = self.urls[start_index % len(self.urls)]
        self.url = self.base_url

    def _endpoint_reachable(self, url: str) -> bool:
        """Probe one endpoint with the health route its protocol exposes."""
        endpoints = (
            (f"{normalize_openai_base_url(url)}/models", f"{url}/health")
            if self.protocol == "openai"
            else (f"{url}/health",)
        )
        for endpoint in endpoints:
            try:
                response = self._session.get(endpoint, timeout=5.0)
            except requests.RequestException as exc:
                logger.warning("TextPecker health check failed for %s: %s", endpoint, exc)
                continue
            if response.ok:
                return True
        return False

    def _keep_reachable(self, urls: List[str]) -> List[str]:
        """Drop unreachable reward servers, logging each skipped endpoint."""
        reachable = [url for url in urls if self._endpoint_reachable(url)]
        if not reachable:
            raise RuntimeError(f"none of the TextPecker reward servers are reachable: {urls}")
        if len(reachable) < len(urls):
            logger.warning(
                "TextPecker reward pool running with %s/%s servers: %s",
                len(reachable),
                len(urls),
                reachable,
            )
        return reachable

    def health_check(self) -> bool:
        """True when at least one server in the pool answers."""
        return any(self._endpoint_reachable(url) for url in self.urls)

    def _build_payload(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "prompts": prompt,
            "image": [_image_to_b64(img) for img in image] if image else None,
            "images": [_image_to_b64(img) for img in image] if image else None,
            "video": [_video_to_b64(v) for v in video] if video else None,
            "condition_images": (
                [[_image_to_b64(img) for img in imgs] for imgs in condition_images]
                if condition_images
                else None
            ),
            "condition_videos": (
                [[_video_to_b64(v) for v in vs] for vs in condition_videos]
                if condition_videos
                else None
            ),
            "metadata": metadata,
            "target": targets,
            "targets": targets,
            "extra": {"targets": targets, "metadata": metadata},
        }
        return payload

    async def _evaluate_image(self, prompt: str, image_data_url: str) -> str:
        if self.pool is None:
            raise RuntimeError(
                f"TextPecker OpenAI pool is not initialized; _evaluate_image "
                f"requires protocol='openai', got protocol={self.protocol!r}"
            )
        return await self.pool.chat_with_failover(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            retries=self.retries,
            temperature=0,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            log_prefix="TextPecker vLLM",
        )

    async def _evaluate_batch(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        image_urls = [_image_to_data_url(img) for img in images]
        tasks = [
            self._evaluate_image(prompt, image_url)
            for prompt, image_url in zip(prompts, image_urls)
        ]
        return await asyncio.gather(*tasks)

    def compute_openai(
        self,
        image: List[Image.Image],
        targets: List[str],
        recognition_prompts: Optional[List[str]] = None,
        get_score_v2: Optional[Callable[..., Tuple[float, float, Dict[str, Any]]]] = None,
        qua_amplify_factor: float = 5.0,
    ) -> Dict[str, Any]:
        vllm_prompts = recognition_prompts or [_TEXTPECKER_REC_PROMPT for _ in targets]
        text_outputs = _run_async_task(self._evaluate_batch(vllm_prompts, image))
        if get_score_v2 is None:
            return {"text_outputs": text_outputs}

        rewards = []
        quality_scores = []
        semantic_scores = []
        full_results = []
        normalized_outputs: List[str] = []
        for response, target in zip(text_outputs, targets):
            # Repair unescaped quotes / truncated braces before TextPecker scoring
            # so scoring sees the same text dump_ocr / lenient parse would keep.
            normalized = normalize_ocr_json_answer(response)
            normalized_outputs.append(normalized)
            try:
                quality_score, semantic_score, cls_results = get_score_v2(
                    normalized,
                    target,
                    qua_amplify_factor=qua_amplify_factor,
                )
            except Exception as exc:  # pylint: disable=broad-except
                # Preserve the per-sample fallback for response processing failures.
                logger.warning(
                    "Failed to postprocess TextPecker response: %s\nraw=%s\nnormalized=%s",
                    exc,
                    response,
                    normalized,
                )
                quality_score, semantic_score, cls_results = 0.0, 0.0, {"error": str(exc)}
            quality_scores.append(float(quality_score))
            semantic_scores.append(float(semantic_score))
            full_results.append(cls_results)
            rewards.append(0.5 * float(quality_score) + 0.5 * float(semantic_score))
        return {
            "rewards": rewards,
            "quality_scores": quality_scores,
            "semantic_scores": semantic_scores,
            # Keep the server's raw answers for debugging; scoring used the
            # normalized form above.
            "text_outputs": text_outputs,
            "text_outputs_normalized": normalized_outputs,
            "full_results": full_results,
        }

    def compute_reward_server(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        targets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.protocol == "openai":
            raise RuntimeError(
                f"compute_reward_server targets a /compute reward server, but "
                f"this client was built with protocol={self.protocol!r} "
                f"(endpoints={self.urls})"
            )
        payload = self._build_payload(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
            metadata=metadata,
            targets=targets,
        )

        last_error: Optional[Exception] = None
        index = self._rotation.next_index()
        for attempt in range(self.retries):
            try:
                response = self._session.post(
                    f"{self.urls[index]}/compute", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                if data.get("error"):
                    raise RuntimeError(f"Server error: {data['error']}")
            except requests.RequestException as exc:
                last_error = exc
                failed_url = self.urls[index]
                self._rotation.mark_failed(index)
                if attempt == self.retries - 1:
                    break
                index = self._rotation.next_index()
                # Another server is up -> switch over right away; only wait when
                # the whole pool is cooling down.
                wait = 0.0 if self._rotation.has_available() else min(5 * (2**attempt), 30)
                logger.warning(
                    "TextPecker compute attempt %s/%s on %s failed: %s; " "retrying on %s in %ss",
                    attempt + 1,
                    self.retries,
                    failed_url,
                    exc,
                    self.urls[index],
                    wait,
                )
                if wait:
                    time.sleep(wait)
                continue
            self._rotation.mark_healthy(index)
            return data
        raise RuntimeError(
            f"TextPecker compute failed after {self.retries} attempts across "
            f"{self.urls}: {last_error}"
        ) from last_error

    def compute(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        targets: Optional[List[str]] = None,
        recognition_prompts: Optional[List[str]] = None,
        get_score_v2: Optional[Callable[..., Tuple[float, float, Dict[str, Any]]]] = None,
        qua_amplify_factor: float = 5.0,
    ) -> Dict[str, Any]:
        if self.protocol == "openai":
            if image is None or targets is None:
                raise ValueError(
                    "image and targets are required for TextPecker OpenAI/vLLM protocol"
                )
            return self.compute_openai(
                image=image,
                targets=targets,
                recognition_prompts=recognition_prompts,
                get_score_v2=get_score_v2,
                qua_amplify_factor=qua_amplify_factor,
            )
        return self.compute_reward_server(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
            metadata=metadata,
            targets=targets,
        )


class TextPeckerRemoteRewardModel(PointwiseRewardModel):
    """Pointwise remote reward model for TextPecker-compatible scoring servers."""

    required_fields: Tuple[str, ...] = ("prompt", "image")
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        server_urls = _resolve_server_urls(
            getattr(config, "server_url", None), getattr(config, "server_urls", None)
        )
        rank = getattr(accelerator, "process_index", 0) or 0

        self.component_weights = _normalize_component_weights(
            config.extra_kwargs.get("component_weights")
            or config.extra_kwargs.get("score_weights")
            or config.extra_kwargs.get("textpecker_weights")
        )
        self.normalize_component_weights = bool(
            config.extra_kwargs.get("normalize_component_weights", False)
        )
        self.prefer_server_rewards = bool(config.extra_kwargs.get("prefer_server_rewards", False))
        self.target_keys = tuple(config.extra_kwargs.get("target_keys", _DEFAULT_TARGET_KEYS))
        self.prompt_target_joiner = str(config.extra_kwargs.get("prompt_target_joiner", " "))
        self.check_health = bool(config.extra_kwargs.get("check_health", True))
        self.protocol = str(config.extra_kwargs.get("protocol", "openai"))
        self.qua_amplify_factor = float(config.extra_kwargs.get("qua_amplify_factor", 5.0))
        self.recognition_prompt = str(
            config.extra_kwargs.get("recognition_prompt", _TEXTPECKER_REC_PROMPT)
        )
        self.get_score_v2 = None
        if self.protocol == "openai":
            self.get_score_v2 = _load_get_score_v2(config.extra_kwargs.get("parse_utils_path"))

        # Unreachable servers are logged and dropped here rather than aborting
        # the run; runtime failures fail over to the next server in the pool.
        self.client = TextPeckerRemoteClient(
            server_urls=server_urls,
            timeout=float(getattr(config, "timeout", 180.0)),
            retries=int(getattr(config, "retry_attempts", 3)),
            protocol=self.protocol,
            model_name=str(config.extra_kwargs.get("model_name", "TextPecker")),
            api_key=str(config.extra_kwargs.get("api_key", "EMPTY")),
            max_tokens=int(config.extra_kwargs.get("max_tokens", 2048)),
            repetition_penalty=float(config.extra_kwargs.get("repetition_penalty", 1.2)),
            use_env_proxies=bool(config.extra_kwargs.get("use_env_proxies", False)),
            start_index=rank,
            failure_cooldown=float(config.extra_kwargs.get("server_failure_cooldown", 60.0)),
            probe_health=self.check_health,
        )
        self._chosen_url = self.client.base_url
        logger.info(
            "[rank %s] TextPecker reward pool = %s (primary %s, %s/%s configured "
            "servers reachable)",
            rank,
            self.client.urls,
            self.client.base_url,
            len(self.client.urls),
            len(server_urls),
        )

    def _postprocess_response(
        self, data: Dict[str, Any], batch_size: int
    ) -> Tuple[List[float], Dict[str, Any]]:
        response_data = data
        rewards_value = _find_first(response_data, _REWARD_KEYS)
        if isinstance(rewards_value, dict):
            response_data = {**rewards_value, **response_data}
            rewards_value = _find_first(response_data, _REWARD_KEYS[1:])

        semantic_scores = _as_float_list(
            _find_first(response_data, _SEMANTIC_KEYS), "semantic_scores"
        )
        quality_scores = _as_float_list(_find_first(response_data, _QUALITY_KEYS), "quality_scores")

        if _is_nested_score_triplet(rewards_value):
            server_rewards = _as_float_list(rewards_value[0], "rewards[0]")
            quality_scores = quality_scores or _as_float_list(rewards_value[1], "rewards[1]")
            semantic_scores = semantic_scores or _as_float_list(rewards_value[2], "rewards[2]")
        else:
            server_rewards = (
                _as_float_list(rewards_value, "rewards") if rewards_value is not None else None
            )

        if (
            not self.prefer_server_rewards
            and semantic_scores is not None
            and quality_scores is not None
        ):
            rewards = aggregate_textpecker_scores(
                semantic_scores=semantic_scores,
                quality_scores=quality_scores,
                component_weights=self.component_weights,
                normalize_component_weights=self.normalize_component_weights,
            )
        elif server_rewards is not None:
            rewards = server_rewards
        else:
            raise ValueError(
                "TextPecker server response must contain either rewards/scores/avg "
                "or both semantic_scores and quality_scores"
            )

        if len(rewards) != batch_size:
            raise ValueError(
                f"TextPecker server returned {len(rewards)} rewards for batch size {batch_size}"
            )

        # Surface the per-sample recognized text (TextPecker's OCR output) when the
        # protocol returns it (openai/vllm path).
        recognized_text = data.get("text_outputs")
        if recognized_text is not None and not isinstance(recognized_text, list):
            recognized_text = [recognized_text]

        extra_info = {
            "server_rewards": server_rewards,
            "semantic_scores": semantic_scores,
            "quality_scores": quality_scores,
        }
        if recognized_text is not None:
            extra_info["recognized_text"] = recognized_text
        return rewards, extra_info

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        target: Optional[Union[List[str], str]] = None,
        targets: Optional[Union[List[str], str]] = None,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        resolved_targets = _extract_targets_from_metadata(
            prompts=prompt,
            metadata=metadata,
            target=target,
            targets=targets,
            target_keys=self.target_keys,
            prompt_target_joiner=self.prompt_target_joiner,
        )
        response = self.client.compute(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
            metadata=metadata,
            targets=resolved_targets,
            recognition_prompts=[self.recognition_prompt for _ in resolved_targets],
            get_score_v2=self.get_score_v2,
            qua_amplify_factor=self.qua_amplify_factor,
        )
        rewards, extra_info = self._postprocess_response(response, len(prompt))
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            extra_info=extra_info,
        )
