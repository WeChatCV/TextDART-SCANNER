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

# src/flow_factory/trainers/scanner.py
"""SCANNER trainer for text rendering (TextDART).

GRPO-Guard with a spatio-temporal advantage instead of one scalar per image:

* Spatial: the reward model scores every requested text target separately.
  Rewards are z-scored per ``(prompt group, target_id)`` and painted onto the
  latent cells covered by that target's region, so each glyph is credited
  independently of where it was placed. Unrequested ("extra") text is scored
  against a clean reference of ``1.0``. A target that a rollout dropped while
  group peers rendered it gets ``missing_target_reward`` on the peers' expected
  location. Background cells carry no spatial advantage.
* Temporal: at SDE step ``t`` the painted map is mixed with the scalar (GDPO)
  advantage, ``A_t = w_t * A_scalar + (1 - w_t) * A_dense``, where ``w_t`` is the
  min-max normalized Flow-SDE exploration scale (1 at the noisiest step).

The policy loss is the spatial GRPO-Guard loss: the Guard log-ratio exponent is
averaged over each target's cells (one ratio per target, one for background),
PPO-clipped and reweighted by ``1/dt``. The KL anchor averages text-box and
background cells separately and weights them with ``kl_beta_box_scale`` /
``kl_beta_bg_scale``. Prompt groups without a usable signal (OCR failure in any
rollout, or an all-zero map in every rollout) are excluded from the loss.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm as tqdm_
from PIL import Image

from ..advantage.target_advantage_processor import TargetAdvantageProcessor, gather_group_keys
from ..logger.formatting import LogImage, LogTable
from ..samples import BaseSample
from ..scheduler.abc import flow_sde_std_dev_t
from ..scheduler.dense_flow_match_euler_discrete import DenseFlowMatchEulerDiscreteSDEScheduler
from ..utils.base import create_generator, filter_kwargs
from ..utils.latent_coverage import box_to_latent_coverage, quad_to_latent_coverage
from ..utils.logger_utils import setup_logger
from .grpo import GRPOGuardTrainer

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)

SCANNER_MIX_EPS = 1e-8


# ============================ Region geometry ============================
def _region_image_frame(sample: BaseSample, region: dict) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(image_W, image_H)`` for pixel coords, else ``(None, None)`` for 0-1000."""
    if str(region.get("coord_space", "norm_1000")) != "pixel":
        return None, None
    size = region.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    width = getattr(sample, "width", None)
    height = getattr(sample, "height", None)
    if width and height:
        return int(width), int(height)
    image = getattr(sample, "image", None)
    if isinstance(image, Image.Image):
        return int(image.size[0]), int(image.size[1])
    raise ValueError(
        "region coord_space='pixel' but no image_size / sample.width|height / "
        "sample.image available to map absolute pixels onto the latent grid"
    )


def _region_coverage(  # pylint: disable=invalid-name
    region: dict,
    sample: BaseSample,
    latent_W: int,
    latent_H: int,
    device: torch.device,
) -> torch.Tensor:
    """0/1 latent coverage of one region: prefer ``quad_2d``, else ``box_2d``."""
    img_w, img_h = _region_image_frame(sample, region)
    quad = region.get("quad_2d")
    if isinstance(quad, (list, tuple)) and len(quad) == 8:
        return quad_to_latent_coverage(
            quad, latent_W, latent_H, device=device, image_W=img_w, image_H=img_h
        )
    box = region.get("box_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return box_to_latent_coverage(
            box, latent_W, latent_H, device=device, image_W=img_w, image_H=img_h
        )
    return torch.zeros((latent_H, latent_W), device=device, dtype=torch.float32)


def _region_aabb_frac(sample: BaseSample, region: dict) -> Optional[Tuple[float, ...]]:
    """Axis-aligned box of a real region as image fractions ``(x0, y0, x1, y1)``."""
    if region.get("_synthetic_missing"):
        return None
    img_w, img_h = _region_image_frame(sample, region)
    denom_w = float(img_w) if img_w else 1000.0
    denom_h = float(img_h) if img_h else 1000.0
    quad = region.get("quad_2d")
    if isinstance(quad, (list, tuple)) and len(quad) == 8:
        xs = [float(quad[k]) for k in range(0, 8, 2)]
        ys = [float(quad[k]) for k in range(1, 8, 2)]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        box = region.get("box_2d")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            return None
        x0, x1 = sorted((float(box[0]), float(box[2])))
        y0, y1 = sorted((float(box[1]), float(box[3])))

    def clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    return (
        clamp01(x0 / denom_w),
        clamp01(y0 / denom_h),
        clamp01(x1 / denom_w),
        clamp01(y1 / denom_h),
    )


def region_target_columns(region: dict) -> List[int]:
    """Real target columns a region occupies (all members of a merged cluster)."""
    if not isinstance(region, dict):
        raise TypeError(f"expected dict for region, got {type(region).__name__}: {region!r}")
    cluster = region.get("cluster_target_ids")
    if isinstance(cluster, (list, tuple)) and cluster:
        members = sorted({int(x) for x in cluster if int(x) >= 0})
        if members:
            return members
    tid = int(region.get("target_id", -1))
    return [tid] if tid >= 0 else []


def connected_target_clusters(adjacency: torch.Tensor) -> List[List[int]]:
    """Connected components of a ``(T, T)`` bool adjacency (identity implied)."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be a square (T, T) matrix, got {tuple(adjacency.shape)}")
    n = int(adjacency.shape[0])
    adj = adjacency.to(dtype=torch.bool) | torch.eye(n, dtype=torch.bool, device=adjacency.device)
    seen = [False] * n
    clusters: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: List[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            for nbr in adj[node].nonzero(as_tuple=True)[0].tolist():
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(int(nbr))
        clusters.append(sorted(members))
    return clusters


# ============================ Spatial Guard ratio ============================
def _reduce_mean_to_spatial(tensor: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H, W) for image latents; (B, L, D) -> (B, L) for sequences."""
    if tensor.ndim == 4:
        return tensor.mean(dim=1)
    if tensor.ndim == 3:
        return tensor.mean(dim=2)
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def _per_sample_vector(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    """Reduce a scalar or ``(B, 1, ..., 1)`` scheduler tensor to ``(B,)``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected Tensor for {name}, got {type(value).__name__}: {value!r}")
    if value.ndim == 0:
        return value.expand(batch_size)
    if value.shape[0] != batch_size or value.numel() != batch_size:
        raise ValueError(
            f"expected {name} as one scalar per sample for batch_size={batch_size}, "
            f"got shape {tuple(value.shape)}"
        )
    return value.reshape(batch_size)


def guard_scale_factor(dt: torch.Tensor, std_dev_t: torch.Tensor) -> torch.Tensor:
    """Per-sample GRPO-Guard scale ``sqrt(-dt) * std_dev_t`` with shape ``(B,)``."""
    if dt.ndim == 0 and std_dev_t.ndim == 0:
        raise ValueError(
            "expected batched dt or std_dev_t, got scalars "
            f"dt {tuple(dt.shape)} and std_dev_t {tuple(std_dev_t.shape)}"
        )
    batch_size = int(dt.shape[0]) if dt.ndim > 0 else int(std_dev_t.shape[0])
    return torch.sqrt(-_per_sample_vector(dt, batch_size, "dt")) * _per_sample_vector(
        std_dev_t, batch_size, "std_dev_t"
    )


def mean_by_owner(values: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """Mean ``values`` within each ``(batch, owner_id)`` group and broadcast back.

    ``owner`` is ``(B, H, W)`` integer ids; ``-1`` is background (its own group).
    """
    if values.shape != owner.shape or owner.ndim != 3:
        raise ValueError(
            f"expected values and owner of equal (B, H, W) shape, got "
            f"{tuple(values.shape)} and {tuple(owner.shape)}"
        )
    if owner.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"expected integer owner dtype, got {owner.dtype}")
    groups = owner.to(dtype=torch.int64) + 1
    if bool((groups < 0).any()):
        raise ValueError(f"owner ids must be >= -1, got min {int(owner.min().item())}")
    batch = owner.shape[0]
    n_groups = int(groups.max().item()) + 1
    flat_g = groups.reshape(batch, -1)
    flat_v = values.reshape(batch, -1)
    sums = torch.zeros(batch, n_groups, device=values.device, dtype=values.dtype)
    counts = torch.zeros(batch, n_groups, device=values.device, dtype=values.dtype)
    sums.scatter_add_(1, flat_g, flat_v)
    counts.scatter_add_(1, flat_g, torch.ones_like(flat_v))
    means = sums / counts.clamp(min=1.0)
    return means.gather(1, flat_g).reshape(owner.shape)


def merged_guard_ratio(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    owner: torch.Tensor,
    scale_factor: torch.Tensor,
    mse: torch.Tensor,
) -> torch.Tensor:
    """One GRPO-Guard ratio per target-owner group, broadcast to ``(B, H, W)``."""
    if not (log_prob.shape == old_log_prob.shape == mse.shape):
        raise ValueError(
            f"log_prob {tuple(log_prob.shape)}, old_log_prob {tuple(old_log_prob.shape)} "
            f"and mse {tuple(mse.shape)} must share one shape"
        )
    scale = _per_sample_vector(scale_factor, int(log_prob.shape[0]), "scale_factor")
    scale = scale.reshape(log_prob.shape[0], *([1] * (log_prob.ndim - 1)))
    exponent = (log_prob - old_log_prob) * scale + mse / (2 * scale)
    return torch.exp(mean_by_owner(exponent, owner))


# ============================ Temporal mix ============================
def exploration_omega(
    sigmas: torch.Tensor,
    step_indices: Sequence[int],
    noise_level: float,
) -> torch.Tensor:
    """Scalar-advantage weight ``w_t`` for each SDE step in the training window.

    ``n_t = std_dev_t * sqrt(-dt)`` is the Flow-SDE exploration scale; it is made
    monotone (running min) and min-max normalized so the noisiest window step
    gets 1 and the quietest 0 (0.5 everywhere for a degenerate window).
    """
    if not step_indices:
        raise ValueError("step_indices must not be empty")
    max_index = max(step_indices)
    if max_index + 1 >= int(sigmas.numel()):
        raise ValueError(
            f"need sigmas of length >= {max_index + 2} for train step {max_index}, "
            f"got {tuple(sigmas.shape)}"
        )
    sigma = sigmas[list(step_indices)]
    sigma_next = sigmas[[int(step) + 1 for step in step_indices]]
    std_dev_t = flow_sde_std_dev_t(sigma, sigmas[1], noise_level)
    n_window = std_dev_t * torch.sqrt((sigma - sigma_next).clamp(min=0.0))
    n_window = torch.cummin(n_window, dim=0).values
    denom = n_window.max() - n_window.min()
    if float(denom) <= SCANNER_MIX_EPS:
        return torch.full_like(n_window, 0.5)
    return (n_window - n_window.min()) / denom


def diverging_adv_heatmap(
    value_map: np.ndarray, pil_size: Tuple[int, int], abs_max: float
) -> Image.Image:
    """NEAREST-upscale a latent advantage map with a shared diverging color scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    scale = max(float(abs_max), 1e-8)
    normalized = np.clip((value_map + scale) / (2.0 * scale), 0.0, 1.0)
    colored = (cm.RdBu_r(normalized)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored).resize((int(pil_size[0]), int(pil_size[1])), Image.NEAREST)


def _decode_sample_image(sample: BaseSample, resolution: int) -> Optional[Image.Image]:
    """Return ``sample.image`` as an RGB PIL image, or ``None`` when absent."""
    image = getattr(sample, "image", None)
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError(
            "sample.image must be PIL.Image / torch.Tensor / np.ndarray, got "
            f"{type(image).__name__} (prompt={str(sample.prompt)[:60]!r})"
        )
    img_np = image.float().cpu().numpy() if isinstance(image, torch.Tensor) else image
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
        img_np = img_np.transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    return Image.fromarray(img_np).convert("RGB").resize((resolution, resolution))


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected numeric {name}, got {type(value).__name__}: {value!r}")


# ============================ Trainer ============================
class SCANNERTrainer(GRPOGuardTrainer):
    """GRPO-Guard with per-target spatial advantages mixed with the scalar one over time.

    ``train.extra_kwargs`` knobs:
        target_sigma_min: Std floor of the per-target z-score (default 1e-4).
        spatial_adv_clip: Clip for per-target and scalar advantages (default 5.0, 0 = off).
        spatial_box_weight: Weight of the text-box term of the painted map (default 1.0).
        kl_beta_box_scale / kl_beta_bg_scale: KL multipliers on box / background cells
            (defaults 1.0 / 5.0), on top of ``kl_beta``.
        missing_target_min_present: Peers that must render a target before a dropped
            copy is penalized (default 1).
        missing_target_reward: Reward assigned to a dropped target (default 0.0).
        scanner_log_max_samples: Rollouts shown in the per-step advantage panel (default 2).
    """

    sample_callback_kwargs: List[str] = ["log_prob_spatial", "next_latents_mean"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ek = self.training_args.extra_kwargs
        self.target_sigma_min = float(ek.get("target_sigma_min", 1e-4))
        self.spatial_adv_clip = float(ek.get("spatial_adv_clip", 5.0))
        self.spatial_box_weight = float(ek.get("spatial_box_weight", 1.0))
        self.kl_beta_box_scale = float(ek.get("kl_beta_box_scale", 1.0))
        self.kl_beta_bg_scale = float(ek.get("kl_beta_bg_scale", 5.0))
        self.missing_target_min_present = int(ek.get("missing_target_min_present", 1))
        self.missing_target_reward = float(ek.get("missing_target_reward", 0.0))
        self.scanner_log_max_samples = int(ek.get("scanner_log_max_samples", 2))
        if self.spatial_adv_clip < 0.0:
            raise ValueError(
                f"spatial_adv_clip must be >= 0 (0 disables), got {self.spatial_adv_clip}"
            )
        if self.kl_beta_box_scale < 0.0 or self.kl_beta_bg_scale < 0.0:
            raise ValueError(
                "kl_beta_box_scale / kl_beta_bg_scale must be >= 0, got "
                f"{self.kl_beta_box_scale} / {self.kl_beta_bg_scale}"
            )
        if self.missing_target_min_present < 1:
            raise ValueError(
                f"missing_target_min_present must be >= 1, got {self.missing_target_min_present}"
            )
        if self.scanner_log_max_samples < 1:
            raise ValueError(
                f"scanner_log_max_samples must be >= 1, got {self.scanner_log_max_samples}"
            )

        # Spatial log-prob (B, H, W) / (B, L) alongside the scalar one.
        scheduler = DenseFlowMatchEulerDiscreteSDEScheduler.from_scheduler(self.adapter.scheduler)
        self.adapter.scheduler = scheduler
        self.adapter.pipeline.scheduler = scheduler

        self.target_advantage_processor = TargetAdvantageProcessor(verbose=True)
        logger.info(
            "SCANNER initialized: target_sigma_min=%s, spatial_adv_clip=%s, "
            "spatial_box_weight=%s, kl_beta_box_scale=%s, kl_beta_bg_scale=%s, "
            "missing_target(min_present=%s, reward=%s)",
            self.target_sigma_min,
            self.spatial_adv_clip,
            self.spatial_box_weight,
            self.kl_beta_box_scale,
            self.kl_beta_bg_scale,
            self.missing_target_min_present,
            self.missing_target_reward,
        )

    def _clip(self, values: torch.Tensor) -> torch.Tensor:
        if self.spatial_adv_clip > 0.0:
            return values.clamp(-self.spatial_adv_clip, self.spatial_adv_clip)
        return values

    # =========================== Reward / advantage ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Scalar GDPO advantages (parent), then the per-target spatial advantage maps."""
        super().prepare_feedback(samples)

        reward_per_target, c = self._build_reward_per_target(samples)
        n_missing = self._apply_missing_target_penalty(samples, reward_per_target, c)
        gathered_missing = self.accelerator.gather(
            torch.tensor([float(n_missing)], device=self.accelerator.device)
        )
        self.log_data(
            {"train/missing_target_injected": float(gathered_missing.sum().item())},
            step=self.step,
        )

        ocr_failed = torch.tensor(
            [bool(s.extra_kwargs.get("ocr_failed", False)) for s in samples],
            device=self.accelerator.device,
            dtype=torch.bool,
        )
        unique_ids = [s.unique_id for s in samples]
        adv_per_target, ocr_drop = self.target_advantage_processor.compute_target_advantages(
            accelerator=self.accelerator,
            reward_per_target=reward_per_target,
            unique_ids=unique_ids,
            ocr_failed=ocr_failed,
            sigma_min=self.target_sigma_min,
        )
        adv_per_target = self._clip(adv_per_target)
        adv_per_target = self._apply_extra_vs_one(samples, adv_per_target, unique_ids, ocr_drop)

        latent_h, latent_w = self._infer_latent_size(samples)
        adv_maps, box_masks = self._build_advantage_maps(
            samples, adv_per_target, latent_h, latent_w, c
        )
        self._apply_zero_adv_drop(samples, adv_maps, ocr_drop, box_masks)

        num_train_timesteps = len(self.adapter.scheduler.train_timesteps)
        for i, s in enumerate(samples):
            s.extra_kwargs["advantage_spatial"] = (
                adv_maps[i].unsqueeze(0).expand(num_train_timesteps, -1, -1).contiguous()
            )
            s.extra_kwargs["box_mask_spatial"] = box_masks[i]

    def _infer_latent_size(self, samples: List[BaseSample]) -> Tuple[int, int]:
        """(H, W) of the latent grid from the stored spatial log-prob."""
        for s in samples:
            lps = s.extra_kwargs.get("log_prob_spatial")
            if lps is None:
                continue
            if lps.ndim == 3:
                return int(lps.shape[-2]), int(lps.shape[-1])
            if lps.ndim == 2:
                length = int(lps.shape[-1])
                side = int(math.isqrt(length))
                for h in range(side, 0, -1):
                    if length % h == 0:
                        return h, length // h
        raise ValueError(
            "no sample carries extra_kwargs['log_prob_spatial'] to infer the latent size"
        )

    def _build_reward_per_target(self, samples: List[BaseSample]) -> Tuple[torch.Tensor, int]:
        """``(B_local, C)`` per-target rewards (NaN = absent); column ``C-1`` is extra text.

        ``C`` is the global max target count + 1 so the matrix gathers across ranks.
        """
        device = self.accelerator.device
        local_max = max((int(s.extra_kwargs.get("num_targets", 0)) for s in samples), default=0)
        gathered_max = self.accelerator.gather(
            torch.tensor([local_max], device=device, dtype=torch.long)
        )
        n_real = int(gathered_max.max().item())
        c = n_real + 1
        reward_per_target = torch.full(
            (len(samples), c), float("nan"), device=device, dtype=torch.float32
        )
        for i, s in enumerate(samples):
            target_rewards = s.extra_kwargs.get("target_rewards", {}) or {}
            if not isinstance(target_rewards, dict):
                raise TypeError(
                    "expected dict for extra_kwargs['target_rewards'], got "
                    f"{type(target_rewards).__name__} (unique_id={s.unique_id!r})"
                )
            for key, val in target_rewards.items():
                if key == "extra":
                    reward_per_target[i, c - 1] = float(val)
                    continue
                ki = int(key)
                if 0 <= ki < n_real:
                    reward_per_target[i, ki] = float(val)
        return reward_per_target, c

    def _build_target_boxes_frac(self, samples: List[BaseSample], C: int) -> torch.Tensor:  # pylint: disable=invalid-name
        """``(B_local, C, 4)`` enclosing AABB (image fractions) of each rendered target."""
        device = self.accelerator.device
        boxes = torch.full((len(samples), C, 4), float("nan"), device=device, dtype=torch.float32)
        for i, s in enumerate(samples):
            acc: Dict[int, List[float]] = {}
            for region in s.extra_kwargs.get("target_regions", []) or []:
                members = [c for c in region_target_columns(region) if 0 <= c < C - 1]
                frac = _region_aabb_frac(s, region) if members else None
                if frac is None:
                    continue
                for col in members:
                    if col in acc:
                        a = acc[col]
                        acc[col] = [
                            min(a[0], frac[0]),
                            min(a[1], frac[1]),
                            max(a[2], frac[2]),
                            max(a[3], frac[3]),
                        ]
                    else:
                        acc[col] = list(frac)
            for col, b in acc.items():
                boxes[i, col] = torch.tensor(b, device=device, dtype=torch.float32)
        return boxes

    def _apply_missing_target_penalty(  # pylint: disable=invalid-name
        self,
        samples: List[BaseSample],
        reward_per_target: torch.Tensor,
        C: int,
    ) -> int:
        """Penalize dropped targets at the location where group peers rendered them.

        Targets merged into one region by the reward model (``cluster_target_ids``)
        form one cluster. For every ``(prompt, cluster)`` rendered by at least
        ``missing_target_min_present`` rollouts, each rollout that dropped it gets
        ``missing_target_reward`` on every member column plus one synthetic region
        (0-1000 coordinates) enclosing the peers' boxes. Mutates
        ``reward_per_target`` and ``samples``; returns the local injection count.
        """
        accel = self.accelerator
        device = accel.device
        b_local = len(samples)
        n_real = C - 1
        if n_real < 1:
            return 0

        gathered_boxes = accel.gather(self._build_target_boxes_frac(samples, C).contiguous())
        gathered_keys = gather_group_keys(accel, [s.unique_id for s in samples])
        cluster_adj = torch.zeros((b_local, n_real, n_real), dtype=torch.bool, device=device)
        for i, sample in enumerate(samples):
            for region in sample.extra_kwargs.get("target_regions", []) or []:
                members = [col for col in region_target_columns(region) if 0 <= col < n_real]
                for src in members:
                    for dst in members:
                        cluster_adj[i, src, dst] = True
        gathered_adj = accel.gather(cluster_adj.contiguous())
        present = ~torch.isnan(gathered_boxes).any(dim=2)
        b_global = gathered_boxes.shape[0]

        inject = torch.full((b_global, C, 4), float("nan"), device=device, dtype=torch.float32)
        set_missing = torch.zeros((b_global, C), dtype=torch.bool, device=device)
        cluster_rep = torch.full((b_global, C), -1, device=device, dtype=torch.long)
        for group_id in torch.unique(gathered_keys):
            idx = (gathered_keys == group_id).nonzero(as_tuple=True)[0]
            for members in connected_target_clusters(gathered_adj[idx].any(dim=0)):
                pres_any = present[idx][:, members].any(dim=1)
                pres_rows = idx[pres_any]
                miss_rows = idx[~pres_any]
                if pres_rows.numel() < self.missing_target_min_present or miss_rows.numel() == 0:
                    continue
                peer_boxes = []
                for col in members:
                    pb = gathered_boxes[pres_rows, col]
                    valid = ~torch.isnan(pb).any(dim=1)
                    if bool(valid.any().item()):
                        peer_boxes.append(pb[valid])
                if not peer_boxes:
                    continue
                pb = torch.cat(peer_boxes, dim=0)
                expected = torch.stack(
                    [pb[:, 0].min(), pb[:, 1].min(), pb[:, 2].max(), pb[:, 3].max()]
                )
                for col in members:
                    inject[miss_rows, col] = expected
                    set_missing[miss_rows, col] = True
                    cluster_rep[miss_rows, col] = int(min(members))

        local_inject = inject.reshape(accel.num_processes, b_local, C, 4)[accel.process_index]
        local_missing = set_missing.reshape(accel.num_processes, b_local, C)[accel.process_index]
        local_rep = cluster_rep.reshape(accel.num_processes, b_local, C)[accel.process_index]

        count = 0
        for i in range(b_local):
            extra = samples[i].extra_kwargs
            regions = extra.get("target_regions")
            if regions is None:
                regions = extra["target_regions"] = []
            rewards_dict = extra.get("target_rewards")
            if not isinstance(rewards_dict, dict):
                rewards_dict = extra["target_rewards"] = {}
            missing_cols = [c for c in range(n_real) if bool(local_missing[i, c].item())]
            for c in missing_cols:
                reward_per_target[i, c] = self.missing_target_reward
                rewards_dict[str(c)] = self.missing_target_reward
            seen_reps = set()
            for c in missing_cols:
                rep = int(local_rep[i, c].item())
                if rep < 0 or rep in seen_reps:
                    continue
                seen_reps.add(rep)
                members = [m for m in range(n_real) if int(local_rep[i, m].item()) == rep]
                b = local_inject[i, c].tolist()
                synthetic = {
                    "target_id": rep,
                    "box_2d": [b[0] * 1000.0, b[1] * 1000.0, b[2] * 1000.0, b[3] * 1000.0],
                    "region_qua": 0.0,
                    "region_sem": 0.0,
                    "text": "",
                    "_synthetic_missing": True,
                }
                if len(members) >= 2:
                    synthetic["cluster_target_ids"] = members
                regions.append(synthetic)
                count += 1
        return count

    def _zscore_by_prompt_group(self, values: torch.Tensor, unique_ids: List) -> torch.Tensor:
        """Group z-score of a per-rollout ``(B_local,)`` value across ranks."""
        if values.ndim != 1 or len(unique_ids) != values.shape[0]:
            raise ValueError(
                f"values must be (B,) aligned with unique_ids, got {tuple(values.shape)} "
                f"vs {len(unique_ids)} ids"
            )
        gathered = self.accelerator.gather(values.contiguous())
        gathered_keys = gather_group_keys(self.accelerator, unique_ids).to(values.device)
        adv = torch.zeros_like(gathered)
        for group_id in torch.unique(gathered_keys):
            idx = (gathered_keys == group_id).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            col = gathered[idx]
            adv[idx] = (col - col.mean()) / torch.clamp(col.std(), min=self.target_sigma_min)
        return adv.reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]

    def _apply_extra_vs_one(
        self,
        samples: List[BaseSample],
        advantage_per_target: torch.Tensor,
        unique_ids: List,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score extra text against a clean reference of ``1.0``.

        Each rollout's extra reward is the merged quality of its extra regions
        (``region_qua`` in [0, 1]), or ``1.0`` when it rendered no extra text. The
        group z-score of that value is written into the extra column, so extra
        text is penalized relative to the clean rollouts; rollouts without extra
        text paint nothing.
        """
        rewards: List[float] = []
        has_extra_flags: List[bool] = []
        for sample in samples:
            extras = [
                r
                for r in (sample.extra_kwargs.get("target_regions") or [])
                if int(r.get("target_id", -1)) == -1
            ]
            if not extras:
                rewards.append(1.0)
                has_extra_flags.append(False)
                continue
            qua = extras[0].get("region_qua")
            if qua is None:
                qua = (sample.extra_kwargs.get("target_rewards") or {}).get("extra")
            if qua is None:
                qua = 0.0
            if not isinstance(qua, (int, float)):
                raise TypeError(
                    f"expected numeric merged extra quality, got {type(qua).__name__}: {qua!r}"
                )
            rewards.append(max(0.0, min(1.0, float(qua))))
            has_extra_flags.append(True)

        device, dtype = advantage_per_target.device, advantage_per_target.dtype
        extra_reward = torch.tensor(rewards, device=device, dtype=dtype)
        has_extra = torch.tensor(has_extra_flags, device=device, dtype=torch.bool)
        extra_adv = self._clip(self._zscore_by_prompt_group(extra_reward, unique_ids))
        extra_adv = extra_adv * (~drop_mask).to(dtype=dtype)

        out = advantage_per_target.clone()
        out[:, -1] = extra_adv * has_extra.to(dtype=dtype)
        present_adv = extra_adv[has_extra]
        self.log_data(
            {
                "train/extra_vs_one_adv_mean": float(extra_adv.mean().item()),
                "train/extra_vs_one_frac": float(has_extra.float().mean().item()),
                "train/extra_vs_one_reward_mean": float(extra_reward.mean().item()),
                "train/extra_vs_one_present_adv_mean": (
                    float(present_adv.mean().item()) if present_adv.numel() > 0 else 0.0
                ),
            },
            step=self.step,
        )
        return out

    def _resolve_target_advantage_map(  # pylint: disable=invalid-name
        self,
        sample: BaseSample,
        adv_row: torch.Tensor,
        latent_H: int,
        latent_W: int,
        C: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Paint one rollout's per-target advantages onto the latent grid.

        Overlapping regions resolve to the region with the smallest footprint.
        Synthetic (missing-target) regions exclude cells covered by rendered
        text. Returns ``(adv_map, box_mask, owner_area, num_present, owner_col)``;
        ``owner_col`` is the owning column per cell (``-1`` = background).
        """
        device = self.accelerator.device
        shape = (latent_H, latent_W)
        regions = sample.extra_kwargs.get("target_regions", []) or []

        def coverage_of(region: dict) -> torch.Tensor:
            return _region_coverage(region, sample, latent_W, latent_H, device)

        present_union: Optional[torch.Tensor] = None
        painted: List[Tuple[int, float, torch.Tensor]] = []
        for region in regions:
            tid = int(region.get("target_id", -1))
            col = (C - 1) if tid == -1 else tid
            if not (0 <= col < C):
                continue
            w = coverage_of(region)
            if region.get("_synthetic_missing"):
                if present_union is None:
                    present_union = torch.zeros_like(w)
                    for other in regions:
                        if not other.get("_synthetic_missing"):
                            present_union = torch.maximum(present_union, coverage_of(other))
                    present_union = present_union.clamp(max=1.0)
                w = (w * (1.0 - present_union)).clamp(min=0.0)
            if bool(torch.any(w > 0)):
                painted.append((col, float(adv_row[col].item()), w))

        owner_col = torch.full(shape, -1, device=device, dtype=torch.long)
        owner_adv = torch.zeros(shape, device=device, dtype=torch.float32)
        owner_area = torch.ones(shape, device=device, dtype=torch.float32)
        cur_area = torch.full(shape, float("inf"), device=device, dtype=torch.float32)
        for col, adv, w in painted:
            m = w > 0
            region_area = float(m.sum().item())
            take = m & ((owner_col < 0) | (cur_area > region_area))
            cur_area = torch.where(take, torch.full_like(cur_area, region_area), cur_area)
            owner_col = torch.where(take, torch.full_like(owner_col, col), owner_col)
            owner_adv = torch.where(take, torch.full_like(owner_adv, adv), owner_adv)

        box_mask = owner_col >= 0
        present_cols = torch.unique(owner_col[box_mask])
        for col in present_cols.tolist():
            col_mask = owner_col == col
            owner_area = torch.where(
                col_mask, torch.full_like(owner_area, float(col_mask.sum().item())), owner_area
            )
        adv_map = torch.where(box_mask, owner_adv, torch.zeros_like(owner_adv))
        return adv_map, box_mask, owner_area, int(present_cols.numel()), owner_col

    def _build_advantage_maps(  # pylint: disable=invalid-name
        self,
        samples: List[BaseSample],
        advantage_per_target: torch.Tensor,
        latent_H: int,
        latent_W: int,
        C: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training advantage maps and text-box masks, both ``(B_local, H, W)``.

        Each box cell is divided by its target's footprint and by the number of
        present targets, so after the loss mean over ``H*W`` every target
        contributes ``spatial_box_weight * A_target / num_present`` regardless of
        glyph size. Also stores ``target_owner_spatial`` on every sample.
        """
        hw = float(latent_H * latent_W)
        train_maps: List[torch.Tensor] = []
        box_masks: List[torch.Tensor] = []
        for i, s in enumerate(samples):
            adv_map, box_mask, owner_area, num_present, owner_col = (
                self._resolve_target_advantage_map(
                    s, advantage_per_target[i], latent_H, latent_W, C
                )
            )
            s.extra_kwargs["target_owner_spatial"] = owner_col
            scale_box = self.spatial_box_weight * hw / float(max(num_present, 1))
            train_maps.append(torch.where(box_mask, adv_map / owner_area * scale_box, adv_map))
            box_masks.append(box_mask.to(torch.float32))
        return torch.stack(train_maps), torch.stack(box_masks)

    def _apply_zero_adv_drop(
        self,
        samples: List[BaseSample],
        adv_maps: torch.Tensor,
        ocr_drop_mask: torch.Tensor,
        box_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Drop whole prompt groups without a text signal and write ``train_keep``.

        A group is dropped when any rollout failed OCR or every rollout has an
        all-zero advantage map. Logged ratios are over prompt groups.
        """
        if adv_maps.ndim != 3 or box_masks.shape != adv_maps.shape:
            raise ValueError(
                f"expected adv_maps and box_masks of equal (B, H, W) shape, got "
                f"{tuple(adv_maps.shape)} and {tuple(box_masks.shape)}"
            )
        if ocr_drop_mask.shape[0] != adv_maps.shape[0] or len(samples) != adv_maps.shape[0]:
            raise ValueError(
                f"batch mismatch: adv_maps {adv_maps.shape[0]}, ocr_drop_mask "
                f"{ocr_drop_mask.shape[0]}, samples {len(samples)}"
            )
        eps = 1e-12
        device = adv_maps.device
        box_bool = box_masks > 0
        box_zero = adv_maps.abs().masked_fill(~box_bool, 0.0).amax(dim=(1, 2)) <= eps
        map_zero = adv_maps.abs().amax(dim=(1, 2)) <= eps

        gathered_keys = gather_group_keys(self.accelerator, [s.unique_id for s in samples])
        gathered_box = self.accelerator.gather(box_zero.float().contiguous())
        gathered_map = self.accelerator.gather(map_zero.float().contiguous())
        gathered_ocr = self.accelerator.gather(
            ocr_drop_mask.to(device=device, dtype=torch.float32).contiguous()
        )
        gathered_drop = torch.zeros(gathered_keys.shape[0], dtype=torch.bool, device=device)
        n_groups = n_box_zero = n_map_zero = n_ocr = n_drop = 0
        for group_id in torch.unique(gathered_keys):
            idx = (gathered_keys == group_id).nonzero(as_tuple=True)[0]
            n_groups += 1
            all_map_zero = float(gathered_map[idx].min().item()) >= 1.0
            ocr_group = float(gathered_ocr[idx].max().item()) >= 1.0
            n_box_zero += int(float(gathered_box[idx].min().item()) >= 1.0)
            n_map_zero += int(all_map_zero)
            n_ocr += int(ocr_group)
            if ocr_group or all_map_zero:
                n_drop += 1
                gathered_drop[idx] = True
        if n_groups < 1:
            raise ValueError(f"expected >= 1 prompt group after gather, got n_groups={n_groups}")

        denom = float(n_groups)
        self.log_data(
            {
                "train/group_zero_box_adv_ratio": n_box_zero / denom,
                "train/group_zero_adv_map_ratio": n_map_zero / denom,
                "train/drop_ratio_ocr": n_ocr / denom,
                "train/drop_ratio": n_drop / denom,
            },
            step=self.step,
        )
        drop_mask = gathered_drop.reshape(self.accelerator.num_processes, -1)[
            self.accelerator.process_index
        ]
        for i, s in enumerate(samples):
            s.extra_kwargs["train_keep"] = torch.tensor(
                0.0 if bool(drop_mask[i].item()) else 1.0, device=device, dtype=torch.float32
            )
        return drop_mask

    # =========================== Temporal mix ============================
    def _train_step_indices(self) -> List[int]:
        steps = self.adapter.scheduler.train_timesteps
        if isinstance(steps, torch.Tensor):
            return [int(step) for step in steps.detach().cpu().tolist()]
        if not isinstance(steps, (list, tuple)):
            raise TypeError(
                f"expected list/tuple/tensor train_timesteps, got {type(steps).__name__}"
            )
        return [int(step) for step in steps]

    def _apply_temporal_mix(self, samples: List[BaseSample]) -> None:
        """Replace each ``advantage_spatial`` (T, H, W) by ``w_t * A_scalar + (1 - w_t) * A_dense``."""
        if not samples:
            raise ValueError("scanner received an empty sample list")
        dense = torch.stack([s.extra_kwargs["advantage_spatial"] for s in samples], dim=0)
        step_indices = self._train_step_indices()
        scheduler = self.adapter.scheduler
        omega = exploration_omega(
            scheduler.sigmas.detach().float().cpu(), step_indices, float(scheduler.noise_level)
        )
        if dense.ndim != 4 or int(dense.shape[1]) != int(omega.numel()):
            raise ValueError(
                f"advantage_spatial must stack to (B, T, H, W) with T={int(omega.numel())}, "
                f"got {tuple(dense.shape)}"
            )
        for s in samples:
            if "advantage" not in s.extra_kwargs:
                raise KeyError(
                    f"scanner requires extra_kwargs['advantage'] (unique_id={s.unique_id!r})"
                )
        scalar = torch.tensor(
            [_as_float(s.extra_kwargs["advantage"], "advantage") for s in samples],
            device=dense.device,
            dtype=dense.dtype,
        )
        scalar = self._clip(scalar)
        weight = omega.to(device=dense.device, dtype=dense.dtype).view(1, -1, 1, 1)
        mixed = weight * scalar.view(-1, 1, 1, 1) + (1.0 - weight) * dense
        for index, sample in enumerate(samples):
            sample.extra_kwargs["advantage_spatial"] = mixed[index]

        payload = {
            "train/scanner_omega_scalar_mean": float(omega.mean().item()),
            "train/scanner_scalar_adv_mean": float(scalar.mean().item()),
        }
        for step, w in zip(step_indices, omega.tolist()):
            payload[f"train/scanner_omega_scalar_step/{int(step)}"] = float(w)
        self.log_data(payload, step=self.step)
        self._log_temporal_visuals(samples, mixed, omega, step_indices)

    def _log_temporal_visuals(
        self,
        samples: List[BaseSample],
        mixed: torch.Tensor,
        omega: torch.Tensor,
        step_indices: Sequence[int],
    ) -> None:
        """Log each shown rollout next to its mixed advantage map at every SDE step."""
        if not self.accelerator.is_main_process:
            return
        resolution = self.training_args.resolution
        if isinstance(resolution, (list, tuple)):
            resolution = resolution[0]
        columns = ["rollout"] + [f"t{int(step)}" for step in step_indices]
        omega_list = [float(w) for w in omega.detach().cpu().tolist()]
        gallery: List[LogImage] = []
        rows: List[List[LogImage]] = []
        for sample_index in range(min(self.scanner_log_max_samples, len(samples))):
            sample = samples[sample_index]
            prompt_short = str(getattr(sample, "prompt", None) or f"sample_{sample_index}")[:120]
            pil_img = _decode_sample_image(sample, int(resolution))
            if pil_img is None:
                logger.warning(
                    "Sample %d (prompt=%r) has no image; skipping visuals.",
                    sample_index,
                    prompt_short,
                )
                continue
            step_maps = mixed[sample_index].detach().float().cpu().numpy()
            abs_max = max(abs(float(step_maps.min())), abs(float(step_maps.max())), 1e-8)
            scalar_adv = _as_float(sample.extra_kwargs.get("advantage", 0.0), "advantage")
            rollout = LogImage(
                pil_img, caption=f"[rollout] A_scalar={scalar_adv:.3f} | {prompt_short}"
            )
            row = [rollout]
            for step_pos, step_index in enumerate(step_indices):
                adv_map = step_maps[step_pos]
                row.append(
                    LogImage(
                        diverging_adv_heatmap(adv_map, pil_img.size, abs_max),
                        caption=(
                            f"[A t={int(step_index)} w_scalar={omega_list[step_pos]:.3f}] "
                            f"min={float(adv_map.min()):.3f} max={float(adv_map.max()):.3f} "
                            f"mean={float(adv_map.mean()):.3f}"
                        ),
                    )
                )
            gallery.extend(row)
            rows.append(row)
        if not rows:
            return
        self.log_data(
            {
                "scanner/visuals": gallery,
                "scanner/step_adv": LogTable(
                    columns=columns, rows=rows, target_height=int(resolution)
                ),
            },
            step=self.step,
        )

    # =========================== Optimization ============================
    @staticmethod
    def _kept_mean(per_sample: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean over kept samples (``keep`` in {0, 1}); 0 when every sample is dropped."""
        if keep is None:
            return per_sample.mean()
        keep_f = keep.to(dtype=per_sample.dtype).reshape(-1)
        if keep_f.shape[0] != per_sample.shape[0]:
            raise ValueError(
                f"train_keep length {keep_f.shape[0]} != batch size {per_sample.shape[0]}"
            )
        return (per_sample * keep_f).sum() / keep_f.sum().clamp(min=1.0)

    def _weighted_masked_kl(
        self,
        sq_err: torch.Tensor,
        box_mask_spatial: torch.Tensor,
        sample_keep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``kl_beta_box_scale * mean_box + kl_beta_bg_scale * mean_bg`` of a squared error."""
        flat = _reduce_mean_to_spatial(sq_err).reshape(sq_err.shape[0], -1)
        mask = box_mask_spatial.reshape(sq_err.shape[0], -1).to(flat.dtype)
        if mask.shape != flat.shape:
            kl_per = flat.mean(dim=1)
        else:
            box_mean = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            bg_mean = (flat * (1.0 - mask)).sum(dim=1) / (1.0 - mask).sum(dim=1).clamp(min=1.0)
            kl_per = self.kl_beta_box_scale * box_mean + self.kl_beta_bg_scale * bg_mean
        return self._kept_mean(kl_per, sample_keep)

    def optimize(self, samples: List[BaseSample]) -> None:
        """Spatial GRPO-Guard policy optimization on the temporally mixed advantage maps."""
        self._apply_temporal_mix(samples)
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            self.adapter.train()
            loss_info = defaultdict(list)
            with self.autocast():
                for batch_idx in tqdm(
                    range(num_batches),
                    total=num_batches,
                    desc=f"Epoch {self.epoch} Training",
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    start = batch_idx * per_device_batch_size
                    batch = BaseSample.stack(
                        [
                            s.to(device)
                            for s in shuffled_samples[start : start + per_device_batch_size]
                        ]
                    )
                    latents_index_map = batch["latent_index_map"]
                    log_probs_index_map = batch["log_prob_index_map"]
                    callback_index_map = batch["callback_index_map"][0]
                    train_keep = batch["train_keep"]
                    owner = batch["target_owner_spatial"]
                    for idx, timestep_index in enumerate(
                        tqdm(
                            self.adapter.scheduler.train_timesteps,
                            desc=f"Epoch {self.epoch} Timestep",
                            position=1,
                            leave=False,
                            disable=not self.show_progress_bar,
                        )
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Inputs and rollout-time statistics.
                            old_log_prob = batch["log_probs"][
                                :, log_probs_index_map[timestep_index]
                            ]
                            old_log_prob_spatial = batch["log_prob_spatial"][
                                :, callback_index_map[timestep_index]
                            ]
                            old_next_latents_mean = batch["next_latents_mean"][
                                :, callback_index_map[timestep_index]
                            ]
                            num_timesteps = batch["timesteps"].shape[1]
                            t = batch["timesteps"][:, timestep_index]
                            t_next = (
                                batch["timesteps"][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.tensor(0, device=device)
                            )
                            forward_inputs = {
                                **self.training_args,
                                "t": t,
                                "t_next": t_next,
                                "latents": batch["all_latents"][
                                    :, latents_index_map[timestep_index]
                                ],
                                "next_latents": batch["all_latents"][
                                    :, latents_index_map[timestep_index + 1]
                                ],
                                "compute_log_prob": True,
                                "noise_level": self.adapter.scheduler.noise_level,
                                **batch,
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)

                            # 2. Forward pass.
                            return_kwargs = [
                                "log_prob",
                                "log_prob_spatial",
                                "next_latents_mean",
                                "std_dev_t",
                                "dt",
                            ]
                            if self.enable_kl_loss and self.training_args.kl_type == "v-based":
                                return_kwargs.append("noise_pred")
                            forward_inputs["return_kwargs"] = return_kwargs
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Spatial Guard ratio, one per target (sequence models: (B, L) -> (B, H, W)).
                            adv = batch["advantage_spatial"][:, idx]
                            log_prob_spatial = output.log_prob_spatial
                            mse_spatial = _reduce_mean_to_spatial(
                                (output.next_latents_mean - old_next_latents_mean) ** 2
                            )
                            if log_prob_spatial.ndim == 2:
                                h, w = adv.shape[-2], adv.shape[-1]
                                log_prob_spatial = log_prob_spatial[:, : h * w].reshape(-1, h, w)
                                old_log_prob_spatial = old_log_prob_spatial[:, : h * w].reshape(
                                    -1, h, w
                                )
                                mse_spatial = mse_spatial[:, : h * w].reshape(-1, h, w)
                            ratio = merged_guard_ratio(
                                log_prob_spatial,
                                old_log_prob_spatial,
                                owner,
                                guard_scale_factor(output.dt, output.std_dev_t),
                                mse_spatial,
                            )

                            # 4. PPO-clipped loss over kept samples + GRPO-Guard 1/dt reweighting.
                            ratio_clip_range = self.training_args.clip_range
                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * torch.clamp(
                                ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1]
                            )
                            per_sample_policy = torch.maximum(unclipped_loss, clipped_loss).mean(
                                dim=tuple(range(1, clipped_loss.ndim))
                            )
                            policy_loss = self._kept_mean(per_sample_policy, train_keep)
                            policy_loss = policy_loss / (torch.sqrt(-output.dt).mean() ** 2)
                            if float(train_keep.sum()) == 0.0:
                                # Keep every parameter in the graph so distributed backward does not hang.
                                policy_loss = policy_loss + 0.0 * output.log_prob.mean()
                            loss = policy_loss

                            # 5. Box/background-weighted KL to the reference model.
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs["compute_log_prob"] = False
                                    if self.training_args.kl_type == "v-based":
                                        ref_forward_inputs["return_kwargs"] = ["noise_pred"]
                                    elif self.training_args.kl_type == "x-based":
                                        ref_forward_inputs["return_kwargs"] = ["next_latents_mean"]
                                    else:
                                        raise ValueError(
                                            (
                                                f"kl_type must be 'v-based' or 'x-based', got "
                                                f"{self.training_args.kl_type!r}"
                                            )
                                        )
                                    ref_output = self.adapter.forward(**ref_forward_inputs)
                                # Computed outside `torch.no_grad()` so the KL term regularizes the policy.
                                if self.training_args.kl_type == "v-based":
                                    kl_sq = (output.noise_pred - ref_output.noise_pred) ** 2
                                else:
                                    kl_sq = (
                                        output.next_latents_mean - ref_output.next_latents_mean
                                    ) ** 2
                                kl_div = self._weighted_masked_kl(
                                    kl_sq, batch["box_mask_spatial"], train_keep
                                )
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss = loss + kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                            # 6. Log info.
                            ratio_mean = torch.exp(output.log_prob - old_log_prob)
                            loss_info["ratio_mean"].append(ratio_mean.mean().detach())
                            loss_info["pixel_ratio_mean"].append(ratio.mean().detach())
                            loss_info["pixel_ratio_min"].append(ratio.min().detach())
                            loss_info["pixel_ratio_max"].append(ratio.max().detach())
                            loss_info["policy_loss"].append(policy_loss.detach())
                            loss_info["loss"].append(loss.detach())
                            loss_info["adv_spatial_mean"].append(adv.mean().detach())
                            loss_info["adv_spatial_std"].append(adv.std().detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info["clip_frac_total"].append(
                                (clip_frac_high + clip_frac_low).detach()
                            )

                            # 7. Backward and optimizer step.
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                loss_info = {k: torch.stack(v).mean() for k, v in loss_info.items()}
                                loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                                loss_info["grad_norm"] = grad_norm
                                self.log_data(
                                    {f"train/{k}": v for k, v in loss_info.items()}, step=self.step
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
