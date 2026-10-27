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

# src/flow_factory/advantage/target_advantage_processor.py
"""Per-target group-normalized advantages.

Instead of normalizing per pixel across rollouts (which would require text boxes
to land on the same pixels in every rollout), rewards are normalized per
``target_id`` within each prompt group. A rollout group shares one prompt and
hence one ordered target list, so ``target_id`` is consistent across rollouts.

Input contract: ``reward_per_target`` is a ``(B_local, C)`` tensor whose columns
``0 .. C-2`` are real targets and column ``C-1`` is the extra (over-generation)
bucket; ``NaN`` marks a target the rollout did not render.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator

from .uid_hash import uid_to_group_key

logger = logging.getLogger(__name__)


def gather_group_keys(accelerator: Accelerator, unique_ids: List) -> torch.Tensor:
    """All-gather cross-rank-stable integer group keys for ``unique_ids``."""
    keys = torch.tensor(
        [uid_to_group_key(uid) for uid in unique_ids],
        dtype=torch.long,
        device=accelerator.device,
    )
    return accelerator.gather(keys)


def _local_slice(accelerator: Accelerator, values: torch.Tensor) -> torch.Tensor:
    return values.reshape(accelerator.num_processes, -1, *values.shape[1:])[
        accelerator.process_index
    ]


class TargetAdvantageProcessor:
    """Per-target (position-independent) group-normalized advantages."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def compute_target_advantages(
        self,
        accelerator: Accelerator,
        reward_per_target: torch.Tensor,
        unique_ids: List,
        ocr_failed: Optional[torch.Tensor] = None,
        sigma_min: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z-score every target column within each prompt group.

        Only rollouts that rendered the target (non-``NaN``) take part in its
        column; a column with fewer than 2 such rollouts gets advantage 0.

        Args:
            accelerator: Accelerator for cross-rank gather.
            reward_per_target: ``(B_local, C)`` rewards, ``NaN`` for absent cells.
            unique_ids: ``(B_local,)`` prompt group ids.
            ocr_failed: ``(B_local,)`` bool, True when the reward server's answer
                was unusable for that rollout. Every group containing such a
                rollout is dropped (advantages zeroed, ``drop_mask`` set).
            sigma_min: Std floor for stable normalization.

        Returns:
            ``(advantage_per_target, drop_mask)`` for the local rank, shaped
            ``(B_local, C)`` and ``(B_local,)``.
        """
        if reward_per_target.ndim != 2:
            raise ValueError(
                f"reward_per_target must be (B, C), got shape {tuple(reward_per_target.shape)}"
            )
        device = reward_per_target.device
        b_local, c_value = reward_per_target.shape

        gathered = accelerator.gather(reward_per_target.contiguous())
        gathered_keys = gather_group_keys(accelerator, unique_ids).to(device)
        if ocr_failed is None:
            ocr_failed = torch.zeros(b_local, dtype=torch.bool, device=device)
        gathered_failed = accelerator.gather(ocr_failed.to(device=device, dtype=torch.bool))

        advantages = torch.zeros_like(gathered)
        drop_mask = torch.zeros(gathered.shape[0], dtype=torch.bool, device=device)
        group_sizes: List[int] = []
        for group_id in torch.unique(gathered_keys):
            idx = (gathered_keys == group_id).nonzero(as_tuple=True)[0]
            group_sizes.append(int(idx.numel()))
            valid = ~torch.isnan(gathered[idx])
            for c in range(c_value):
                rows = idx[valid[:, c]]
                if rows.numel() >= 2:
                    col = gathered[rows, c]
                    std = torch.clamp(col.std(), min=sigma_min)
                    advantages[rows, c] = (col - col.mean()) / std
            if bool(gathered_failed[idx].any().item()):
                advantages[idx] = 0.0
                drop_mask[idx] = True

        adv_local = _local_slice(accelerator, advantages)
        drop_local = _local_slice(accelerator, drop_mask)
        if self.verbose and accelerator.is_local_main_process:
            logger.info(
                "TargetAdvantage: %d groups, sizes=%s, C=%d, sigma_min=%s, "
                "adv_mean=%.4f, adv_std=%.4f, dropped=%d",
                len(group_sizes),
                group_sizes,
                c_value,
                sigma_min,
                float(adv_local.mean()),
                float(adv_local.std()),
                int(drop_mask.sum().item()),
            )
        return adv_local, drop_local
