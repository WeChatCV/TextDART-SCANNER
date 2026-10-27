# src/flow_factory/scheduler/dense_flow_match_euler_discrete.py
"""
Dense Flow-Matching Euler Discrete SDE Scheduler.

Extends FlowMatchEulerDiscreteSDEScheduler to compute spatial log_prob (B, H, W)
alongside the standard scalar log_prob (B,) for spatial-advantage training
(``scanner``).

The spatial log_prob is the per-pixel Gaussian log-probability, averaged over channels
but preserving the spatial (H, W) dimensions. This enables pixel-level policy gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from ..utils.base import to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from .abc import SDESchedulerOutput, flow_sde_std_dev_t
from .flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler

logger = setup_logger(__name__)


@dataclass
class DenseSDESchedulerOutput(SDESchedulerOutput):
    """SDE step output extended with spatial log_prob."""

    log_prob_spatial: Optional[torch.FloatTensor] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DenseSDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


def _spatial_log_prob(log_prob_full: torch.Tensor) -> torch.Tensor:
    """Reduce log_prob to spatial dimensions.

    4D (B, C, H, W) → mean over C → (B, H, W)   [image models: SD3, etc.]
    3D (B, L, D)     → mean over D → (B, L)      [sequence models: FLUX, etc.]
    """
    if log_prob_full.ndim == 4:
        return log_prob_full.mean(dim=1)
    elif log_prob_full.ndim == 3:
        return log_prob_full.mean(dim=2)
    else:
        return log_prob_full.mean(dim=tuple(range(1, log_prob_full.ndim)))


def _zeros_spatial(ref_tensor: torch.Tensor) -> torch.Tensor:
    """Create zero-filled spatial tensor matching ref_tensor's layout."""
    if ref_tensor.ndim == 4:
        return torch.zeros(
            ref_tensor.shape[0],
            ref_tensor.shape[2],
            ref_tensor.shape[3],
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )
    else:
        return torch.zeros(
            ref_tensor.shape[0],
            ref_tensor.shape[1],
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )


class DenseFlowMatchEulerDiscreteSDEScheduler(FlowMatchEulerDiscreteSDEScheduler):
    """
    Scheduler that additionally computes spatial log_prob (B, H, W).

    Identical to FlowMatchEulerDiscreteSDEScheduler except:
    - log_prob computation is split into spatial (mean over C) and scalar (mean over all)
    - Returns DenseSDESchedulerOutput with log_prob_spatial field
    """

    @classmethod
    def from_scheduler(
        cls, scheduler: FlowMatchEulerDiscreteSDEScheduler
    ) -> "DenseFlowMatchEulerDiscreteSDEScheduler":
        """Create a DenseFlowMatchEulerDiscreteSDEScheduler from an existing scheduler."""
        new_scheduler = cls(
            noise_level=scheduler.noise_level,
            sde_steps=scheduler._sde_steps.tolist() if scheduler._sde_steps is not None else None,
            num_sde_steps=scheduler._num_sde_steps,
            seed=scheduler.seed,
            dynamics_type=scheduler.dynamics_type,
            # ``sde_window_mode`` / ``sde_window_seed`` are plain instance
            # attributes (not register_to_config), so they are NOT carried by
            # ``**scheduler.config``. Copy them explicitly (with getattr fallback
            # for older base schedulers) or a non-default window mode would be
            # silently reset to the default when the trainer swaps the scheduler.
            sde_window_mode=getattr(scheduler, "sde_window_mode", "random"),
            sde_window_seed=getattr(scheduler, "sde_window_seed", 0),
            # Carry the per-rank window offset too, otherwise the swapped
            # scheduler would fall back to rank 0 on every process and collapse the
            # per-rank window spread (flow_grpo's `process_index`) into one window.
            sde_window_rank=getattr(scheduler, "sde_window_rank", 0),
            **scheduler.config,
        )
        # Copy runtime state
        if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
            new_scheduler.timesteps = scheduler.timesteps
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
            new_scheduler.sigmas = scheduler.sigmas
        new_scheduler._is_eval = scheduler._is_eval
        return new_scheduler

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor] = None,
        compute_log_prob: bool = False,
        noise_level: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        timestep_next: Optional[Union[float, torch.Tensor]] = None,
        dynamics_type: Optional[Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"]] = None,
        sigma_max: Optional[float] = None,
        return_dict: bool = True,
        return_kwargs: Optional[List[str]] = None,
    ):
        """
        SDE step with both scalar and spatial log_prob computation.

        Additional output vs parent:
            log_prob_spatial: (B, H, W) - per-pixel log_prob averaged over channels
        """
        if return_kwargs is None:
            return_kwargs = ['next_latents']
        # ---- Timestep resolution (same as parent) ----
        if timestep_next is None:
            if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
            ):
                step_index = [int(timestep)]
            elif isinstance(timestep, torch.Tensor):
                if timestep.ndim == 0:
                    step_index = [self.index_for_timestep(timestep)]
                elif timestep.ndim == 1:
                    step_index = [self.index_for_timestep(t) for t in timestep]
                else:
                    raise ValueError(
                        f"`timestep` must be a scalar or 1D tensor, got shape {tuple(timestep.shape)}."
                    )
            elif isinstance(timestep, float):
                step_index = [self.index_for_timestep(timestep)]
            else:
                raise TypeError(
                    f"`timestep` must be float, or torch.Tensor, got {type(timestep).__name__}."
                )

            timestep = self.timesteps[step_index]
            timestep_next = torch.as_tensor(
                [
                    (
                        self.timesteps[i + 1]
                        if i + 1 < len(self.timesteps)
                        else torch.tensor(0, device=timestep.device)
                    )
                    for i in step_index
                ],
                device=timestep.device,
            )
            sigma = self.sigmas[step_index]
            sigma_prev = self.sigmas[[i + 1 for i in step_index]]
        else:
            sigma = timestep / 1000
            sigma_prev = timestep_next / 1000

        # ---- Numerical preparation ----
        _input_dtype = latents.dtype
        noise_pred = noise_pred.float()
        latents = latents.float()
        if next_latents is not None:
            next_latents = next_latents.float()

        # ---- Variable preparation ----
        dynamics_type = dynamics_type or self.dynamics_type
        if self.is_eval or dynamics_type == "ODE":
            noise_level = 0.0
        elif noise_level is None:
            noise_level = self.get_noise_level_for_sigma(sigma)

        noise_level = to_broadcast_tensor(noise_level, latents)
        sigma = to_broadcast_tensor(sigma, latents)
        sigma_prev = to_broadcast_tensor(sigma_prev, latents)
        dt = sigma_prev - sigma

        # Initialize spatial log_prob
        log_prob_spatial = None

        # ---- Compute next sample ----
        if dynamics_type == "ODE":
            next_latents_mean = latents + noise_pred * dt
            std_dev_t = torch.zeros_like(sigma)
            if next_latents is None:
                next_latents = next_latents_mean
            if compute_log_prob:
                logger.warning(
                    "`log_prob` is meaningless when `dynamics_type` is set `ODE`, setting to zero."
                )
                log_prob = torch.zeros(
                    (next_latents.shape[0]), dtype=next_latents.dtype, device=next_latents.device
                )
                log_prob_spatial = _zeros_spatial(next_latents)

        elif dynamics_type == "Flow-SDE":
            sigma_max = sigma_max or self.sigmas[1].item()
            sigma_max = to_broadcast_tensor(sigma_max, latents)
            std_dev_t = flow_sde_std_dev_t(sigma, sigma_max, noise_level)

            next_latents_mean = (
                latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
                next_latents = next_latents.to(_input_dtype).float()

            if compute_log_prob:
                std_variance = std_dev_t * torch.sqrt(-1 * dt)
                log_prob_full = (
                    -((next_latents.detach() - next_latents_mean) ** 2) / (2 * std_variance**2)
                    - torch.log(std_variance)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                # Spatial: mean over channels, keep (B, H, W)
                log_prob_spatial = _spatial_log_prob(log_prob_full)
                # Scalar: mean over all spatial dims
                log_prob = log_prob_full.mean(dim=tuple(range(1, log_prob_full.ndim)))

        elif dynamics_type == "Dance-SDE":
            pred_original_sample = latents - sigma * noise_pred
            std_dev_t = noise_level
            log_term = (
                0.5 * noise_level**2 * (latents - pred_original_sample * (1 - sigma)) / sigma**2
            )
            next_latents_mean = latents + (noise_pred + log_term) * dt
            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
                next_latents = next_latents.to(_input_dtype).float()

            if compute_log_prob:
                std_variance = std_dev_t * torch.sqrt(-1 * dt)
                log_prob_full = (
                    (-((next_latents.detach() - next_latents_mean) ** 2) / (2 * std_variance**2))
                    - torch.log(std_variance)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob_spatial = _spatial_log_prob(log_prob_full)
                log_prob = log_prob_full.mean(dim=tuple(range(1, log_prob_full.ndim)))

        elif dynamics_type == "CPS":
            std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2)
            x0 = latents - sigma * noise_pred
            x1 = latents + noise_pred * (1 - sigma)
            next_latents_mean = x0 * (1 - sigma_prev) + x1 * torch.sqrt(
                sigma_prev**2 - std_dev_t**2
            )

            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * variance_noise
                next_latents = next_latents.to(_input_dtype).float()

            if compute_log_prob:
                log_prob_full = -((next_latents.detach() - next_latents_mean) ** 2)
                log_prob_spatial = _spatial_log_prob(log_prob_full)
                log_prob = log_prob_full.mean(dim=tuple(range(1, log_prob_full.ndim)))

        if not compute_log_prob:
            log_prob = None
            # Always provide a zero-filled log_prob_spatial so CallbackCollector
            # doesn't skip it (maintaining index alignment with callback_index_map)
            if log_prob_spatial is None:
                log_prob_spatial = _zeros_spatial(next_latents)

        if not return_dict:
            return (
                next_latents,
                next_latents_mean,
                noise_pred,
                log_prob,
                std_dev_t,
                dt,
                log_prob_spatial,
            )

        d = {}
        for k in return_kwargs:
            if k in locals():
                d[k] = locals()[k]
            else:
                logger.warning(
                    f"Requested return keyword '{k}' is not available in the step output."
                )

        return DenseSDESchedulerOutput.from_dict(d)
