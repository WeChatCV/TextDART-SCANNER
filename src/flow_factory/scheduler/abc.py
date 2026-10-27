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

# src/flow_factory/scheduler/abc.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional, Union

import torch
from diffusers.utils.outputs import BaseOutput

# Tolerance for the "sigma reached the top of the schedule" branch of the
# Flow-SDE denominator ``1 - where(sigma == 1, sigmas[1], sigma)``.
# ``mu``/``shift`` and ``linspace`` are not always exactly representable in
# fp32, so ``sigmas[0]`` can land on 0.99999994 instead of 1.0. An exact
# ``== 1.0`` would then skip the substitution and divide by ``~6e-8`` --
# above the ``1e-8`` clamp -- silently turning ``std_dev_t`` into ~4e3.
SIGMA_ONE_TOL = 1e-6


def flow_sde_std_dev_t(
    sigma: torch.Tensor,
    sigma_one_subst: torch.Tensor,
    noise_level: float | torch.Tensor,
) -> torch.Tensor:
    """Flow-SDE ``std_dev_t = sqrt(sigma / (1 - sigma_eff)) * noise_level``.

    Single source of truth for the "Flow-GRPO SDE" scale, shared by the
    schedulers (``dynamics_type='Flow-SDE'``) and by the temporal advantage
    mix of ``scanner``, so the two can never drift apart.

    Args:
        sigma: Current sigma (broadcastable with ``sigma_one_subst``).
        sigma_one_subst: Value substituted for ``sigma`` on steps at the top
            of the schedule (``sigma >= 1 - SIGMA_ONE_TOL``), where
            ``1 - sigma`` vanishes. Schedulers pass ``sigmas[1]``.
        noise_level: SDE noise scale (``a``). Schedulers broadcast it to
            ``(B, 1, 1)`` via ``to_broadcast_tensor`` before calling, so this
            must stay a tensor op -- ``float(noise_level)`` raises
            "only one element tensors can be converted to Python scalars"
            as soon as ``B > 1``.
    """
    denom = 1.0 - torch.where(sigma >= (1.0 - SIGMA_ONE_TOL), sigma_one_subst, sigma)
    return torch.sqrt(sigma / denom.clamp(min=1e-8)) * noise_level


@dataclass
class SDESchedulerOutput(BaseOutput):
    """Single SDE step output with latents, statistics, and log probability."""

    next_latents: Optional[torch.FloatTensor] = None
    next_latents_mean: Optional[torch.FloatTensor] = None
    std_dev_t: Optional[torch.FloatTensor] = None
    dt: Optional[torch.FloatTensor] = None
    log_prob: Optional[torch.FloatTensor] = None
    noise_pred: Optional[torch.FloatTensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDESchedulerOutput":
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


class SDESchedulerMixin(ABC):
    """
    Abstract mixin for SDE-capable schedulers in RL fine-tuning.

    Extends `diffusers` schedulers with stochastic sampling, noise injection control,
    and log probability computation for policy gradient methods.

    Usage:
        class MySDEScheduler(DiffusersScheduler, SDESchedulerMixin):
            ...

    Attributes:
        sigmas: Noise schedule sigma values (from `diffusers`).
        timesteps: Discrete timesteps (from `diffusers`).
        noise_level: Noise injection scale for SDE sampling.
        sde_steps: Indices of steps eligible for SDE noise.
        seed: Random seed for stochastic step selection.
        dynamics_type: SDE variant ("Flow-SDE", "Dance-SDE", "CPS", "ODE").
    """

    # From diffusers schedulers
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    config: Any

    # SDE-specific
    noise_level: float
    _sde_steps: Optional[torch.Tensor]
    _num_sde_steps: Optional[int]
    seed: int
    dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"]
    _is_eval: bool

    # ==================== Mode Management ====================
    @property
    @abstractmethod
    def is_eval(self) -> bool:
        """Whether scheduler is in deterministic eval mode."""
        ...

    @abstractmethod
    def eval(self) -> None:
        """Switch to deterministic ODE sampling (no noise injection)."""
        ...

    @abstractmethod
    def train(self, mode: bool = True) -> None:
        """Switch to stochastic SDE sampling."""
        ...

    @abstractmethod
    def rollout(self, mode: bool = True) -> None:
        """Switch to rollout mode (alias for train)."""
        ...

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set random seed for stochastic step selection."""
        ...

    # ==================== Step Selection ====================
    @property
    @abstractmethod
    def sde_steps(self) -> torch.Tensor:
        """Step indices eligible for SDE noise injection."""
        ...

    @property
    @abstractmethod
    def num_sde_steps(self) -> int:
        """Number of training steps with SDE noise."""
        ...

    @property
    @abstractmethod
    def current_sde_steps(self) -> torch.Tensor:
        """Step indices where SDE noise is applied under current seed."""
        ...

    @property
    @abstractmethod
    def train_timesteps(self) -> torch.Tensor:
        """Step indices for training timesteps."""
        ...

    @abstractmethod
    def get_train_timesteps(self) -> torch.Tensor:
        """Timestep values [0, 1000] for training steps."""
        ...

    @abstractmethod
    def get_train_sigmas(self) -> torch.Tensor:
        """Sigma values for training steps."""
        ...

    # ==================== Noise Level ====================
    @abstractmethod
    def get_noise_levels(self) -> torch.Tensor:
        """Noise level for each timestep (0 if not in `current_sde_steps`)."""
        ...

    @abstractmethod
    def get_noise_level_for_timestep(
        self, timestep: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Get noise level for specific timestep(s)."""
        ...

    @abstractmethod
    def get_noise_level_for_sigma(self, sigma: float) -> float:
        """Get noise level for specific sigma value."""
        ...
