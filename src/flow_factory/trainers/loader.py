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

# src/flow_factory/trainers/loader.py
"""
Trainer loader factory for extensibility.
Supports multiple RL algorithms via registry pattern.
"""

import os
from datetime import timedelta

from accelerate import Accelerator, DistributedDataParallelKwargs, PartialState
from accelerate.utils import ProjectConfiguration, broadcast_object_list, set_seed

from ..hparams import Arguments
from ..models.loader import load_model
from ..utils.logger_utils import setup_logger
from .abc import BaseTrainer
from .registry import get_trainer_class, list_registered_trainers

logger = setup_logger(__name__)

ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True
)  # Fix issue that Qwen-Image uses different cache context for CFG forwards


def sync_run_name(config: Arguments) -> None:
    """Make every rank agree on ``log.run_name`` before any path is derived from it.

    Each rank parses the YAML on its own, so an auto-generated run name (which
    ends in a local-clock timestamp) differs between ranks that started in
    different seconds. Since every rank writes its own DeepSpeed/FSDP shard to
    ``{save_dir}/{run_name}/checkpoints``, that would scatter a single
    checkpoint across as many directories as there were distinct start seconds.
    Broadcasting rank 0's name is a no-op when the name is set explicitly.
    """
    # Accelerator() reuses an already-created PartialState, so initializing the
    # distributed state here (needed for the broadcast) is safe.
    state_kwargs = {}
    timeout_minutes = os.environ.get("FF_PROCESS_GROUP_TIMEOUT_MIN")
    if timeout_minutes is not None:
        if int(timeout_minutes) <= 0:
            raise ValueError("FF_PROCESS_GROUP_TIMEOUT_MIN must be a positive integer.")
        state_kwargs["timeout"] = timedelta(minutes=int(timeout_minutes))
    state = PartialState(**state_kwargs)
    if state.num_processes == 1:
        return

    names = [config.log_args.run_name]
    broadcast_object_list(names, from_process=0)
    if names[0] != config.log_args.run_name:
        logger.info(
            "Overriding locally generated run_name %r with rank 0's %r so all "
            "ranks share one run directory.",
            config.log_args.run_name,
            names[0],
        )
        config.log_args.run_name = names[0]


def load_trainer(config: Arguments) -> BaseTrainer:
    """
    Factory function to instantiate trainer based on algorithm type.

    Uses registry pattern for automatic trainer discovery and loading.
    Supports both built-in trainers and custom algorithms via python paths.

    Args:
        config: Configuration containing trainer_type and all hyperparameters

    Returns:
        An instance of a BaseTrainer subclass

    Raises:
        ImportError: If the trainer is not registered or cannot be imported

    Examples:
        # Using built-in trainer
        config.training_args.trainer_type = "grpo"
        trainer = load_trainer(config)

        # Using custom trainer
        config.training_args.trainer_type = "my_package.trainers.PPOTrainer"
        trainer = load_trainer(config)
    """
    sync_run_name(config)

    # Initialize Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.log_args.save_dir, config.log_args.run_name),
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(config.training_args.seed, device_specific=True)

    # Initialize model adapter
    adapter = load_model(config=config, accelerator=accelerator)

    # Get trainer class from registry
    trainer_type = config.training_args.trainer_type

    try:
        trainer_cls = get_trainer_class(trainer_type)
    except ImportError as e:
        registered_trainers = list(list_registered_trainers().keys())
        raise ImportError(
            f"Failed to load trainer '{trainer_type}'. "
            f"Available trainers: {registered_trainers}"
        ) from e

    return trainer_cls(
        config=config,
        accelerator=accelerator,
        adapter=adapter,
    )
