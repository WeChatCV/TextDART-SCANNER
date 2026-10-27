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

# src/flow_factory/hparams/log_args.py
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import yaml

from .abc import ArgABC


@dataclass
class LogArguments(ArgABC):
    r"""Arguments pertaining to logging and checkpoint saving."""

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training run."},
    )
    project: str = field(
        default="Flow-Factory",
        metadata={"help": "Project name for logging platforms."},
    )
    logging_backend: Optional[Literal["wandb", "swanlab", "none"]] = field(
        default=None,
        metadata={"help": "Logging backend to use."},
    )

    save_dir: str = field(
        default="save",
        metadata={"help": "Directory to save logs and checkpoints. None for no saving."},
    )

    resume_run: bool = field(
        default=False,
        metadata={
            "help": (
                "Continue a previous run instead of starting a fresh one. "
                "When True and `model.resume_path` points to a "
                "`{save_dir}/{run_name}/checkpoints/checkpoint-{epoch}` directory, "
                "the run directory (`save_dir` + `run_name`) is reused so new "
                "checkpoints keep landing in the SAME `checkpoints/` folder, the "
                "epoch/step counters are restored, and the logging backend resumes "
                "the same run (see `wandb_run_id`). Options: true, false."
            )
        },
    )

    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Wandb run id to resume (e.g. '2kqx15x1', the suffix of the "
                "`wandb/run-<timestamp>-<id>` directory). Only used when "
                "`resume_run` is True and `logging_backend` is 'wandb'. New "
                "points append to the existing curves on the same ``_step`` "
                "x-axis. Wandb cannot rewrite a step that was already logged."
            )
        },
    )

    save_freq: int = field(
        default=10,
        metadata={"help": "Model saving frequency (in epochs). 0 for no saving."},
    )

    save_model_only: bool = field(
        default=True,
        metadata={
            "help": "Whether to save the model only, or the complete training state (model and optimizer)."
        },
    )

    save_samples: bool = field(
        default=False,
        metadata={"help": "Whether to save generated sample images/videos to disk each epoch."},
    )

    save_samples_dir: str = field(
        default="images",
        metadata={
            "help": "Base directory for saved samples. Layout: {save_samples_dir}/{run_name}/{epoch}/{group_id}/"
        },
    )

    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print detailed progress during training."},
    )

    def __post_init__(self):

        # Expand path to user's path
        self.save_dir = os.path.expanduser(self.save_dir)
        # If save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()
