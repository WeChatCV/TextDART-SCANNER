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

# src/flow_factory/logger/wandb.py
from typing import Any, Dict, Optional

import wandb

from .abc import Logger
from .formatting import LogImage, LogTable, LogVideo


def build_wandb_init_kwargs(log_args: Any, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build ``wandb.init`` kwargs for a new or resumed run.

    Resume uses ``id`` + ``resume='allow'`` so later points keep the same
    wandb ``_step`` x-axis. Do not use ``resume_from`` / rewind: that API is
    private preview and 400s on standard accounts.

    Args:
        log_args: Log arguments. Uses ``project``, ``run_name``, ``resume_run``,
            and ``wandb_run_id``.
        config_dict: Full config dumped into wandb's run config.

    Returns:
        Keyword arguments for ``wandb.init``.
    """
    init_kwargs: Dict[str, Any] = dict(
        project=log_args.project,
        name=log_args.run_name,
        config=config_dict,
    )
    if log_args.resume_run and log_args.wandb_run_id:
        init_kwargs["id"] = log_args.wandb_run_id
        init_kwargs["resume"] = "allow"
    return init_kwargs


class WandbLogger(Logger):
    def _init_platform(self):
        log_args = self.config.log_args
        init_kwargs = build_wandb_init_kwargs(log_args, self.config.to_dict())
        wandb.init(**init_kwargs)
        self.platform = wandb

    def _convert_to_platform(
        self, value: Any, height: Optional[int] = None, width: Optional[int] = None
    ) -> Any:
        if isinstance(value, LogImage):
            return wandb.Image(value.get_value(height, width), caption=value.caption)

        if isinstance(value, LogVideo):
            return wandb.Video(
                value.get_value(format="mp4", height=height, width=width),
                caption=value.caption,
                format="mp4",
            )

        if isinstance(value, LogTable):
            # For LogTable, all items have the same height for better formatting
            h = height or value.target_height  # Use specified height or default
            data = [
                [
                    self._convert_to_platform(item, height=h) if item is not None else None
                    for item in row
                ]
                for row in value.rows
            ]
            return wandb.Table(columns=value.columns, data=data)

        return value

    def _log_impl(self, data: Dict, step: int) -> None:
        self.platform.log(data, step=step)

    def finish(self) -> None:
        if wandb.run is not None:
            wandb.finish()
