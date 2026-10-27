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

from typing import List, Optional, Union

import torch

# src/flow_factory/rewards/my_reward.py
from PIL import Image

from .abc import GroupwiseRewardModel, PointwiseRewardModel, RewardModelOutput


class MyPointwiseRewardModel(PointwiseRewardModel):

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Return zero rewards for each prompt.

        Media arguments follow the reward model interface.
        """

        # Ensure inputs are lists, each of length `self.config.batch_size`
        # Implement your custom reward computation here
        rewards = torch.zeros(len(prompt), device=self.device)

        # Wrap rewards in RewardModelOutput
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},  # Add any extra info if needed
        )


class MyGroupwiseRewardModel(GroupwiseRewardModel):

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
    ) -> RewardModelOutput:
        """Return the positional index of each prompt as its reward.

        Media arguments follow the reward model interface.
        """

        # Ensure inputs are lists, each of length `group_size`
        # Implement your custom reward computation here.
        rewards = torch.arange(
            len(prompt)
        )  # A trivia reward assignment (0, 1, 2, .... group_size - 1)

        # Wrap rewards in RewardModelOutput, make sure the order of `rewards` align the original prompt
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},  # Add any extra info if needed
        )
