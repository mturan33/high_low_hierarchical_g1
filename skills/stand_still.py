"""
Stand-Still Skill
==================
Hold the robot in place with zero velocity commands.
Used between other skills or as a "stop" command.
"""

from __future__ import annotations

import torch
from typing import Optional

from .base_skill import BaseSkill, SkillResult, SkillStatus
from ..config.skill_config import StandStillConfig
from ..config.joint_config import MIN_BASE_HEIGHT, CONTROL_DT


class StandStillSkill(BaseSkill):
    """Stand still for a specified duration."""

    def __init__(
        self,
        config: Optional[StandStillConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(name="stand_still", device=device)
        self.cfg = config or StandStillConfig()
        self._max_steps = self.cfg.max_steps
        self._target_steps: int = 0

    def reset(self, duration_s: float = None, **kwargs) -> None:
        """
        Initialize stand_still skill.

        Args:
            duration_s: How long to stand (seconds). Default from config.
        """
        super().reset()
        duration = duration_s if duration_s is not None else self.cfg.duration_s
        self._target_steps = int(duration / CONTROL_DT)
        print(f"[StandStill] Duration: {duration:.1f}s ({self._target_steps} steps)")

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """Execute one stand_still step (zero velocity)."""
        super().step(obs_dict)

        root_pos = obs_dict["root_pos"]
        base_height = obs_dict.get("base_height", root_pos[:, 2])

        # Check fall
        if (base_height < MIN_BASE_HEIGHT).any():
            zero_cmd = torch.zeros(1, 3, device=self.device)
            return zero_cmd, True, self._make_failure(reason="Robot fell")

        # Zero velocity command
        num_envs = root_pos.shape[0]
        zero_cmd = torch.zeros(num_envs, 3, device=self.device)

        # Check if duration complete
        if self._step_count >= self._target_steps:
            return zero_cmd, True, self._make_success(
                reason=f"Stood for {self._step_count * CONTROL_DT:.1f}s",
            )

        return zero_cmd, False, self._make_running()

    def get_affordance(self, state: dict) -> float:
        """Standing still is almost always possible."""
        robot = state.get("robot", {})
        if robot.get("stance") == "squatting":
            return 0.5  # Can stand still while squatting but less stable
        return 0.99
