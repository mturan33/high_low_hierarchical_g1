"""
Turn-To Skill
==============
Rotate the robot in place to face a target heading.

Architecture:
    target_heading → VelocityCommandGenerator → [0, 0, vyaw] → LocoPolicy → joint targets
"""

from __future__ import annotations

import torch
import math
from typing import Optional

from .base_skill import BaseSkill, SkillResult, SkillStatus
from low_level.velocity_command import VelocityCommandGenerator, get_yaw_from_quat, normalize_angle
from config.skill_config import TurnToConfig
from config.joint_config import MIN_BASE_HEIGHT


class TurnToSkill(BaseSkill):
    """Turn in place to a target heading."""

    def __init__(
        self,
        config: Optional[TurnToConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(name="turn_to", device=device)
        self.cfg = config or TurnToConfig()
        self._max_steps = self.cfg.max_steps

        self.cmd_gen = VelocityCommandGenerator(
            kp_angular=self.cfg.kp_angular,
            max_ang_vel_z=self.cfg.max_yaw_rate,
            device=device,
        )

        self._target_heading: Optional[torch.Tensor] = None

    def reset(self, heading: float = None, target_x: float = None, target_y: float = None, **kwargs) -> None:
        """
        Initialize turn_to skill.

        Args:
            heading: Target heading in radians (world frame).
            target_x, target_y: Alternative — compute heading from position.
                                 Requires robot position to be passed in first step.
        """
        super().reset()

        if heading is not None:
            self._target_heading = torch.tensor(
                [heading], dtype=torch.float32, device=self.device
            )
            self._compute_from_position = False
            print(f"[TurnTo] Target heading: {math.degrees(heading):.1f} deg")
        elif target_x is not None and target_y is not None:
            self._target_pos = torch.tensor(
                [[target_x, target_y]], dtype=torch.float32, device=self.device
            )
            self._compute_from_position = True
            self._target_heading = None
            print(f"[TurnTo] Face toward: ({target_x:.2f}, {target_y:.2f})")
        else:
            raise ValueError("Must provide either 'heading' or 'target_x/target_y'")

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """Execute one turn_to step."""
        super().step(obs_dict)

        timeout = self._check_timeout()
        if timeout is not None:
            zero_cmd = torch.zeros(1, 3, device=self.device)
            return zero_cmd, True, timeout

        root_pos = obs_dict["root_pos"]
        root_quat = obs_dict["root_quat"]
        base_height = obs_dict.get("base_height", root_pos[:, 2])
        robot_yaw = get_yaw_from_quat(root_quat)

        # Check fall
        if (base_height < MIN_BASE_HEIGHT).any():
            zero_cmd = torch.zeros(1, 3, device=self.device)
            return zero_cmd, True, self._make_failure(reason="Robot fell")

        # Compute target heading from position if needed
        if self._compute_from_position and self._target_heading is None:
            delta = self._target_pos - root_pos[:, :2]
            self._target_heading = torch.atan2(delta[:, 1], delta[:, 0])

        # Compute turn command
        target = self._target_heading.expand(robot_yaw.shape[0])
        cmd_vel, heading_error = self.cmd_gen.compute_turn_command(
            robot_yaw, target
        )

        # Check arrival
        if (torch.abs(heading_error) < self.cfg.heading_threshold).all():
            zero_cmd = torch.zeros_like(cmd_vel)
            return zero_cmd, True, self._make_success(
                reason="Reached target heading",
                final_error=heading_error.abs().mean().item(),
            )

        return cmd_vel, False, self._make_running(
            heading_error=heading_error.abs().mean().item(),
        )

    def get_affordance(self, state: dict) -> float:
        """Turn is almost always possible when standing."""
        robot = state.get("robot", {})
        if robot.get("stance") == "squatting":
            return 0.2
        return 0.95
