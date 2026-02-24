"""
Base Skill Interface
=====================
Abstract base class for all skill primitives.
Inspired by SayCan (Ahn et al. 2022) affordance model.

Each skill:
  1. Takes parameters (target position, heading, etc.)
  2. Runs a control loop using the locomotion policy
  3. Returns success/failure with diagnostics
  4. Provides an affordance score (probability of success given current state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch


class SkillStatus(Enum):
    """Skill execution status."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SkillResult:
    """Result of a skill execution."""
    status: SkillStatus
    steps_taken: int = 0
    time_elapsed: float = 0.0
    reason: str = ""
    data: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == SkillStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status in (SkillStatus.FAILED, SkillStatus.TIMEOUT)

    def __repr__(self) -> str:
        return (
            f"SkillResult(status={self.status.value}, "
            f"steps={self.steps_taken}, reason='{self.reason}')"
        )


class BaseSkill(ABC):
    """
    Abstract base class for skill primitives.

    Subclasses must implement:
      - reset(**params): Initialize skill with parameters
      - step(obs_dict) -> (action, done, result): Execute one control step
      - get_affordance(state) -> float: Estimate success probability

    The skill executor calls:
      1. skill.reset(target_x=1.0, target_y=2.0)
      2. Loop: action, done, result = skill.step(obs_dict)
      3. Check result.status
    """

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = torch.device(device)
        self._step_count = 0
        self._max_steps = 1000
        self._is_active = False

    @abstractmethod
    def reset(self, **params) -> None:
        """
        Initialize skill with parameters.

        Args:
            **params: Skill-specific parameters (e.g., target_x, target_y)
        """
        self._step_count = 0
        self._is_active = True

    @abstractmethod
    def step(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, bool, SkillResult]:
        """
        Execute one control step.

        Args:
            obs_dict: Current observations from environment:
                - "root_pos": [num_envs, 3] — robot base position
                - "root_quat": [num_envs, 4] — robot base orientation (wxyz)
                - "base_ang_vel": [num_envs, 3] — angular velocity in body frame
                - "projected_gravity": [num_envs, 3] — gravity in body frame
                - "joint_pos": [num_envs, 29] — joint positions
                - "joint_vel": [num_envs, 29] — joint velocities
                - "base_height": [num_envs] — base height above ground

        Returns:
            velocity_command: [num_envs, 3] = [vx, vy, vyaw]
            done: Whether skill has terminated
            result: SkillResult with status and diagnostics
        """
        self._step_count += 1

    def get_affordance(self, state: dict) -> float:
        """
        Estimate the probability of successfully executing this skill
        from the current state (SayCan-style).

        Args:
            state: Current semantic map state dict

        Returns:
            affordance: Float in [0, 1], higher = more likely to succeed
        """
        # Default: always possible
        return 1.0

    @property
    def is_active(self) -> bool:
        """Whether the skill is currently executing."""
        return self._is_active

    @property
    def step_count(self) -> int:
        """Number of steps taken in current execution."""
        return self._step_count

    def _check_timeout(self) -> Optional[SkillResult]:
        """Check if max steps exceeded."""
        if self._step_count >= self._max_steps:
            self._is_active = False
            return SkillResult(
                status=SkillStatus.TIMEOUT,
                steps_taken=self._step_count,
                reason=f"Exceeded max steps ({self._max_steps})",
            )
        return None

    def _make_success(self, reason: str = "", **data) -> SkillResult:
        """Create a success result."""
        self._is_active = False
        return SkillResult(
            status=SkillStatus.SUCCESS,
            steps_taken=self._step_count,
            reason=reason,
            data=data,
        )

    def _make_failure(self, reason: str = "", **data) -> SkillResult:
        """Create a failure result."""
        self._is_active = False
        return SkillResult(
            status=SkillStatus.FAILED,
            steps_taken=self._step_count,
            reason=reason,
            data=data,
        )

    def _make_running(self, **data) -> SkillResult:
        """Create a running result (not done yet)."""
        return SkillResult(
            status=SkillStatus.RUNNING,
            steps_taken=self._step_count,
            data=data,
        )
