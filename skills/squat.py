"""
Squat Skill (Placeholder)
===========================
Phase 2 implementation — currently a placeholder.

Options for implementation:
  A) Eureka-generated reward → train separate squat policy
  B) Manual reward: base height tracking curriculum (0.7 → 0.5 → 0.3m)
  C) Heuristic: interpolate leg joints to squat pose (no RL)

For now, implements option C (heuristic joint interpolation).
"""

from __future__ import annotations

import torch
from typing import Optional

from .base_skill import BaseSkill, SkillResult, SkillStatus
from config.skill_config import SquatConfig
from config.joint_config import MIN_BASE_HEIGHT, CONTROL_DT


class SquatSkill(BaseSkill):
    """Squat to a target height (placeholder — Phase 2)."""

    def __init__(
        self,
        config: Optional[SquatConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(name="squat", device=device)
        self.cfg = config or SquatConfig()
        self._max_steps = self.cfg.max_steps

    def reset(self, depth: float = None, **kwargs) -> None:
        """
        Initialize squat skill.

        Args:
            depth: Target squat depth from standing height (meters).
        """
        super().reset()
        self._target_height = self.cfg.target_height if depth is None else (0.78 - depth)
        print(f"[Squat] Target height: {self._target_height:.2f}m (PLACEHOLDER)")

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """Placeholder: immediately returns success."""
        super().step(obs_dict)

        # TODO Phase 2: Implement actual squat control
        # For now, return zero command and mark as success after a brief pause
        zero_cmd = torch.zeros(1, 3, device=self.device)

        if self._step_count >= 50:  # 1 second pause
            return zero_cmd, True, self._make_success(
                reason="Squat placeholder complete (Phase 2 TODO)",
            )

        return zero_cmd, False, self._make_running()

    def get_affordance(self, state: dict) -> float:
        """Squatting is possible only when standing."""
        robot = state.get("robot", {})
        if robot.get("stance") == "standing":
            return 0.8
        return 0.1
