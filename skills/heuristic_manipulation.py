"""
Heuristic Manipulation Skills (Placeholder)
=============================================
Phase 3 implementation — distance-based grasp and place.

No RL involved. Uses simple distance thresholds:
  - Grasp: if hand-object distance < threshold → attach object
  - Place: if robot near surface + holding object → detach object

These will be replaced by RL-trained skills in future phases.
"""

from __future__ import annotations

import torch
from typing import Optional

from .base_skill import BaseSkill, SkillResult, SkillStatus
from ..config.skill_config import HeuristicGraspConfig, HeuristicPlaceConfig


class HeuristicGraspSkill(BaseSkill):
    """Heuristic grasp: attach object if hand is close enough (Phase 3)."""

    def __init__(
        self,
        config: Optional[HeuristicGraspConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(name="grasp", device=device)
        self.cfg = config or HeuristicGraspConfig()
        self._max_steps = self.cfg.max_steps
        self._object_id: Optional[str] = None

    def reset(self, object_id: str, **kwargs) -> None:
        """
        Initialize grasp skill.

        Args:
            object_id: ID of the object to grasp (from semantic map).
        """
        super().reset()
        self._object_id = object_id
        print(f"[Grasp] Target object: {object_id} (PLACEHOLDER)")

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """Placeholder: check hand-object distance."""
        super().step(obs_dict)

        zero_cmd = torch.zeros(1, 3, device=self.device)

        # TODO Phase 3: Implement actual grasp
        # 1. Get hand position from FK
        # 2. Get object position from semantic map
        # 3. If distance < threshold, call env.attach_object()
        # 4. Return success

        if self._step_count >= 25:
            return zero_cmd, True, self._make_success(
                reason=f"Grasp placeholder complete for '{self._object_id}' (Phase 3 TODO)",
            )

        return zero_cmd, False, self._make_running()

    def get_affordance(self, state: dict) -> float:
        """Grasp requires squatting near the object."""
        robot = state.get("robot", {})

        # Must not already be holding something
        if robot.get("holding") is not None:
            return 0.0

        # Should be squatting for ground objects
        if robot.get("stance") != "squatting":
            return 0.2

        return 0.7


class HeuristicPlaceSkill(BaseSkill):
    """Heuristic place: detach held object above surface (Phase 3)."""

    def __init__(
        self,
        config: Optional[HeuristicPlaceConfig] = None,
        device: str = "cuda",
    ):
        super().__init__(name="place", device=device)
        self.cfg = config or HeuristicPlaceConfig()
        self._max_steps = self.cfg.max_steps
        self._surface_id: Optional[str] = None

    def reset(self, surface_id: str, **kwargs) -> None:
        """
        Initialize place skill.

        Args:
            surface_id: ID of the surface to place on (from semantic map).
        """
        super().reset()
        self._surface_id = surface_id
        print(f"[Place] Target surface: {surface_id} (PLACEHOLDER)")

    def step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, bool, SkillResult]:
        """Placeholder: detach object."""
        super().step(obs_dict)

        zero_cmd = torch.zeros(1, 3, device=self.device)

        # TODO Phase 3: Implement actual place
        # 1. Verify holding an object
        # 2. Check proximity to target surface
        # 3. Call env.detach_object()

        if self._step_count >= 25:
            return zero_cmd, True, self._make_success(
                reason=f"Place placeholder complete on '{self._surface_id}' (Phase 3 TODO)",
            )

        return zero_cmd, False, self._make_running()

    def get_affordance(self, state: dict) -> float:
        """Place requires holding an object near a surface."""
        robot = state.get("robot", {})

        if robot.get("holding") is None:
            return 0.0  # Must be holding something

        if robot.get("stance") != "standing":
            return 0.3

        return 0.8
