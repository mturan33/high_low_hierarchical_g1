"""
DEX3 Finger Controller
=======================
Heuristic open/close controller for DEX3 3-finger hands.

Each hand has 7 joints:
  - index (2): index_0, index_1
  - middle (2): middle_0, middle_1
  - thumb (3): thumb_0, thumb_1, thumb_2

Control modes:
  - OPEN:  all fingers at 0.0 (flat open)
  - CLOSE: fingers curl to grasp position
  - INTERPOLATE: smooth transition between open/close

The controller outputs joint position targets for the 14 finger joints
that get applied alongside the body joint targets from the locomotion policy.
"""

from __future__ import annotations

import torch
from enum import Enum
from typing import Optional

from ..config.joint_config import (
    NUM_DEX3_JOINTS,
    NUM_DEX3_JOINTS_PER_HAND,
    DEX3_FINGER_CLOSE,
    DEX3_JOINT_NAMES,
)


class GripperState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    OPENING = "opening"


class FingerController:
    """
    Heuristic DEX3 finger controller.

    Provides smooth open/close control for both hands independently.
    Outputs 14 joint position targets per env.

    Usage:
        controller = FingerController(num_envs=16, device="cuda:0")

        # Open all fingers
        targets = controller.get_targets()  # [16, 14] all zeros

        # Close right hand
        controller.close(hand="right")
        for _ in range(50):
            targets = controller.get_targets()  # smoothly interpolates
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        close_speed: float = 0.05,  # radians per control step
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.close_speed = close_speed

        # Current finger positions [num_envs, 14]
        self.current_pos = torch.zeros(
            num_envs, NUM_DEX3_JOINTS, dtype=torch.float32, device=self.device
        )

        # Target finger positions [num_envs, 14]
        self.target_pos = torch.zeros(
            num_envs, NUM_DEX3_JOINTS, dtype=torch.float32, device=self.device
        )

        # Close position tensor [14]
        self._close_positions = torch.tensor(
            [DEX3_FINGER_CLOSE[name] for name in DEX3_JOINT_NAMES],
            dtype=torch.float32, device=self.device,
        )

        # Per-hand state tracking
        self.left_state = GripperState.OPEN
        self.right_state = GripperState.OPEN

        print(f"[FingerController] DEX3 hands: {NUM_DEX3_JOINTS} joints, "
              f"speed={close_speed} rad/step")

    def open(self, hand: str = "both", env_ids: Optional[torch.Tensor] = None):
        """
        Command fingers to open.

        Args:
            hand: "left", "right", or "both"
            env_ids: specific env indices (None = all envs)
        """
        ids = slice(None) if env_ids is None else env_ids

        if hand in ("left", "both"):
            self.target_pos[ids, :NUM_DEX3_JOINTS_PER_HAND] = 0.0
            self.left_state = GripperState.OPENING
        if hand in ("right", "both"):
            self.target_pos[ids, NUM_DEX3_JOINTS_PER_HAND:] = 0.0
            self.right_state = GripperState.OPENING

    def close(self, hand: str = "both", env_ids: Optional[torch.Tensor] = None):
        """
        Command fingers to close (grasp).

        Args:
            hand: "left", "right", or "both"
            env_ids: specific env indices (None = all envs)
        """
        ids = slice(None) if env_ids is None else env_ids

        if hand in ("left", "both"):
            self.target_pos[ids, :NUM_DEX3_JOINTS_PER_HAND] = \
                self._close_positions[:NUM_DEX3_JOINTS_PER_HAND]
            self.left_state = GripperState.CLOSING
        if hand in ("right", "both"):
            self.target_pos[ids, NUM_DEX3_JOINTS_PER_HAND:] = \
                self._close_positions[NUM_DEX3_JOINTS_PER_HAND:]
            self.right_state = GripperState.CLOSING

    def get_targets(self) -> torch.Tensor:
        """
        Get current finger joint position targets.

        Smoothly interpolates towards target positions.

        Returns:
            finger_targets: [num_envs, 14] joint position targets
        """
        # Smooth interpolation toward target
        diff = self.target_pos - self.current_pos
        step = torch.clamp(diff, -self.close_speed, self.close_speed)
        self.current_pos = self.current_pos + step

        # Update states
        left_done = (torch.abs(diff[:, :NUM_DEX3_JOINTS_PER_HAND]).max() < 0.01)
        right_done = (torch.abs(diff[:, NUM_DEX3_JOINTS_PER_HAND:]).max() < 0.01)

        if left_done:
            if self.left_state == GripperState.CLOSING:
                self.left_state = GripperState.CLOSED
            elif self.left_state == GripperState.OPENING:
                self.left_state = GripperState.OPEN
        if right_done:
            if self.right_state == GripperState.CLOSING:
                self.right_state = GripperState.CLOSED
            elif self.right_state == GripperState.OPENING:
                self.right_state = GripperState.OPEN

        return self.current_pos.clone()

    def is_closed(self, hand: str = "both") -> bool:
        """Check if fingers are fully closed."""
        if hand == "left":
            return self.left_state == GripperState.CLOSED
        elif hand == "right":
            return self.right_state == GripperState.CLOSED
        else:
            return (self.left_state == GripperState.CLOSED and
                    self.right_state == GripperState.CLOSED)

    def is_open(self, hand: str = "both") -> bool:
        """Check if fingers are fully open."""
        if hand == "left":
            return self.left_state == GripperState.OPEN
        elif hand == "right":
            return self.right_state == GripperState.OPEN
        else:
            return (self.left_state == GripperState.OPEN and
                    self.right_state == GripperState.OPEN)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset fingers to open position."""
        if env_ids is None:
            self.current_pos.zero_()
            self.target_pos.zero_()
            self.left_state = GripperState.OPEN
            self.right_state = GripperState.OPEN
        else:
            self.current_pos[env_ids] = 0.0
            self.target_pos[env_ids] = 0.0
