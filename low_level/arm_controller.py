"""
Arm Controller (Pose-Based + Simple IK)
=========================================
Controls the 14 arm joints (7 per arm) of the G1 29-DoF robot
for manipulation tasks.

Strategy:
  - Uses predefined arm poses for common actions (reach forward, reach down, etc.)
  - Smooth interpolation between poses
  - No external IK library dependency (pure PyTorch, Windows compatible)

Arm joints per side (7):
  shoulder_pitch, shoulder_roll, shoulder_yaw,
  elbow,
  wrist_roll, wrist_pitch, wrist_yaw

The controller outputs 14 joint position targets that override
the locomotion policy's arm outputs during manipulation mode.
"""

from __future__ import annotations

import torch
import math
from enum import Enum
from typing import Optional


class ArmPose(Enum):
    """Predefined arm poses."""
    DEFAULT = "default"           # Arms at sides (loco policy default)
    REACH_FORWARD = "reach_fwd"   # Both arms reach forward
    REACH_DOWN = "reach_down"     # Both arms reach down (table height)
    REACH_TABLE = "reach_table"   # Arms reach to table surface level
    LEFT_REACH = "left_reach"     # Only left arm reaches
    RIGHT_REACH = "right_reach"   # Only right arm reaches
    CARRY = "carry"               # Arms in carrying position (elbows bent, hands forward)


# ============================================================================
# Predefined joint poses [7 values per arm]:
#   shoulder_pitch, shoulder_roll, shoulder_yaw,
#   elbow,
#   wrist_roll, wrist_pitch, wrist_yaw
#
# These are in ISAAC LAB joint ordering (resolved at runtime).
# Values are absolute positions in radians.
# ============================================================================

# Default standing pose (from UNITREE_G1_29DOF_CFG init_state)
_DEFAULT_LEFT = [0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0]
_DEFAULT_RIGHT = [0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0]

# Reach forward: arms straight ahead at chest height, palms down
_REACH_FWD_LEFT = [-0.8, 0.15, 0.0, 0.3, 0.0, -0.5, 0.0]
_REACH_FWD_RIGHT = [-0.8, -0.15, 0.0, 0.3, 0.0, -0.5, 0.0]

# Reach down: arms reaching to table level (~0.75m high)
_REACH_DOWN_LEFT = [-1.0, 0.1, 0.0, 0.6, 0.0, -0.8, 0.0]
_REACH_DOWN_RIGHT = [-1.0, -0.1, 0.0, 0.6, 0.0, -0.8, 0.0]

# Reach table: arms extended to table surface, palms facing down
_REACH_TABLE_LEFT = [-0.9, 0.1, 0.0, 0.4, 0.0, -1.2, 0.0]
_REACH_TABLE_RIGHT = [-0.9, -0.1, 0.0, 0.4, 0.0, -1.2, 0.0]

# Carry: elbows bent, hands in front of torso
_CARRY_LEFT = [-0.4, 0.2, 0.0, 1.4, 0.0, -0.3, 0.0]
_CARRY_RIGHT = [-0.4, -0.2, 0.0, 1.4, 0.0, -0.3, 0.0]

# Pose lookup table: {pose_name: (left_7, right_7)}
_POSE_TABLE = {
    ArmPose.DEFAULT: (_DEFAULT_LEFT, _DEFAULT_RIGHT),
    ArmPose.REACH_FORWARD: (_REACH_FWD_LEFT, _REACH_FWD_RIGHT),
    ArmPose.REACH_DOWN: (_REACH_DOWN_LEFT, _REACH_DOWN_RIGHT),
    ArmPose.REACH_TABLE: (_REACH_TABLE_LEFT, _REACH_TABLE_RIGHT),
    ArmPose.CARRY: (_CARRY_LEFT, _CARRY_RIGHT),
    # Single-arm poses use default for the other arm
    ArmPose.LEFT_REACH: (_REACH_FWD_LEFT, _DEFAULT_RIGHT),
    ArmPose.RIGHT_REACH: (_DEFAULT_LEFT, _REACH_FWD_RIGHT),
}


class ArmController:
    """
    Pose-based arm controller with smooth interpolation.

    Outputs 14 joint position targets for the arm joints
    (in the same ordering as the Isaac Lab articulation's arm joints).

    Usage:
        controller = ArmController(num_envs=16, device="cuda:0")

        # Set target pose
        controller.set_pose(ArmPose.REACH_FORWARD)

        # Each step, get interpolated targets
        arm_targets = controller.get_targets()  # [16, 14]
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cuda:0",
        interp_speed: float = 0.03,  # radians per control step (smooth)
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.interp_speed = interp_speed

        # Current arm positions [num_envs, 14] (left_7 + right_7)
        default_pose = torch.tensor(
            _DEFAULT_LEFT + _DEFAULT_RIGHT,
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).expand(num_envs, -1).clone()

        self.current_pos = default_pose.clone()
        self.target_pos = default_pose.clone()

        self._current_pose = ArmPose.DEFAULT
        self._transition_done = True

        print(f"[ArmController] 14 arm joints, interp_speed={interp_speed} rad/step")

    def set_pose(
        self,
        pose: ArmPose,
        env_ids: Optional[torch.Tensor] = None,
    ):
        """
        Set target arm pose for all or specific envs.

        Args:
            pose: Target predefined pose
            env_ids: specific env indices (None = all)
        """
        if pose not in _POSE_TABLE:
            raise ValueError(f"Unknown pose: {pose}")

        left, right = _POSE_TABLE[pose]
        target = torch.tensor(
            left + right,
            dtype=torch.float32, device=self.device,
        )

        ids = slice(None) if env_ids is None else env_ids
        self.target_pos[ids] = target
        self._current_pose = pose
        self._transition_done = False

        print(f"[ArmController] Target pose: {pose.value}")

    def set_custom_targets(
        self,
        arm_targets: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ):
        """
        Set custom arm joint targets directly.

        Args:
            arm_targets: [num_envs, 14] or [14] absolute joint positions
            env_ids: specific env indices (None = all)
        """
        ids = slice(None) if env_ids is None else env_ids
        if arm_targets.ndim == 1:
            arm_targets = arm_targets.unsqueeze(0)
        self.target_pos[ids] = arm_targets
        self._transition_done = False

    def get_targets(self) -> torch.Tensor:
        """
        Get current arm joint targets with smooth interpolation.

        Returns:
            arm_targets: [num_envs, 14] joint position targets
        """
        diff = self.target_pos - self.current_pos
        step = torch.clamp(diff, -self.interp_speed, self.interp_speed)
        self.current_pos = self.current_pos + step

        # Check if transition is done
        max_diff = torch.abs(diff).max().item()
        if max_diff < 0.01:
            self._transition_done = True

        return self.current_pos.clone()

    @property
    def is_done(self) -> bool:
        """Check if arm has reached target pose."""
        return self._transition_done

    @property
    def current_pose(self) -> ArmPose:
        return self._current_pose

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset arms to default pose."""
        default = torch.tensor(
            _DEFAULT_LEFT + _DEFAULT_RIGHT,
            dtype=torch.float32, device=self.device,
        )
        if env_ids is None:
            self.current_pos[:] = default
            self.target_pos[:] = default
        else:
            self.current_pos[env_ids] = default
            self.target_pos[env_ids] = default
        self._current_pose = ArmPose.DEFAULT
        self._transition_done = True
