"""
Velocity Command Generator
============================
Converts high-level targets (position, heading) into velocity commands
[vx, vy, vyaw] that the locomotion policy understands.

Uses P-controllers with configurable gains and velocity limits.
"""

from __future__ import annotations

import torch
import math
from typing import Optional

from ..config.joint_config import CMD_RANGE_LIMIT


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def get_yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw angle from quaternion [w, x, y, z] or [x, y, z, w].

    Args:
        quat: Quaternion tensor [num_envs, 4]

    Returns:
        yaw: Yaw angle [num_envs]
    """
    # Isaac Lab uses [w, x, y, z] convention
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


class VelocityCommandGenerator:
    """
    Generates velocity commands to navigate to a target position.

    Takes robot state (position, orientation) and target position,
    outputs [vx, vy, vyaw] velocity commands for the locomotion policy.
    """

    def __init__(
        self,
        kp_linear: float = 1.0,
        kp_angular: float = 0.8,
        max_lin_vel_x: float = 0.8,
        max_lin_vel_y: float = 0.25,
        max_ang_vel_z: float = 0.2,
        min_lin_vel: float = 0.05,
        heading_deadzone: float = 0.05,
        device: str = "cuda",
    ):
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_lin_vel_x = max_lin_vel_x
        self.max_lin_vel_y = max_lin_vel_y
        self.max_ang_vel_z = max_ang_vel_z
        self.min_lin_vel = min_lin_vel
        self.heading_deadzone = heading_deadzone
        self.device = torch.device(device)

    def compute_walk_command(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity command to walk toward target position.

        Args:
            robot_pos: Robot XY position [num_envs, 2]
            robot_yaw: Robot heading angle [num_envs]
            target_pos: Target XY position [num_envs, 2]

        Returns:
            cmd_vel: Velocity command [num_envs, 3] = [vx, vy, vyaw]
            distance: Distance to target [num_envs]
        """
        # Vector from robot to target in world frame
        delta_world = target_pos - robot_pos  # [num_envs, 2]
        distance = torch.norm(delta_world, dim=-1)  # [num_envs]

        # Desired heading (world frame)
        target_heading = torch.atan2(delta_world[:, 1], delta_world[:, 0])

        # Heading error
        heading_error = normalize_angle(target_heading - robot_yaw)

        # Transform delta to body frame for forward/lateral velocity
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)
        dx_body = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        dy_body = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]

        # P-controller for linear velocity (in body frame)
        vx = torch.clamp(
            self.kp_linear * dx_body,
            -self.max_lin_vel_x,
            self.max_lin_vel_x,
        )
        vy = torch.clamp(
            self.kp_linear * dy_body,
            -self.max_lin_vel_y,
            self.max_lin_vel_y,
        )

        # P-controller for angular velocity
        vyaw = torch.clamp(
            self.kp_angular * heading_error,
            -self.max_ang_vel_z,
            self.max_ang_vel_z,
        )

        # Apply heading deadzone
        vyaw = torch.where(
            torch.abs(heading_error) < self.heading_deadzone,
            torch.zeros_like(vyaw),
            vyaw,
        )

        cmd_vel = torch.stack([vx, vy, vyaw], dim=-1)  # [num_envs, 3]
        return cmd_vel, distance

    def compute_turn_command(
        self,
        robot_yaw: torch.Tensor,
        target_heading: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity command to turn in place to target heading.

        Args:
            robot_yaw: Current heading [num_envs]
            target_heading: Desired heading [num_envs]

        Returns:
            cmd_vel: Velocity command [num_envs, 3] = [0, 0, vyaw]
            heading_error: Remaining error [num_envs]
        """
        heading_error = normalize_angle(target_heading - robot_yaw)

        vyaw = torch.clamp(
            self.kp_angular * heading_error,
            -self.max_ang_vel_z,
            self.max_ang_vel_z,
        )

        cmd_vel = torch.zeros(robot_yaw.shape[0], 3, device=self.device)
        cmd_vel[:, 2] = vyaw

        return cmd_vel, heading_error

    def compute_stand_command(self, num_envs: int = 1) -> torch.Tensor:
        """Return zero velocity command (stand still)."""
        return torch.zeros(num_envs, 3, device=self.device)
