"""
Locomotion Policy Wrapper
==========================
Loads a trained RSL-RL checkpoint from unitree_rl_lab and provides
a clean inference interface for the skill layer.

The policy expects:
  - Observation: 96 dims per frame × 5 history frames = 480 dims
  - Action: 29 joint position targets (scaled by 0.25)

Observation vector per frame (96 dims):
  [0:3]   base_ang_vel (scaled by 0.2)
  [3:6]   projected_gravity
  [6:9]   velocity_commands [vx, vy, vyaw]
  [9:38]  joint_pos_rel (current - default)
  [38:67] joint_vel_rel (scaled by 0.05)
  [67:96] last_action
"""

from __future__ import annotations

import os
import torch
import numpy as np
from collections import deque
from typing import Optional

from config.joint_config import (
    NUM_ALL_JOINTS,
    OBS_DIM_PER_FRAME,
    OBS_HISTORY_LENGTH,
    OBS_DIM_TOTAL,
    ACTION_DIM,
    ACTION_SCALE,
    DEFAULT_JOINT_LIST,
    OBS_SCALES,
)


class LocomotionPolicy:
    """
    Wraps a trained RSL-RL ActorCritic policy for inference.

    Usage:
        policy = LocomotionPolicy("path/to/model_1500.pt")

        # Each step:
        action = policy.get_action(
            base_ang_vel=robot.ang_vel,           # [3]
            projected_gravity=robot.proj_gravity,  # [3]
            joint_pos=robot.joint_pos,             # [29]
            joint_vel=robot.joint_vel,             # [29]
            velocity_command=[0.5, 0.0, 0.0],     # [vx, vy, vyaw]
        )
        # action shape: [29] — joint position targets
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_envs: int = 1,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs

        # Default joint positions as tensor
        self.default_joint_pos = torch.tensor(
            DEFAULT_JOINT_LIST, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(num_envs, -1)

        # Observation history buffer (deque of last N frames)
        self.obs_history: deque[torch.Tensor] = deque(maxlen=OBS_HISTORY_LENGTH)

        # Previous action buffer
        self.prev_action = torch.zeros(
            num_envs, ACTION_DIM, dtype=torch.float32, device=self.device
        )

        # Load model
        self.policy = self._load_policy(checkpoint_path)

        print(f"[LocomotionPolicy] Loaded from: {checkpoint_path}")
        print(f"[LocomotionPolicy] Device: {self.device}, Envs: {num_envs}")
        print(f"[LocomotionPolicy] Obs: {OBS_DIM_PER_FRAME}×{OBS_HISTORY_LENGTH}={OBS_DIM_TOTAL}, Act: {ACTION_DIM}")

    def _load_policy(self, checkpoint_path: str) -> torch.nn.Module:
        """Load RSL-RL ActorCritic policy from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # RSL-RL saves the full model state dict
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Try loading as JIT model first (exported policy)
        if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".jit"):
            try:
                model = torch.jit.load(checkpoint_path, map_location=self.device)
                model.eval()
                print("[LocomotionPolicy] Loaded as TorchScript model")
                return model
            except Exception:
                pass

        # Load as RSL-RL ActorCritic state dict
        try:
            from rsl_rl.modules import ActorCritic

            # Reconstruct network architecture (must match training config)
            # unitree_rl_lab uses: actor=[512,256,128], critic=[512,256,128], ELU
            model = ActorCritic(
                num_actor_obs=OBS_DIM_TOTAL,
                num_critic_obs=OBS_DIM_TOTAL,
                num_actions=ACTION_DIM,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation="elu",
                init_noise_std=1.0,
            )

            # Load weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            print("[LocomotionPolicy] Loaded as RSL-RL ActorCritic")
            return model

        except ImportError:
            raise ImportError(
                "rsl_rl not found. Install with: pip install rsl_rl"
            )

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset observation history and previous actions for given envs."""
        if env_ids is None:
            self.obs_history.clear()
            self.prev_action.zero_()
        else:
            # Partial reset — zero out the prev_action for specified envs
            self.prev_action[env_ids] = 0.0
            # For history, we can't partially reset a deque easily,
            # so we zero out those env rows in all history frames
            for i, frame in enumerate(self.obs_history):
                frame[env_ids] = 0.0

    def _build_observation(
        self,
        base_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build single-frame observation vector (96 dims).

        All inputs should be [num_envs, dim] tensors on self.device.
        """
        # Apply observation scales
        ang_vel_scaled = base_ang_vel * OBS_SCALES["ang_vel"]
        joint_vel_scaled = joint_vel * OBS_SCALES["joint_vel"]

        # Joint positions relative to default
        joint_pos_rel = joint_pos - self.default_joint_pos[:base_ang_vel.shape[0]]

        # Concatenate: [ang_vel(3), gravity(3), cmd(3), jpos(29), jvel(29), prev_act(29)] = 96
        obs_frame = torch.cat([
            ang_vel_scaled,       # 3
            projected_gravity,    # 3
            velocity_command,     # 3
            joint_pos_rel,        # 29
            joint_vel_scaled,     # 29
            self.prev_action[:base_ang_vel.shape[0]],  # 29
        ], dim=-1)

        return obs_frame  # [num_envs, 96]

    def _stack_history(self, current_frame: torch.Tensor) -> torch.Tensor:
        """Stack observation history (5 frames) into flat vector."""
        self.obs_history.append(current_frame.clone())

        # Pad with zeros if not enough history yet
        while len(self.obs_history) < OBS_HISTORY_LENGTH:
            self.obs_history.appendleft(
                torch.zeros_like(current_frame)
            )

        # Stack: oldest first → newest last
        stacked = torch.cat(list(self.obs_history), dim=-1)
        return stacked  # [num_envs, 480]

    @torch.no_grad()
    def get_action(
        self,
        base_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        velocity_command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run policy inference.

        Args:
            base_ang_vel: Angular velocity in body frame [num_envs, 3]
            projected_gravity: Gravity vector in body frame [num_envs, 3]
            joint_pos: Current joint positions [num_envs, 29]
            joint_vel: Current joint velocities [num_envs, 29]
            velocity_command: Target velocity [num_envs, 3] = [vx, vy, vyaw]

        Returns:
            joint_targets: Joint position targets [num_envs, 29]
                          (absolute positions, not relative to default)
        """
        # Build current frame observation
        obs_frame = self._build_observation(
            base_ang_vel, projected_gravity,
            joint_pos, joint_vel, velocity_command,
        )

        # Stack with history
        obs_stacked = self._stack_history(obs_frame)

        # Run policy
        action = self.policy.act_inference(obs_stacked)

        # Store for next frame's observation
        self.prev_action[:action.shape[0]] = action.clone()

        # Convert to absolute joint position targets
        # action is in [-1, 1] range, scale by ACTION_SCALE and add default
        joint_targets = (
            self.default_joint_pos[:action.shape[0]]
            + action * ACTION_SCALE
        )

        return joint_targets

    def get_action_from_numpy(
        self,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        velocity_command: np.ndarray,
    ) -> np.ndarray:
        """Numpy convenience wrapper for get_action."""
        def to_tensor(x):
            t = torch.from_numpy(np.asarray(x, dtype=np.float32))
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(self.device)

        joint_targets = self.get_action(
            to_tensor(base_ang_vel),
            to_tensor(projected_gravity),
            to_tensor(joint_pos),
            to_tensor(joint_vel),
            to_tensor(velocity_command),
        )

        return joint_targets.cpu().numpy().squeeze()
