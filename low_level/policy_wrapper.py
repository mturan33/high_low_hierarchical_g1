"""
Locomotion Policy Wrapper
==========================
Loads a trained RSL-RL checkpoint from unitree_rl_lab and provides
a clean inference interface for the skill layer.

The policy expects:
  - Observation: 480 dims (6 terms × 5 history frames, per-term stacking)
  - Action: 29 joint position targets (scaled by 0.25)

Observation layout (per-term history stacking, matching IsaacLab ObservationManager):
  [0:15]    base_ang_vel × 5 history   (3×5, scaled by 0.2)
  [15:30]   projected_gravity × 5      (3×5)
  [30:45]   velocity_commands × 5      (3×5)
  [45:190]  joint_pos_rel × 5          (29×5)
  [190:335] joint_vel_rel × 5          (29×5, scaled by 0.05)
  [335:480] last_action × 5            (29×5)

IMPORTANT: IsaacLab uses per-term stacking (each term's history is flattened
independently, then all terms are concatenated). This is different from
per-frame stacking where full observation frames are concatenated.
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

# Observation term sizes in order (must match velocity_env_cfg.py PolicyCfg)
OBS_TERM_SIZES = [3, 3, 3, 29, 29, 29]  # ang_vel, gravity, cmd, jpos, jvel, action
OBS_TERM_NAMES = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos_rel", "joint_vel_rel", "last_action",
]


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

        # Per-term history buffers (matching IsaacLab's per-term CircularBuffer)
        # Each term has its own deque of length OBS_HISTORY_LENGTH
        self._term_history: list[deque] = [
            deque(maxlen=OBS_HISTORY_LENGTH) for _ in OBS_TERM_SIZES
        ]

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
        """
        Load trained actor network from RSL-RL checkpoint.

        The checkpoint contains the full ActorCritic state dict. We extract
        only the actor weights and build a standalone nn.Sequential for inference.
        This avoids dependency on the RSL-RL ActorCritic constructor API (which
        changed across versions).

        Architecture (from unitree_rl_lab training config):
          Actor: 480 -> 512 -> ELU -> 256 -> ELU -> 128 -> ELU -> 29
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            full_state_dict = checkpoint["model_state_dict"]
        else:
            full_state_dict = checkpoint

        # Build actor network matching the trained architecture
        # Keys in checkpoint: actor.0.weight (512,480), actor.2.weight (256,512),
        #   actor.4.weight (128,256), actor.6.weight (29,128)
        # This is nn.Sequential: [Linear, ELU, Linear, ELU, Linear, ELU, Linear]
        actor_hidden_dims = [512, 256, 128]
        layers = []
        input_dim = OBS_DIM_TOTAL  # 480
        for hidden_dim in actor_hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, ACTION_DIM))  # output layer

        actor = torch.nn.Sequential(*layers)

        # Load only the actor weights from the full state dict
        actor_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith("actor."):
                actor_state_dict[key[len("actor."):]] = value

        actor.load_state_dict(actor_state_dict)
        actor.to(self.device)
        actor.eval()

        print(f"[LocomotionPolicy] Actor: {OBS_DIM_TOTAL} -> {actor_hidden_dims} -> {ACTION_DIM}")
        print(f"[LocomotionPolicy] Iteration: {checkpoint.get('iter', 'unknown')}")
        return actor

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset observation history and previous actions for given envs."""
        if env_ids is None:
            for buf in self._term_history:
                buf.clear()
            self.prev_action.zero_()
        else:
            # Partial reset -- zero out the prev_action for specified envs
            self.prev_action[env_ids] = 0.0
            # Zero out those env rows in all history frames for each term
            for buf in self._term_history:
                for frame in buf:
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
        Build the full 480-dim observation using per-term history stacking.

        IsaacLab's ObservationManager stacks history PER-TERM (each term's
        CircularBuffer is flattened independently), NOT per-frame. The final
        observation is: [term0_history(size0×5), term1_history(size1×5), ...]

        All inputs should be [num_envs, dim] tensors on self.device.
        """
        n = base_ang_vel.shape[0]

        # Compute current observation terms (order must match velocity_env_cfg.py PolicyCfg)
        current_terms = [
            base_ang_vel * OBS_SCALES["ang_vel"],           # [n, 3]  scaled by 0.2
            projected_gravity,                               # [n, 3]
            velocity_command,                                # [n, 3]
            joint_pos - self.default_joint_pos[:n],          # [n, 29] relative to default
            joint_vel * OBS_SCALES["joint_vel"],             # [n, 29] scaled by 0.05
            self.prev_action[:n].clone(),                    # [n, 29]
        ]

        # Append each term to its own history buffer
        for i, term_val in enumerate(current_terms):
            self._term_history[i].append(term_val.clone())

        # Pad with zeros if not enough history yet
        for i, buf in enumerate(self._term_history):
            while len(buf) < OBS_HISTORY_LENGTH:
                buf.appendleft(torch.zeros(n, OBS_TERM_SIZES[i],
                                           dtype=torch.float32, device=self.device))

        # Per-term stacking: flatten each term's history, then concatenate all terms
        # This matches IsaacLab's: circular_buffer.buffer.reshape(num_envs, -1)
        stacked_terms = []
        for buf in self._term_history:
            # [t-4, t-3, t-2, t-1, t] -> [n, size*5]
            stacked_terms.append(torch.cat(list(buf), dim=-1))

        return torch.cat(stacked_terms, dim=-1)  # [num_envs, 480]

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
        # Build full 480-dim observation with per-term history stacking
        obs_stacked = self._build_observation(
            base_ang_vel, projected_gravity,
            joint_pos, joint_vel, velocity_command,
        )

        # Run actor network directly (standalone nn.Sequential)
        action = self.policy(obs_stacked)

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
