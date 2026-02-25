"""
Hierarchical G1 Environment (DEX3)
====================================
Isaac Lab environment for hierarchical control of the G1 humanoid
with DEX3 3-finger hands (43 DoF = 29 body + 14 fingers).

Scene:
  - G1 robot with DEX3 hands on flat terrain
  - Table (kinematic rigid body, 3.5m ahead)
  - Cup on the table (dynamic rigid body)
  - Dome light

Control pipeline (walking mode):
  velocity_command [vx, vy, vyaw]
      -> LocomotionPolicy (pre-trained, 480->29 body joints)
      -> FingerController (heuristic open/close, 14 finger joints)
          -> Robot PD actuators (43 total)

Control pipeline (manipulation mode):
  arm_targets [14 arm joints] (from IK/heuristic)
      + LocomotionPolicy controls legs+waist only (15 joints)
      + FingerController (14 finger joints)
          -> Robot PD actuators (43 total)

Usage:
    env = HierarchicalG1Env(sim, HierarchicalSceneCfg(), checkpoint_path, num_envs=16)
    obs = env.reset()

    # Walking mode (loco policy controls everything including arms):
    obs = env.step(vel_cmd)

    # Manipulation mode (override arm joints, close fingers):
    env.set_manipulation_mode(True)
    env.finger_controller.close(hand="right")
    obs = env.step_manipulation(vel_cmd, arm_targets)
"""

from __future__ import annotations

import torch
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_DEX3_CFG

from ..config.joint_config import NUM_ALL_JOINTS, NUM_DEX3_JOINTS

# Physics / control constants (must match training config)
PHYSICS_DT = 0.005       # 200 Hz physics
DECIMATION = 4            # 4 physics steps per control step
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50 Hz


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------
@configclass
class HierarchicalSceneCfg(InteractiveSceneCfg):
    """Scene: flat ground + G1-DEX3 robot + table + cup + light."""

    # -- Ground plane with high friction (matching training) --
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # -- G1 Robot with DEX3 hands (43 DoF) --
    robot: ArticulationCfg = UNITREE_G1_29DOF_DEX3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # -- Table: 3.5 m ahead of robot spawn --
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.18),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.5, 0.0, 0.375),
        ),
    )

    # -- Red cup sitting on the table --
    cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.12, 0.12),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.5, 0.0, 0.80),
        ),
    )

    # -- Dome light --
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Environment Wrapper
# ---------------------------------------------------------------------------
class HierarchicalG1Env:
    """
    Wraps an Isaac Lab InteractiveScene with a pre-trained locomotion
    policy and DEX3 finger controller for hierarchical skill-based
    control of the G1 humanoid.

    Two operating modes:
      1. Walking mode: loco policy controls all 29 body joints, fingers idle
      2. Manipulation mode: loco policy controls legs+waist (15),
         arm targets provided externally (14), fingers from controller (14)
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene_cfg: HierarchicalSceneCfg,
        checkpoint_path: str,
        num_envs: int = 16,
        device: str = "cuda:0",
    ):
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.decimation = DECIMATION
        self.physics_dt = PHYSICS_DT
        self.control_dt = CONTROL_DT
        self.step_count = 0

        # Operating mode
        self._manipulation_mode = False

        # -- Create scene --
        scene_cfg.num_envs = num_envs
        scene_cfg.env_spacing = 5.0
        self.scene = InteractiveScene(scene_cfg)

        # -- Get entity handles --
        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.cup: RigidObject = self.scene["cup"]

        # -- Load locomotion policy (29 body joints) --
        from ..low_level.policy_wrapper import LocomotionPolicy

        self.loco_policy = LocomotionPolicy(
            checkpoint_path=checkpoint_path,
            device=device,
            num_envs=num_envs,
        )

        # -- Create finger controller (14 DEX3 joints) --
        from ..low_level.finger_controller import FingerController

        self.finger_controller = FingerController(
            num_envs=num_envs,
            device=device,
        )

        # -- Create arm controller (14 arm joints, pose-based) --
        from ..low_level.arm_controller import ArmController

        self.arm_controller = ArmController(
            num_envs=num_envs,
            device=device,
        )

        # Joint index mappings (set after first reset when articulation is ready)
        self._body_joint_ids: Optional[torch.Tensor] = None
        self._finger_joint_ids: Optional[torch.Tensor] = None
        self._arm_joint_ids: Optional[torch.Tensor] = None
        self._leg_waist_joint_ids: Optional[torch.Tensor] = None
        self._arm_local_indices: Optional[torch.Tensor] = None

        # Placeholder for initial positions (set in reset)
        self._initial_pos: Optional[torch.Tensor] = None
        self._is_reset = False

        print(f"[HierarchicalG1Env] {num_envs} envs, device={device}")
        print(f"[HierarchicalG1Env] Control: {1.0/self.control_dt:.0f} Hz "
              f"({self.decimation}x decimation)")

    def _resolve_joint_indices(self):
        """
        Find body vs finger joint indices in the articulation.

        Isaac Lab's joint ordering comes from the USD, which may differ
        from the SDK ordering. We match by joint name keywords.
        """
        joint_names = self.robot.joint_names
        total_joints = len(joint_names)

        finger_keywords = ["_hand_index_", "_hand_middle_", "_hand_thumb_"]

        body_ids = []
        finger_ids = []
        for i, name in enumerate(joint_names):
            if any(kw in name for kw in finger_keywords):
                finger_ids.append(i)
            else:
                body_ids.append(i)

        self._body_joint_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self._finger_joint_ids = torch.tensor(finger_ids, device=self.device, dtype=torch.long)

        # Within body joints, find arm vs leg+waist
        arm_keywords = ["_shoulder_", "_elbow_", "_wrist_"]
        arm_ids_global = []
        leg_waist_ids_global = []
        arm_local_indices = []
        for local_idx, global_idx in enumerate(body_ids):
            name = joint_names[global_idx]
            if any(kw in name for kw in arm_keywords):
                arm_ids_global.append(global_idx)
                arm_local_indices.append(local_idx)
            else:
                leg_waist_ids_global.append(global_idx)

        self._arm_joint_ids = torch.tensor(arm_ids_global, device=self.device, dtype=torch.long)
        self._leg_waist_joint_ids = torch.tensor(
            leg_waist_ids_global, device=self.device, dtype=torch.long
        )
        self._arm_local_indices = torch.tensor(
            arm_local_indices, device=self.device, dtype=torch.long
        )

        print(f"[HierarchicalG1Env] Joint mapping: "
              f"{len(body_ids)} body ({len(leg_waist_ids_global)} leg+waist, "
              f"{len(arm_ids_global)} arm) + {len(finger_ids)} finger = {total_joints}")
        print(f"[HierarchicalG1Env] Body joint names: {[joint_names[i] for i in body_ids[:5]]}...")
        print(f"[HierarchicalG1Env] Finger joint names: {[joint_names[i] for i in finger_ids[:5]]}...")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def set_manipulation_mode(self, enabled: bool):
        """Toggle between walking mode and manipulation mode."""
        self._manipulation_mode = enabled
        mode_name = "MANIPULATION" if enabled else "WALKING"
        print(f"[HierarchicalG1Env] Mode: {mode_name}")

    def reset(self) -> dict:
        """Reset the environment and return initial observations."""
        # First-time reset: initialize the simulation
        if not self._is_reset:
            self.sim.reset()
            self._is_reset = True

        # Reset all scene entities to their initial states
        indices = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(indices)
        self.table.reset(indices)
        self.cup.reset(indices)

        # Write resets to sim and step once to apply
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.physics_dt)

        # Resolve joint indices (only once)
        if self._body_joint_ids is None:
            self._resolve_joint_indices()

        # Reset policy history and finger controller
        self.loco_policy.reset()
        self.finger_controller.reset()
        self.step_count = 0
        self._manipulation_mode = False

        # CRITICAL: override policy default_joint_pos with the robot's
        # actual default positions (Isaac Lab joint ordering, body joints only)
        default_jpos_all = self.robot.data.default_joint_pos[:self.num_envs].clone()
        default_jpos_body = default_jpos_all[:, self._body_joint_ids]
        self.loco_policy.default_joint_pos = default_jpos_body

        # Store initial XY positions for relative target computation
        self._initial_pos = self.robot.data.root_pos_w[:, :2].clone()

        return self.get_obs()

    def step(self, velocity_command: torch.Tensor) -> dict:
        """
        Step the environment in WALKING mode.

        Loco policy controls all 29 body joints.
        Finger controller provides finger targets independently.

        Args:
            velocity_command: [num_envs, 3] velocity in body frame [vx, vy, vyaw]

        Returns:
            obs_dict with robot proprioception.
        """
        obs = self.get_obs()

        # Locomotion policy: velocity command -> 29 body joint targets
        with torch.inference_mode():
            body_targets = self.loco_policy.get_action(
                base_ang_vel=obs["base_ang_vel"],
                projected_gravity=obs["projected_gravity"],
                joint_pos=obs["joint_pos_body"],
                joint_vel=obs["joint_vel_body"],
                velocity_command=velocity_command,
            )

        # Finger targets from controller
        finger_targets = self.finger_controller.get_targets()

        # Combine into full 43-DoF targets
        self._apply_joint_targets(body_targets, finger_targets)

        # Physics sub-stepping with decimation
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()

        # Update scene data
        self.scene.update(self.control_dt)

        self.step_count += 1
        return self.get_obs()

    def step_manipulation(
        self,
        velocity_command: torch.Tensor,
        arm_targets: torch.Tensor,
    ) -> dict:
        """
        Step the environment in MANIPULATION mode.

        Loco policy controls legs+waist (15 joints).
        Arm targets provided externally (14 joints).
        Finger controller provides finger targets (14 joints).

        Args:
            velocity_command: [num_envs, 3] velocity for leg balance
            arm_targets: [num_envs, 14] absolute arm joint positions

        Returns:
            obs_dict with robot proprioception.
        """
        obs = self.get_obs()

        # Locomotion policy: get full 29 body targets (we'll override arms)
        with torch.inference_mode():
            body_targets = self.loco_policy.get_action(
                base_ang_vel=obs["base_ang_vel"],
                projected_gravity=obs["projected_gravity"],
                joint_pos=obs["joint_pos_body"],
                joint_vel=obs["joint_vel_body"],
                velocity_command=velocity_command,
            )

        # Override arm joints with external targets
        body_targets[:, self._arm_local_indices] = arm_targets

        # Finger targets from controller
        finger_targets = self.finger_controller.get_targets()

        # Combine into full 43-DoF targets
        self._apply_joint_targets(body_targets, finger_targets)

        # Physics sub-stepping with decimation
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()

        # Update scene data
        self.scene.update(self.control_dt)

        self.step_count += 1
        return self.get_obs()

    def _apply_joint_targets(
        self,
        body_targets: torch.Tensor,
        finger_targets: torch.Tensor,
    ):
        """
        Apply body + finger joint position targets to the 43-DoF robot.

        Args:
            body_targets: [num_envs, 29] body joint positions
            finger_targets: [num_envs, 14] finger joint positions
        """
        n = body_targets.shape[0]
        total_joints = len(self._body_joint_ids) + len(self._finger_joint_ids)

        full_targets = torch.zeros(
            n, total_joints, dtype=torch.float32, device=self.device
        )

        # Place body targets at correct indices
        full_targets[:, self._body_joint_ids] = body_targets
        # Place finger targets at correct indices
        full_targets[:, self._finger_joint_ids] = finger_targets

        self.robot.set_joint_position_target(full_targets)

    def get_obs(self) -> dict:
        """
        Get current robot observations.

        Returns dict compatible with the skill interface:
            root_pos          : [N, 3]   world position
            root_quat         : [N, 4]   orientation (w, x, y, z)
            base_ang_vel      : [N, 3]   angular velocity in body frame
            projected_gravity : [N, 3]   gravity vector in body frame
            joint_pos_body    : [N, 29]  body joint positions (Isaac Lab order)
            joint_vel_body    : [N, 29]  body joint velocities (Isaac Lab order)
            joint_pos_finger  : [N, 14]  finger joint positions
            joint_vel_finger  : [N, 14]  finger joint velocities
            joint_pos         : [N, 29]  alias for joint_pos_body (backward compat)
            joint_vel         : [N, 29]  alias for joint_vel_body (backward compat)
            base_height       : [N]      robot base height
        """
        all_pos = self.robot.data.joint_pos
        all_vel = self.robot.data.joint_vel

        body_pos = all_pos[:, self._body_joint_ids]
        body_vel = all_vel[:, self._body_joint_ids]
        finger_pos = all_pos[:, self._finger_joint_ids]
        finger_vel = all_vel[:, self._finger_joint_ids]

        return {
            "root_pos": self.robot.data.root_pos_w,
            "root_quat": self.robot.data.root_quat_w,
            "base_ang_vel": self.robot.data.root_ang_vel_b,
            "projected_gravity": self.robot.data.projected_gravity_b,
            "joint_pos_body": body_pos,
            "joint_vel_body": body_vel,
            "joint_pos_finger": finger_pos,
            "joint_vel_finger": finger_vel,
            # Backward compatibility (skills expect "joint_pos" / "joint_vel")
            "joint_pos": body_pos,
            "joint_vel": body_vel,
            "base_height": self.robot.data.root_pos_w[:, 2],
        }

    @property
    def initial_positions(self) -> torch.Tensor:
        """Initial XY positions of all robots [num_envs, 2]."""
        if self._initial_pos is None:
            raise RuntimeError("Call reset() first")
        return self._initial_pos

    def close(self):
        """Clean up resources."""
        pass
