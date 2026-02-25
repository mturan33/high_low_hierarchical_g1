"""
Hierarchical G1 Environment
============================
Isaac Lab environment for hierarchical control of the G1 humanoid.

Scene:
  - G1 robot on flat terrain (29-DoF, matching unitree_rl_lab config)
  - Table (kinematic rigid body, 3.5m ahead)
  - Cup on the table (dynamic rigid body)
  - Dome light

Control pipeline:
  velocity_command [vx, vy, vyaw]
      -> LocomotionPolicy (pre-trained, 480->29)
          -> joint position targets
              -> Robot PD actuators

Usage:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    env = HierarchicalG1Env(sim, HierarchicalSceneCfg(), checkpoint_path, num_envs=16)
    obs = env.reset()

    for _ in range(1000):
        vel_cmd = torch.tensor([[0.5, 0.0, 0.0]]).expand(16, -1)
        obs = env.step(vel_cmd)
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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

# Physics / control constants (must match training config)
PHYSICS_DT = 0.005       # 200 Hz physics
DECIMATION = 4            # 4 physics steps per control step
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50 Hz


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------
@configclass
class HierarchicalSceneCfg(InteractiveSceneCfg):
    """Scene: flat ground + G1 robot + table + cup + light."""

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

    # -- G1 Robot (same config as velocity training) --
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
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
    policy for hierarchical skill-based control of the G1 humanoid.

    The environment takes velocity commands [vx, vy, vyaw] and internally
    converts them to 29-DoF joint position targets via the trained policy.
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

        # -- Create scene --
        scene_cfg.num_envs = num_envs
        scene_cfg.env_spacing = 5.0
        self.scene = InteractiveScene(scene_cfg)

        # -- Get entity handles --
        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.cup: RigidObject = self.scene["cup"]

        # -- Load locomotion policy --
        from ..low_level.policy_wrapper import LocomotionPolicy

        self.loco_policy = LocomotionPolicy(
            checkpoint_path=checkpoint_path,
            device=device,
            num_envs=num_envs,
        )

        # Placeholder for initial positions (set in reset)
        self._initial_pos: Optional[torch.Tensor] = None
        self._is_reset = False

        print(f"[HierarchicalG1Env] {num_envs} envs, device={device}")
        print(f"[HierarchicalG1Env] Control: {1.0/self.control_dt:.0f} Hz "
              f"({self.decimation}x decimation)")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
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

        # Reset policy history
        self.loco_policy.reset()
        self.step_count = 0

        # CRITICAL: override policy default_joint_pos with the robot's
        # actual default positions (Isaac Lab joint ordering, not SDK ordering)
        default_jpos = self.robot.data.default_joint_pos[:self.num_envs].clone()
        self.loco_policy.default_joint_pos = default_jpos

        # Store initial XY positions for relative target computation
        self._initial_pos = self.robot.data.root_pos_w[:, :2].clone()

        return self.get_obs()

    def step(self, velocity_command: torch.Tensor) -> dict:
        """
        Step the environment with a velocity command.

        Runs DECIMATION physics sub-steps (50 Hz control loop).

        Args:
            velocity_command: [num_envs, 3] velocity in body frame [vx, vy, vyaw]

        Returns:
            obs_dict with robot proprioception.
        """
        obs = self.get_obs()

        # Locomotion policy: velocity command -> joint position targets
        with torch.inference_mode():
            joint_targets = self.loco_policy.get_action(
                base_ang_vel=obs["base_ang_vel"],
                projected_gravity=obs["projected_gravity"],
                joint_pos=obs["joint_pos"],
                joint_vel=obs["joint_vel"],
                velocity_command=velocity_command,
            )

        # Apply to robot
        self.robot.set_joint_position_target(joint_targets)

        # Physics sub-stepping with decimation
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()

        # Update scene data
        self.scene.update(self.control_dt)

        self.step_count += 1
        return self.get_obs()

    def get_obs(self) -> dict:
        """
        Get current robot observations.

        Returns dict compatible with the skill interface:
            root_pos          : [N, 3]  world position
            root_quat         : [N, 4]  orientation (w, x, y, z)
            base_ang_vel      : [N, 3]  angular velocity in body frame
            projected_gravity : [N, 3]  gravity vector in body frame
            joint_pos         : [N, 29] joint positions (Isaac Lab order)
            joint_vel         : [N, 29] joint velocities (Isaac Lab order)
            base_height       : [N]     robot base height
        """
        return {
            "root_pos": self.robot.data.root_pos_w,
            "root_quat": self.robot.data.root_quat_w,
            "base_ang_vel": self.robot.data.root_ang_vel_b,
            "projected_gravity": self.robot.data.projected_gravity_b,
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
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
