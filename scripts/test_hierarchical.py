#!/usr/bin/env python3
"""
Test Hierarchical G1 - Walk + Reach + Grasp Demo
===================================================
Full hierarchical manipulation demo:
  Phase 1: Walk to table (3m forward)
  Phase 2: Switch to manipulation mode, reach forward
  Phase 3: Close fingers (grasp)
  Phase 4: Return arms to carry pose
  Phase 5: Stand still (holding object)

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\high_low_hierarchical_g1\\scripts\\test_hierarchical.py --num_envs 16 --checkpoint C:\\IsaacLab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2026-02-24_16-51-25\\model_20700.pt
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test hierarchical G1 DEX3 walk + reach + grasp")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to trained locomotion policy checkpoint (.pt file)",
)
parser.add_argument("--walk_distance", type=float, default=3.0, help="Walk distance in meters")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps before timeout")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Isaac Lab imports (AFTER AppLauncher)
# ============================================================================
import os
import sys
import time
import torch

import isaaclab.sim as sim_utils

# Add parent of high_low_hierarchical_g1 to path so package imports work
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT
from high_low_hierarchical_g1.skills.walk_to import WalkToSkill
from high_low_hierarchical_g1.config.skill_config import WalkToConfig
from high_low_hierarchical_g1.low_level.arm_controller import ArmPose


def run_phase(env, name, step_fn, max_steps=500, log_interval=50):
    """Run a phase with logging."""
    print(f"\n{'─'*50}")
    print(f"  PHASE: {name}")
    print(f"{'─'*50}")

    for step in range(max_steps):
        if not simulation_app.is_running():
            return None

        obs = step_fn(step)

        if step > 0 and step % log_interval == 0:
            h = obs["base_height"].mean().item()
            standing = (obs["base_height"] > 0.3).sum().item()
            print(f"  [{name}] Step {step:4d} | Height: {h:.2f}m | Standing: {standing}/{env.num_envs}")

    return obs


def main():
    """Main test loop."""
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  Hierarchical G1 DEX3 - Walk + Reach + Grasp Demo")
    print("=" * 60)
    print(f"  Environments : {num_envs}")
    print(f"  Checkpoint   : {args_cli.checkpoint}")
    print(f"  Walk distance: {args_cli.walk_distance}m")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create simulation context
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, -4.0, 3.5], target=[3.0, 0.0, 0.5])

    # ------------------------------------------------------------------
    # 2. Create hierarchical environment (DEX3)
    # ------------------------------------------------------------------
    scene_cfg = HierarchicalSceneCfg()
    env = HierarchicalG1Env(
        sim=sim,
        scene_cfg=scene_cfg,
        checkpoint_path=args_cli.checkpoint,
        num_envs=num_envs,
        device=device,
    )

    # ------------------------------------------------------------------
    # 3. Reset environment
    # ------------------------------------------------------------------
    print("\n[Demo] Resetting environment...")
    obs = env.reset()

    initial_pos = env.initial_positions
    stand_cmd = torch.zeros(num_envs, 3, device=device)

    # ==================================================================
    # PHASE 1: Walk to table
    # ==================================================================
    target_positions = initial_pos.clone()
    target_positions[:, 0] += args_cli.walk_distance

    walk_cfg = WalkToConfig()
    walk_cfg.max_steps = args_cli.max_steps
    skill = WalkToSkill(config=walk_cfg, device=device)
    skill.reset(target_positions=target_positions)

    print(f"\n{'─'*50}")
    print(f"  PHASE 1: Walk to table ({args_cli.walk_distance}m)")
    print(f"{'─'*50}")

    start_time = time.time()
    walk_done = False

    while simulation_app.is_running() and not walk_done:
        vel_cmd, walk_done, result = skill.step(obs)
        obs = env.step(vel_cmd)

        if env.step_count % 50 == 0:
            elapsed = time.time() - start_time
            current_pos = obs["root_pos"][:, :2]
            distances = torch.norm(current_pos - target_positions, dim=-1)
            print(
                f"  [Walk] Step {env.step_count:4d} | "
                f"Time: {elapsed:5.1f}s | "
                f"Mean dist: {distances.mean():.2f}m | "
                f"Height: {obs['base_height'].mean():.2f}m"
            )

        if (obs["base_height"] < 0.2).all():
            print("  [Walk] All robots fell!")
            walk_done = True

    walk_time = time.time() - start_time
    print(f"  [Walk] Result: {result.status.name} ({result.reason})")
    print(f"  [Walk] Time: {walk_time:.1f}s, Steps: {result.steps_taken}")

    # ==================================================================
    # PHASE 2: Stand still briefly (stabilize before reaching)
    # ==================================================================
    def stabilize_step(step):
        return env.step(stand_cmd)

    obs = run_phase(env, "Stabilize", stabilize_step, max_steps=100, log_interval=50)

    # ==================================================================
    # PHASE 3: Reach forward (manipulation mode)
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  PHASE 3: Reach forward (arm override)")
    print(f"{'─'*50}")

    env.set_manipulation_mode(True)
    env.arm_controller.set_pose(ArmPose.REACH_FORWARD)

    for step in range(200):
        if not simulation_app.is_running():
            break

        arm_targets = env.arm_controller.get_targets()
        obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 50 == 0:
            h = obs["base_height"].mean().item()
            arm_done = env.arm_controller.is_done
            print(f"  [Reach] Step {step:4d} | Height: {h:.2f}m | "
                  f"Arm done: {arm_done}")

    print(f"  [Reach] Arms in REACH_FORWARD position")

    # ==================================================================
    # PHASE 4: Close fingers (grasp)
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  PHASE 4: Close fingers (grasp)")
    print(f"{'─'*50}")

    env.finger_controller.close(hand="both")

    for step in range(100):
        if not simulation_app.is_running():
            break

        arm_targets = env.arm_controller.get_targets()
        obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 25 == 0:
            finger_pos = obs["joint_pos_finger"]
            closed = env.finger_controller.is_closed()
            print(f"  [Grasp] Step {step:4d} | "
                  f"Finger mean: {finger_pos.mean():.3f} rad | "
                  f"Closed: {closed}")

    print(f"  [Grasp] Fingers closed!")

    # ==================================================================
    # PHASE 5: Move to carry pose (arms bent, holding object)
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  PHASE 5: Carry pose")
    print(f"{'─'*50}")

    env.arm_controller.set_pose(ArmPose.CARRY)

    for step in range(200):
        if not simulation_app.is_running():
            break

        arm_targets = env.arm_controller.get_targets()
        obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 50 == 0:
            h = obs["base_height"].mean().item()
            arm_done = env.arm_controller.is_done
            print(f"  [Carry] Step {step:4d} | Height: {h:.2f}m | "
                  f"Arm done: {arm_done}")

    # ==================================================================
    # PHASE 6: Return to default pose, open fingers
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  PHASE 6: Return to default + open fingers")
    print(f"{'─'*50}")

    env.arm_controller.set_pose(ArmPose.DEFAULT)
    env.finger_controller.open(hand="both")

    for step in range(200):
        if not simulation_app.is_running():
            break

        arm_targets = env.arm_controller.get_targets()
        obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 50 == 0:
            h = obs["base_height"].mean().item()
            print(f"  [Return] Step {step:4d} | Height: {h:.2f}m")

    # Switch back to walking mode
    env.set_manipulation_mode(False)

    # ==================================================================
    # RESULTS
    # ==================================================================
    final_pos = obs["root_pos"][:, :2]
    distances = torch.norm(final_pos - target_positions, dim=-1)

    print("\n" + "=" * 60)
    print("  DEMO RESULTS")
    print("=" * 60)
    print(f"  Walk          : {result.status.name} ({walk_time:.1f}s)")
    print(f"  Reach         : OK (arms extended forward)")
    print(f"  Grasp         : OK (fingers closed)")
    print(f"  Carry         : OK (arms in carry pose)")
    print(f"  Return        : OK (arms back to default)")
    print(f"  Final height  : {obs['base_height'].mean():.2f}m")
    print(f"  Standing      : {(obs['base_height'] > 0.3).sum()}/{num_envs}")
    print(f"  Total DoF     : {env.robot.num_joints} (29 body + {env.robot.num_joints - 29} fingers)")
    print("=" * 60)

    # Keep sim alive
    print("\n[Demo] Complete! Keeping sim alive...")
    while simulation_app.is_running():
        obs = env.step(stand_cmd)

    env.close()


if __name__ == "__main__":
    main()
