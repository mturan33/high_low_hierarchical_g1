#!/usr/bin/env python3
"""
Test Hierarchical G1 Environment - Walk To Skill
===================================================
Spawns 16 G1 robots on flat terrain with tables and cups.
Each robot walks 3 meters forward using the trained locomotion policy.

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\high_low_hierarchical_g1\\scripts\\test_hierarchical.py --num_envs 16 --checkpoint C:\\IsaacLab\\logs\\rsl_rl\\unitree_g1_29dof_velocity\\2026-02-24_16-51-25\\model_20700.pt
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test hierarchical G1 walk_to skill")
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
# (avoids collision with OpenCV's config module)
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # high_low_hierarchical_g1/
_PKG_PARENT = os.path.dirname(_PKG_DIR)  # direct/
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT
from high_low_hierarchical_g1.skills.walk_to import WalkToSkill
from high_low_hierarchical_g1.config.skill_config import WalkToConfig


def main():
    """Main test loop."""
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  Hierarchical G1 - Walk To Skill Test")
    print("=" * 60)
    print(f"  Environments : {num_envs}")
    print(f"  Checkpoint   : {args_cli.checkpoint}")
    print(f"  Walk distance: {args_cli.walk_distance}m")
    print(f"  Max steps    : {args_cli.max_steps}")
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

    # Set camera for good viewing angle
    sim.set_camera_view(
        eye=[8.0, -4.0, 3.5],
        target=[3.0, 0.0, 0.5],
    )

    # ------------------------------------------------------------------
    # 2. Create hierarchical environment
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
    print("\n[Test] Resetting environment...")
    obs = env.reset()

    initial_pos = env.initial_positions  # [num_envs, 2]
    print(f"[Test] Robot initial positions (first 4):")
    for i in range(min(4, num_envs)):
        print(f"  Env {i}: ({initial_pos[i, 0]:.2f}, {initial_pos[i, 1]:.2f})")

    # ------------------------------------------------------------------
    # 4. Setup walk_to skill - 3m ahead of each robot's start position
    # ------------------------------------------------------------------
    target_positions = initial_pos.clone()
    target_positions[:, 0] += args_cli.walk_distance  # X forward

    walk_cfg = WalkToConfig()
    walk_cfg.max_steps = args_cli.max_steps
    skill = WalkToSkill(config=walk_cfg, device=device)
    skill.reset(target_positions=target_positions)

    print(f"\n[Test] Walk targets (first 4):")
    for i in range(min(4, num_envs)):
        print(f"  Env {i}: ({target_positions[i, 0]:.2f}, {target_positions[i, 1]:.2f})")

    # ------------------------------------------------------------------
    # 5. Run skill loop
    # ------------------------------------------------------------------
    print("\n[Test] Starting walk_to skill...")
    start_time = time.time()
    done = False

    while simulation_app.is_running() and not done:
        # Skill generates velocity command from current state
        vel_cmd, done, result = skill.step(obs)

        # Environment steps with velocity command
        obs = env.step(vel_cmd)

        # Progress logging
        if env.step_count % 50 == 0:
            elapsed = time.time() - start_time
            current_pos = obs["root_pos"][:, :2]
            distances = torch.norm(current_pos - target_positions, dim=-1)
            print(
                f"[Test] Step {env.step_count:4d} | "
                f"Time: {elapsed:5.1f}s | "
                f"Mean dist: {distances.mean():.2f}m | "
                f"Min dist: {distances.min():.2f}m | "
                f"Height: {obs['base_height'].mean():.2f}m"
            )

        # Safety: if all robots fell, stop
        if (obs["base_height"] < 0.2).all():
            print("[Test] All robots fell! Stopping.")
            done = True

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    final_pos = obs["root_pos"][:, :2]
    distances = torch.norm(final_pos - target_positions, dim=-1)
    travel = torch.norm(final_pos - initial_pos, dim=-1)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Status       : {result.status.name}")
    print(f"  Reason       : {result.reason}")
    print(f"  Steps taken  : {result.steps_taken}")
    print(f"  Time elapsed : {elapsed:.1f}s")
    print(f"  Mean distance to target: {distances.mean():.2f}m")
    print(f"  Mean distance traveled : {travel.mean():.2f}m")
    print(f"  Mean final height      : {obs['base_height'].mean():.2f}m")
    print(f"  Robots still standing  : {(obs['base_height'] > 0.3).sum()}/{num_envs}")
    print("=" * 60)

    # Keep visualization alive
    if result.status.name == "SUCCESS":
        print("\n[Test] SUCCESS! Robots reached target. Keeping sim alive...")
    else:
        print(f"\n[Test] {result.status.name}: {result.reason}")

    # Stand still after reaching target
    stand_cmd = torch.zeros(num_envs, 3, device=device)
    while simulation_app.is_running():
        obs = env.step(stand_cmd)

    env.close()


if __name__ == "__main__":
    main()
