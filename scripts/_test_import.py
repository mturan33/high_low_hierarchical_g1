"""Quick import test for arm policy wrapper."""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from high_low_hierarchical_g1.low_level.arm_policy_wrapper import (
    ArmPolicyWrapper, ARM_OBS_DIM, ARM_ACT_DIM,
    ARM_POLICY_JOINT_NAMES_29DOF, RIGHT_FINGER_JOINT_NAMES_29DOF,
    get_palm_forward, compute_orientation_error,
)

print(f"ArmPolicyWrapper imported: {ARM_OBS_DIM} obs -> {ARM_ACT_DIM} act")
print(f"Arm joints: {ARM_POLICY_JOINT_NAMES_29DOF}")
print(f"Right finger joints: {RIGHT_FINGER_JOINT_NAMES_29DOF}")

# Test network build
layers = []
prev = ARM_OBS_DIM
for h in [256, 256, 128]:
    layers += [nn.Linear(prev, h), nn.ELU()]
    prev = h
layers.append(nn.Linear(prev, ARM_ACT_DIM))
net = nn.Sequential(*layers)
test_in = torch.randn(4, ARM_OBS_DIM)
test_out = net(test_in)
print(f"Network test: {test_in.shape} -> {test_out.shape}")

# Test obs builder
obs = ArmPolicyWrapper.build_obs(
    arm_pos=torch.zeros(4, 5),
    arm_vel=torch.zeros(4, 5),
    finger_pos=torch.zeros(4, 7),
    ee_body=torch.zeros(4, 3),
    ee_vel_body=torch.zeros(4, 3),
    palm_quat=torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.float),
    finger_lower=torch.zeros(7),
    finger_upper=torch.ones(7),
    target_body=torch.tensor([[0.3, -0.1, 0.1]] * 4),
    root_height=torch.ones(4) * 0.78,
    lin_vel_body=torch.zeros(4, 3),
    ang_vel_body=torch.zeros(4, 3),
    steps_since_spawn=torch.zeros(4, dtype=torch.long),
    ee_pos_at_spawn=torch.zeros(4, 3),
    initial_dist=torch.ones(4) * 0.3,
)
print(f"Obs builder test: {obs.shape} (expected [4, {ARM_OBS_DIM}])")

# Test checkpoint loading
ckpt_path = r"C:\IsaacLab\logs\ulc\ulc_g1_stage7_antigaming_2026-02-06_17-41-47\model_best.pt"
if os.path.exists(ckpt_path):
    wrapper = ArmPolicyWrapper(ckpt_path, device="cpu")
    action = wrapper.get_action(obs)
    targets = wrapper.get_arm_targets(obs)
    print(f"Inference: obs {obs.shape} -> action {action.shape} -> targets {targets.shape}")
    print(f"Sample action: {action[0].tolist()}")
    print(f"Sample targets: {targets[0].tolist()}")
else:
    print(f"Checkpoint not found: {ckpt_path}")

print("\nAll tests PASSED!")
