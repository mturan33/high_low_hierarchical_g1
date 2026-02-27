"""
G1 29-DoF Joint Configuration
==============================
Joint definitions, default poses, and observation dimensions for the
Unitree G1-29DoF robot as used by unitree_rl_lab.

Source: unitree_rl_lab/assets/robots/unitree.py (UNITREE_G1_29DOF_CFG)

Total: 29 joints = 12 legs + 3 waist + 14 arms (7 per side)
"""

# ============================================================================
# JOINT NAMES — SDK ordering (matches unitree_rl_lab)
# ============================================================================

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
]  # 12 joints

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
]  # 3 joints

ARM_JOINT_NAMES_LEFT = [
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
]  # 7 joints

ARM_JOINT_NAMES_RIGHT = [
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",   # 23
    "right_shoulder_yaw_joint",    # 24
    "right_elbow_joint",           # 25
    "right_wrist_roll_joint",      # 26
    "right_wrist_pitch_joint",     # 27
    "right_wrist_yaw_joint",       # 28
]  # 7 joints

ARM_JOINT_NAMES = ARM_JOINT_NAMES_LEFT + ARM_JOINT_NAMES_RIGHT  # 14 joints
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES  # 29 joints

# ============================================================================
# DEX3 FINGER JOINT NAMES (7 per hand, 14 total)
# ============================================================================

DEX3_JOINT_NAMES_LEFT = [
    "left_hand_index_0_joint",    # 0
    "left_hand_index_1_joint",    # 1
    "left_hand_middle_0_joint",   # 2
    "left_hand_middle_1_joint",   # 3
    "left_hand_thumb_0_joint",    # 4
    "left_hand_thumb_1_joint",    # 5
    "left_hand_thumb_2_joint",    # 6
]  # 7 joints

DEX3_JOINT_NAMES_RIGHT = [
    "right_hand_index_0_joint",   # 0
    "right_hand_index_1_joint",   # 1
    "right_hand_middle_0_joint",  # 2
    "right_hand_middle_1_joint",  # 3
    "right_hand_thumb_0_joint",   # 4
    "right_hand_thumb_1_joint",   # 5
    "right_hand_thumb_2_joint",   # 6
]  # 7 joints

DEX3_JOINT_NAMES = DEX3_JOINT_NAMES_LEFT + DEX3_JOINT_NAMES_RIGHT  # 14 joints
NUM_DEX3_JOINTS = len(DEX3_JOINT_NAMES)       # 14
NUM_DEX3_JOINTS_PER_HAND = 7

# Full 43-DoF joint list (body + fingers)
ALL_JOINT_NAMES_DEX3 = ALL_JOINT_NAMES + DEX3_JOINT_NAMES  # 43 joints
NUM_ALL_JOINTS_DEX3 = 43

# Finger close positions (radians) — derived from actual USD joint limits
# Sign convention determined by joint limit ranges:
#   Right index/middle: range [0, +1.57]    → close = POSITIVE
#   Right thumb_0:      range [-1.05, +1.05] → close = NEGATIVE (opposition)
#   Right thumb_1:      range [-1.05, +0.61] → close = NEGATIVE
#   Right thumb_2:      range [-1.75, 0.00]  → close = NEGATIVE (only neg possible!)
#   Left hand: MIRRORED — index/middle close NEGATIVE, thumb_1/thumb_2 close POSITIVE
DEX3_FINGER_OPEN = {j: 0.0 for j in DEX3_JOINT_NAMES}
DEX3_FINGER_CLOSE = {
    # Left hand — index/middle close NEGATIVE, thumb_1/2 close POSITIVE
    "left_hand_index_0_joint": -0.8,     # range [-1.571, 0.000]
    "left_hand_index_1_joint": -1.0,     # range [-1.745, 0.000]
    "left_hand_middle_0_joint": -0.8,    # range [-1.571, 0.000]
    "left_hand_middle_1_joint": -1.0,    # range [-1.745, 0.000]
    "left_hand_thumb_0_joint": 0.6,      # range [-1.047, +1.047] (opposition)
    "left_hand_thumb_1_joint": 0.8,      # range [-0.611, +1.047]
    "left_hand_thumb_2_joint": 0.6,      # range [+0.000, +1.745]
    # Right hand — index/middle close POSITIVE, thumb_1/2 close NEGATIVE
    "right_hand_index_0_joint": 0.8,     # range [+0.000, +1.571]
    "right_hand_index_1_joint": 1.0,     # range [+0.000, +1.745]
    "right_hand_middle_0_joint": 0.8,    # range [+0.000, +1.571]
    "right_hand_middle_1_joint": 1.0,    # range [+0.000, +1.745]
    "right_hand_thumb_0_joint": -0.6,    # range [-1.047, +1.047] (opposition)
    "right_hand_thumb_1_joint": -0.8,    # range [-1.047, +0.611]
    "right_hand_thumb_2_joint": -0.6,    # range [-1.745, +0.000]
}

# ============================================================================
# JOINT COUNTS
# ============================================================================

NUM_LEG_JOINTS = len(LEG_JOINT_NAMES)       # 12
NUM_WAIST_JOINTS = len(WAIST_JOINT_NAMES)   # 3
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)       # 14
NUM_ALL_JOINTS = 29

# ============================================================================
# DEFAULT POSES — from UNITREE_G1_29DOF_CFG init_state
# ============================================================================

DEFAULT_JOINT_POSES = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.97,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# As ordered list (matching ALL_JOINT_NAMES order)
DEFAULT_JOINT_LIST = [DEFAULT_JOINT_POSES.get(j, 0.0) for j in ALL_JOINT_NAMES]

# ============================================================================
# OBSERVATION DIMENSIONS — unitree_rl_lab format
# ============================================================================

# Per-frame observation (96 dim total):
#   base_ang_vel:    3
#   proj_gravity:    3
#   velocity_cmds:   3
#   joint_pos_rel:  29
#   joint_vel_rel:  29
#   last_action:    29
OBS_DIM_PER_FRAME = 96

# History stacking
OBS_HISTORY_LENGTH = 5
OBS_DIM_TOTAL = OBS_DIM_PER_FRAME * OBS_HISTORY_LENGTH  # 480

# Action dimension
ACTION_DIM = NUM_ALL_JOINTS  # 29
ACTION_SCALE = 0.25  # Joint position target scale

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

STANDING_HEIGHT = 0.80       # Default standing height (m)
TARGET_BASE_HEIGHT = 0.78    # Reward target height (m)
MIN_BASE_HEIGHT = 0.20       # Termination threshold (m)
MAX_TILT_ANGLE = 0.8         # Termination: max body tilt (rad)

GAIT_PERIOD = 0.8            # Gait cycle period (s)
GAIT_OFFSET = [0.0, 0.5]    # Left/right phase offset

# Control frequency
SIM_DT = 0.005               # Physics timestep (s)
DECIMATION = 4                # Control decimation
CONTROL_DT = SIM_DT * DECIMATION  # 0.02s = 50 Hz

# ============================================================================
# VELOCITY COMMAND RANGES — from unitree_rl_lab env_cfg
# ============================================================================

# Initial (curriculum start)
CMD_RANGE_INIT = {
    "lin_vel_x": (-0.1, 0.1),   # m/s
    "lin_vel_y": (-0.1, 0.1),   # m/s
    "ang_vel_z": (-0.1, 0.1),   # rad/s
}

# Limit (curriculum max)
CMD_RANGE_LIMIT = {
    "lin_vel_x": (-0.5, 1.0),   # m/s
    "lin_vel_y": (-0.3, 0.3),   # m/s
    "ang_vel_z": (-0.2, 0.2),   # rad/s
}

# Observation scaling factors
OBS_SCALES = {
    "ang_vel": 0.2,
    "joint_vel": 0.05,
}
