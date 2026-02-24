#!/usr/bin/env bash
# =============================================================================
# Play/Visualize trained G1 29-DoF Locomotion Policy
# =============================================================================
#
# Usage:
#   bash play_loco.sh                            # Latest checkpoint
#   bash play_loco.sh --load_run my_run_name     # Specific run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_DIR="${SCRIPT_DIR}/../../../../../../.."
UNITREE_RL_LAB_DIR="${ISAACLAB_DIR}/source/unitree_rl_lab"

echo "=============================================="
echo "  G1 29-DoF Locomotion Play"
echo "=============================================="

python "${UNITREE_RL_LAB_DIR}/scripts/rsl_rl/play.py" \
    --task Unitree-G1-29dof-Velocity \
    --num_envs ${NUM_ENVS:-16} \
    "$@"
