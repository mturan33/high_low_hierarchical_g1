#!/usr/bin/env bash
# =============================================================================
# Train G1 29-DoF Locomotion Policy using unitree_rl_lab
# =============================================================================
#
# Prerequisites:
#   1. unitree_rl_lab installed (pip install -e source/unitree_rl_lab/source/unitree_rl_lab/)
#   2. G1 robot USD model downloaded from HuggingFace:
#      git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
#   3. UNITREE_MODEL_DIR set in unitree_rl_lab/assets/robots/unitree.py
#
# Usage:
#   cd C:\IsaacLab
#   bash source/isaaclab_tasks/isaaclab_tasks/direct/high_low_hierarchical_g1/scripts/train_loco.sh
#
# Or on Windows:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p source\unitree_rl_lab\scripts\rsl_rl\train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 4096

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_DIR="${SCRIPT_DIR}/../../../../../../.."
UNITREE_RL_LAB_DIR="${ISAACLAB_DIR}/source/unitree_rl_lab"

echo "=============================================="
echo "  G1 29-DoF Locomotion Training"
echo "=============================================="
echo "Isaac Lab: ${ISAACLAB_DIR}"
echo "unitree_rl_lab: ${UNITREE_RL_LAB_DIR}"
echo ""

# Train
python "${UNITREE_RL_LAB_DIR}/scripts/rsl_rl/train.py" \
    --headless \
    --task Unitree-G1-29dof-Velocity \
    --num_envs ${NUM_ENVS:-4096} \
    --max_iterations ${MAX_ITER:-2000} \
    "$@"
