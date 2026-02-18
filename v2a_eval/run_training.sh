#!/bin/bash
################################################################################
# V2A Articulation-Aware Training
# Single joint, clean eval setup
# No multi-joint, no recovery phase, no wandb required
################################################################################

set -e

print_step() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

################################################################################
# Arguments
################################################################################

SCENE_DIR=${1:?"Usage: $0 <scene_dir> [cuda_device]"}
CUDA_DEVICE=${2:-0}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Derived paths
CANONICAL_DIR="${SCENE_DIR}/canonical"
ARTI_TRANSFORMS="${SCENE_DIR}/articulated/transforms_post.json"
RANSAC_DIR="${SCENE_DIR}/articulated/ransac_joints"
OBJ_MASK="${RANSAC_DIR}/obj_masks/obj_revolute.pt" 
# Training settings
CURRENT_TIME=$(date +"%Y-%m-%d_%H%M%S")
TRAIN_ITR=10000
STOP_SPLIT_AT=8000
DEPTH_SCALE=1.0
USE_DEPTH=False

################################################################################
# VALIDATION
################################################################################

print_step "VALIDATION: Checking prerequisites for ${SCENE_DIR}"

# Check canonical GS checkpoint exists
# Find the most recent checkpoint under canonical/
CANONICAL_CHECKPOINT=$(find "${CANONICAL_DIR}" -name "nerfstudio_models" -type d \
    | sort | tail -1)

CANONICAL_CONFIG=$(find "${CANONICAL_DIR}" -name "config.yml" \
    | sort | tail -1)

if [ -z "$CANONICAL_CHECKPOINT" ]; then
    echo "❌ ERROR: Canonical GS checkpoint not found under: ${CANONICAL_DIR}"
    echo "   Run: ns-train qed-splatter --data ${CANONICAL_DIR}/transforms.json"
    exit 1
fi

if [ -z "$CANONICAL_CONFIG" ]; then
    echo "❌ ERROR: Canonical GS config.yml not found under: ${CANONICAL_DIR}"
    exit 1
fi

if [ ! -f "$ARTI_TRANSFORMS" ]; then
    echo "❌ ERROR: Articulation transforms not found: ${ARTI_TRANSFORMS}"
    echo "   Run: python -m v2a_eval.merge_arti_data_v2a --scene_dir ${SCENE_DIR}"
    exit 1
fi

# Determine joint type and find correct mask
if [ -f "${RANSAC_DIR}/obj_masks/obj_revolute.pt" ]; then
    OBJ_MASK="${RANSAC_DIR}/obj_masks/obj_revolute.pt"
    JOINT_PREFIX="revolute"
elif [ -f "${RANSAC_DIR}/obj_masks/obj_prismatic.pt" ]; then
    OBJ_MASK="${RANSAC_DIR}/obj_masks/obj_prismatic.pt"
    JOINT_PREFIX="prismatic"
else
    echo "❌ ERROR: No object mask found in: ${RANSAC_DIR}/obj_masks/"
    echo "   Expected: obj_revolute.pt or obj_prismatic.pt"
    exit 1
fi

echo "✓ Canonical checkpoint: ${CANONICAL_CHECKPOINT}"
echo "✓ Canonical config:     ${CANONICAL_CONFIG}"
echo "✓ Arti transforms:      ${ARTI_TRANSFORMS}"
echo "✓ Object mask:          ${OBJ_MASK} (${JOINT_PREFIX})"

################################################################################
# PHASE 1: Canonical 3DGS (already trained - just verify)
################################################################################

print_step "PHASE 1: Canonical 3DGS - Already trained ✓"
echo "  Using checkpoint: ${CANONICAL_CHECKPOINT}"
echo "  Skipping re-training."

################################################################################
# PHASE 2: Articulation-Aware Training
################################################################################

print_step "PHASE 2: Articulation-Aware Training"
echo "  Scene:      $(basename ${SCENE_DIR})"
echo "  Joint type: ${JOINT_PREFIX}"
echo "  Iterations: ${TRAIN_ITR}"
echo "  Device:     GPU ${CUDA_DEVICE}"

ns-train arti_splatfacto \
    --vis viewer \
    --output-dir "${SCENE_DIR}/arti_model" \
    --experiment-name "$(basename ${SCENE_DIR})" \
    --timestamp "$CURRENT_TIME" \
    --steps-per-eval-all-images 1000 \
    --max-num-iterations $TRAIN_ITR \
    --machine.num-devices 1 \
    --load-dir "$CANONICAL_CHECKPOINT" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.stop-split-at $STOP_SPLIT_AT \
    --pipeline.model.obj-mask-file "$OBJ_MASK" \
    --pipeline.model.use-depth $USE_DEPTH \
    --pipeline.model.active-joint-id "joint_0" \
    --pipeline.model.training-mode "articulation" \
    --pipeline.model.cull_alpha_thresh 0.005 \
    arti-dataparser \
    --data "$SCENE_DIR/articulated" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train-split-fraction 0.95 \
    --depth-unit-scale-factor $DEPTH_SCALE

echo ""
echo "========================================================================"
echo "✓ Articulation training complete: $(basename ${SCENE_DIR})"
echo "  Output: ${SCENE_DIR}/arti_model"
echo "========================================================================"