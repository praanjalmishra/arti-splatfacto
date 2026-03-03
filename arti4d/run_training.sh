#!/bin/bash
#############################################################################
# ArtiSplatfacto Training Script
#############################################################################

set -euo pipefail

print_step() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

docker_run() {
    docker compose -f /local/home/pmishra/arti-splatfacto/docker-compose.yml \
        exec -w /workspace/arti-splatfacto arti-splatfacto "$@"
}


################################################################################
# Arguments
################################################################################

# Usage:
#   ./run_training.sh <scene_dir> [joint_idx] [cuda_device]

SCENE_DIR="$1"
JOINT_IDX=${2:-0}
CUDA_DEVICE=${3:-0}

if [ -z "${SCENE_DIR:-}" ]; then
    echo "Usage: ./run_training.sh <scene_dir> [joint_idx] [cuda_device]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

################################################################################
# Paths
################################################################################

# Example SCENE_DIR:
# /media/.../arti4d/arti4d_processed/din080/scene_xxx

HOST_SCENE="${SCENE_DIR}"

SCENE_NAME="$(basename "$SCENE_DIR")"
SCENE_TYPE="$(basename "$(dirname "$SCENE_DIR")")"

# Container path (based on your volume mount)
CONTAINER_SCENE="/workspace/arti4d/arti4d_processed/${SCENE_TYPE}/${SCENE_NAME}"
CONTAINER_CANONICAL="${CONTAINER_SCENE}/canonical"
CONTAINER_JOINT="${CONTAINER_SCENE}/articulated_joint_${JOINT_IDX}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

################################################################################
# Derived paths
################################################################################

ARTI_TRANSFORMS="${CONTAINER_JOINT}/transforms_post.json"
RANSAC_DIR="${CONTAINER_JOINT}/ransac_joints"

CURRENT_TIME=$(date +"%Y-%m-%d_%H%M%S")
TRAIN_ITR=15000
STOP_SPLIT_AT=14000
USE_DEPTH=True

################################################################################
# VALIDATION
################################################################################

print_step "VALIDATION: Checking prerequisites"

CANONICAL_CHECKPOINT=$(find "${CONTAINER_CANONICAL}" -name "nerfstudio_models" -type d -printf "%T@ %p\n" \
    | sort -n | tail -1 | cut -d' ' -f2-)

CANONICAL_CONFIG=$(find "${CONTAINER_CANONICAL}" -name "config.yml" -printf "%T@ %p\n" \
    | sort -n | tail -1 | cut -d' ' -f2-)

if [[ -z "${CANONICAL_CHECKPOINT:-}" ]]; then
    echo "ERROR: Canonical checkpoint not found under ${CONTAINER_CANONICAL}"
    exit 1
fi

if [[ -z "${CANONICAL_CONFIG:-}" ]]; then
    echo "ERROR: Canonical config.yml not found under ${CONTAINER_CANONICAL}"
    exit 1
fi

if [[ ! -f "${ARTI_TRANSFORMS}" ]]; then
    echo "ERROR: Articulation transforms not found: ${ARTI_TRANSFORMS}"
    exit 1
fi

# Detect joint mask
if [[ -f "${RANSAC_DIR}/obj_masks/obj_revolute.pt" ]]; then
    OBJ_MASK="${RANSAC_DIR}/obj_masks/obj_revolute.pt"
    JOINT_PREFIX="revolute"
elif [[ -f "${RANSAC_DIR}/obj_masks/obj_prismatic.pt" ]]; then
    OBJ_MASK="${RANSAC_DIR}/obj_masks/obj_prismatic.pt"
    JOINT_PREFIX="prismatic"
else
    echo "ERROR: No object mask found in ${RANSAC_DIR}/obj_masks/"
    exit 1
fi

echo "✓ Canonical checkpoint: ${CANONICAL_CHECKPOINT}"
echo "✓ Canonical config:     ${CANONICAL_CONFIG}"
echo "✓ Arti transforms:      ${ARTI_TRANSFORMS}"
echo "✓ Object mask:          ${OBJ_MASK} (${JOINT_PREFIX})"

################################################################################
# PHASE 1: Canonical 3DGS (Verification Only)
################################################################################

print_step "PHASE 1: Canonical 3DGS - Already Trained ✓"
echo "Using checkpoint: ${CANONICAL_CHECKPOINT}"

################################################################################
# PHASE 2: Articulation-Aware Training
################################################################################

print_step "PHASE 2: Articulation-Aware Training"

echo "Scene:      $(basename "${CONTAINER_SCENE}")"
echo "Joint type: ${JOINT_PREFIX}"
echo "Joint ID:   joint_${JOINT_IDX}"
echo "Iterations: ${TRAIN_ITR}"
echo "GPU:        ${CUDA_DEVICE}"

docker_run ns-train arti_splatfacto \
    --vis viewer+wandb \
    --timestamp "${CURRENT_TIME}" \
    --steps-per-eval-all-images 1000 \
    --max-num-iterations "${TRAIN_ITR}" \
    --machine.num-devices 1 \
    --load-dir "${CANONICAL_CHECKPOINT}" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.stop-split-at "${STOP_SPLIT_AT}" \
    --pipeline.model.obj-mask-file "${OBJ_MASK}" \
    --pipeline.model.use-depth "${USE_DEPTH}" \
    --pipeline.model.active-joint-id "joint_${JOINT_IDX}" \
    --pipeline.model.training-mode "articulation" \
    --pipeline.model.cull_alpha_thresh 0.005 \
    --output-dir "${CONTAINER_JOINT}" \
    --experiment-name "" \
    --timestamp "" \
    arti-dataparser \
    --data "${CONTAINER_JOINT}" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train-split-fraction 0.95 \
    # --depth-unit-scale-factor $DEPTH_SCALE

################################################################################
# DONE
################################################################################

echo ""
echo "========================================================================"
echo "✓ Articulation training complete"
echo "Output directory: ${CONTAINER_SCENE}/arti_model"
echo "========================================================================"