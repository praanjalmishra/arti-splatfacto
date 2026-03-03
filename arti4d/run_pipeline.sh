#!/bin/bash
################################################################################
# ArtiSplatfacto Single Scene Evaluation Pipeline
#
# Usage:
#   ./run_pipeline.sh [joint_idx]
#   ./run_pipeline.sh 0    <- default
#   ./run_pipeline.sh 2    <- joint 2
#
# Path layout (from docker-compose.yml volume mounts):
#   HOST ~/nerfstudio-data  ->  CONTAINER /workspace/data       (DATA_DIR)
#   HOST ~/arti-outputs     ->  CONTAINER /workspace/outputs    (OUTPUT_DIR)
#
# Step execution environment:
#   ns-train, change_detection, joint_estimation, merge  -> CONTAINER (docker)
#   sam2, tapip3d                                        -> HOST (conda)
#
# All per-joint outputs live inside articulated_joint_<N>/ :
#   frames/            <- from ARTi4D converter
#   depth/             <- from ARTi4D converter
#   transforms.json    <- from ARTi4D converter
#   cd_mask/           <- Step 2: change detection
#   mask/              <- Step 3: SAM2 VOS
#   tapip3d/           <- Step 4: point tracking
#   ransac_joints/     <- Step 5: RANSAC joint estimation
#   transforms_post.json <- Step 6: final merge
#
# Canonical model output:
#   canonical/splatfacto/  <- Step 1: ns-train
################################################################################

set -euo pipefail

JOINT_IDX=${2:-0}
SCENE_DIR="$1"

if [ -z "$SCENE_DIR" ]; then
    echo "Usage: ./run_pipeline.sh <scene_dir> [joint_idx]"
    exit 1
fi

SCENE_NAME="$(basename "$SCENE_DIR")"
SCENE_TYPE="$(basename "$(dirname "$SCENE_DIR")")"

# ── Host paths (for conda steps: sam2, tapip3d) ───────────────────────────────

# ── Host paths ───────────────────────────────────────────────

HOST_SCENE="${SCENE_DIR}"
HOST_CANONICAL="${HOST_SCENE}/canonical"
HOST_JOINT="${HOST_SCENE}/articulated_joint_${JOINT_IDX}"

SAM2_DIR="/local/home/pmishra/arti-splatfacto/third_party/sam2"

HOST_CANONICAL="${HOST_SCENE}/canonical"
HOST_JOINT="${HOST_SCENE}/articulated_joint_${JOINT_IDX}"

# ── Container paths (same directories via volume mount) ───────────────────────
CONTAINER_SCENE="/workspace/arti4d/arti4d_processed/${SCENE_TYPE}/${SCENE_NAME}"
CONTAINER_CANONICAL="${CONTAINER_SCENE}/canonical"
CONTAINER_JOINT="${CONTAINER_SCENE}/articulated_joint_${JOINT_IDX}"

# ── Helpers ───────────────────────────────────────────────────────────────────

print_step() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

docker_run() {
    docker compose -f /local/home/pmishra/arti-splatfacto/docker-compose.yml \
        exec -w /workspace/arti-splatfacto arti-splatfacto "$@"
}

# ── Preflight ─────────────────────────────────────────────────────────────────

print_step "Preflight checks"

for dir in "${HOST_CANONICAL}" "${HOST_JOINT}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: directory not found: $dir"
        exit 1
    fi
    if [ ! -f "${dir}/transforms.json" ]; then
        echo "ERROR: missing transforms.json in $dir"
        exit 1
    fi
done

echo "  scene:                    ${SCENE_NAME}"
echo "  joint:                    ${JOINT_IDX}"
echo "  canonical (host):         ${HOST_CANONICAL}"
echo "  canonical (container):    ${CONTAINER_CANONICAL}"
echo "  joint dir (host):         ${HOST_JOINT}"
echo "  joint dir (container):    ${CONTAINER_JOINT}"

################################################################################
# STEP 1: Train Canonical 3DGS  (container)
################################################################################

print_step "STEP 1: Canonical training check"

CKPT_PATH="${HOST_CANONICAL}/qed-splatter"

# Detect checkpoint (nerfstudio saves inside nested dirs)
CKPT_FILE=$(find "$CKPT_PATH" -name "*.ckpt" 2>/dev/null | head -n 1 || true)

if [ -n "$CKPT_FILE" ]; then
    echo "  Canonical checkpoint found:"
    echo "  $CKPT_FILE"
    echo "  Skipping canonical training."
else
    echo "  No checkpoint found. Training canonical model..."

    docker_run ns-train qed-splatter \
        --vis viewer \
        --pipeline.model.use-scale-regularization True \
        --pipeline.model.output-depth-during-training True \
        --pipeline.model.camera-optimizer.mode=off \
        --pipeline.model.use-bilateral-grid True \
        --experiment-name "" \
        --timestamp "" \
        --output-dir "${CONTAINER_CANONICAL}" \
        --viewer.quit-on-train-completion True \
        --max-num-iterations 6000 \
        nerfstudio-data \
            --data "${CONTAINER_CANONICAL}" \
            --auto-scale-poses False \
            --center-method none \
            --orientation-method none

    echo "  Canonical model saved: ${HOST_CANONICAL}/qed-splatter"
fi


# ################################################################################
# # STEP 2: Change Detection  (container)
# # Output: articulated_joint_<N>/cd_mask/
# ################################################################################

print_step "STEP 2: Change detection (canonical vs articulated_joint_${JOINT_IDX})"

docker_run python -m arti4d.run_change_detection \
    --canonical_model "${CONTAINER_CANONICAL}/qed-splatter" \
    --joint_dir       "${CONTAINER_JOINT}" \
    --joint           "${JOINT_IDX}" \
    --debug

echo "  Change detection complete → ${HOST_JOINT}/cd_mask/"


# ################################################################################
# # STEP 3: SAM2 VOS  (host conda: sam2)
# # Output: articulated_joint_<N>/mask/
# ################################################################################

print_step "STEP 3: SAM2 VOS  (host conda: sam2)"

set +u
eval "$(conda shell.bash hook)"
conda activate sam2
set -u

python -m arti4d.run_sam2_vos \
    --joint_dir "${HOST_JOINT}" \
    --sam2_dir  "${SAM2_DIR}"

set +u
conda deactivate
set -u
echo "  SAM2 complete → ${HOST_JOINT}/mask/"


################################################################################
# STEP 4: Point Tracking  (host conda: tapip3d)
# Output: articulated_joint_<N>/tapip3d/
################################################################################

print_step "STEP 4: Point tracking  (host conda: tapip3d)"

set +u
eval "$(conda shell.bash hook)"
conda activate tapip3d
set -u

python -m arti4d.run_point_tracking \
    --joint_dir         "${HOST_JOINT}" \
    --max_query_points  500

set +u
conda deactivate
set -u

echo "  Point tracking complete → ${HOST_JOINT}/tapip3d/"
################################################################################
# STEP 5: RANSAC Joint Estimation  (container)
# Output: articulated_joint_<N>/ransac_joints/
################################################################################

print_step "STEP 5: RANSAC joint estimation"

docker_run python -m arti4d.run_joint_estimation \
    --joint_dir "${CONTAINER_JOINT}"

echo "  Joint estimation complete → ${HOST_JOINT}/ransac_joints/"


################################################################################
# STEP 6: Merge — write transforms_post.json  (container)
# Output: articulated_joint_<N>/transforms_post.json
################################################################################

print_step "STEP 6: Merging articulation data"

docker_run python -m arti4d.run_merge_arti_data \
    --joint_dir "${CONTAINER_JOINT}"

echo "  Merge complete → ${HOST_JOINT}/transforms_post.json"


################################################################################
# DONE
################################################################################

print_step "PIPELINE COMPLETE  —  ${SCENE_NAME}  joint ${JOINT_IDX}"
echo ""
echo "  Canonical model:     ${HOST_CANONICAL}/splatfacto/"
echo "  transforms_post.json: ${HOST_JOINT}/transforms_post.json"
echo ""