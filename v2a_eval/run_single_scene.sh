#!/bin/bash
################################################################################
# V2A Single Scene Evaluation Pipeline
# Simple version using docker-compose exec + local conda
################################################################################

set -e

SCENE_NAME=${1:?"Usage: $0 <scene_name>"}

# Container paths (via volume mounts in docker-compose.yml)
SCENE_INPUT="/workspace/videoartgs-data/v2a/${SCENE_NAME}"
SCENE_OUTPUT="/workspace/arti-splatfacto/outputs/v2a_eval/${SCENE_NAME}"

# Host paths (for local conda steps)
HOST_OUTPUT="/local/home/pmishra/arti-splatfacto/outputs/v2a_eval/${SCENE_NAME}"
SAM2_DIR="/local/home/pmishra/arti-splatfacto/third_party/sam2"

print_step() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

################################################################################
# STEP 1: Prepare Scene
################################################################################

print_step "STEP 1: Preparing scene ${SCENE_NAME}"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m v2a_eval.prepare_scene \
    --scene "${SCENE_INPUT}" \
    --output "${SCENE_INPUT}" \

echo "✓ Scene prepared"

################################################################################
# STEP 2: Train Canonical 3DGS
################################################################################

print_step "STEP 2: Training canonical 3DGS"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    ns-train qed-splatter \
    --vis viewer \
    --output-dir "${SCENE_OUTPUT}/canonical" \
    --experiment-name "model" \
    --max-num-iterations 20000 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --data "${SCENE_OUTPUT}/canonical/transforms.json" \
    --auto-scale-poses False \
    --center-method none \
    --orientation-method none

echo "✓ Canonical 3DGS trained"

################################################################################
# STEP 3: Change Detection
################################################################################

print_step "STEP 3: Change detection"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m v2a_eval.run_change_detection \
    --scene "${SCENE_OUTPUT}" \
    --debug

echo "✓ Change detection complete"

################################################################################
# STEP 4: SAM2 VOS (local conda env)
################################################################################

print_step "STEP 4: SAM2 VOS (local conda: sam2)"

eval "$(conda shell.bash hook)"
conda activate sam2

python -m v2a_eval.run_sam2_vos \
    --scene "${HOST_OUTPUT}" \
    --sam2_dir "${SAM2_DIR}"

conda deactivate

echo "✓ SAM2 complete"

################################################################################
# STEP 5: Point Tracking (local conda env)
################################################################################

print_step "STEP 5: Point tracking (local conda: tapip3d)"

conda activate tapip3d

python -m v2a_eval.run_point_tracking \
    --scene "${HOST_OUTPUT}" \
    --max_query_points 300

conda deactivate

echo "✓ Point tracking complete"

################################################################################
# STEP 6: RANSAC Joint Estimation
################################################################################

print_step "STEP 6: RANSAC joint estimation"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m v2a_eval.run_joint_estimation \
    --scene "${SCENE_OUTPUT}" \
    --no_viz

echo "✓ Joint estimation complete"

################################################################################
# STEP 7: Merge Articulation Data
################################################################################

print_step "STEP 7: Merging articulation data"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m v2a_eval.merge_arti_data_v2a \
    --joint_schemas "${SCENE_OUTPUT}/ransac_joints/joint_schemas.json" \
    --scene_dir "${SCENE_OUTPUT}"

echo "✓ Articulation data merged"

################################################################################
# STEP 8: Train Articulation-Aware 3DGS
################################################################################

print_step "STEP 8: Training articulation-aware 3DGS"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    bash v2a_eval/run_training.sh \
    "${SCENE_OUTPUT}" \
    0

echo "✓ Articulation training complete"

################################################################################
# DONE
################################################################################

print_step "✅ PIPELINE COMPLETE: ${SCENE_NAME}"

echo "Outputs: ${HOST_OUTPUT}/"