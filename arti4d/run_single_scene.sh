#!/bin/bash
################################################################################
# V2A Single Scene Evaluation Pipeline
# Simple version using docker-compose exec + local conda
################################################################################

set -e

SCENE_NAME=${1:?"Usage: $0 <scene_name>"}

# Container paths (via volume mounts in docker-compose.yml)
SCENE_INPUT="/workspace/data/arti4d/nerf_output/din080_scene1"
SCENE_OUTPUT="/workspace/outputs/arti4d/${SCENE_NAME}"

# Host paths (for local conda steps)
HOST_OUTPUT="/local/home/pmishra/arti-outputs/art4d/${SCENE_NAME}"
SAM2_DIR="/local/home/pmishra/arti-splatfacto/third_party/sam2"

print_step() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}


# ################################################################################
# # STEP 1: Train Canonical 3DGS
# ################################################################################

print_step "STEP 2: Training canonical 3DGS"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
ns-train qed-splatter \
    --vis viewer \
    --pipeline.model.use-scale-regularization=True \
    --pipeline.model.output-depth-during-training=True \
    --pipeline.model.rasterize-mode=antialiased \
    --pipeline.model.camera-optimizer.mode=SO3xR3 \
    --pipeline.model.use-bilateral-grid=True \
    --experiment-name "" \
    --timestamp "" \
    --output-dir "$OUTPUT_PATH" \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 5000 \
    nerfstudio-data \
    --data "$SCENE_INPUT" \
    --train-split-fraction 1.0

echo "✓ Canonical 3DGS trained"

################################################################################
# STEP 2: Change Detection
################################################################################

print_step "STEP 3: Change detection"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m aria_hoi.run_change_detection \
    --scene "${SCENE_OUTPUT}" \
    --debug

echo "✓ Change detection complete"

################################################################################
# STEP 4: SAM2 VOS (local conda env)
################################################################################

print_step "STEP 4: SAM2 VOS (local conda: sam2)"

eval "$(conda shell.bash hook)"
conda activate sam2

python -m aria_hoi.run_sam2_vos \
    --scene "${HOST_OUTPUT}" \
    --sam2_dir "${SAM2_DIR}"

conda deactivate

echo "✓ SAM2 complete"

################################################################################
# STEP 5: Point Tracking (local conda env)
################################################################################

print_step "STEP 5: Point tracking (local conda: tapip3d)"

conda activate tapip3d

python -m aria_hoi.run_point_tracking \
    --scene "${HOST_OUTPUT}" \
    --max_query_points 300

conda deactivate

echo "✓ Point tracking complete"

################################################################################
# STEP 6: RANSAC Joint Estimation
################################################################################

print_step "STEP 6: RANSAC joint estimation"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m aria_hoi.run_joint_estimation \
    --scene "${SCENE_OUTPUT}" \

echo "✓ Joint estimation complete"

################################################################################
# STEP 7: Merge Articulation Data
################################################################################

print_step "STEP 7: Merging articulation data"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    python -m aria_hoi.merge_arti_data_v2a \
    --scene_dir "${SCENE_OUTPUT}"

echo "✓ Articulation data merged"

################################################################################
# STEP 8: Train Articulation-Aware 3DGS
################################################################################

print_step "STEP 8: Training articulation-aware 3DGS"

docker compose exec -w /workspace/arti-splatfacto arti-splatfacto \
    bash aria_hoi/run_training.sh \
    "${SCENE_OUTPUT}" \
    0

echo "✓ Articulation training complete"

################################################################################
# DONE
################################################################################

print_step "✅ PIPELINE COMPLETE: ${SCENE_NAME}"

echo "Outputs: ${HOST_OUTPUT}/"