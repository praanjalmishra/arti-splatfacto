#!/usr/bin/env bash
set -e

DATA_DIR="$(realpath "${1:-"eval_renders/StorageFurniture_35059"}")"
CUDA_DEVICE=${2:-0}
SKIP_RENDERING=${3:-true}
CHANGE_CFG=${5:-""}

OBJECT_ID=$(basename "$DATA_DIR" | cut -d'_' -f2)
PRE_DIR="${DATA_DIR}/pre"
POST_DIR="${DATA_DIR}/post"

# Environment
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/EfficientSAM"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/sam2"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/TAPIP3D"

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

print_step() { echo -e "\n=== $1 ==="; }

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Missing file: $1"
        exit 1
    fi
}

print_step "Processing Object: $OBJECT_ID"
echo "Using dataset at: $DATA_DIR"
echo "Pre dir: $PRE_DIR"
echo "Post dir: $POST_DIR"

if [ "$SKIP_RENDERING" = "true" ]; then
    check_file "${PRE_DIR}/transforms.json"
    check_file "${POST_DIR}/transforms.json"
    cp "${POST_DIR}/transforms.json" "${POST_DIR}/transforms_reloc.json"
else
    echo "ERROR: Rendering mode not supported in eval pipeline"
    exit 1
fi

# Step 2: Train static 3DGS -----------------------------------------------------
print_step "Training Static 3DGS"

# Define checkpoint search path
CKPT_PATH=$(find "${PRE_DIR}/splatfacto/eval_${OBJECT_ID}" -name "*.ckpt" 2>/dev/null | head -1)

if [ -n "$CKPT_PATH" ]; then
    echo "Checkpoint already exists at: $CKPT_PATH"
    echo "→ Skipping training step."
else
    echo "No checkpoint found — starting training..."
    ns-train qed-splatter \
        --output-dir "${PRE_DIR}/splatfacto" \
        --experiment-name "eval_${OBJECT_ID}" \
        --timestamp $TIMESTAMP \
        --steps_per_eval_all_images 500 \
        --pipeline.model.cull_alpha_thresh 0.005 \
        --pipeline.model.cull_scale_thresh 0.1 \
        --max-num-iterations 6000 \
        --pipeline.model.warmup_length 200 \
        --pipeline.model.use_scale_regularization True \
        --pipeline.model.background_color black \
        --machine.num-devices 1 \
        --viewer.quit-on-train-completion True \
        nerfstudio-data \
        --data "${PRE_DIR}/transforms.json" \
        --depth_unit_scale_factor 1.0 \
        --auto-scale-poses False \
        --center-method none \
        --orientation-method none \
        --load-3D-points True
fi

# Find (or reuse) checkpoint directory
CHECKPOINT=$(find "${PRE_DIR}/splatfacto/eval_${OBJECT_ID}" -name "config.yml" 2>/dev/null | head -1 | xargs dirname)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No valid checkpoint directory found after training."
    exit 1
fi

echo "Checkpoint found at: $CHECKPOINT"
check_file "${CHECKPOINT}/config.yml"

# Step 3: Change detection
print_step "Change Detection"

python change_det/change_detection_sam.py \
    --config "${CHECKPOINT}/config.yml" \
    --output "${POST_DIR}" \
    --transform "${POST_DIR}/transforms_reloc.json" \
    --params "$CHANGE_CFG" \
    --debug

# Step 4: SAM2 segmentation
print_step "SAM2 Segmentation"

eval "$(conda shell.bash hook)"
conda activate sam2

python /local/home/pmishra/cvg/sam2/tools/run_sam2_vos_wrapper.py \
    --frames_dir "${POST_DIR}/frames" \
    --mask_path "${POST_DIR}/cd_mask" \
    --output_mask_dir "${POST_DIR}/mask"

# Step 5: Point tracking (FIXED PATH HANDLING)
print_step "TAPIP3D Tracking"

conda activate tapip3d

# Store current directory and use absolute paths to avoid duplication
CURRENT_DIR="$(pwd)"

# Create output directory
mkdir -p "${POST_DIR}/tapip3d"

# Verify input files exist
check_file "${POST_DIR}/transforms_reloc.json"
check_file "${POST_DIR}/mask/frame_00001.png"

echo "TAPIP3D inputs:"
echo "  transforms: ${POST_DIR}/transforms_reloc.json"
echo "  mask: ${POST_DIR}/mask/frame_00001.png"
echo "  output: ${POST_DIR}/tapip3d"

cd /local/home/pmishra/cvg/TAPIP3D
python inference_nerf_mask.py \
    --input_path "${POST_DIR}/transforms_reloc.json" \
    --mask_path "${POST_DIR}/mask/frame_00001.png" \
    --output_dir "${POST_DIR}/tapip3d" \
    --max_query_points 100

# Return to original directory
cd "$CURRENT_DIR"

# Step 6: Joint estimation
print_step "Joint Estimation"

conda activate nerfstudio_env

python -m joint_estimator.main \
    --tapip3d_result "${POST_DIR}/tapip3d" \
    --out_dir "${POST_DIR}/ransac_joints" \
    --camera_metadata "${POST_DIR}/transforms_reloc.json" \
    --data_dir "${POST_DIR}" \
    --no_viz

# Step 7: Merge results
print_step "Merging Results"

python joint_estimator/merge_arti_data.py \
    --joint_schemas "${POST_DIR}/ransac_joints/joint_schemas.json" \
    --transforms_post "${POST_DIR}/transforms_reloc.json" \
    --output "${POST_DIR}/transforms_post.json" \
    --use_masks

print_step "Complete: $OBJECT_ID"
echo "Results: ${POST_DIR}"