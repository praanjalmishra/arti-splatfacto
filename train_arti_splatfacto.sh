#!/bin/bash

################################################################################
# Articulated 3D Reconstruction Pipeline
# Unsupervised articulated 3DGS from monocular dynamic capture
################################################################################

set -e  # Exit on error
################################################################################
# Configuration
################################################################################

# Parse command line arguments
CUDA_DEVICE=${2:-0}
SKIP_PRECHANGE=${3:-false}  # Set to 'true' to skip pre-change processing
SKIP_3DGS_TRAIN=${4:-false}  # Set to 'true' to skip 3DGS training

# Project paths
PROJECT_ROOT="/local/home/pmishra/cvg/arti-splatfacto"
OUTPUT_FOLDER="${PROJECT_ROOT}/outputs"
NERFSTUDIO_FOLDER="/local/home/pmishra/cvg/nerfstudio"

# Environment setup
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/EfficientSAM"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/sam2"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/TAPIP3D"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Timestamp for runs
CURRENT_TIME=$(date +%Y-%m-%d_%H%M%S)

################################################################################
# Pre-change sequence parameters
################################################################################

PRECHANGE_DATA_DIR="data_real/day8/pre_raw"
PRECHANGE_R3D_PATH="${PRECHANGE_DATA_DIR}/pre.r3d"
PRECHANGE_OUTPUT_DIR="data_real/day8/pre"
PRECHANGE_STRIDE=11
PRECHANGE_START=80
PRECHANGE_END=2600
PRECHANGE_DOWNSAMPLE=2
PRECHANGE_PC_SUBSAMPLE=8
PRECHANGE_VOXEL_DOWNSAMPLE=0.02

# Pre-trained checkpoint (update after first run)
PRETRAINED_CONFIG="${PRECHANGE_OUTPUT_DIR}/splatfacto/qed-splatter/2025-11-14_131705/config.yml"
CHECKPOINT_PATH="${PRECHANGE_OUTPUT_DIR}/splatfacto/qed-splatter/2025-11-14_131705/nerfstudio_models"

################################################################################
# Multi-view dynamic sequence parameters
################################################################################

# MULTI_DATA_DIR="data_real/day6/prism_raw"
# MULTI_R3D_PATH="${MULTI_DATA_DIR}/2025-11-04--10-55-19.r3d"
# MULTI_OUTPUT="data_real/day6/prism"
# MULTI_START=200
# MULTI_END=1500
# MULTI_STRIDE=7
# MULTI_VOXEL_DOWNSAMPLE=0.02




MULTI_DATA_DIR="data_real/day8/post_3_raw"
MULTI_R3D_PATH="${MULTI_DATA_DIR}/post_3.r3d"
MULTI_OUTPUT="data_real/day8/post_3"
MULTI_START=50
MULTI_END=1500
MULTI_STRIDE=7
MULTI_VOXEL_DOWNSAMPLE=0.02



NUM_RETRIEVAL=20
RANSAC_THRESH=12.0
BATCHSIZE_REFINEMENT=8
MAX_QUERY_POINTS=150


print_step() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        exit 1
    fi
}

check_dir_exists() {
    if [ ! -d "$1" ]; then
        echo "ERROR: Required directory not found: $1"
        exit 1
    fi
}

################################################################################
# STEP 1: Process Pre-Change RGBD Data
################################################################################

# if [ "$SKIP_PRECHANGE" != "true" ]; then
#     print_step "STEP 1: Processing pre-change RGBD data"
    
#     python process_data/r3d_to_nerf.py \
#         --r3d_data_dir "$PRECHANGE_R3D_PATH" \
#         --output_dir "$PRECHANGE_OUTPUT_DIR" \
#         --stride $PRECHANGE_STRIDE \
#         --start $PRECHANGE_START \
#         --end $PRECHANGE_END \
#         --downsample_factor $PRECHANGE_DOWNSAMPLE \
#         --pc_subsample $PRECHANGE_PC_SUBSAMPLE \
#         --voxel_downsample $PRECHANGE_VOXEL_DOWNSAMPLE
    
#     echo "✓ Pre-change data processed"
# fi

# exit 0

# ################################################################################
# # STEP 2: Process Multi-View RGBD Data
# ################################################################################

print_step "STEP 2: Processing multi-view RGBD data"

# python process_data/r3d_to_nerf.py \
#     --r3d_data_dir "$MULTI_R3D_PATH" \
#     --output_dir "$MULTI_OUTPUT" \
#     --stride $MULTI_STRIDE \
#     --start $MULTI_START \
#     --end $MULTI_END \
#     --downsample_factor $PRECHANGE_DOWNSAMPLE \
#     --voxel_downsample $MULTI_VOXEL_DOWNSAMPLE

# echo "✓ Multi-view data processed"


# # ################################################################################
# # ################################################################################
## new colmap and hloc script


# python -m localization.main_pipeline \
#     --pre "$PRECHANGE_OUTPUT_DIR" \
#     --post "$MULTI_OUTPUT"

# exit 0
# --- Activate TAPIP3D environment ---
CONDA_ENV="nerfstudio_env"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi




# ################################################################################
# # STEP 4: Train Static 3DGS with Splatfacto + Depth
# ################################################################################

# if [ "$SKIP_3DGS_TRAIN" != "true" ]; then
#     print_step "STEP 4: Training static 3DGS with qed-splatter "

#     ns-train qed-splatter --vis viewer \
#         --output-dir "$PRECHANGE_OUTPUT_DIR/splatfacto" \
#         --experiment-name "" \
#         --timestamp $CURRENT_TIME \
#         --steps_per_eval_all_images 500 \
#         --pipeline.model.cull_alpha_thresh 0.005 \
#         --max-num-iterations 20000 \
#         --pipeline.model.warmup_length 200 \
#         --machine.num-devices 1 \
#         --viewer.quit-on-train-completion True \
#         nerfstudio-data --data "$PRECHANGE_OUTPUT_DIR/transforms_colmap_metric.json" \
#         --depth_unit_scale_factor 1.0 \
#         --auto-scale-poses=False --center-method none --orientation-method none \
#         --load-3D-points True \
#         --train_split_fraction 0.9 \

# fi

# exit 0
    
#     echo "✓ Static 3DGS training complete"
#     echo ""
#     echo "IMPORTANT: Update PRETRAINED_CONFIG and CHECKPOINT_PATH variables with the new checkpoint path"
#     echo "New checkpoint should be in: $PRECHANGE_OUTPUT_DIR/splatfacto/qed-splatter/$CURRENT_TIME/"
# fi

#     ns-train splatfacto --vis viewer+wandb \
#         --output-dir "$PRECHANGE_OUTPUT_DIR/splatfacto" \
#         --experiment-name "" \
#         --timestamp $CURRENT_TIME \
#         --steps_per_eval_all_images 500 \
#         --pipeline.model.cull_alpha_thresh 0.005 \
#         --max-num-iterations 30000 \
#         --pipeline.model.warmup_length 200 \
#         --machine.num-devices 1 \
#         --viewer.quit-on-train-completion True \
#         nerfstudio-data --data "$PRECHANGE_OUTPUT_DIR/transforms_colmap.json" \
#         --depth_unit_scale_factor 1.0 \
#         --auto-scale-poses=False --center-method none --orientation-method none \
#         --load-3D-points True \
#         --train_split_fraction 0.9 \
    
#     echo "✓ Static 3DGS training complete"
#     echo ""
#     echo "IMPORTANT: Update PRETRAINED_CONFIG and CHECKPOINT_PATH variables with the new checkpoint path"
#     echo "New checkpoint should be in: $PRECHANGE_OUTPUT_DIR/splatfacto/qed-splatter/$CURRENT_TIME/"
# fi

# python debug_renders/compare_renders.py --data-dir  $MULTI_OUTPUT --trans $MULTI_OUTPUT/transforms_localized.json --out $MULTI_OUTPUT/debug_renders_arkit


# exit 0

################################################################################
# STEP 6: Change Detection Using DINO Features
################################################################################

print_step "STEP 6: Running 3DGS change detection"

# Validate required files
check_file_exists "$PRETRAINED_CONFIG"
check_file_exists "$MULTI_OUTPUT/transforms_reloc.json"

# mkdir -p "${CHANGE_OUTPUT}"

# python change_det/change_detection_sam.py \
#     --config "${PRETRAINED_CONFIG}" \
#     --output "${MULTI_OUTPUT}" \
#     --transform "${MULTI_OUTPUT}/transforms_reloc.json" \
#     --debug

# echo "✓ Change detection complete"


# exit 0
################################################################################
# STEP 7: SAM2 Video Object Segmentation
################################################################################

# print_step "STEP 7: Generating SAM2 masks via video object segmentation"

# Define paths
FRAMES_DIR="${MULTI_OUTPUT}/frames"
END_MASK_PATH="${MULTI_OUTPUT}/cd_mask"
OUTPUT_MASK_DIR="${MULTI_OUTPUT}/mask"


CONDA_ENV="sam2"


if [ -z "${CONDA_EXE:-}" ]; then
    echo "🔍 Locating conda base..."
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

echo "🔧 Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

SAM2_DIR="/local/home/pmishra/cvg/sam2"

# python "$SAM2_DIR/tools/run_sam2_vos_wrapper.py" \
#     --frames_dir "$FRAMES_DIR" \
#     --mask_path "$END_MASK_PATH" \
#     --output_mask_dir "$OUTPUT_MASK_DIR" \

echo "✓ SAM2 mask generation complete"
conda deactivate

# ###############################################################################
# STEP 8: TAPIP3D Inference
# ###############################################################################


TAPIP3D_DIR="/local/home/pmishra/cvg/TAPIP3D"
TAPIP3D_OUTPUT_DIR="${MULTI_OUTPUT}/tapip3d/"
TRANSFORMS_PATH="${MULTI_OUTPUT}/transforms_reloc.json"
MASK_SAMPLE="${OUTPUT_MASK_DIR}/frame_00001.png"

check_file_exists "$MASK_SAMPLE"
check_dir_exists "$TAPIP3D_DIR"


# --- Activate TAPIP3D environment ---
CONDA_ENV="tapip3d"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

set +u
conda activate "$CONDA_ENV"
set -u

TRANSFORMS_PATH=$(realpath "$MULTI_OUTPUT/transforms_reloc.json")
MASK_SAMPLE=$(realpath "$OUTPUT_MASK_DIR/frame_00001.png")
TAPIP_OUTPUT_DIR=$(realpath "$MULTI_OUTPUT/tapip3d")
mkdir -p "$TAPIP_OUTPUT_DIR"

cd "$TAPIP3D_DIR"
# PYTHONPATH="/local/home/pmishra/cvg/TAPIP3D" \
# python inference_nerf_mask.py \
#     --input_path "$TRANSFORMS_PATH" \
#     --mask_path "$MASK_SAMPLE" \
#     --output_dir "$TAPIP_OUTPUT_DIR" \
#     --max_query_points $MAX_QUERY_POINTS


# exit 0
################################################################################
# STEP 9: 4D RANSAC Joint Discovery
################################################################################

print_step "STEP 9: Running 4D RANSAC for joint parameter estimation"



# --- Activate TAPIP3D environment ---
CONDA_ENV="nerfstudio_env"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

set +u
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_ROOT"  
JOINT_OUTPUT_DIR="${MULTI_OUTPUT}/ransac_joints"
mkdir -p "$JOINT_OUTPUT_DIR"


# python -m joint_estimator.main \
#     --tapip3d_result "$TAPIP_OUTPUT_DIR" \
#     --out_dir "$JOINT_OUTPUT_DIR" \
#     --camera_metadata "$TRANSFORMS_PATH" \
#     --data_dir "$MULTI_OUTPUT" \
#     --use_temp

echo "✓ 4D RANSAC joint discovery complete: $JOINT_OUTPUT_DIR"


# ################################################################################
# # STEP 10: Merge Articulation Data into Transforms
# ################################################################################

print_step "STEP 10: Merging articulation data into transforms"

# python joint_estimator/merge_arti_data.py \
#     --joint_schemas "${JOINT_OUTPUT_DIR}/joint_schemas.json" \
#     --transforms_post "${MULTI_OUTPUT}/transforms_reloc.json" \
#     --output "${MULTI_OUTPUT}/transforms_post.json" \
#     --use_masks

# echo "✓ Articulation data merged"


################################################################################
# STEP 11: Train Articulated Splatfacto
################################################################################

print_step "STEP 11: Training articulated splatfacto model"


TRANSFORM_JSON="${MULTI_OUTPUT}/transforms_post.json"
OBJ_MASK_FILE="${MULTI_OUTPUT}/ransac_joints/obj_masks/obj_revolute.pt"
USE_DEPTH=True
DEPTH_SCALE=1.0

# ns-train arti_splatfacto \
#     --vis viewer+wandb \
#     --output-dir "${MULTI_OUTPUT}/artisplatfacto" \
#     --experiment-name "" \
#     --timestamp "$CURRENT_TIME" \
#     --steps-per-eval-all-images 100 \
#     --max-num-iterations 20000 \
#     --machine.num-devices 1 \
#     --viewer.quit-on-train-completion True \
#     --load-dir "$CHECKPOINT_PATH" \
#     --pipeline.model.obj-mask-file "$OBJ_MASK_FILE" \
#     --pipeline.model.use-depth $USE_DEPTH \
#     arti-dataparser \
#     --data "$MULTI_OUTPUT" \
#     --auto-scale-poses=False \
#     --center-method none \
#     --orientation-method none \
#     --train_split_fraction 0.9 \
#     --depth_unit_scale_factor $DEPTH_SCALE

# echo "✓ Articulated splatfacto training complete"


LATEST_TRAIN_RUN=$(ls -td "${MULTI_OUTPUT}/artisplatfacto/arti_splatfacto"/* | head -n 1)
CHECKPOINT_PATH="${LATEST_TRAIN_RUN}/nerfstudio_models"

echo "Using checkpoint from: $CHECKPOINT_PATH"

#Run recovery phase
# ns-train arti_splatfacto_recovery \
#     --vis viewer+wandb \
#     --output-dir "${MULTI_OUTPUT}/artisplatfacto_refined" \
#     --experiment-name "" \
#     --timestamp "$CURRENT_TIME" \
#     --steps-per-eval-all-images 100 \
#     --max-num-iterations 2000 \
#     --machine.num-devices 1 \
#     --viewer.quit-on-train-completion True \
#     --load-dir "$CHECKPOINT_PATH" \
#     --pipeline.model.obj-mask-file "$OBJ_MASK_FILE" \
#     --pipeline.model.use-depth "$USE_DEPTH" \
#     arti-dataparser \
#     --data "$MULTI_OUTPUT" \
#     --auto-scale-poses=False \
#     --center-method none \
#     --orientation-method none \
#     --train_split_fraction 0.9 \
#     --depth_unit_scale_factor "$DEPTH_SCALE"


# LATEST_TRAIN_RUN=$(ls -td "${MULTI_OUTPUT}/artisplatfacto_refined/arti_splatfacto_recovery"/* | head -n 1)
CHECKPOINT_PATH="${LATEST_TRAIN_RUN}/nerfstudio_models"

python arti_splatfacto/render.py "$LATEST_TRAIN_RUN/config.yml" --output_dir "${MULTI_OUTPUT}/renders" --render_depth




exit 0
################################################################################
# Pipeline Complete
################################################################################

print_step "PIPELINE COMPLETE!"


echo "Pipeline finished at: $(date)"


TRANSFORM_JSON="${MULTI_OUTPUT}/transforms_post.json"
OBJ_MASK_FILE="${MULTI_OUTPUT}/ransac_joints/obj_masks/obj_revolute.pt"
USE_DEPTH=True
DEPTH_SCALE=1.0

CHECKPOINT_PATH_2="/local/home/pmishra/cvg/arti-splatfacto/data_real/day8/post/artisplatfacto_refined/arti_splatfacto_recovery/2025-11-17_172007/nerfstudio_models"

ns-train arti_splatfacto \
    --vis viewer+wandb \
    --output-dir "${MULTI_OUTPUT}/artisplatfacto" \
    --experiment-name "" \
    --timestamp "$CURRENT_TIME" \
    --steps-per-eval-all-images 100 \
    --max-num-iterations 20000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    --load-dir "$CHECKPOINT_PATH_2" \
    --pipeline.model.obj-mask-file "$OBJ_MASK_FILE" \
    --pipeline.model.use-depth $USE_DEPTH \
    arti-dataparser \
    --data "$MULTI_OUTPUT" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train_split_fraction 0.9 \
    --depth_unit_scale_factor $DEPTH_SCALE

echo "✓ Articulated splatfacto training complete"


# Find the newest training run folder (sorted by modification time)
LATEST_TRAIN_RUN=$(ls -td "${MULTI_OUTPUT}/artisplatfacto/arti_splatfacto"/* | head -n 1)
CHECKPOINT_PATH="${LATEST_TRAIN_RUN}/nerfstudio_models"

echo "Using checkpoint from: $CHECKPOINT_PATH"

Run recovery phase
ns-train arti_splatfacto_recovery \
    --vis viewer+wandb \
    --output-dir "${MULTI_OUTPUT}/artisplatfacto_refined" \
    --experiment-name "" \
    --timestamp "$CURRENT_TIME" \
    --steps-per-eval-all-images 100 \
    --max-num-iterations 2500 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    --load-dir "$CHECKPOINT_PATH" \
    --pipeline.model.obj-mask-file "$OBJ_MASK_FILE" \
    --pipeline.model.use-depth "$USE_DEPTH" \
    arti-dataparser \
    --data "$MULTI_OUTPUT" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train_split_fraction 0.9 \
    --depth_unit_scale_factor "$DEPTH_SCALE"