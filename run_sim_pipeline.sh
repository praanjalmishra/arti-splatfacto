#!/bin/bash

################################################################################
# Articulated 3D Reconstruction Pipeline - SIM DATA VERSION
# Process synthetic PartNet-Mobility data through artisplatfacto pipeline
################################################################################

set -e  # Exit on error

################################################################################
# Configuration
################################################################################

# Parse command line arguments
PARTNET_OBJ=${1:-"35059"}  # PartNet object ID
CUDA_DEVICE=${2:-0}
SKIP_RENDERING=${3:-false}  # Set to 'true' to skip rendering if data exists
SKIP_3DGS_TRAIN=${4:-false}  # Set to 'true' to skip 3DGS training

# Project paths
PROJECT_ROOT="/local/home/pmishra/cvg/arti-splatfacto"
PARTNET_ROOT="data_itaco/video2articulation/partnet-mobility-v0"  # UPDATE THIS PATH
OUTPUT_FOLDER="${PROJECT_ROOT}/outputs_sim"

# Environment setup
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/EfficientSAM"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/sam2"
export PYTHONPATH="${PYTHONPATH}:/local/home/pmishra/cvg/TAPIP3D"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Timestamp for runs
CURRENT_TIME=$(date +%Y-%m-%d_%H%M%S)

################################################################################
# Sim Data Parameters
################################################################################

# PartNet object paths
PARTNET_DIR="${PARTNET_ROOT}/${PARTNET_OBJ}"
PARTNET_URDF="${PARTNET_DIR}/mobility.urdf"
PARTNET_META="${PARTNET_DIR}/meta_${PARTNET_OBJ}.json"

# Output directories
SIM_OUTPUT_DIR="${OUTPUT_FOLDER}/${PARTNET_OBJ}"
PRECHANGE_OUTPUT_DIR="${SIM_OUTPUT_DIR}/pre"
POSTCHANGE_OUTPUT_DIR="${SIM_OUTPUT_DIR}/post"

# Rendering parameters
JOINT_ID=0
N_FRAMES=200  # Number of frames in articulation sequence

# Pipeline parameters
MAX_QUERY_POINTS=100

################################################################################
# Helper Functions
################################################################################

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
# STEP 0: Validate PartNet Data
################################################################################

print_step "STEP 0: Validating PartNet-Mobility data"

check_dir_exists "$PARTNET_DIR"
check_file_exists "$PARTNET_URDF"
check_file_exists "$PARTNET_META"

echo "✓ PartNet object validated: ${PARTNET_OBJ}"

################################################################################
# STEP 1: Render Synthetic Data
################################################################################

if [ "$SKIP_RENDERING" != "true" ]; then
    print_step "STEP 1: Rendering synthetic PartNet-Mobility data"
    
    # python render_viewpoints.py \
    #     --partnet_dir "$PARTNET_DIR" \
    #     --meta "$PARTNET_META" \
    #     --output_dir "$SIM_OUTPUT_DIR" \
    #     --joint_id $JOINT_ID \
    #     --n_frames $N_FRAMES
    
    echo "✓ Rendering complete"
    echo "  Pre-change data: ${SIM_OUTPUT_DIR}/pre/"
    echo "  Post-change data: ${SIM_OUTPUT_DIR}/post/"
else
    print_step "STEP 1: Skipping rendering (using existing data)"
    
    check_dir_exists "${SIM_OUTPUT_DIR}/pre"
    check_dir_exists "${SIM_OUTPUT_DIR}/post"
    check_file_exists "${SIM_OUTPUT_DIR}/pre/transforms.json"
    check_file_exists "${SIM_OUTPUT_DIR}/post/transforms_post.json"
fi

################################################################################
# STEP 2: Convert to Pipeline Format (if needed)
################################################################################

print_step "STEP 2: Preparing data for pipeline"

# The rendering script already outputs in the correct format, but we may need
# to add some compatibility conversions here

# Ensure the transforms files have the right structure
python - <<EOF
import json
from pathlib import Path

# Load pre-change transforms
pre_path = Path("${SIM_OUTPUT_DIR}/pre/transforms.json")
with open(pre_path) as f:
    pre_data = json.load(f)

# Ensure required fields exist
if 'frames' not in pre_data:
    raise ValueError("Pre-change transforms missing 'frames' field")

# Verify depth files exist
frames_with_depth = []
for frame in pre_data['frames']:
    depth_file_path = Path("${SIM_OUTPUT_DIR}/pre") / frame['depth_file_path']
    if depth_file_path.exists():
        frames_with_depth.append(frame)
    else:
        print(f"Warning: Depth file missing: {depth_file_path}")

print(f"✓ Pre-change: {len(frames_with_depth)}/{len(pre_data['frames'])} frames with depth")

# Load post-change transforms
post_path = Path("${SIM_OUTPUT_DIR}/post/transforms_post.json")
with open(post_path) as f:
    post_data = json.load(f)

# Verify post-change structure
if 'frames' not in post_data:
    raise ValueError("Post-change transforms missing 'frames' field")
    
if 'articulations_gt' not in post_data:
    print("Warning: No articulations_gt field found")

print(f"✓ Post-change: {len(post_data['frames'])} frames")
print(f"✓ Ground truth articulations available: {'articulations_gt' in post_data}")

EOF

echo "✓ Data validation complete"

################################################################################
# STEP 3: Camera Registration (Sim data already aligned)
################################################################################

print_step "STEP 3: Camera registration"

# For sim data, cameras are already in the same coordinate frame
# Copy post transforms as "reloc" transforms for pipeline compatibility
cp "${SIM_OUTPUT_DIR}/post/transforms_post.json" \
   "${SIM_OUTPUT_DIR}/post/transforms_reloc.json"

# echo "✓ Camera registration skipped (sim data already aligned)"

################################################################################
# STEP 4: Train Static 3DGS with Splatfacto + Depth
################################################################################

if [ "$SKIP_3DGS_TRAIN" != "true" ]; then
    print_step "STEP 4: Training static 3DGS with qed-splatter"

    # ns-train qed-splatter --vis viewer \
    #     --output-dir "$PRECHANGE_OUTPUT_DIR/splatfacto" \
    #     --experiment-name "sim_${PARTNET_OBJ}" \
    #     --timestamp $CURRENT_TIME \
    #     --steps_per_eval_all_images 500 \
    #     --pipeline.model.cull_alpha_thresh 0.005 \
    #     --pipeline.model.cull_scale_thresh 0.1 \
    #     --max-num-iterations 6000 \
    #     --pipeline.model.warmup_length 200 \
    #     --pipeline.model.use_scale_regularization True \
    #     --machine.num-devices 1 \
    #     --viewer.quit-on-train-completion True \
    #     nerfstudio-data --data "$PRECHANGE_OUTPUT_DIR/transforms.json" \
    #     --depth_unit_scale_factor 1.0 \
    #     --auto-scale-poses False \
    #     --center-method none \
    #     --orientation-method none \
    #     --load-3D-points True \
    #     --train_split_fraction 0.9

    # Find the checkpoint directory
    CHECKPOINT_DIR=$(find "$PRECHANGE_OUTPUT_DIR/splatfacto/sim_${PARTNET_OBJ}/qed-splatter" -type f -name "config.yml" | head -1 | xargs dirname)
    PRETRAINED_CONFIG="${CHECKPOINT_DIR}/config.yml"
    VANILLA_GS_CHECKPOINT="${CHECKPOINT_DIR}/nerfstudio_models"
    
    echo "✓ 3DGS training complete"
    echo "  Config: $PRETRAINED_CONFIG"
else
    print_step "STEP 4: Skipping 3DGS training"
    
    # Find existing checkpoint
    CHECKPOINT_DIR=$(find "$PRECHANGE_OUTPUT_DIR/splatfacto/${PARTNET_OBJ}/qed-splatter" -type f -name "config.yml" | head -1 | xargs dirname)
    PRETRAINED_CONFIG="${CHECKPOINT_DIR}/config.yml"
    check_file_exists "$PRETRAINED_CONFIG"
fi

################################################################################
# STEP 5: Change Detection Using Pre-trained 3DGS
################################################################################

print_step "STEP 5: Running 3DGS-based change detection"

check_file_exists "$PRETRAINED_CONFIG"

# python change_det/change_detection_sam.py \
#     --config "${PRETRAINED_CONFIG}" \
#     --output "${POSTCHANGE_OUTPUT_DIR}" \
#     --transform "${POSTCHANGE_OUTPUT_DIR}/transforms_reloc.json" \
#     --debug

# echo "✓ Change detection complete"

################################################################################
# STEP 6: SAM2 Video Object Segmentation
################################################################################

print_step "STEP 6: Generating SAM2 masks via video object segmentation"

# Define paths
FRAMES_DIR="${POSTCHANGE_OUTPUT_DIR}/frames"
END_MASK_PATH="${POSTCHANGE_OUTPUT_DIR}/cd_mask"
OUTPUT_MASK_DIR="${POSTCHANGE_OUTPUT_DIR}/mask"

# Activate SAM2 environment
CONDA_ENV="sam2"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV"

SAM2_DIR="/local/home/pmishra/cvg/sam2"

# python "$SAM2_DIR/tools/run_sam2_vos_wrapper.py" \
#     --frames_dir "$FRAMES_DIR" \
#     --mask_path "$END_MASK_PATH" \
#     --output_mask_dir "$OUTPUT_MASK_DIR"

# echo "✓ SAM2 mask generation complete"
# conda deactivate

################################################################################
# STEP 7: TAPIP3D Point Tracking
################################################################################

print_step "STEP 7: Running TAPIP3D for point tracking"

TAPIP3D_DIR="/local/home/pmishra/cvg/TAPIP3D"
TAPIP3D_OUTPUT_DIR="${POSTCHANGE_OUTPUT_DIR}/tapip3d/"
TRANSFORMS_PATH="${POSTCHANGE_OUTPUT_DIR}/transforms_reloc.json"
MASK_SAMPLE="${OUTPUT_MASK_DIR}/frame_00001.png"

check_file_exists "$MASK_SAMPLE"
check_dir_exists "$TAPIP3D_DIR"

# Activate TAPIP3D environment
CONDA_ENV="tapip3d"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV"

TRANSFORMS_PATH=$(realpath "$POSTCHANGE_OUTPUT_DIR/transforms_reloc.json")
MASK_SAMPLE=$(realpath "$OUTPUT_MASK_DIR/frame_00001.png")
TAPIP_OUTPUT_DIR=$(realpath "$POSTCHANGE_OUTPUT_DIR/tapip3d")
mkdir -p "$TAPIP_OUTPUT_DIR"

cd "$TAPIP3D_DIR"
PYTHONPATH="/local/home/pmishra/cvg/TAPIP3D" \
# python inference_nerf_mask.py \
#     --input_path "$TRANSFORMS_PATH" \
#     --mask_path "$MASK_SAMPLE" \
#     --output_dir "$TAPIP_OUTPUT_DIR" \
#     --max_query_points $MAX_QUERY_POINTS

conda deactivate
cd "$PROJECT_ROOT"

echo "✓ TAPIP3D tracking complete"

################################################################################
# STEP 8: 4D RANSAC Joint Discovery
################################################################################

print_step "STEP 8: Running 4D RANSAC for joint parameter estimation"

# Activate nerfstudio environment
CONDA_ENV="nerfstudio_env"
if [ -z "${CONDA_EXE:-}" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV"

JOINT_OUTPUT_DIR="${POSTCHANGE_OUTPUT_DIR}/ransac_joints"
mkdir -p "$JOINT_OUTPUT_DIR"

# python -m joint_estimator.main \
#     --tapip3d_result "$TAPIP_OUTPUT_DIR" \
#     --out_dir "$JOINT_OUTPUT_DIR" \
#     --camera_metadata "$TRANSFORMS_PATH" \
#     --data_dir "$POSTCHANGE_OUTPUT_DIR" \
#     # --use_temp

# echo "✓ 4D RANSAC joint discovery complete"

# ################################################################################
# # STEP 9: Merge Articulation Data into Transforms
# ################################################################################

# print_step "STEP 9: Merging articulation data into transforms"

# python joint_estimator/merge_arti_data.py \
#     --joint_schemas "${JOINT_OUTPUT_DIR}/joint_schemas.json" \
#     --transforms_post "${POSTCHANGE_OUTPUT_DIR}/transforms_reloc.json" \
#     --output "${POSTCHANGE_OUTPUT_DIR}/transforms_post.json" \
#     --use_masks

# echo "✓ Articulation data merged"

################################################################################
# STEP 10: Evaluate Against Ground Truth (Sim Data Only)
################################################################################

# print_step "STEP 10: Evaluating against ground truth"

# python - <<EOF
# import json
# import numpy as np
# from pathlib import Path

# # Load ground truth
# gt_path = Path("${SIM_OUTPUT_DIR}/post/transforms_post.json")
# with open(gt_path) as f:
#     gt_data = json.load(f)

# # Load predicted
# pred_path = Path("${POSTCHANGE_OUTPUT_DIR}/transforms_post_final.json")
# with open(pred_path) as f:
#     pred_data = json.load(f)

# print("\n" + "="*80)
# print("GROUND TRUTH vs PREDICTED COMPARISON")
# print("="*80)

# if 'articulations_gt' in gt_data:
#     gt_arti = gt_data['articulations_gt'][0]
    
#     print("\nGround Truth:")
#     print(f"  Joint Type: {gt_arti['joint_type_gt']}")
#     print(f"  Joint Axis: {gt_arti['joint_axis_gt']}")
#     print(f"  Joint Pivot: {gt_arti['joint_pivot_gt']}")
#     print(f"  Joint Limits: {gt_arti['joint_limits_gt']}")
    
#     if 'articulations' in pred_data:
#         pred_arti = pred_data['articulations'][0]
        
#         print("\nPredicted:")
#         print(f"  Joint Type: {pred_arti.get('joint_type', 'N/A')}")
#         print(f"  Joint Axis: {pred_arti.get('joint_axis', 'N/A')}")
#         print(f"  Joint Pivot: {pred_arti.get('joint_pivot', 'N/A')}")
        
#         # Compute errors
#         gt_axis = np.array(gt_arti['joint_axis_gt'])
#         pred_axis = np.array(pred_arti.get('joint_axis', [0, 0, 1]))
#         axis_error = np.arccos(np.clip(np.dot(gt_axis, pred_axis), -1, 1))
        
#         gt_pivot = np.array(gt_arti['joint_pivot_gt'])
#         pred_pivot = np.array(pred_arti.get('joint_pivot', [0, 0, 0]))
#         pivot_error = np.linalg.norm(gt_pivot - pred_pivot)
        
#         print("\nErrors:")
#         print(f"  Axis angular error: {np.rad2deg(axis_error):.2f}°")
#         print(f"  Pivot position error: {pivot_error:.4f} units")
#     else:
#         print("\nNo predicted articulations found!")
# else:
#     print("\nNo ground truth articulations available")

# print("="*80 + "\n")
# EOF

################################################################################
# Pipeline Complete
################################################################################

print_step "Pipeline Complete!"

echo "Results saved to: ${SIM_OUTPUT_DIR}"
echo ""
echo "Directory structure:"
echo "  ${SIM_OUTPUT_DIR}/"
echo "  ├── pre/                    # Static pre-change data"
echo "  │   ├── frames/             # RGB images"
echo "  │   ├── depth/              # Depth maps (.npy)"
echo "  │   ├── transforms.json     # Camera poses + intrinsics"
echo "  │   └── fused_pc.ply        # Initial point cloud"
echo "  └── post/                   # Dynamic articulation sequence"
echo "      ├── frames/             # RGB images"
echo "      ├── depth/              # Depth maps (.npy)"
echo "      ├── mask_gt/            # Ground truth masks"
echo "      ├── mask/               # SAM2 predicted masks"
echo "      ├── tapip3d/            # Point tracks"
echo "      ├── ransac_joints/      # Estimated joint parameters"
echo "      └── transforms_post_final.json  # Final output with articulations"
echo ""
echo "Next steps:"
echo "  1. Visualize results with your viewer"
echo "  2. Train articulated 3DGS with the final transforms"
echo "  3. Compare against ground truth metrics"




USE_DEPTH=True
DEPTH_SCALE=1.0
TRAIN_ITR=15000
STOP_SPLIT_AT=14000
TRAIN_ITR_RECOVERY=2500

# # Pre-trained checkpoint (vanilla Splatfacto)
# PRECHANGE_OUTPUT_DIR="data_real/scene3/pre"

# # # Pre-trained checkpoint (update after first run)
# PRETRAINED_CONFIG="${PRECHANGE_OUTPUT_DIR}/splatfacto/qed-splatter/2025-12-01_160950/config.yml"
# VANILLA_GS_CHECKPOINT="${PRECHANGE_OUTPUT_DIR}/splatfacto/qed-splatter/2025-12-01_160950/nerfstudio_models"

# # Joint 0 settings
# JOINT_0_DATA="data_real/scene3/prism"
JOINT_3D_MASK="${POSTCHANGE_OUTPUT_DIR}/ransac_joints/obj_masks/obj_revolute.pt"

# # Joint 1 settings
# JOINT_1_DATA="data_real/scene3/arti_2"
# JOINT_1_MASK="${JOINT_1_DATA}/ransac_joints/obj_masks/obj_revolute.pt"


################################################################################
# PHASE 1: SEQUENTIAL ARTICULATION (Geometry + Motion Learning)
################################################################################

print_step "PHASE 1: SEQUENTIAL ARTICULATION TRAINING"

# ---------------------------------------------------------------------------
# Joint 0 - Articulation (Partition from vanilla GS)
# ---------------------------------------------------------------------------
print_step "Stage 1.1: Joint 0 Articulation Training"

ns-train arti_splatfacto \
    --vis viewer+wandb \
    --output-dir "${POSTCHANGE_OUTPUT_DIR}" \
    --experiment-name "joint_0_articulation" \
    --timestamp "$CURRENT_TIME" \
    --steps-per-eval-all-images 100 \
    --max-num-iterations $TRAIN_ITR \
    --machine.num-devices 1 \
    --load-dir "$VANILLA_GS_CHECKPOINT" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.stop-split-at $STOP_SPLIT_AT \
    --pipeline.model.obj-mask-file "$JOINT_3D_MASK" \
    --pipeline.model.use-depth $USE_DEPTH \
    --pipeline.model.active-joint-id "joint_0" \
    --pipeline.model.training-mode "articulation" \
    --pipeline.model.cull_alpha_thresh 0.005 \
    arti-dataparser \
    --data "$POSTCHANGE_OUTPUT_DIR" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train-split-fraction 0.95 \
    --depth-unit-scale-factor $DEPTH_SCALE

echo "✓ Joint 0 articulation complete"

# Get Joint 0 articulation checkpoint
JOINT_0_ARTIC_RUN=$(ls -td "${POSTCHANGE_OUTPUT_DIR}/joint_0_articulation/arti_splatfacto"/* | head -n 1)
JOINT_0_ARTIC_CHECKPOINT="${JOINT_0_ARTIC_RUN}/nerfstudio_models"
JOINT_0_ARTIC_CONFIG="${JOINT_0_ARTIC_RUN}/config.yml"

if [ ! -d "$JOINT_0_ARTIC_CHECKPOINT" ]; then
    echo "❌ ERROR: Joint 0 articulation checkpoint not found!"
    exit 1
fi

echo "Joint 0 articulation checkpoint: $JOINT_0_ARTIC_CHECKPOINT"
echo "Joint 0 articulation config: $JOINT_0_ARTIC_CONFIG"

# ---------------------------------------------------------------------------
# Joint 0 - Recovery (Refine Joint 0's appearance)
# ---------------------------------------------------------------------------
print_step "Stage 1.2: Joint 0 Recovery Training"

ns-train arti_splatfacto_recovery \
    --vis viewer+wandb \
    --output-dir "${POSTCHANGE_OUTPUT_DIR}" \
    --experiment-name "joint_0_recovery" \
    --timestamp "$CURRENT_TIME" \
    --steps-per-eval-all-images 100 \
    --max-num-iterations $TRAIN_ITR_RECOVERY \
    --machine.num-devices 1 \
    --load-dir "$JOINT_0_ARTIC_CHECKPOINT" \
    --viewer.quit-on-train-completion True \
    --pipeline.model.obj-mask-file "$JOINT_0_MASK" \
    --pipeline.model.use-depth $USE_DEPTH \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.depth-lambda 0.5 \
    --pipeline.model.active-joint-id "joint_0" \
    --pipeline.model.training-mode "recovery" \
    --pipeline.model.cull_alpha_thresh 0.005 \
    arti-dataparser \
    --data "$JOINT_0_DATA" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --train-split-fraction 0.9 \
    --depth-unit-scale-factor $DEPTH_SCALE

echo "✓ Joint 0 recovery complete"

# Get Joint 0 final checkpoint
JOINT_0_FINAL_RUN=$(ls -td "${POSTCHANGE_OUTPUT_DIR}/joint_0_articulation/arti_splatfacto_recovery"/* | head -n 1)
JOINT_0_FINAL_CHECKPOINT="${JOINT_0_FINAL_RUN}/nerfstudio_models"
JOINT_0_FINAL_CONFIG="${JOINT_0_FINAL_RUN}/config.yml"

if [ ! -d "$JOINT_0_FINAL_CHECKPOINT" ]; then
    echo "❌ ERROR: Joint 0 recovery checkpoint not found!"
    exit 1
fi

echo "Joint 0 final checkpoint: $JOINT_0_FINAL_CHECKPOINT"
echo "Joint 0 final config: $JOINT_0_FINAL_CONFIG"

print_step "✓ Joint 0 Complete (Articulation + Recovery)"