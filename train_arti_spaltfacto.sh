#!/bin/bash

# ------------------ Default Argument Values ------------------
DATA_FOLDER=${1:-"/local/home/pmishra/cvg/3dgscd/data/sim_data/gs_sim_2"}
CUDA_VISIBLE_DEVICES=${2:-0}
POST_TRAIN_IDX=${3:-"7 10 12 14 15 18 20"}  # Optional: list of image indices used for post-finetuning
OUTPUT_FOLDER=${4:-"/local/home/pmishra/cvg/arti-splatfacto/outputs"}
NERFSTUDIO_FOLDER=${5:-"/local/home/pmishra/nerfstudio"}
MERGE_SCRIPT=${6:-"$NERFSTUDIO_FOLDER/scripts/merge_sim_data.py"}
EDIT_SCRIPT=${7:-"$NERFSTUDIO_FOLDER/scripts/edit_sim_data.py"}

# ------------------ Environment Setup ------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerfstudio_env || { echo "Failed to activate conda env"; exit 1; }

cd "$NERFSTUDIO_FOLDER" || { echo "Failed to cd into Nerfstudio folder"; exit 1; }

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
current_time=$(date +%Y-%m-%d_%H%M%S)

# ------------------ Paths ------------------
# TRANSFORM_JSON="$DATA_FOLDER/transforms_finetune.json"
TRANSFORM_JSON="$DATA_FOLDER/transforms_finetune_with_timestamp.json"
OBJ_MASK_FILE="$DATA_FOLDER/obj3Dseg0_updated.pt"
CHECKPOINT_PATH="/local/home/pmishra/nerfstudio/outputs/gs_sim_2/splatfacto/2025-07-12_162025/nerfstudio_models"

# Check existence of important files
if [[ ! -f "$TRANSFORM_JSON" ]]; then
    echo "[ERROR] transforms_finetune.json not found at $TRANSFORM_JSON"
    exit 1
fi

if [[ ! -f "$OBJ_MASK_FILE" ]]; then
    echo "[ERROR] Object mask file not found at $OBJ_MASK_FILE"
    exit 1
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "[ERROR] Checkpoint directory not found at $CHECKPOINT_PATH"
    exit 1
fi

# ------------------ Launch Training ------------------
ns-train arti_splatfacto \
    --vis viewer+wandb \
    --experiment-name "$(basename "$DATA_FOLDER")"_finetune \
    --output-dir "$OUTPUT_FOLDER" \
    --timestamp "$current_time" \
    --steps-per-eval-all-images 500 \
    --pipeline.model.cull-alpha-thresh 0.001 \
    --pipeline.model.densify-grad-thresh 0.0001 \
    --pipeline.model.obj-mask-file "$OBJ_MASK_FILE" \
    --max-num-iterations 30000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    --load-dir "$CHECKPOINT_PATH" \
    nerfstudio-data \
    --data "$TRANSFORM_JSON" \
    --auto-scale-poses=False \
    --center-method none \
    --orientation-method none \
    --load-3D-points False
