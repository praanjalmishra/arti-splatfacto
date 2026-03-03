#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_all_joints.sh <scene_dir> [cuda_device]

SCENE_DIR="$1"
CUDA_DEVICE=${2:-0}

if [ -z "${SCENE_DIR:-}" ]; then
    echo "Usage: ./run_all_joints.sh <scene_dir> [cuda_device]"
    exit 1
fi

SKIP_JOINTS=( )

shopt -s nullglob

for dir in "$SCENE_DIR"/articulated_joint_*; do
    [[ -d "$dir" ]] || continue

    j="${dir##*_}"

    if [[ " ${SKIP_JOINTS[*]} " =~ " ${j} " ]]; then
        echo "Skipping joint $j"
        continue
    fi

    echo "===================================================="
    echo "Processing joint $j"
    echo "Scene: $SCENE_DIR"
    echo "GPU:   $CUDA_DEVICE"
    echo "===================================================="

    # Resume protection: skip if articulation training already exists
    if compgen -G "$dir/arti_splatfacto/*/nerfstudio_models/*.ckpt" > /dev/null; then
        echo "Joint $j already trained → skipping"
        continue
    fi


    ./arti4d/run_pipeline.sh "$SCENE_DIR" "$j"


    ./arti4d/run_training.sh "$SCENE_DIR" "$j" "$CUDA_DEVICE"

done

echo "All joints complete for: $SCENE_DIR"

echo ""
echo "===================================================="
echo "Running joint estimation evaluation"
echo "===================================================="

if ./arti4d/run_joint_eval.sh "$SCENE_DIR"; then
    echo "✓ Evaluation finished for: $SCENE_DIR"
else
    echo "✗ Evaluation failed for: $SCENE_DIR"
fi