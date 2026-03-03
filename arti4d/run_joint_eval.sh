#!/bin/bash
set -uo pipefail   # do NOT use -e (we don’t want batch to stop on eval failure)

# Usage:
#   ./run_joint_eval.sh <scene_dir>

SCENE_DIR="$1"

if [ -z "${SCENE_DIR:-}" ]; then
    echo "Usage: ./run_joint_eval.sh <scene_dir>"
    exit 1
fi

SCENE_NAME="$(basename "$SCENE_DIR")"
SCENE_TYPE="$(basename "$(dirname "$SCENE_DIR")")"

GT_JSON="${SCENE_DIR}/GT_joint_info.json"
PRED_ROOT="${SCENE_DIR}"

JSON_OUT="${SCENE_DIR}/joint_eval_summary.json"
TXT_OUT="${SCENE_DIR}/joint_eval_summary.txt"

if [[ ! -f "$GT_JSON" ]]; then
    echo "No GT_joint_info.json found in $SCENE_DIR"
    exit 1
fi

echo "======================================================="
echo "Running joint evaluation"
echo "Scene: ${SCENE_TYPE}/${SCENE_NAME}"
echo "======================================================="

if python arti4d/eval_joints.py \
        --gt_json   "$GT_JSON" \
        --pred_root "$PRED_ROOT" \
        --json_out  "$JSON_OUT" \
        | tee "$TXT_OUT"; then

    echo "✓ Evaluation complete: ${SCENE_TYPE}/${SCENE_NAME}"
    echo "  JSON: $JSON_OUT"
    echo "  TXT : $TXT_OUT"

else
    echo "✗ Evaluation failed: ${SCENE_TYPE}/${SCENE_NAME}"
fi