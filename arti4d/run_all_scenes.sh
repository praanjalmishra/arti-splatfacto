#!/bin/bash
set -uo pipefail   # do NOT use -e (we want to continue on failure)

ROOT="/media/pmishra/SANDISK1TB/arti4d/arti4d_processed"
LOG_DIR="/media/pmishra/SANDISK1TB/arti4d/arti4d_processed/logs_arti4d_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

# Collect all scenes automatically
SCENES=()
for scene_type in "$ROOT"/*; do
    [[ -d "$scene_type" ]] || continue
    for scene_dir in "$scene_type"/scene_*; do
        [[ -d "$scene_dir" ]] || continue
        SCENES+=("$scene_dir")
    done
done

TOTAL=${#SCENES[@]}
SUCCESS=0
FAILED=0

echo "Starting ARTi4D batch pipeline"
echo "Total scenes: $TOTAL"
echo "Logs directory: $LOG_DIR"
echo ""

for i in "${!SCENES[@]}"; do
    SCENE_DIR="${SCENES[$i]}"
    NUM=$((i + 1))

    SCENE_NAME="$(basename "$SCENE_DIR")"
    SCENE_TYPE="$(basename "$(dirname "$SCENE_DIR")")"

    ID="${SCENE_TYPE}_${SCENE_NAME}"

    echo ""
    echo "========================================================================"
    echo "[$NUM/$TOTAL] Processing: $ID"
    echo "========================================================================"

    LOG_FILE="${LOG_DIR}/${ID}.log"

    if bash ./arti4d/run_all_joints.sh "$SCENE_DIR" > "$LOG_FILE" 2>&1; then
        echo "✓ SUCCESS: $ID"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ FAILED: $ID (see $LOG_FILE)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================================================"
echo "Batch Complete"
echo "========================================================================"
echo "Success: $SUCCESS/$TOTAL"
echo "Failed:  $FAILED/$TOTAL"
echo "Logs:    $LOG_DIR/"

SUMMARY_FILE="${LOG_DIR}/batch_summary.txt"

{
echo "ARTi4D Batch Summary"
echo "Date: $(date)"
echo ""
echo "Success: $SUCCESS/$TOTAL"
echo "Failed:  $FAILED/$TOTAL"
echo ""
echo "Failed scenes:"
grep "✗ FAILED" "$LOG_DIR"/*.log 2>/dev/null || true
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"



echo ""
echo "========================================================================"
echo "Running GLOBAL joint evaluation aggregation"
echo "========================================================================"

GLOBAL_TXT="${LOG_DIR}/global_joint_eval_summary.txt"
GLOBAL_CSV="${LOG_DIR}/global_joint_eval_summary.csv"

if python ./arti4d/aggregate_global_eval.py \
        --root "$ROOT" \
        --out_txt "$GLOBAL_TXT" \
        --out_csv "$GLOBAL_CSV"; then

    echo ""
    echo "✓ Global evaluation complete"
    echo "  TXT : $GLOBAL_TXT"
    echo "  CSV : $GLOBAL_CSV"

else
    echo ""
    echo "✗ Global aggregation failed"
fi