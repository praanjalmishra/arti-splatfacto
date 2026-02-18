#!/bin/bash
################################################################################
# Run V2A evaluation on all scenes
################################################################################

set -e

DATA_ROOT="/local/home/pmishra/VideoArtGS/data/v2a"
GPU_DEVICE=${1:-0}

SCENES=(
    100068_joint_0_bg_view_0
    100071_joint_0_bg_view_1
    100072_joint_0_bg_view_0
    100087_joint_0_bg_view_0
    100092_joint_0_bg_view_0
    100106_joint_0_bg_view_1
    100128_joint_0_bg_view_1
    100133_joint_0_bg_view_0
    10040_joint_1_bg_view_0
    100664_joint_0_bg_view_1
    10143_joint_0_bg_view_1
    101886_joint_0_bg_view_0
    101948_joint_0_bg_view_1
    10306_joint_1_bg_view_1
    103273_joint_1_bg_view_1
    103283_joint_1_bg_view_1
    103351_joint_0_bg_view_0
    103490_joint_0_bg_view_0
    103528_joint_0_bg_view_1
    10356_joint_0_bg_view_1
    10373_joint_0_bg_view_0
    103778_joint_0_bg_view_0
    10495_joint_0_bg_view_1
    10558_joint_0_bg_view_0
    10559_joint_0_bg_view_1
    10567_joint_0_bg_view_0
    10867_joint_0_bg_view_1
    10895_joint_1_bg_view_0
    10973_joint_0_bg_view_1
    11304_joint_0_bg_view_0
    11304_joint_1_bg_view_0
    11700_joint_0_bg_view_1
    12071_joint_0_bg_view_0
    12531_joint_0_bg_view_1
    12552_joint_0_bg_view_0
    12552_joint_0_bg_view_1
    12562_joint_0_bg_view_1
    12565_joint_1_bg_view_0
    19179_joint_1_bg_view_1
    19855_joint_0_bg_view_1
    19898_joint_1_bg_view_1
    19898_joint_3_bg_view_0
    19898_joint_4_bg_view_1
    20745_joint_0_bg_view_1
    20985_joint_1_bg_view_0
    22241_joint_0_bg_view_1
    22339_joint_0_bg_view_1
    22367_joint_0_bg_view_0
    22367_joint_2_bg_view_0
    22433_joint_0_bg_view_0
    22433_joint_0_bg_view_1
    22433_joint_1_bg_view_0
    23372_joint_1_bg_view_1
    23724_joint_0_bg_view_1
    23724_joint_2_bg_view_0
    23807_joint_1_bg_view_1
    26525_joint_0_bg_view_1
    26608_joint_0_bg_view_1
    26657_joint_1_bg_view_0
    27267_joint_0_bg_view_1
    35059_joint_0_bg_view_0
    40453_joint_1_bg_view_1
    41083_joint_3_bg_view_1
    41510_joint_1_bg_view_1
    44781_joint_0_bg_view_1
    44817_joint_1_bg_view_1
    44962_joint_2_bg_view_1
    45001_joint_1_bg_view_0
    45132_joint_2_bg_view_0
    45146_joint_0_bg_view_0
    7265_joint_0_bg_view_0
    7265_joint_0_bg_view_1
    9987_joint_0_bg_view_1
)


LOG_DIR="outputs/v2a_eval/logs"
mkdir -p "$LOG_DIR"

TOTAL=${#SCENES[@]}
SUCCESS=0
FAILED=0

for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    NUM=$((i + 1))
    
    echo ""
    echo "========================================================================"
    echo "[$NUM/$TOTAL] Processing: $SCENE"
    echo "========================================================================"
    
    LOG_FILE="${LOG_DIR}/${SCENE}.log"
    
    if bash v2a_eval/run_single_scene.sh "$SCENE" "$GPU_DEVICE" > "$LOG_FILE" 2>&1; then
        echo "✓ SUCCESS: $SCENE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "✗ FAILED: $SCENE (see $LOG_FILE)"
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