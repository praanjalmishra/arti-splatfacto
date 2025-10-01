#!/bin/bash

DATA_DIR="/local/home/pmishra/cvg/arti-splatfacto/data_ransac/sync_data/multiview"

echo "Testing refinement..."
echo ""

# Test on first 20 frames only (for quick validation)
# Create a temporary subset directory
TEMP_DIR="${DATA_DIR}/test_subset"
mkdir -p "${TEMP_DIR}/rgb"
mkdir -p "${TEMP_DIR}/masks_pre"
mkdir -p "${TEMP_DIR}/masks_post"

# Copy first 20 frames
echo "Creating test subset (20 frames)..."
for i in $(seq 0 299); do
    frame=$(printf "frame_%06d.png" $i)
    
    if [ -f "${DATA_DIR}/rgb/${frame}" ]; then
        cp "${DATA_DIR}/rgb/${frame}" "${TEMP_DIR}/rgb/"
        cp "${DATA_DIR}/masks_pre/${frame}" "${TEMP_DIR}/masks_pre/"
        cp "${DATA_DIR}/masks_post/${frame}" "${TEMP_DIR}/masks_post/"
    fi
done

echo "Running Stage 1 on PRE masks (test subset)..."
python mask_refinement.py \
    "${TEMP_DIR}" \
    --pose pre \
    --batch-size 16 \
    --min-sam-score 0.85 \
    --overlays

echo ""
echo "Running Stage 1 on POST masks (test subset)..."
python mask_refinement.py \
    "${TEMP_DIR}" \
    --pose post \
    --batch-size 16 \
    --min-sam-score 0.85 \
    --overlays

echo ""
echo "Test complete! Check outputs:"
echo "  - Refined masks: ${TEMP_DIR}/masks_*_refined/"
echo "  - Debug overlays: ${TEMP_DIR}/debug_overlays/"
echo ""
# echo "If results look good, run on full dataset:"
# echo "  python mask_refinement.py ${DATA_DIR} --pose pre --overlays"
# echo "  python mask_refinement.py ${DATA_DIR} --pose post --overlays"