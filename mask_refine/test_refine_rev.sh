#!/bin/bash

DATA_DIR="/local/home/pmishra/cvg/arti-splatfacto/data_ransac/sync_data_rev/multiview"

echo "Running refinement on full dataset..."
echo ""

# Run refinement on PRE masks
python mask_refine/mask_refinement.py \
    "${DATA_DIR}" \
    --pose pre \
    --batch-size 16 \
    --min-sam-score 0.80 \
    --overlays \

echo ""
# Run refinement on POST masks
python mask_refine/mask_refinement.py \
    "${DATA_DIR}" \
    --pose post \
    --batch-size 16 \
    --min-sam-score 0.80 \
    --overlays \

echo ""
echo "Refinement complete! Check outputs:"
echo "  - Refined PRE masks:  ${DATA_DIR}/masks_pre/"
echo "  - Refined POST masks: ${DATA_DIR}/masks_post/"
echo "  - Coarse backups:     ${DATA_DIR}/masks_pre_coarse/, ${DATA_DIR}/masks_post_coarse/"
echo "  - Debug overlays:     ${DATA_DIR}/debug_overlays/"
echo ""
