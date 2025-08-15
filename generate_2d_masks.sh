#!/bin/bash

# Usage script for generating 2D masks

# Set your data directory (where transforms_post.json is located)
DATA_DIR="data/gs_t_multi_5_post"  # Change this to your actual path

# Output directory for masks
echo "Generating 2D masks from 3D articulated object..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the mask generation script
python arti_splatfacto/utils/generate_2d_masks.py "$DATA_DIR" 
