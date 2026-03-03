#!/bin/bash

DATA_PATH="/workspace/data/arti4d/nerf_output/din080_scene1/canonical"
OUTPUT_PATH="$DATA_PATH"

ns-train qed-splatter \
    --vis viewer+wandb \
    --pipeline.model.use-scale-regularization=True \
    --pipeline.model.output-depth-during-training=True \
    --pipeline.model.rasterize-mode=antialiased \
    --pipeline.model.camera-optimizer.mode=SO3xR3 \
    --pipeline.model.use-bilateral-grid=True \
    --experiment-name "" \
    --timestamp "" \
    --output-dir "$OUTPUT_PATH" \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 5000 \
    nerfstudio-data \
    --data "$DATA_PATH" \
    --train-split-fraction 0.9