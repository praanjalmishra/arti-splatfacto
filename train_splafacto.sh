#!/bin/bash
DATA_FOLDER=${1:-"/local/home/pmishra/cvg/arti-splatfacto/data/gs_t"}
CUDA_VISIBLE_DEVICES=${2:-0}
OUTPUT_FOLDER=${4:-"/local/home/pmishra/cvg/arti-splatfacto/outputs"}
NERFSTUDIO_FOLDER=${5:-"/local/home/pmishra/nerfstudio"}

current_time=$(date +%Y-%m-%d_%H%M%S)


# ------------------ Launch Training ------------------
# ns-train splatfacto \
#     --vis viewer+wandb \
#     --experiment-name "$(basename "$DATA_FOLDER")_splatfacto" \
#     --data "$DATA_FOLDER" \
#     --output-dir "$OUTPUT_FOLDER" \
#     --max-num-iterations 20000 \
#     --machine.num-devices 1 \
#     nerfstudio-data \
#     --load-3D-points True

ns-train splatfacto --vis viewer+wandb \
    --experiment-name "$(basename "$DATA_FOLDER")_splatfacto" \
    --output-dir ${OUTPUT_FOLDER} \
    --timestamp $current_time \
    --steps_per_eval_all_images 1000 \
    --pipeline.model.cull_alpha_thresh 0.005 \
    --max-num-iterations 20000 \
    --machine.num-devices 1 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data ${DATA_FOLDER}/transforms_pre.json \
    --auto-scale-poses=False --center-method none --orientation-method none \
    --load-3D-points True \
    --train_split_fraction 0.9