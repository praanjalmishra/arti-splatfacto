#!/usr/bin/env bash

set -euo pipefail

ARTI_ROOT=/workspace/arti4d/arti4d/raw
OUT_ROOT=/workspace/arti4d/arti4d_processed
SCRIPT=/workspace/arti-splatfacto/arti4d/convert_arti4d_to_nerf.py

LOG_DIR="$OUT_ROOT/logs_processing"
mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/process_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RUN_LOG"
}

log "==== Arti4D Processing Started ===="
log "ARTI_ROOT=$ARTI_ROOT"
log "OUT_ROOT=$OUT_ROOT"
log "SCRIPT=$SCRIPT"

# Copy metadata
cp "$ARTI_ROOT/metadata.yaml" "$OUT_ROOT/"
log "Metadata copied"

for scene_dir in "$ARTI_ROOT"/*/; do
  scene_name=$(basename "$scene_dir")
  log "Processing scene: $scene_name"

  for seq_dir in "${scene_dir}"scene_*; do
    if [ -d "$seq_dir" ]; then
      seq_name=$(basename "$seq_dir")
      out_dir="$OUT_ROOT/$scene_name/$seq_name"
      mkdir -p "$out_dir"

      log "  → Sequence: $seq_name"

      SEQ_LOG="$LOG_DIR/${scene_name}_${seq_name}.log"

      {
        echo "===== $scene_name / $seq_name ====="
        echo "Start: $(date)"
        python "$SCRIPT" \
          --scene_dir "$seq_dir" \
          --output_dir "$out_dir" \
          --metadata "$ARTI_ROOT/metadata.yaml" \
          --only_verified \
          --max_frames 500
        echo "End: $(date)"
      } &>> "$SEQ_LOG"

      log "  ✓ Finished $scene_name / $seq_name"
    fi
  done
done

log "==== Arti4D Processing Completed Successfully ===="