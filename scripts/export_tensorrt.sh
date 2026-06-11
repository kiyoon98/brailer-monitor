#!/usr/bin/env bash
# Export trained YOLO weights to TensorRT engine for Jetson Thor.
set -euo pipefail

WEIGHTS="${1:-models/brailer_seg.pt}"
OUTPUT_DIR="${2:-models}"
IMGSZ="${3:-640}"
DEVICE="${4:-0}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Weights not found: $WEIGHTS"
  echo "Train first: python -m brailer_monitor train --dataset config/dataset.yaml"
  exit 1
fi

python -m brailer_monitor export "$WEIGHTS" \
  --output-dir "$OUTPUT_DIR" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE"

echo "Done. Use models/*.engine with: python -m brailer_monitor analyze video.mp4 --model models/brailer_seg.engine"
