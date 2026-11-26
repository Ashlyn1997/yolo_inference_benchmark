#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" -m yolo_benchmark.cli env > "$OUTPUT_DIR/environment_${STAMP}.json"
nvidia-smi --query-gpu=timestamp,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader > "$OUTPUT_DIR/nvidia_smi_${STAMP}.csv"

printf 'Environment: %s\nGPU snapshot: %s\n' \
  "$OUTPUT_DIR/environment_${STAMP}.json" \
  "$OUTPUT_DIR/nvidia_smi_${STAMP}.csv"
