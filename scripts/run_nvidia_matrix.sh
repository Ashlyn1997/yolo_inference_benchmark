#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE="${SOURCE:-${PROJECT_DIR}/datasets/coco128/images/train2017}"
RUNS="${RUNS:-500}"
WARMUP="${WARMUP:-50}"
DEVICE="${DEVICE:-cuda:0}"

cd "$PROJECT_DIR"
mkdir -p results

run_case() {
  local label="$1"
  local engine="$2"
  local batch="$3"
  local output="results/${label}.json"

  if [[ ! -f "$engine" ]]; then
    printf 'Skip %s: missing %s\n' "$label" "$engine"
    return 0
  fi

  "$PYTHON_BIN" -m yolo_benchmark.cli benchmark \
    --engine "$engine" \
    --source "$SOURCE" \
    --imgsz 640 \
    --batch "$batch" \
    --device "$DEVICE" \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --output "$output"
}

run_case nvidia_trt_fp32_b1 artifacts/yolo11n.engine 1
run_case nvidia_trt_fp16_b1 artifacts/yolo11n_trt_fp16_source.engine 1
run_case nvidia_trt_fp16_b4 artifacts/yolo11n_trt_fp16_b4_source.engine 4
run_case nvidia_trt_fp16_b8 artifacts/yolo11n_trt_fp16_b8_source.engine 8
run_case nvidia_trt_int8_b4 artifacts/yolo11n_trt_int8_b4_coco128_source.engine 4

"$PYTHON_BIN" scripts/generate_nvidia_report.py --results-dir results --output reports/nvidia_summary.md
