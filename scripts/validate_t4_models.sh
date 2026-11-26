#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-${PROJECT_DIR}/artifacts/yolo11n.pt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/t4}"
MODEL_NAME="${MODEL_NAME:-yolo11n}"
DYNAMIC_MAX_BATCH="${DYNAMIC_MAX_BATCH:-8}"
DATA="${DATA:-coco128.yaml}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/results/t4/validation_${RUN_STAMP}}"
PLOTS_DIR="${PLOTS_DIR:-${PROJECT_DIR}/runs/t4}"

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR" "$PLOTS_DIR"

validate_case() {
  local label="$1"
  local model_path="$2"
  local precision="$3"
  local batch="$4"
  if [[ ! -f "$model_path" ]]; then
    printf 'Skip %s: missing %s\n' "$label" "$model_path"
    return 0
  fi
  "$PYTHON_BIN" -m yolo_benchmark.cli validate \
    --model "$model_path" \
    --data "$DATA" \
    --device 0 \
    --imgsz 640 \
    --batch "$batch" \
    --precision "$precision" \
    --project "$PLOTS_DIR" \
    --name "$label" \
    --output "$RUN_DIR/${label}.json"
}

validate_case t4_pytorch_fp32 "$MODEL" fp32 1
validate_case t4_ort_fp32 "$ARTIFACT_DIR/${MODEL_NAME}_dynamic_b${DYNAMIC_MAX_BATCH}.onnx" fp32 1
validate_case t4_trt_fp32 "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp32_b1.engine" fp32 1
validate_case t4_trt_fp16 "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_b1.engine" fp16 1
validate_case t4_trt_int8 "$ARTIFACT_DIR/${MODEL_NAME}_t4_int8_b1.engine" int8 1

printf 'T4 validation results: %s\n' "$RUN_DIR"
