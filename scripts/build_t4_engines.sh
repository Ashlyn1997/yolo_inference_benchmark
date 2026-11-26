#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-${PROJECT_DIR}/artifacts/yolo11n.pt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/t4}"
MODEL_NAME="${MODEL_NAME:-yolo11n}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
DYNAMIC_MAX_BATCH="${DYNAMIC_MAX_BATCH:-8}"
WORKSPACE_GIB="${WORKSPACE_GIB:-4}"
CALIBRATION_DATA="${CALIBRATION_DATA:-}"
ALLOW_NON_T4="${ALLOW_NON_T4:-0}"

cd "$PROJECT_DIR"
mkdir -p "$ARTIFACT_DIR"

environment_json="$($PYTHON_BIN -m yolo_benchmark.cli env)"
if [[ "$ALLOW_NON_T4" != "1" && "$environment_json" != *"T4"* ]]; then
  printf 'Refusing T4 build: detected environment does not contain "T4".\n%s\n' "$environment_json" >&2
  printf 'Set ALLOW_NON_T4=1 only when intentionally testing the workflow elsewhere.\n' >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Missing source model: %s\nRun the download command from docs/T4_RUNBOOK.md first.\n' "$MODEL" >&2
  exit 2
fi

export_model() {
  local output_name="$1"
  shift
  "$PYTHON_BIN" -m yolo_benchmark.cli export \
    --model "$MODEL" \
    --output-dir "$ARTIFACT_DIR" \
    --output-name "$output_name" \
    --imgsz "$IMAGE_SIZE" \
    --device 0 \
    --workspace "$WORKSPACE_GIB" \
    "$@"
}

# ONNX is retained as a portable interchange artifact; all TensorRT engines
# below are built on the target T4 and must not replace historical L20 engines.
export_model "${MODEL_NAME}_dynamic_b${DYNAMIC_MAX_BATCH}.onnx" \
  --formats onnx --batch "$DYNAMIC_MAX_BATCH" --dynamic

export_model "${MODEL_NAME}_t4_fp32_b1.engine" --formats engine --batch 1
export_model "${MODEL_NAME}_t4_fp16_b1.engine" --formats engine --batch 1 --half
export_model "${MODEL_NAME}_t4_fp16_b4.engine" --formats engine --batch 4 --half
export_model "${MODEL_NAME}_t4_fp16_b8.engine" --formats engine --batch 8 --half
export_model "${MODEL_NAME}_t4_fp16_dynamic_b${DYNAMIC_MAX_BATCH}.engine" \
  --formats engine --batch "$DYNAMIC_MAX_BATCH" --dynamic --half

if [[ -n "$CALIBRATION_DATA" ]]; then
  export_model "${MODEL_NAME}_t4_int8_b1.engine" \
    --formats engine --batch 1 --int8 --data "$CALIBRATION_DATA"
  export_model "${MODEL_NAME}_t4_int8_b4.engine" \
    --formats engine --batch 4 --int8 --data "$CALIBRATION_DATA"
else
  printf 'Skip INT8: set CALIBRATION_DATA to a representative dataset YAML.\n'
fi

printf 'T4 artifacts written under %s\n' "$ARTIFACT_DIR"
