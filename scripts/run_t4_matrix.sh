#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-${PROJECT_DIR}/artifacts/yolo11n.pt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/t4}"
MODEL_NAME="${MODEL_NAME:-yolo11n}"
DYNAMIC_MAX_BATCH="${DYNAMIC_MAX_BATCH:-8}"
SOURCE="${SOURCE:-synthetic}"
RUNS="${RUNS:-500}"
WARMUP="${WARMUP:-50}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-cuda:0}"
GPU_INDEX="${GPU_INDEX:-0}"
GPU_MONITOR_INTERVAL_MS="${GPU_MONITOR_INTERVAL_MS:-200}"
EXPECTED_GPU="${EXPECTED_GPU:-T4}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/results/t4/${RUN_STAMP}}"

cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR/raw" "$RUN_DIR/gpu"
"$PYTHON_BIN" -m yolo_benchmark.cli env --gpu-index "$GPU_INDEX" > "$RUN_DIR/environment.json"

expected_gpu_args=()
if [[ -n "$EXPECTED_GPU" ]]; then
  expected_gpu_args=(--expected-gpu "$EXPECTED_GPU")
fi

run_case() {
  local label="$1"
  local backend_flag="$2"
  local model_path="$3"
  local batch="$4"
  local precision="$5"
  local use_half="$6"

  if [[ ! -f "$model_path" ]]; then
    printf 'Skip %s: missing %s\n' "$label" "$model_path"
    return 0
  fi
  for repeat in $(seq 1 "$REPEATS"); do
    local tag="${label}_repeat${repeat}"
    half_args=()
    if [[ "$use_half" == "1" ]]; then
      half_args=(--half)
    fi
    "$PYTHON_BIN" -m yolo_benchmark.cli benchmark \
      "$backend_flag" "$model_path" \
      --source "$SOURCE" \
      --imgsz 640 \
      --batch "$batch" \
      --device "$DEVICE" \
      --warmup "$WARMUP" \
      --runs "$RUNS" \
      --precision "$precision" \
      --run-tag "$tag" \
      --gpu-index "$GPU_INDEX" \
      --gpu-monitor-interval-ms "$GPU_MONITOR_INTERVAL_MS" \
      --gpu-metrics-dir "$RUN_DIR/gpu" \
      --raw-output "$RUN_DIR/raw/${tag}.csv" \
      --output "$RUN_DIR/${tag}.json" \
      "${expected_gpu_args[@]}" \
      "${half_args[@]}"
  done
}

run_case t4_pytorch_fp32_b1 --pt "$MODEL" 1 fp32 0
run_case t4_pytorch_fp16_b1 --pt "$MODEL" 1 fp16 1
run_case t4_ort_fp32_b1 --onnx "$ARTIFACT_DIR/${MODEL_NAME}_dynamic_b${DYNAMIC_MAX_BATCH}.onnx" 1 fp32 0
run_case t4_trt_fp32_b1 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp32_b1.engine" 1 fp32 0
run_case t4_trt_fp16_b1 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_b1.engine" 1 fp16 0
run_case t4_trt_fp16_b4 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_b4.engine" 4 fp16 0
run_case t4_trt_fp16_b8 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_b8.engine" 8 fp16 0
run_case t4_trt_fp16_dynamic_b1 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_dynamic_b${DYNAMIC_MAX_BATCH}.engine" 1 fp16 0
run_case t4_trt_fp16_dynamic_b4 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_dynamic_b${DYNAMIC_MAX_BATCH}.engine" 4 fp16 0
run_case t4_trt_fp16_dynamic_b8 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_fp16_dynamic_b${DYNAMIC_MAX_BATCH}.engine" 8 fp16 0
run_case t4_trt_int8_b1 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_int8_b1.engine" 1 int8 0
run_case t4_trt_int8_b4 --engine "$ARTIFACT_DIR/${MODEL_NAME}_t4_int8_b4.engine" 4 int8 0

"$PYTHON_BIN" scripts/generate_nvidia_report.py \
  --results-dir "$RUN_DIR" \
  --output "$RUN_DIR/summary.md"

printf 'T4 benchmark run: %s\n' "$RUN_DIR"
