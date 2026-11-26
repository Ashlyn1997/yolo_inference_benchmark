# NVIDIA Docker 与一键测试说明

## 目的

本目录用于固定 NVIDIA TensorRT FAE 测试环境，避免 Conda、Python、CUDA 动态库和 ONNX Runtime Execution Provider 不一致。

TensorRT Engine 绑定 GPU 架构和 TensorRT 版本，因此请在最终部署机或版本严格兼容的同类机器中构建 Engine。

## 1. 选择基础镜像

使用 NVIDIA NGC TensorRT 容器，并选择与目标服务器驱动兼容的 tag。不要在未知兼容性的情况下使用最新 tag。

```bash
docker login nvcr.io
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:<compatible-tag> \
  -f docker/Dockerfile \
  -t yolo-nvidia-benchmark .
```

NVIDIA 容器运行需要 NVIDIA Container Toolkit。启动时使用 `--gpus all` 暴露 GPU。

## 2. 环境检查

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace/yolo-inference-benchmark" \
  -w /workspace/yolo-inference-benchmark \
  yolo-nvidia-benchmark env
```

确认输出包含 CUDA 可用、正确 GPU 名称以及可导入的 TensorRT。

## 3. 运行 NVIDIA 性能矩阵

主机 Conda 环境复跑时：

```bash
PYTHON_BIN=/data/conda/yolo-trt/bin/python scripts/capture_nvidia_environment.sh
PYTHON_BIN=/data/conda/yolo-trt/bin/python scripts/run_nvidia_matrix.sh
```

Docker 中执行：

```bash
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/workspace/yolo-inference-benchmark" \
  -w /workspace/yolo-inference-benchmark \
  yolo-nvidia-benchmark scripts/run_nvidia_matrix.sh
```

脚本会跳过不存在的 Engine，运行可用的 FP32、FP16 和 INT8 基线，并生成 `reports/nvidia_summary.md`。

T4 请使用独立流程，避免覆盖 L20 历史资产：

```bash
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/workspace/yolo-inference-benchmark" \
  -w /workspace/yolo-inference-benchmark \
  yolo-nvidia-benchmark scripts/build_t4_engines.sh

docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/workspace/yolo-inference-benchmark" \
  -w /workspace/yolo-inference-benchmark \
  -e SOURCE=/workspace/yolo-inference-benchmark/datasets/business_val/images \
  yolo-nvidia-benchmark scripts/run_t4_matrix.sh
```

具体参数和输出结构见 `docs/T4_RUNBOOK.md`。

## 4. 注意事项

- `artifacts/`、`datasets/`、`results/`、`runs/` 通过挂载保存，不会被构建进镜像。
- 当前 INT8 仅以 COCO128 作为实验校准，不能作为生产精度结论。
- 最终报告应保留环境快照、原始 JSON、精度验证目录和生成的汇总报告。
