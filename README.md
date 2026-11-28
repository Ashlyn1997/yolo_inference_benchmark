# YOLO PyTorch / ONNX Runtime / TensorRT 部署对比

这是一个完全独立的 YOLO 推理部署实验目录，不依赖也不修改上层模型训练项目。它使用统一的图像预处理和 NMS 后处理，只比较三个后端的模型执行部分：

- PyTorch CUDA
- ONNX Runtime CUDA Execution Provider
- TensorRT Engine（原生 Python API）

支持环境诊断、模型导出、结构化精度验证、单图推理、GPU 资源采样以及带预热的 P50/P95/P99 基准测试。默认测试数据是固定随机种子的合成图像，适合检查流程；正式报告应换成冻结的真实业务图片。

仓库保留原 NVIDIA L20 实验资产；新增的 T4 流程使用独立的 `artifacts/t4`、`results/t4` 和 `runs/t4`，不会覆盖 L20 数据。完整步骤见 [NVIDIA T4 Benchmark Runbook](docs/T4_RUNBOOK.md)。

T4 输出回传优化的完整排查过程、原始数据核验和结果图见 [T4 实验报告](reports/T4报告.md#10-输出回传优化与最终统一条件复测2026-11-28)。最终三轮独立进程复测中，Batch 8 复用 pinned 输出使推理吞吐提高 15.76%、E2E 耗时降低 6.59%；收益适用于报告中的固定批次同步消费条件。

## 目录结构

```text
yolo_inference_benchmark/
├── yolo_benchmark/
│   ├── backends.py       # 三种推理后端
│   ├── pipeline.py       # 统一 letterbox、解码和 NMS
│   ├── benchmark.py      # 延迟、吞吐和 JSON 报告
│   ├── gpu_monitor.py    # nvidia-smi 资源采样
│   ├── validation.py     # mAP/precision/recall 结构化验证
│   └── cli.py            # env/export/validate/infer/benchmark 命令
├── scripts/
│   ├── build_t4_engines.sh
│   ├── validate_t4_models.sh
│   └── run_t4_matrix.sh
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── pyproject.toml
```

模型默认写入已忽略的 `artifacts/`；运行结果写入 `results/`，可按需要提交经过筛选的 JSON、CSV 和报告。T4 默认使用带时间戳的子目录，避免覆盖已有结果。

## 1. 环境准备

建议使用 NVIDIA CUDA 或 TensorRT 官方容器，避免宿主机驱动、CUDA Toolkit、PyTorch、ONNX Runtime 和 TensorRT 版本混装。容器和 Python wheel 必须按当前目标机的实际 driver 与 compute capability 选择；历史 L20 环境的版本只能用于解释历史结果，不能直接当作 T4 安装约束。

在本目录创建独立虚拟环境：

```bash
cd yolo_inference_benchmark
python3 -m venv .venv
source .venv/bin/activate
```

先根据CUDA环境安装匹配的PyTorch，再安装本项目：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

上面的PyTorch索引只是CUDA 12.1示例，应该按照目标服务器环境选择。TensorRT建议通过匹配版本的NVIDIA容器、系统包或CUDA 12 wheel安装。

检查实际环境：

```bash
python -m yolo_benchmark.cli env
```

重点确认：

- `torch.cuda_available`为`true`
- ONNX Runtime包含`CUDAExecutionProvider`
- TensorRT能够导入
- GPU 显示为当前目标卡；运行 T4 专用脚本时必须包含 `T4`

## 2. 下载模型

使用项目命令下载官方YOLO11n权重，文件会固定保存到`artifacts/yolo11n.pt`：

```bash
python -m yolo_benchmark.cli download \
  --model yolo11n.pt \
  --output-dir artifacts
```

如果文件已经存在，命令会直接复用；需要重新下载时增加`--force`。也可以将自己的Ultralytics检测模型手动放入`artifacts/`。

## 3. 导出模型

导出固定Batch 1、640×640的ONNX和FP16 TensorRT Engine：

```bash
python -m yolo_benchmark.cli export \
  --model artifacts/yolo11n.pt \
  --formats onnx engine \
  --output-dir artifacts \
  --imgsz 640 \
  --batch 1 \
  --half
```

动态Batch Engine示例：

```bash
python -m yolo_benchmark.cli export \
  --model artifacts/yolo11n.pt \
  --formats onnx engine \
  --output-dir artifacts \
  --imgsz 640 \
  --batch 16 \
  --dynamic \
  --half
```

INT8必须使用与生产数据分布相近的校准集：

```bash
python -m yolo_benchmark.cli export \
  --model artifacts/best.pt \
  --formats engine \
  --output-dir artifacts \
  --imgsz 640 \
  --batch 16 \
  --dynamic \
  --int8 \
  --data /path/to/dataset.yaml
```

不要用随机图片或COCO替代业务数据做正式INT8校准。导出Engine的机器应与部署机器具有相同GPU架构和兼容的TensorRT环境。

## 4. 单图验证

先分别跑三种后端，确认输出框数量和位置基本一致：

```bash
python -m yolo_benchmark.cli infer \
  --backend pytorch \
  --model artifacts/yolo11n.pt \
  --source /path/to/test.jpg \
  --output results/pytorch.jpg \
  --half

python -m yolo_benchmark.cli infer \
  --backend onnxruntime \
  --model artifacts/yolo11n.onnx \
  --source /path/to/test.jpg \
  --output results/onnxruntime.jpg

python -m yolo_benchmark.cli infer \
  --backend tensorrt \
  --model artifacts/yolo11n.engine \
  --source /path/to/test.jpg \
  --output results/tensorrt.jpg
```

这里的通用后处理针对Ultralytics检测模型的常见原始输出：`[B, 4+C, N]`、`[B, N, 4+C]`以及端到端`[B, N, 6]`。分割、姿态估计、OBB或自定义检测头需要增加对应解码器。

完整验证集需要使用结构化 `validate` 命令，而不是只看单张图片：

```bash
python -m yolo_benchmark.cli validate \
  --model artifacts/yolo11n.engine \
  --data /path/to/dataset.yaml \
  --batch 1 \
  --precision fp16 \
  --project runs/validation \
  --name tensorrt_fp16 \
  --output results/validation/tensorrt_fp16.json
```

## 5. 公平基准测试

三种后端一起测试：

```bash
python -m yolo_benchmark.cli benchmark \
  --pt artifacts/yolo11n.pt \
  --onnx artifacts/yolo11n.onnx \
  --engine artifacts/yolo11n.engine \
  --source /path/to/images \
  --imgsz 640 \
  --batch 1 \
  --half \
  --warmup 50 \
  --runs 500 \
  --output results/batch1.json
```

测试Batch 1/4/8/16时，需要使用支持对应Batch范围的动态模型，或者分别构建固定Batch Engine：

```bash
for batch in 1 4 8 16; do
  python -m yolo_benchmark.cli benchmark \
    --pt artifacts/yolo11n.pt \
    --onnx artifacts/yolo11n.onnx \
    --engine artifacts/yolo11n.engine \
    --source /path/to/images \
    --imgsz 640 \
    --batch "$batch" \
    --half \
    --warmup 50 \
    --runs 500 \
    --output "results/batch${batch}.json"
done
```

报告包括：

- 平均推理延迟
- P50、P95、P99推理延迟
- 纯推理吞吐量
- 预处理、后处理和端到端平均延迟
- PyTorch可见的CUDA显存占用
- GPU、Python、PyTorch及CUDA环境信息

可选的原始时延和 GPU 采样：

```bash
python -m yolo_benchmark.cli benchmark \
  --engine artifacts/t4/yolo11n_t4_fp16_b1.engine \
  --source /path/to/images \
  --batch 1 --warmup 50 --runs 500 \
  --precision fp16 --expected-gpu T4 \
  --raw-output results/t4/manual/raw.csv \
  --gpu-metrics-dir results/t4/manual/gpu \
  --run-tag t4_trt_fp16_b1 \
  --output results/t4/manual/result.json
```

该 JSON 还会保存模型/Engine SHA256、compute capability、driver、总显存、功耗上限、库版本、GPU 平均/峰值指标和 `img/s/W`。

## 6. 测试规范

为了让数据可以用于FAE项目报告：

1. 测试前用`nvidia-smi`记录GPU、驱动、功耗和其他进程。
2. 三个后端使用相同图片、分辨率、Batch、置信度和IoU阈值。
3. 至少预热50次、正式运行500次。
4. 分开报告纯推理和端到端耗时。
5. 性能测试期间不要让其他进程占用GPU算力。
6. FP16/INT8除了速度，还要在验证集上测mAP50-95。
7. 重复三轮并报告均值及波动，不只选择最好的一次。

本工具返回CPU侧NumPy输出，以便三个后端共用后处理。因此“纯推理延迟”包含输出同步和GPU到CPU传输，符合常见同步API服务的观测方式。如果要研究完全异步流水线，应另做CUDA Stream、Pinned Memory和GPU NMS实验，并单独标注测量口径。

## 7. 运行单元测试

```bash
pytest -q
```

单元测试不需要模型。后处理使用项目内的纯PyTorch NMS，不依赖torchvision的C++/CUDA扩展。

## T4 最终结果可视化

CPU 环境只安装可选绘图依赖即可，不需要 CUDA、TensorRT 或训练依赖：

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_t4_pinned_results.py \
  results/t4/pinned_final_20261128_062057_514198
```

脚本先从逐轮 CSV 重算并核对 summary/comparison JSON，再输出吞吐、后端调用延迟、
E2E 延迟和阶段分解四张 180 DPI PNG，保存到该结果目录的 `plots/`。
误差线为各轮均值的总体标准差，不是置信区间。核验记录为 `plots/verification.json`，
原始 CSV/JSON 不会被改写。只做核验可追加 `--verify-only`，无需安装绘图库。

图已嵌入 [T4 实验报告](reports/T4报告.md)，具体结论和输出缓冲区生命周期限制以报告为准。

接下来的三档输出缓冲区实验使用独立脚本，不覆盖上述历史数据：

```bash
python scripts/run_t4_output_buffer_comparison.py \
  --source /path/to/coco2017/val2017
```

默认 Batch 4/8 × original/reuse_pageable/reuse_pinned × 3 轮，共 18 次独立进程。
正式结果不含 CUDA Event 插桩；额外的 Event 诊断用
`scripts/profile_t4_output_buffers.py --cuda-events` 单独执行。
完整 Kaggle 单元格、计时边界、输出目录和验收说明见
[T4 操作手册第 10 节](docs/T4_RUNBOOK.md#10-三档输出缓冲区对照与-cuda-event-诊断新增实验)。

## 常见问题

### ONNX Runtime回退到CPU

运行`env`命令检查Provider。必须看到`CUDAExecutionProvider`；如果只有`CPUExecutionProvider`，通常是安装了CPU版ONNX Runtime，或者CUDA/cuDNN动态库版本不匹配。

### TensorRT Engine无法加载

Engine不是通用模型文件。应检查TensorRT版本、GPU架构、操作系统以及构建时是否启用了相应兼容选项。最稳妥的方法是在最终目标 GPU 的部署环境中重新构建 Engine；L20 Engine 不应直接用于 T4 benchmark。

### 动态Batch执行失败

确认ONNX导出时使用了`--dynamic`，TensorRT Engine的Optimization Profile也覆盖了目标Batch。固定Batch 1的Engine不能直接用Batch 8测试。

### 三种后端框的位置有细微差别

FP16和INT8会带来数值变化，NMS阈值附近的框可能不同。应比较完整验证集mAP，而不是只看一张图片。
