# NVIDIA T4 Benchmark Runbook

本流程在不覆盖 L20 历史结果的前提下，在 NVIDIA T4 上重新构建 TensorRT Engine，执行精度验证、三轮性能测试和 GPU 资源采样。

## 1. 隔离规则

- L20 历史 JSON、图片、报告和 `runs/detect/` 保持不变。
- T4 Engine 写入 `artifacts/t4/`。
- T4 性能结果写入 `results/t4/<timestamp>/`。
- T4 验证图写入 `runs/t4/`。
- TensorRT Engine 必须在 T4 目标环境中重新构建，不能复用 L20 Engine。

## 2. 环境检查

选择与宿主机 driver 兼容、并仍支持 Turing SM 7.5 的 NVIDIA TensorRT 环境。不要仅按“最新版”选择镜像。

```bash
cd yolo_inference_benchmark
python -m yolo_benchmark.cli env --gpu-index 0
nvidia-smi
```

输出至少应确认：

- `gpu` 包含 `T4`；
- `gpu_compute_capability` 为 `7.5`；
- CUDA 可用；
- ONNX Runtime 能加载 CUDA Execution Provider；
- TensorRT 可以导入。

下载模型：

```bash
python -m yolo_benchmark.cli download \
  --model yolo11n.pt \
  --output-dir artifacts
```

## 3. 在 T4 上构建制品

FP32、FP16 和 dynamic Engine：

```bash
./scripts/build_t4_engines.sh
```

需要 INT8 时必须提供代表性校准集：

```bash
CALIBRATION_DATA=/path/to/business_dataset.yaml \
  ./scripts/build_t4_engines.sh
```

可用环境变量：

- `MODEL`：源 `.pt` 文件；
- `ARTIFACT_DIR`：默认 `artifacts/t4`；
- `MODEL_NAME`：默认 `yolo11n`；
- `DYNAMIC_MAX_BATCH`：默认 8；
- `WORKSPACE_GIB`：默认 4；
- `CALIBRATION_DATA`：为空时跳过 INT8；
- `ALLOW_NON_T4=1`：只用于在其他机器检查脚本，不能把结果标成 T4。

## 4. 精度验证

```bash
DATA=coco128.yaml ./scripts/validate_t4_models.sh
```

正式交付应将 `DATA` 替换为固定的业务验证集 YAML。脚本分别保存 PyTorch、ONNX Runtime、TensorRT FP32/FP16/INT8 的：

- mAP50-95、mAP50、mAP75；
- mean precision、mean recall；
- 每类别 mAP50-95；
- 模型 SHA256；
- dataset YAML SHA256（本地文件时）；
- 环境版本和 Ultralytics 验证图。

INT8 结果只有同时满足精度阈值和性能目标时才可以作为上线候选。

## 5. 性能矩阵

推荐使用固定真实图片目录：

```bash
SOURCE=/path/to/frozen/images \
REPEATS=3 \
WARMUP=50 \
RUNS=500 \
  ./scripts/run_t4_matrix.sh
```

默认覆盖：

- PyTorch FP32/FP16 Batch 1；
- ONNX Runtime CUDA FP32 Batch 1；
- TensorRT FP32 Batch 1；
- TensorRT FP16 Batch 1/4/8；
- TensorRT dynamic FP16 Batch 1/4/8；
- TensorRT INT8 Batch 1/4（存在 Engine 时）。

每轮结果包含：

```text
results/t4/<timestamp>/
├── environment.json
├── <case>_repeat<N>.json
├── raw/<case>_repeat<N>.csv
├── gpu/<case>_repeat<N>_<backend>.csv
└── summary.md
```

GPU CSV 以 200 ms 采样 utilization、memory、temperature、power、SM clock 和 memory clock。JSON 记录平均/峰值资源数据和 `img/s/W`。GPU monitor 不可用时 benchmark 会继续，但报告中必须说明缺失。

如需在非 T4 环境检查脚本，可设置 `EXPECTED_GPU=`；这样产生的数据不能放进 T4 报告。

## 6. 测量口径

- warmup 完成后才启动 GPU monitor，warmup 不进入正式 GPU 汇总。
- `inference_ms` 是同步后端调用的主机可观察时间，包含输入进入后端、同步和输出回到 CPU；它不是纯 kernel 时间。
- raw CSV 用于检查抖动、异常值和重复实验稳定性。
- summary 同时报告单轮结果和跨 repeat 的均值、标准差、CV。
- L20 与 T4 只能在相同模型、图片、尺寸、batch 和统计方法下比较，并同时列出各自软件版本。

## 7. 进一步定位

发现 ORT 尾延迟、INT8 未加速或大 batch 退化时，先保留当前 raw CSV 和 GPU CSV，再使用 Nsight Systems 或 TensorRT verbose/profile 定位。不要仅凭平均 GPU utilization 推断根因。

建议最终报告分三部分：

1. T4 平台内 PyTorch/ORT/TensorRT 和 FP32/FP16/INT8 对比；
2. L20 与 T4 的相同 workload 对比；
3. 精度、吞吐、尾延迟、显存和功耗效率之间的部署建议。
