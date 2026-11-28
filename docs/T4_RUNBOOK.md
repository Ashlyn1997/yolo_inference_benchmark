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

## 8. Kaggle：原输出与 pinned 输出的前处理细分

当 pinned 输出缩短 inference，却使 E2E 变慢时，使用独立诊断脚本
`scripts/profile_t4_preprocess.py`。先将该新文件同步到 Kaggle 项目；它不会
自动出现在尚未更新的远程仓库中，也不需要重新构建 Engine。

在 Notebook 的 Python 单元格执行：

```python
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

project = Path("/kaggle/working/yolo_inference_benchmark")
script = project / "scripts/profile_t4_preprocess.py"
assert script.is_file(), "请先同步诊断脚本到 Kaggle"
output = project / "results/t4" / (
    "preprocess_breakdown_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
)

for mode in ["original", "pinned"]:
    subprocess.run([
        sys.executable, str(script),
        "--engine", str(project / "artifacts/t4/yolo11n_t4_fp16_b8.engine"),
        "--source", "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017",
        "--batch", "8", "--mode", mode,
        "--output-dir", str(output / mode),
    ], cwd=project, check=True)

print("两种模式全部完成：", output)
```

每种模式均先进行一轮未插桩的完整 pipeline 对照，再进行一轮细分计时，
默认各自 warmup=50、runs=500。正式计时不启用 GPU monitor。脚本验证
前处理数组/坐标元信息不变，并检查同批输入的输出数值接近。

每个模式的目录保存 `summary.json`、`control_latency.csv`、
`profiled_latency.csv` 和 `preprocess_samples.csv`。细分项为：

- `letterbox_ms`：resize 与 padding；
- `color_ms`：BGR 转 RGB；
- `layout_normalize_ms`：原有布局转换、float32 转换和归一化表达式；
- `stack_ms`：NumPy batch 拼接；
- `body_total_ms`：函数体内计时；
- `outer_preprocess_ms`：benchmark 的外层前处理计时；
- `outside_body_ms`：外层减函数体时间，包含返回清理、调用方赋值及插桩开销，
  不能直接当成纯内存释放耗时。

先比较 control 与 profiled 的前处理时间，以及两种模式的整体现象是否仍然
复现；若插桩明显改变结果，不能直接用分段时间解释原测量。这是一次定位运行，
不提供统计显著性结论。pinned 模式返回会复用的 NumPy view，仅用于当前推理后
立即完成后处理的同步流程，不能无条件替换正式后端的独立输出所有权语义。

## 9. 最终统一条件复测（12 次独立进程）

将更新后的 `scripts/profile_t4_preprocess.py` 和新增的
`scripts/run_t4_pinned_comparison.py` 同步到 Kaggle。前者新增
`--benchmark-only`，只测未插桩 control，不再额外跑一次分段 profiling。
两种模式都执行相同的正确性检查，并在 warmup 前释放校验数组。

Notebook 执行：

```python
%cd /kaggle/working/yolo_inference_benchmark

import subprocess
import sys

subprocess.run([
    sys.executable, "scripts/run_t4_pinned_comparison.py",
    "--source", "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017",
], check=True)
```

默认 b4/b8 × original/pinned × 3 轮，共 12 次，每次 warmup=50、runs=500，
交替配置顺序，GPU monitor 关闭。使用已有 Engine，不重新构建。
可用 `--engine-template`、`--batches`、`--repeats`、`--warmup`、`--runs`、
`--device` 和 `--output-dir` 覆盖默认值。非空输出目录会被拒绝以防覆盖。

默认生成 `results/t4/pinned_final_<timestamp>/`，其中每个配置保存
`summary.json`、`control_latency.csv`，另保存日志、`execution_order.json`、
总汇总 `comparison.json` 和 `summary.md`。

汇总检查同 batch 的 Engine SHA256、环境版本、图片尺寸、线程数、请求的
测量设置、校验结果及内存生命周期策略一致。报告同时给出前处理、inference、
后处理、E2E、推理吞吐，以及 inference/E2E 的跨轮 CV；收益为负表示退化，
不会仅根据推理吞吐判定全流程成功。P95 是各轮 P95 的平均值，原始 CSV 可用于
另算合并分位数。当前均值/总体标准差不提供统计显著性结论。

出现 `ALL 12 RUNS COMPLETE` 后本轮才全部结束。最终报告只使用本轮统一条件
的数据，不与保留校验数组的旧实验混合平均。该运行器仍属于同步消费输出的实验
实现，不改变正式 `TensorRTBackend` 的输出所有权契约。
