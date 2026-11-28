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

## 10. 三档输出缓冲区对照与 CUDA Event 诊断（新增实验）

这一节使用新脚本，保留第 9 节 worker、结果和 SHA256。依赖仍为当前项目依赖，
不需要安装 Nsight、NVML 或额外的 CUDA Python 包，也不重新构建 Engine。
请先完成 Engine/工具备份，再把新代码同步到 Kaggle。

| 模式 | CPU 输出内存 | 是否复用 | 输出所有权 |
| --- | --- | --- | --- |
| V1 `original` | Pageable | 否 | 独立输出 |
| V2 `reuse_pageable` | Pageable | 是 | 下一次 infer 前消费完的借用视图 |
| V3 `reuse_pinned` | Pinned | 是 | 下一次 infer 前消费完的借用视图 |

输入传输、输入转换、GPU 输出逐次分配、同步点和后处理保持原有逻辑。
V1 正式 control 直接调用原后端；V2/V3 使用相同实验执行路径，只改变 CPU
输出分配策略。它们仍使用阻塞式 CPU 输出 copy，不是异步流水线。
V2/V1 比较本实现下的复用路径，V3/V2 比较复用前提下的 Pinned 收益；
三档实验不能完整估计两个因素的交互效应，也不提供相同输出所有权的公平服务基准。

### 10.1 先运行未插桩的 18 次正式对照

在 Notebook 新建 Python 单元格：

```python
%cd /kaggle/working/yolo_inference_benchmark

import subprocess
import sys

subprocess.run([
    sys.executable, "-u", "scripts/run_t4_output_buffer_comparison.py",
    "--source", "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017",
    "--batches", "4,8",
    "--repeats", "3",
    "--warmup", "50",
    "--runs", "500",
    "--device", "cuda:0",
], check=True)
```

默认复用 `artifacts/t4/yolo11n_t4_fp16_b4.engine` 和 b8 Engine。
可通过 `--engine-template` 更改路径，例如 `/kaggle/working/.../b{batch}.engine`。
这两份 Engine 应来自本次 T4 FP16 构建；脚本不会重新推断 Engine 内每层精度。
支持 `--imgsz` 和 `--output-dir`，但不能覆盖非空输出目录。

每个配置启动独立进程，三轮循环轮换 V1/V2/V3 的执行先后，每轮交替 batch 顺序。
每次校验原后端与实验输出 allclose，并在 warmup 前释放校验数组；用于正式复用的
CPU output buffer 会保留。正式计时不创建 CUDA Events，不启动 GPU monitor。
仍然使用同一固定图片批次，不代表遍历数据集。

每组开始时显示 `[n/18]` 和日志路径，完成后显示 inference/E2E；单组测量期间
不逐次打印。只有出现 `ALL 18 RUNS COMPLETE` 才表示整轮结束。不要同时运行
其他性能实验或重复启动同一单元格。

```text
results/t4/output_buffers_<timestamp>/
├── provenance.json             # worker/runner/核心代码及 Engine hash
├── execution_order.json        # 命令、顺序、返回码
├── comparison.json             # 三档跨轮汇总、三组配对收益、逐轮记录
├── summary.md
├── logs/
│   └── b8_reuse_pinned_repeat1.log
└── b8_reuse_pinned_repeat1/
    ├── summary.json
    └── control_latency.csv     # 500 条正式样本；其他 17 组结构相同
```

汇总核对同 batch 的 Engine 与实际图片 hash、图片尺寸、环境、线程、CPU affinity、
测量设置和校验策略。源代码或 Engine 在执行期间变化也会拒绝总汇总。
失败时保留已经写出的结果、日志和执行清单，不输出成功总汇总。

报告分别列出 V2/V1、V3/V2、V3/V1 的后端延迟、推理吞吐和 E2E 收益，负值表示
退化。吞吐仍是各轮推理吞吐的平均；P95 是各轮 P95 的平均，不是 pooled P95。
三轮均值、总体标准差和 CV 不构成置信区间或显著性结论。

### 10.2 再运行独立 CUDA Event 诊断

正式结果完成后，在另一个单元格执行六组诊断。每组先跑未插桩 control，再额外
跑 Event phase；两阶段各自 warmup 50 次并正式采样 100 次。诊断有意使用较少
采样来检查机制，不能与上一轮 500 次正式对照混合平均。

```python
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

output = Path("results/t4") / (
    "output_buffer_events_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
)
for batch in (4, 8):
    for mode in ("original", "reuse_pageable", "reuse_pinned"):
        subprocess.run([
            sys.executable, "-u", "scripts/profile_t4_output_buffers.py",
            "--engine", f"artifacts/t4/yolo11n_t4_fp16_b{batch}.engine",
            "--source", "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017",
            "--batch", str(batch), "--mode", mode,
            "--warmup", "50", "--runs", "100", "--device", "cuda:0",
            "--cuda-events",
            "--output-dir", str(output / f"b{batch}_{mode}"),
        ], check=True)

print("ALL 6 DIAGNOSTICS COMPLETE:", output)
```

每组生成：

- `summary.json`、`control_latency.csv`：未插桩 control。
- `cuda_event_samples.csv`：仅正式测量期的逐次 CUDA Event 及主机分段计时。
- `cuda_event_summary.json`：诊断均值、计时解释和 Event phase 的 benchmark 汇总。
- `event_latency.csv`：Event phase 外层前处理、后端、后处理及 E2E 耗时。

CUDA Event 字段：

| 字段 | 测量区间 |
| --- | --- |
| `h2d_event_ms` | 输入 `.to(device)` 前后 |
| `input_conversion_event_ms` | 输入 dtype/contiguous 转换 |
| `execution_event_ms` | TensorRT enqueue 前后的当前 stream 区间 |
| `output_conversion_event_ms` | 输出 `.float()` 前后 |
| `d2h_event_ms` | CPU 输出获取/copy 前后 |

Event phase 不在 warmup 期间记录样本，结束 Event 完成后才读取 elapsed time。
同时保存 `h2d_call_wall_ms`、`shape_buffers_wall_ms`、`enqueue_call_wall_ms`、
`device_sync_wait_wall_ms`、`host_output_call_wall_ms` 等主机时间。
`backend_wall_before_event_readout_ms` 不包含 elapsed-time 读取及采样整理；
`event_latency.csv` 的外层 inference 包含这些诊断开销。

Event 测量的是指定 stream 上两条标记之间的经过时间，可能包含主机提交间隙，
不是所有 kernel 的耗时总和。尤其对于 Pageable 内存，CPU 分配/staging 不会被
完整、独立地归入纯 DMA 时间；不能把 `d2h_event_ms` 当作 PCIe 纯传输带宽。
设备转换无实际工作时，Event 仍可能记录少量标记开销。
各 Event 区间不能直接相加等同 E2E；也不能将主机等待与设备执行相加后重复归因。

诊断保留原有设备同步和阻塞输出 copy，没有引入多 Stream 或异步重叠。
Event 结果只用于帮助归因，正式收益使用 10.1 的未插桩数据。
当前仅检查固定批次输出 allclose，不代表新增了全量 mAP 验证或并发安全接口。

### 10.3 结果保存和下一步

回收整个 `output_buffers_<timestamp>` 和 `output_buffer_events_<timestamp>`，保留
原始 CSV/JSON、日志、源代码版本及 Engine。仓库现已为
`results/t4/output_buffers_*/logs/*.log` 增加忽略规则例外，防止这类实验日志被
通用 `*.log` 规则遗漏；仍应检查备份包确实包含 `logs/`。例外规则不能恢复旧归档中
已经缺失的日志。
无需重建或覆盖历史 `pinned_final_*` 结果。

完成后先核对三档的后端与 E2E 是否同向改善，再看 V2/V1 和 V3/V2 的差异。
这一步不提前承诺提升比例。后续才补不同图片连续推理、输出所有权和全量精度回归。

本地不需要 GPU 的相关测试：

```bash
python -m unittest discover -s tests -p 'test_output_buffer_comparison.py' -v
# 安装了项目完整测试依赖时，也可运行：
pytest -q
```

在 Kaggle 中先用 `%cd /kaggle/working/yolo_inference_benchmark` 切到项目根目录。
若出现 `No module named 'tests.test_output_buffer_comparison'`，先确认新测试文件
已同步到 Kaggle，再使用上面的 discovery 命令，避免与环境内同名 `tests` 包冲突。
项目同时提供 `tests/__init__.py` 和 `scripts/__init__.py` 明确本地包的导入边界。

### 10.4 本地原始数据核验和绘图

回收正式轮和六组 Event 目录后，可在 CPU 上运行：

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_t4_output_buffer_results.py \
  results/t4/output_buffers_20261128_072552_478147 \
  --events-dir results/t4/output_buffer_events_20261128_073920_821735
```

默认写入正式目录的 `plots/`：推理吞吐、后端延迟、E2E、独立 D2H Event 和
b8 原实现逐次采样五张 180 DPI PNG，以及 `verification.json`。
正式图误差线是跨轮总体标准差，Event 图为独立诊断均值；不将两轮混为一轮。
`--verify-only` 可在不安装 matplotlib 时只核验；`--output-dir` 可改输出位置。

核验重算逐轮 CSV 的阶段均值、标准差、分位数和吞吐，再核对配对汇总；也检查
诊断 CSV 与 Event/主机均值、正式执行清单及源码 hash，并明确报告缺失日志。
Event 目录没有独立源码 provenance，Engine/原图也不会在本地重新读取核对；
一致的 metadata 不代表重新执行了推理或 mAP 验证。

## 11. 收尾：不同图片连续推理和输出生命周期

当前输出缓冲区性能实验到此收口，先补小规模正确性验收。INT8、输入四档优化和
多 Stream 不列为当前收尾的必做项。本节不需要重建 Engine，不修改旧 worker、
正式 `TensorRTBackend` 或历史性能结果，也不输出新的性能或 mAP 数字。

新增 `scripts/check_t4_output_correctness.py`，对目录按文件名排序选择 33 张图片，
逐批前进，不循环复用同一批。验证三个路径：

| 路径 | 执行方式 | 保存结果的规则 |
| --- | --- | --- |
| reference | 原 `TensorRTBackend.infer()` | 每次独立 CPU 输出 |
| borrowed | 第 10 节 `reuse_pinned` 实现 | 下一次调用同一 runner 前完成消费 |
| owned | borrowed 返回后立即 `.copy()` | 后续调用不能覆盖已保存结果 |

这三条路径使用同一 Engine、预处理和 CPU 后处理。脚本比对每次 raw 输出和
解码后的框、分数、类别及顺序，记录每张图片的 ID、内容 hash、批次和槽位。
borrowed 检查内存共享及旧视图随当前输出更新；owned/reference 保留首次结果，
在后续每次调用后核对其未变。不同图片偶尔得到相同预测不算失败，不能只依赖
“输出数值是否变化”判断复用行为。

owned 只是同步调用下增加独立副本的示例，未接入正式后端；它不使 TensorRT
execution context 并发安全，也不能直接继承 borrowed 模式的性能收益。
本检查有意保存校验副本，内存生命周期与性能实验不同，不能拿其运行时间作 benchmark。

### 11.1 Kaggle 一次运行两档检查

同步新代码后执行：

```python
%cd /kaggle/working/yolo_inference_benchmark

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path("results/t4") / (
    "output_correctness_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
)
for batch in (4, 8):
    destination = output / f"b{batch}"
    subprocess.run([
        sys.executable, "-u", "scripts/check_t4_output_correctness.py",
        "--engine", f"artifacts/t4/yolo11n_t4_fp16_b{batch}.engine",
        "--source", "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017",
        "--batch", str(batch), "--images", "33", "--device", "cuda:0",
        "--output-dir", str(destination),
    ], check=True)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["passed"] is True and summary["images_checked"] == 33

print("BOTH CORRECTNESS CHECKS PASSED:", output)
```

Batch 4 共 9 批、Batch 8 共 5 批；尾批分别补 3/7 张，填充位置不输出图片结果。
每批 reference 执行一次，borrowed 执行一次，owned 再调用同一 borrowed runner 并
复制一次；因此每批有三次串行推理，均用于正确性对照。无需再执行 18 次性能矩阵。
源目录必须有足够的真实图片，不能用一个文件循环充数。

默认 `rtol=1e-4`、`atol=1e-4`。输出不一致时保留现场并检查图片槽位、NMS 边界、
Engine/环境和原始预测，不为了得到通过结果而直接放宽阈值。

### 11.2 输出与通过标准

```text
results/t4/output_correctness_<timestamp>/
├── b4/
│   ├── provenance.json       # Engine/源码 hash、环境、参数、选图清单
│   ├── batch_checks.csv     # 每批 raw 误差、填充数、生命周期检查
│   ├── image_results.jsonl  # 每张真实图片的 reference/borrowed/owned 检测结果
│   └── summary.json
└── b8/
    └── ...
```

两组均出现 `CORRECTNESS PASSED`，最后出现 `BOTH CORRECTNESS CHECKS PASSED`。
每组应有 33 条图片记录，图片 ID 唯一且顺序与清单一致；汇总中的
`borrowed_shared_buffer_verified`、`first_owned_and_reference_survived_later_calls`
均为 true。`borrowed_contents_changed_observed` 描述是否观察到内容变化，不是独立
通过条件。Engine 和记录的源码在运行期间发生变化也会判失败。

发生失败时进程非零退出，`summary.json` 标记 `passed=false`，保存 `failure.txt`
及已完成批次的记录；启动阶段的路径或环境错误会直接显示在终端。成功不能外推为
完整 COCO mAP、不同 Engine、异步服务或并发场景的正确性。

请回收整个目录。验收通过后，把本项目整理为环境恢复、制品构建、精度与性能
验证、异常排查和部署取舍的完整案例，进入面试准备。

本地单元测试仅需已有 NumPy，不依赖 NVIDIA GPU、TensorRT 或模型：

```bash
python -m unittest discover -s tests -p 'test_output_correctness.py' -v
```
