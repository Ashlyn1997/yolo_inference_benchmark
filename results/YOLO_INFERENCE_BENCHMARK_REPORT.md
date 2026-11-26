# YOLO 推理部署与 TensorRT 性能评测报告

## 1. 项目一句话介绍

这是一个针对 **Ultralytics YOLO11n 目标检测模型** 的 NVIDIA GPU 推理部署评测项目。项目在同一套预处理、解码和 NMS 后处理下，对比：

```text
PyTorch CUDA → ONNX Runtime CUDA Execution Provider → TensorRT Engine
```

目标不是只导出一个 TensorRT Engine，而是建立完整的部署验证闭环：**环境诊断 → 模型导出 → 三后端一致性验证 → 预热后的延迟/吞吐压测 → FP32/FP16/INT8 和固定/动态 Batch 比较 → 结果与环境留档**。

它可迁移到真实的视觉检测业务：模型权重可以替换为业务 `best.pt`，图像源可以替换为业务验证集，测量框架与环境排障方法保持不变。

## 2. 项目背景与问题定义

YOLO 模型在训练阶段常以 `.pt` 权重和 PyTorch 运行，但线上部署通常更关注：

- Batch 1 的响应延迟是否满足实时要求；
- 批处理时每秒可处理多少张图像；
- TensorRT FP16/INT8 是否真的带来收益，精度是否仍可接受；
- ONNX Runtime 是否真的在调用 CUDA，而不是静默回退到 CPU；
- TensorRT Engine 是否和目标 GPU、TensorRT、driver 相匹配；
- 性能差异究竟来自模型执行、CPU 图像处理、NMS、GPU↔CPU 同步，还是测试环境本身。

因此项目没有直接拿 Ultralytics 的命令行速度打印做结论，而是将前后处理统一，单独记录预处理、后端执行、后处理和端到端时间。

## 3. 项目范围与非范围

### 3.1 已实现范围

| 模块 | 实现内容 | 价值 |
| --- | --- | --- |
| 模型 | 官方 `yolo11n.pt` 或可替换的 Ultralytics 检测模型 | 提供可复现基线，也可替换业务权重 |
| 后端 | PyTorch CUDA、ONNX Runtime CUDA EP、TensorRT 原生 Python API | 比较不同部署路径 |
| 预处理 | Letterbox、BGR→RGB、NCHW、`float32 / 255` | 消除后端之间的输入差异 |
| 后处理 | 多输出形状兼容、坐标还原、class-aware NMS | 三后端共享输出解释逻辑 |
| 性能 | warmup、500 次正式运行、均值/P50/P95/P99、吞吐、端到端拆分 | 不只看单次最快延迟 |
| 正确性 | 单图可视化、COCO 验证产物、pipeline 单元测试 | 防止“速度快但框错位” |
| 可运维性 | 环境快照、Dockerfile、NVIDIA 矩阵脚本、JSON/Markdown 报告 | 使性能数字可回溯 |

### 3.2 明确不在当前项目范围

- 多路摄像头解码、RTSP、消息队列、REST/gRPC 服务化和生产编排；
- GPU 端 NMS、零拷贝视频管线、CUDA Graph、Pinned Memory 和异步多 stream；
- 分割、姿态、OBB 或定制 detection head 的解码；
- 多 GPU、MIG、NVLink 或分布式推理；
- 用业务代表性校准集完成的 INT8 精度签核。

因此它是“模型执行与部署后端评测框架”，不是一个完整的视频检测平台。面试时主动说明边界，比把它包装成完整生产系统更可信。

## 4. 实际运行环境与版本矩阵

以下来自 `results/environment-info.json` 和 `results/nvidia-smi.txt`，采集时间为 **2026-08-12**：

| 层级 | 已验证版本 / 配置 |
| --- | --- |
| GPU | NVIDIA L20 × 1 |
| 可见显存 | 46,068 MiB（约 45 GiB；该数值以本项目当次快照为准） |
| Driver | 535.161.07 |
| `nvidia-smi` 所示 CUDA Driver API | 12.2 |
| OS / kernel | Linux 6.8.0-88-generic, x86_64, glibc 2.39 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| Ultralytics | 8.4.118 |
| ONNX Runtime GPU | 1.22.0 |
| TensorRT | 11.2.1.2 |

这里要区分：宿主机 `nvidia-smi` 的 CUDA 12.2 是 driver 能力级别；PyTorch `2.5.1+cu121` 实际使用 CUDA 12.1 runtime。容器/虚拟环境可以带 runtime，但不能替代宿主机 NVIDIA driver；这也是部署排障时必须记录完整矩阵的原因。

Dockerfile 设计为由 `BASE_IMAGE` 传入 NVIDIA TensorRT NGC 镜像，而不是固定一个“最新镜像”。当前仓库没有保存实际 NGC tag 或 image digest，故报告不能假装其已被锁定；生产发布应将 image digest 写入环境清单。

## 5. 模型制品与部署过程

### 5.1 模型与制品流转

```text
Ultralytics YOLO11n .pt
        │
        ├── PyTorch CUDA baseline
        │
        └── Ultralytics export
              ├── ONNX（可选 dynamic batch）→ ONNX Runtime CUDA EP
              └── TensorRT engine（FP32 / FP16 / INT8，fixed 或 dynamic profile）
                                                        │
业务或 COCO 图像 → 统一 preprocess → 后端执行 → 统一 decode / NMS → 可视化与性能报告
```

项目命令默认下载官方 `yolo11n.pt` 到 `artifacts/yolo11n.pt`，再导出 ONNX / TensorRT Engine。压测结果文件表明实际执行过 `coco128` 相关验证和 Engine 测试，但 `artifacts/` 已按设计忽略且不在当前工作区；因此目前不能给出 `.pt`、ONNX、Engine 的 SHA256，也不能给出 Engine build 时的精确 TensorRT builder log。

### 5.2 导出策略

项目支持三类 TensorRT 交付制品：

| 制品 | 用途 | 关键取舍 |
| --- | --- | --- |
| FP32 Engine | 精度/功能基线 | 通常最稳，但性能与显存成本更高 |
| FP16 Engine | L20 常规高性能推理路径 | 需要在验证集上确认精度可接受 |
| INT8 Engine | 低精度性能实验 | 必须用代表性业务数据校准并评测 mAP，不能只跑速度 |
| Fixed batch Engine | 已知、固定批量场景 | profile 简单、可能有更稳定的性能，但不能接收任意 batch |
| Dynamic batch Engine | batch 波动的在线/批处理场景 | 必须保证 Optimization Profile 覆盖 min/opt/max batch；灵活性可能伴随性能差异 |

代码约束了 `--half` 与 `--int8` 互斥，且没有 `--data` 代表性校准集时拒绝 INT8 导出。这个设计避免了“拿随机图像做 INT8 calibration”这种无法用于业务的假实验。

### 5.3 TensorRT Engine 加载兼容处理

Ultralytics 导出的 `.engine` 文件可能在原生 TensorRT plan 前加入长度+JSON 元数据前缀。若把整个文件直接交给 `deserialize_cuda_engine()`，可能导致反序列化失败。

项目的 `TensorRTBackend._unwrap_ultralytics_engine()` 会：

1. 读取前 4 个字节判断是否存在合法元数据长度；
2. 尝试解析 JSON 元数据；
3. 对 Ultralytics wrapper 去除前缀后再反序列化；
4. 对原生 TensorRT plan 保持原样。

该逻辑有单元测试覆盖。这体现的不是“会调用 TensorRT API”而是能处理模型导出工具与原生 runtime 的制品格式边界。

## 6. 推理实现：怎样保证三后端比较公平

### 6.1 统一前后处理

三后端只比较模型执行路径，公共 pipeline 的步骤是：

```text
原始 BGR 图像
  → 等比例 letterbox 到 640×640，边缘填充 (114,114,114)
  → BGR 转 RGB
  → HWC 转 NCHW、归一化到 [0,1]、float32 NumPy batch
  → 后端执行
  → 输出转 float32 NumPy
  → 解码 xywh / xyxy、置信度过滤、class-aware NMS
  → 去除 padding、缩放回原图坐标
```

后处理兼容常见 Ultralytics 检测输出 `[B, 4+C, N]`、`[B, N, 4+C]` 和端到端 `[B, N, 6]`。NMS 用纯 PyTorch 实现，避免依赖 torchvision 的 C++/CUDA 扩展；不同类别的 box 加类别偏移量后做 NMS，避免跨类别错误抑制。

### 6.2 三个后端的实际实现

| 后端 | 执行路径 | 关键防错设计 |
| --- | --- | --- |
| PyTorch | `YOLO(model).model.to(cuda).eval()`，可切 FP16 | `torch.inference_mode()`；测量前后显式同步 |
| ONNX Runtime | `InferenceSession(... CUDAExecutionProvider, CPUExecutionProvider)` | 初始化时先检查可用 provider，再检查 active provider，若 CUDA EP 未真正启用则报错，禁止静默 CPU fallback |
| TensorRT | 原生 Runtime / Engine / ExecutionContext，`execute_async_v3` 或兼容 v2 binding | 每个 batch 设置 input shape，按 Engine 实际 dtype 分配 GPU tensor，显式同步并复制结果到 CPU |

注意：三条路径都会将模型输出取回 CPU，再执行统一后处理。故“纯推理延迟”包含同步和 GPU→CPU 输出传输，适合衡量同步式服务 API 的可观察耗时；它不是纯 GPU kernel 时间，也不是全异步 pipeline 的极限吞吐。

## 7. 性能测量方法

每一组存档 JSON 的条件为：640×640、warmup 50 次、正式测量 500 次。每次记录：

- `preprocess_mean_ms`：letterbox、颜色转换、归一化等 CPU 前处理；
- `inference_mean/p50/p95/p99_ms`：显式 CUDA 同步边界内的后端执行时间；
- `postprocess_mean_ms`：输出解码、NMS、坐标还原；
- `end_to_end_mean_ms`：三段总耗时；
- `throughput_images_per_second`：仅按平均 inference time 计算的模型执行吞吐；
- `cuda_memory_allocated_mib`：PyTorch 可见的 allocated memory，并非所有 TensorRT/driver 显存。

吞吐公式为：

```text
throughput = batch_size × 1000 / inference_mean_ms
```

这解释了一个容易被问到的问题：batch 4 的 inference 吞吐可能很高，但单批端到端完成时间仍会高于 batch 1；选实时服务还是离线批处理，要看 SLA，而不能只挑最高 img/s。

## 8. 已保存的性能结果与结论

### 8.1 Batch 1、FP32 后端比较

来源：`results/coco128_batch1_fp32.json`。三组均为 L20、640×640、50 warmup、500 run。

| 后端 | 平均推理 | P50 / P95 / P99 | 纯推理吞吐 | 平均端到端 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch CUDA | 8.74 ms | 8.50 / 11.48 / 12.56 ms | 114.46 img/s | 13.70 ms | 可用的功能与性能基线 |
| ONNX Runtime CUDA | 5.08 ms | 2.89 / 19.68 / 20.71 ms | 196.93 img/s | 9.64 ms | 平均值更优，但 P95/P99 波动很大 |
| TensorRT FP32 | 2.68 ms | 2.50 / 4.23 / 4.72 ms | 372.57 img/s | 6.56 ms | 当前存档中最快且尾延迟更低 |

在这份同环境存档中，TensorRT FP32 平均推理较 PyTorch 快约 **3.25×**（8.74 / 2.68），较 ONNX Runtime 快约 **1.89×**（5.08 / 2.68）。这是该模型、该输入尺寸、该版本矩阵、该同步测量口径下的结果，不应宣传为任何 YOLO 模型或所有 GPU 的通用倍数。

ONNX Runtime 的 P50 仅 2.89ms、但 P95 为 19.68ms，是重要异常信号：均值不能掩盖尾延迟。当前仓库没有保存这次异常对应的 GPU 利用率、时钟、CPU 负载、provider debug log 或运行时 trace，因此只能确认现象，不能给出确定根因。正确做法是冻结 source、CPU governor、GPU clocks 和容器版本后复跑，并记录 ORT profiling / CUDA trace。

### 8.2 TensorRT FP16：Batch 1 性能

来源：`results/coco128_batch1_trt_fp16.json`。

| 配置 | 平均推理 | P95 | 推理吞吐 | 平均端到端 |
| --- | ---: | ---: | ---: | ---: |
| TensorRT FP32 / Batch 1 | 2.68 ms | 4.23 ms | 372.57 img/s | 6.56 ms |
| TensorRT FP16 / Batch 1 | **1.64 ms** | **2.59 ms** | **608.92 img/s** | **5.52 ms** |

在这两份存档中，FP16 的平均模型执行吞吐相对 FP32 提升约 **1.63×**。端到端没有同等比例缩短，是因为预处理和后处理仍占约 3.85ms；这正是部署优化不能只盯 Engine 的原因。

### 8.3 Fixed / Dynamic Batch 的实测矩阵

以下是 TensorRT FP16 的存档结果。动态 profile 来源为 `coco128_trt_fp16_dynamic_*`，固定 batch 来源为 `coco128_trt_fp16_fixed_*`。

| Engine 类型 | Batch | 平均推理 | P95 | 推理吞吐 | 平均端到端 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dynamic FP16 | 1 | 1.48 ms | 1.50 ms | 675.38 img/s | 3.91 ms |
| Dynamic FP16 | 4 | 3.07 ms | 3.36 ms | **1301.30 img/s** | 13.99 ms |
| Dynamic FP16 | 8 | 11.46 ms | 15.58 ms | 698.13 img/s | 39.32 ms |
| Dynamic FP16 | 16 | 32.94 ms | 33.58 ms | 485.67 img/s | 87.20 ms |
| Fixed FP16 | 4 | 3.07 ms | 3.25 ms | **1302.24 img/s** | 14.41 ms |
| Fixed FP16 | 8 | 9.42 ms | 15.73 ms | 849.23 img/s | 37.64 ms |

可以负责任地得出的结论：

1. 在当前 640×640、同步 Python pipeline 的样本中，**Batch 4 是最高的纯推理吞吐点**，约 1301 img/s。
2. Batch 增至 8/16 后吞吐反而下降，端到端耗时显著增加；不能简单按“batch 越大越快”设生产批量。
3. Fixed Batch 4 与 Dynamic Batch 4 的平均推理几乎相同；Batch 8 下 fixed 的平均吞吐高于 dynamic，但 P95 同样约 15.7ms。是否选择 dynamic，必须由实际 batch 波动和服务 SLA 决定。
4. 此处没有 GPU utilization、功耗、显存和 CPU profile 的连续采样，不能把 Batch 8/16 的下降直接定性为“GPU 已饱和”。更可能是 batch profile、CPU 前后处理、输出回传、调度或资源状态共同影响，需 profile 后判断。

### 8.4 INT8 的现有结果：不能过度解读

来源：`results/coco128_trt_int8_fixed_batch4.json`。

| 配置 | 平均推理 | P95 | 推理吞吐 | 结论 |
| --- | ---: | ---: | ---: | --- |
| TensorRT FP16 fixed / Batch 4 | 3.07 ms | 3.25 ms | 1302.24 img/s | 当前较优性能基线 |
| TensorRT INT8 fixed / Batch 4 | 3.13 ms | 3.39 ms | 1279.16 img/s | 本次实验中未快于 FP16 |

INT8 在存档结果中没有出现性能优势（平均推理还略慢，约 1.8%）。同时项目文档明确标记：该 INT8 仅使用 COCO128 作为实验校准，且当前没有可追溯的 FP16/INT8 `mAP50-95` 数值。因此不能写“INT8 加速且无精度损失”。

严谨的下一步是：使用真实业务代表性校准集构建 INT8 Engine，在同一冻结验证集上输出 FP32/FP16/INT8 的 mAP50-95、precision、recall、按类漏检率，再决定是否上线。

## 9. 正确性验证与精度结论

项目中保留了多个 `runs/detect/coco128_*` 的验证可视化产物，包括 PR、P/R/F1 曲线、confusion matrix，以及 GT/prediction 示例图，说明至少执行过 Ultralytics 检测验证流程。单图推理也会分别输出 PyTorch、ONNX Runtime 和 TensorRT 的可视化框图，便于检查框数量和坐标位置。

但这些目录中没有保留可机器读取的 `results.csv` 或明确的 mAP 标量。因此当前报告的精度结论只能是：

- 已有验证流程和可视化证据；
- **没有足够的结构化结果支持“mAP 为多少”或“FP16/INT8 无精度下降”**；
- 正式交付必须保存验证命令、dataset YAML、权重/Engine hash、Ultralytics 版本、`results.csv` 和按类别指标。

这是 AI 推理部署工程师和芯片 FAE 应有的报告边界：速度、功能正确性和精度是三套证据，不能相互替代。

## 10. 项目中出现的问题、定位与工程处理

### 10.1 Driver、CUDA runtime 与推理库版本匹配

**现象/风险：**宿主机 driver、PyTorch CUDA wheel、ONNX Runtime CUDA EP、TensorRT wheel/容器来自不同版本时，可能报 driver too old、找不到 `libcudnn`/`libcublas`，或后端不可用。

**项目处理：**

- 保存 `nvidia-smi`、Python、PyTorch、`torch.version.cuda`、Ultralytics、ORT、TensorRT 的环境快照；
- 在 NVIDIA Dockerfile 中由 NGC TensorRT 基础镜像提供 TensorRT，避免随意在业务 requirements 中混装；
- 当前实测矩阵落在 L20 + Driver 535.161.07 + CUDA runtime 12.1 + PyTorch 2.5.1+cu121 + TensorRT 11.2.1.2；
- Dockerfile 不固定最新镜像，而要求显式传入与目标 driver 兼容的 `BASE_IMAGE`。

**面试表达：**我把“模型加载失败”先收敛为 driver、user-space CUDA runtime、Python wheel、动态库四层兼容性问题，再用最小 `torch.cuda.is_available()`、ORT provider 和 TensorRT import 逐层验证，而不是先怀疑模型权重。

### 10.2 ONNX Runtime 静默回退 CPU 的风险

**问题：**安装了 CPU 版 `onnxruntime` 或 CUDA/cuDNN 库不匹配时，ORT 可能只使用 `CPUExecutionProvider`，性能对比就失去意义。

**项目处理：**`OnnxRuntimeBackend` 在创建 Session 前检查 `ort.get_available_providers()`，创建后再检查 `session.get_providers()`；如果 active providers 中没有 `CUDAExecutionProvider`，直接抛异常，拒绝继续 benchmark。

**仍需关注：**即使 CUDA EP 已启用，`coco128_batch1_fp32.json` 的 ORT P95/P99 仍明显高于 P50，说明还需要对 provider options、CPU 线程、GPU clock、动态 shape 和运行时 profile 做深入分析。

### 10.3 Ultralytics `.engine` 与原生 TensorRT plan 格式差异

**问题：**部分 Ultralytics Engine 包含 JSON metadata 前缀，原生 TensorRT API 直接反序列化会失败。

**项目处理：**在加载器中识别并剥离 metadata；如果是普通原生 plan 则不改字节流。该兼容分支有单元测试。

### 10.4 Fixed Engine 不能直接拿来测任意 Batch

**问题：**Batch 1 的 fixed Engine 没有 batch 8 的输入 profile；强行改 batch 会导致 shape 设置失败，或无法执行。

**项目处理：**导出阶段明确区分 `--batch` 与 `--dynamic`；运行手册要求固定 batch 分别构建 Engine，动态 Engine 则确保 Optimization Profile 覆盖目标 Batch。压测矩阵中也将 fixed/dynamic 结果分开保存，避免混淆。

### 10.5 INT8 速度不一定优于 FP16，精度更不能默认成立

**问题：**当前 INT8 Batch 4 结果没有快于 FP16，且 COCO128 不代表生产校准分布。

**项目处理：**代码强制 INT8 导出提供 calibration dataset YAML；报告将此 INT8 结果标为实验数据，不作生产精度承诺。

**下一步：**换业务校准集、保存校准 cache，查看 TensorRT build log 中是否存在算子 fallback，结合 Nsight Systems/Compute 或 TensorRT profiler 确认是否真实使用低精度 kernel。

### 10.5.1 没有 L20 时，如何做性能定位

**结论：**可以推进排障和优化准备，但不能在非 L20 机器上宣称 L20 的性能结论。建议按下列层次工作：

| 手上资源 | 可以做什么 | 不能得出的结论 |
| --- | --- | --- |
| 仅本地 CPU / 无 NVIDIA GPU | 验证预处理、后处理、模型导出、Engine metadata 解析、单元测试、输入输出一致性和报告自动化 | GPU kernel 性能、CUDA EP 实际执行、TensorRT 吞吐 |
| 其他 NVIDIA GPU | 验证 TensorRT build/加载、INT8 calibration 流程、Nsight 使用方法、CPU 前后处理比例、算子 fallback 和瓶颈结构 | L20 的绝对 ms、img/s、功耗、显存容量、最优 batch |
| 可远程登录 L20 | 在目标机/同一容器上采集最小 benchmark、Nsight trace 与环境快照 | 不应把单次 trace 泛化成所有业务负载 |
| 无 L20 登录权限 | 将可复现命令、容器 image digest、模型/Engine hash 与采集清单交给有机器权限的同事/客户执行，取得原始 trace 后分析 | 不能伪造 profile 结论 |

实际推荐顺序如下：

1. **先用当前 benchmark 拆分耗时。**项目已输出 preprocess、inference、postprocess、端到端均值和 P95。以 FP16 Batch 1 为例，模型执行约 1.64ms、端到端约 5.52ms，已经说明后端 Engine 不是唯一优化点。
2. **先看 TensorRT builder log 与 runtime profile。**确认 INT8 Engine 的 layer precision、是否发生 FP32/FP16 fallback、Optimization Profile 的 min/opt/max shape；这是低成本、与 GPU 型号关系较弱的第一步。
3. **在目标 L20 上使用 Nsight Systems 做时间线归因。**它回答“时间花在哪”：CPU 预处理、H2D/D2H copy、TensorRT enqueue、CUDA kernel、同步等待，还是线程调度/空洞。对本项目应围绕固定输入、固定 batch、warmup 后的短测量窗口采集，避免 trace 文件过大。
4. **只对明确的热点使用 Nsight Compute。**它回答“某个 kernel 为什么慢”：Tensor Core 是否利用、显存带宽、occupancy、warp stall、算子精度。它开销较大，不应用于全流程盲扫；先由 Nsight Systems 或 TensorRT profiler 锁定热点 kernel/层。
5. **每次 profile 都绑定同一证据包。**保存 `nvidia-smi`、driver、容器 image digest、PyTorch/ORT/TensorRT 版本、模型与 Engine hash、输入尺寸、batch、profile、warmup/runs 和命令行。没有这些，trace 不可比较。

可交付给目标 L20 机器执行的最小采集示例（根据实际 Nsight 安装路径调整）：

```bash
# 先保留环境与基准结果；--runs 要缩小，避免 Nsight trace 过大
python -m yolo_benchmark.cli env > results/env_before_profile.json
nvidia-smi > results/nvidia_smi_before_profile.txt

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output results/trt_fp16_b4 \
  python -m yolo_benchmark.cli benchmark \
    --engine artifacts/yolo11n_trt_fp16_b4.engine \
    --source datasets/business_val/images \
    --imgsz 640 --batch 4 --warmup 20 --runs 50 \
    --output results/trt_fp16_b4_profiled.json
```

在没有 L20 的阶段，最有价值的产出不是猜测“INT8 为什么慢”，而是准备好可在目标机一键执行的复现包，并从已有数据提出可检验假设，例如：`INT8 是否发生 layer fallback`、`Batch 8 是否受 CPU 前后处理/输出回传影响`、`Engine profile 的 opt batch 是否为 4`。拿到目标机 trace 后再逐条验证。

### 10.5.2 Nsight Systems / Nsight Compute 实操：针对本项目的验证流程

两者不要混用：

| 工具 | 首先回答的问题 | 本项目中要观察什么 |
| --- | --- | --- |
| **Nsight Systems (`nsys`)** | 端到端时间花在哪里？CPU 与 GPU 是否有空洞、排队或同步等待？ | preprocess、H2D/D2H、TensorRT enqueue、CUDA kernel、NMS、`torch.cuda.synchronize()` 之间的时间线 |
| **Nsight Compute (`ncu`)** | 已定位的某个 GPU kernel 为什么慢？ | Tensor Core 利用、SM occupancy、显存吞吐、warp stall、实际 precision/kernel 路径 |

正确顺序必须是 **先 `nsys`，后 `ncu`**。如果不先缩小热点，`ncu` 会对大量 TensorRT 内部 kernel 做 replay，耗时长且难以解释。

#### A. 在目标机开始前：最小环境检查

以下命令必须在**实际加载 Engine 的 L20 主机或容器内**执行；不要在 Mac、无 GPU 的跳板机或与生产不同的虚拟环境中采集：

```bash
cd /workspace/yolo-inference-benchmark  # 替换为真实项目路径

nvidia-smi
nsys --version
ncu --version
nsys status --environment
python -m yolo_benchmark.cli env
```

开始采集前应停止其他 GPU 负载，并固定：Engine 文件、图像目录、输入 `640×640`、batch、阈值、软件版本和 warmup 次数。保存 `nvidia-smi` 输出、模型/Engine SHA256、容器 image digest 和完整命令行。

若在 Docker 中运行，`nsys status --environment` 若提示 `perf_event_open` 或 CPU sampling 权限问题，需要由环境管理员调整 Docker capability/seccomp 或宿主机 `perf_event_paranoid` 配置；不要为了图省事在共享生产机上直接使用无边界的 `--privileged`。Nsight Compute 若报 `ERR_NVGPUCTRPERM`，说明当前用户不能读取 GPU performance counters；应由管理员按 NVIDIA 安全策略授予 profiling 权限，或在获授权的维护窗口使用 sudo。

#### B. 用 Nsight Systems 先做一次“窄采集”

先只采 CUDA/NVTX/OS runtime，不启用 CPU sampling，以降低开销和 trace 文件大小。由于项目 benchmark 已有 warmup，因此设置较小的正式运行次数即可：

```bash
mkdir -p results/profiles

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=results/profiles/trt_fp16_b4_nsys \
  python -m yolo_benchmark.cli benchmark \
    --engine artifacts/yolo11n_trt_fp16_b4.engine \
    --source datasets/business_val/images \
    --imgsz 640 --batch 4 \
    --warmup 20 --runs 50 \
    --output results/trt_fp16_b4_profiled.json
```

如果要对比 INT8，只改变 Engine 路径和结果文件名；不要在同一条命令里同时比较多个 Engine。一次 trace 只回答一个问题，例如“FP16 Batch 4 的端到端 CPU/GPU 空洞在哪里？”

采集后将生成 `results/profiles/trt_fp16_b4_nsys.nsys-rep`。可将该文件下载到本地，用 Nsight Systems GUI 打开；也可先在目标机上查看 CLI 支持的统计报告：

```bash
nsys stats --list-reports results/profiles/trt_fp16_b4_nsys.nsys-rep
nsys stats results/profiles/trt_fp16_b4_nsys.nsys-rep
```

在 GUI 中按以下顺序检查：

1. **CUDA API row：**`cudaMemcpy*`、`cudaStreamSynchronize`、`cudaDeviceSynchronize` 是否占据大量 CPU 时间；
2. **CUDA GPU Kernels row：**TensorRT kernel 是否连续执行，是否有明显 GPU idle gap；
3. **CUDA Memory row：**H2D/D2H copy 是否与 kernel 重叠，输出回传是否成为瓶颈；
4. **OS Runtime / CPU thread row：**OpenCV preprocess、Python NMS 是否使 GPU 等待；
5. **分位延迟异常时段：**对照 benchmark 的 P95/P99，确认长尾来自 CPU 抖动、GPU 空洞还是个别长 kernel。

若时间线显示 CPU 函数归因不够清楚，再单独执行有采样的短采集，而不是每次都打开高开销选项：

```bash
nsys profile \
  --trace=cuda,osrt,nvtx \
  --python-backtrace=cuda \
  --python-sampling=true \
  --force-overwrite=true \
  --output=results/profiles/trt_fp16_b4_cpu \
  python -m yolo_benchmark.cli benchmark \
    --engine artifacts/yolo11n_trt_fp16_b4.engine \
    --source datasets/business_val/images \
    --imgsz 640 --batch 4 --warmup 10 --runs 20 \
    --output results/trt_fp16_b4_cpu_profiled.json
```

这个采集用于定位 CPU 调用栈，不应用于发布性能数字；profiling 本身会改变时延。

#### C. 为更容易分析，给项目加 NVTX 区间（建议项）

当前 benchmark 已按代码分别计时 preprocess/inference/postprocess，但时间线里没有业务命名区间。可在 `run_benchmark()` 的三段周围加 NVTX range（仅用于 profiling，不改变推理逻辑）：

```python
import torch

torch.cuda.nvtx.range_push("preprocess")
batch, metas = preprocess(images, size)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("inference")
output = backend.infer(batch)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("postprocess")
postprocess(output, metas, confidence, iou)
torch.cuda.nvtx.range_pop()
```

加完后，用同一条 `nsys --trace=cuda,nvtx,osrt ...` 命令采集，GUI 中就能直接按 `preprocess / inference / postprocess` 解释时间线。NVTX 是分析可读性增强，不是性能优化本身；提交前应通过 `pytest -q` 和未加 profiler 的基准确认无功能回归。

#### D. 根据 Systems 的热点，使用 Nsight Compute 精测一个 kernel

不要对 500 次 benchmark 直接跑 `ncu --set full`。Nsight Compute 为收集硬件指标可能多次 replay kernel，运行时间会显著增加。应从 Systems 中抄下占 GPU 时间最长的 kernel 名称，先用小样本、较少的 launch 确认能够捕获：

```bash
# 先列出本机实际支持的指标/section；不要从网上硬抄某张卡的 metric 名称
ncu --query-metrics | head -n 40
ncu --list-sets

# 第一次只收一个或少数 kernel；<kernel_regex> 必须替换为 Systems 中看到的名字
ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:<kernel_regex>' \
  --launch-skip 20 \
  --launch-count 1 \
  --set full \
  --force-overwrite \
  --export results/profiles/trt_fp16_b4_hotspot \
  python -m yolo_benchmark.cli benchmark \
    --engine artifacts/yolo11n_trt_fp16_b4.engine \
    --source datasets/business_val/images \
    --imgsz 640 --batch 4 --warmup 5 --runs 5 \
    --output results/trt_fp16_b4_ncu.json
```

`--launch-skip` 用来避开 warmup 阶段，`--launch-count` 限制被 profile 的匹配 kernel 次数；二者应根据实际 kernel 重复次数调整。若使用的 Nsight Compute 版本不支持 `--set full` 或导出格式，先执行 `ncu --help` / `ncu --list-sets`，以目标机安装版本输出为准。

在 Nsight Compute GUI 中优先看：

- **Speed Of Light / GPU Speed Of Light：**热点是 compute-bound 还是 memory-bound；
- **Tensor Core utilization / instruction 类型：**FP16/INT8 Engine 是否实际使用预期的低精度 Tensor Core 路径；
- **Memory Workload Analysis：**DRAM/L2 吞吐是否接近可用带宽；
- **Occupancy / Warp State：**低 occupancy、依赖 stall、memory throttle 或同步 stall 的原因；
- **Source / SASS（仅在必要时）：**确认 kernel 属于 TensorRT layer/tactic，不能只依据名字猜测。

#### E. 怎样验证“INT8 为什么没有 FP16 快”

对 FP16 和 INT8 分别执行 **同一批输入、同一 batch、同一 profile、同一热身策略** 的 Systems 采集，填写下表，而不是直接比较一个均值：

| 验证项 | FP16 | INT8 | 要得出的判断 |
| --- | --- | --- | --- |
| TensorRT build log 的 layer precision | 待采集 | 待采集 | INT8 是否发生 FP16/FP32 fallback |
| GPU kernel 总时间 / 空洞 | 待采集 | 待采集 | 是否真的降低 GPU 执行时间 |
| H2D/D2H 与 CPU 前后处理 | 待采集 | 待采集 | 若相同，INT8 不会改善该部分 |
| Top kernel 的 Tensor Core 指令 / 吞吐 | 待采集 | 待采集 | 是否使用预期低精度计算路径 |
| mAP50-95、按类 recall | 待补测 | 待补测 | 性能变化是否以业务精度为代价 |

只有当 build log、Systems、Compute 和验证集精度四类证据一致时，才可以将 INT8 判定为可上线方案。

NVIDIA 官方文档对 Nsight Systems 的 trace、NVTX 和 Python profiling 选项有说明；Nsight Compute 文档说明了 kernel filter、launch skip/count，以及 performance counter 权限限制。[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

### 10.6 端到端优化的瓶颈可能在 CPU，而不只在 Engine

**现象：**FP16 Batch 1 的模型执行约 1.64ms，但端到端均值仍为 5.52ms；前处理+后处理约 3.85ms，占端到端的大部分。

**项目结论：**仅替换 TensorRT Engine 不足以使业务端到端时延同比下降。后续可评估批量预处理、线程池、Pinned Memory、GPU NMS、异步 stream 和更少的输出回传；但这些属于下一阶段优化，当前项目尚未实现，不能写成既有成果。

## 11. 测试质量、局限与补齐计划

| 当前限制 | 为什么影响结论 | 建议补齐 |
| --- | --- | --- |
| 部分基准默认可用合成图像 | 无法代表真实分辨率、目标密度和 NMS 负载 | 以冻结业务图像集压测，并记录 token/图像统计替代信息：分辨率、目标数量、文件大小 |
| 未保留 mAP 标量 | 无法签核精度 | 保留 Ultralytics `results.csv`、按类 PR、mAP50-95 |
| Engine / 权重 hash 缺失 | 无法复现同一制品 | 发布清单写入 `.pt`/ONNX/Engine SHA256、builder log、image digest |
| 无连续 GPU telemetry | 无法判断限速来自 GPU、CPU 还是功耗/时钟 | 记录 GPU util、memory、power、temperature、clock、CPU load |
| 仅同步 Python 调用 | 不代表异步视频服务的极限能力 | 另建 async/pinned-memory/GPU-NMS 实验，并明确新测量口径 |
| 每项存档单轮 500 run | 可能受瞬时负载影响 | 至少 3 轮，报告均值、标准差、min/max，清理其他 GPU 进程 |

## 12. 面向岗位的项目价值

### 12.1 三类岗位的共同能力

| 共同能力 | 本项目证据 |
| --- | --- |
| 模型制品工程化 | `.pt → ONNX → TensorRT Engine`，处理 Ultralytics metadata 封装 |
| 异构软件栈兼容 | Driver / CUDA runtime / PyTorch / ORT / TensorRT 版本矩阵与容器化 |
| 性能方法论 | warmup、500-run 分位延迟、吞吐、端到端拆分、fixed/dynamic batch 矩阵 |
| 质量守门 | 统一预后处理、NMS 单测、检测可视化、精度与速度分开论证 |
| 故障定位 | 防 ORT CPU fallback、Engine shape/profile、INT8 calibration、动态库兼容边界 |

### 12.2 推理部署 / 模型部署工程师侧重点

你可以强调：将 YOLO 从开发态 PyTorch 基线转成面向 L20 的 TensorRT 制品，建立可复现 benchmark，发现并量化 FP16、Batch 和 CPU 前后处理对端到端延迟的影响；同时通过 Docker/环境快照降低部署漂移风险。

合适的简历表述：

> 在 NVIDIA L20 上完成 YOLO11n 的 PyTorch、ONNX Runtime CUDA 与 TensorRT 多后端部署评测，构建统一前后处理、CUDA EP 校验、Engine 兼容加载及 50 次预热/500 次采样基准框架；存档测试中 TensorRT FP16 Batch 1 达 1.64ms 平均推理、608.9 img/s，Batch 4 达约 1302 img/s，并识别端到端 CPU 前后处理与 INT8 校准质量为后续优化重点。

上述数字必须加上“存档测试中 / L20 / 640×640 / 当前版本矩阵”的上下文，不要脱离条件说成通用能力。

### 12.3 AI 芯片 FAE 侧重点

你可以强调：能将客户的“模型慢/Engine 加载失败/精度可疑”转成可复现的软硬件联合矩阵；不仅给建议，还能提供模型、输入、driver/runtime、Engine、profile、延迟分位、吞吐、精度和日志的闭环证据。

合适的表达：

> 基于 YOLO11n 在 L20 上完成 TensorRT FP32/FP16/INT8 与 fixed/dynamic batch 矩阵验证；针对 ORT CUDA provider、TensorRT Engine 制品格式、shape profile 及 CUDA runtime/driver 兼容风险实现前置检查，并用原始 JSON 输出延迟分位与吞吐，为客户硬件选型和性能 bring-up 提供可复现基线。

## 13. 面试官可能追问的问题与参考回答

**Q1：为什么 TensorRT 比 PyTorch 快？**  
**答：**TensorRT 在 build 阶段会做图优化、算子融合、精度选择和 kernel/tactic 选择，并为目标 GPU 构建 Engine；PyTorch eager 路径通常保留更多框架调度开销。本项目的 L20 存档中，Batch 1 FP32 TensorRT 平均执行 2.68ms，PyTorch 为 8.74ms。但我会同时说明测量含同步和输出回传，不把它说成纯 kernel 倍数或通用结论。

**Q2：为什么 Batch 16 的吞吐反而不如 Batch 4？**  
**答：**当前数据只能说明在这套 Engine、640×640 和同步 Python pipeline 下 Batch 4 约 1301 img/s，而 Batch 16 约 486 img/s。原因不能凭数字直接断言为 GPU 饱和，可能包括 TensorRT profile 的 opt shape、CPU 前后处理、输出回传、资源状态或动态 shape 选择。下一步我会固定 profile、采集 GPU/CPU telemetry 和 TensorRT profiler，再进行归因。

**Q3：为什么不只看 QPS/吞吐？**  
**答：**实时摄像头看的是单请求/单帧 P95 与端到端时延；离线批处理才更看 img/s。Batch 4 吞吐最高，不代表适合 30 FPS 的实时单流。项目同时输出 P50/P95/P99 与前后处理耗时，避免只给一个平均吞吐误导业务。

**Q4：如何确认 ONNX Runtime 没有跑 CPU？**  
**答：**不能只看安装包名。我会检查 `ort.get_available_providers()` 中是否有 `CUDAExecutionProvider`，创建 session 后再检查 `session.get_providers()` 是否实际激活 CUDA EP。本项目将缺失 CUDA EP 视为错误并中止压测，而不是接受 CPU fallback 结果。

**Q5：TensorRT Engine 为什么不能拷到任意机器运行？**  
**答：**Engine 通常与 GPU 计算能力、TensorRT 版本、部分 CUDA/driver 环境以及 build profile 相关。最可靠实践是在最终部署机、或架构和软件栈严格兼容的机器上构建并验证；发布时记录 Engine hash、builder log、GPU、TensorRT 和镜像 digest。

**Q6：INT8 结果略慢，你会怎么排？**  
**答：**先确认 Engine 真的是 INT8、build log 是否有算子回退以及校准 cache 是否生效；再检查 TensorRT profile、输入 batch、CPU 拷贝和计时口径；最后在代表性业务校准集和验证集上同时比较 mAP。当前实验没有证据支持 INT8 上线，我不会因为名称是 INT8 就承诺加速。

**Q7：如何把当前工具接到生产服务？**  
**答：**保持 `preprocess → infer → postprocess` 的清晰接口，在上层增加 REST/gRPC、batcher/队列、超时/限流、模型版本路由、Prometheus 指标和健康检查。性能测试要由单进程 CLI 扩展为压测客户端 + 服务端 tracing，并重新定义端到端口径。当前项目只完成底层推理基准，不会假称已部署完整在线服务。

## 14. 后续行动清单

1. 固定业务模型 `.pt`、ONNX、Engine、tokenizer 不涉及本视觉项目，改为保存权重/Engine SHA256 与 dataset YAML hash；
2. 对真实业务验证集跑 FP32/FP16/INT8 的 mAP50-95、按类 precision/recall；
3. 用业务校准集重建 INT8 Engine，保留 TensorRT build log、calibration cache；
4. 在同一容器和空闲 GPU 上重复 3 轮 benchmark，加入 GPU power/clock/util/memory 与 CPU 指标；
5. 对 ORT P95 异常执行 profiling，确认 CUDA EP provider options 和动态库实际加载路径；
6. 依据实时/离线 SLA 选择 Batch，而不是仅以最高 img/s 选择 Batch 4；
7. 若进入服务化，补充异步 batcher、GPU NMS/零拷贝实验，并与当前同步基准分开报告。

## 15. 可追溯资产索引

| 资产 | 作用 |
| --- | --- |
| `yolo_inference_benchmark/yolo_benchmark/backends.py` | PyTorch / ORT / TensorRT 后端与 Engine metadata 兼容逻辑 |
| `yolo_inference_benchmark/yolo_benchmark/pipeline.py` | 统一 preprocess、decode、NMS、坐标还原 |
| `yolo_inference_benchmark/yolo_benchmark/benchmark.py` | warmup、统计分位延迟、吞吐和 JSON 报告 |
| `yolo_inference_benchmark/yolo_benchmark/cli.py` | env / download / export / infer / benchmark 命令 |
| `yolo_inference_benchmark/docker/Dockerfile` | 基于 NVIDIA TensorRT 镜像的可复现环境骨架 |
| `yolo_inference_benchmark/results/environment-info.json` | 实测环境版本矩阵 |
| `yolo_inference_benchmark/results/coco128_*.json` | 原始 benchmark 数据 |
| `yolo_inference_benchmark/runs/detect/coco128_*` | 检测验证曲线、混淆矩阵和预测可视化 |
| `yolo_inference_benchmark/scripts/run_nvidia_matrix.sh` | 可重复执行的 NVIDIA TensorRT 测试矩阵 |
