# NVIDIA TensorRT Benchmark Summary

Generated: 2026-11-28 05:16 UTC

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0

## Performance Results

| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monitor_off_b8_repeat1 | tensorrt | fp16 | 8 | 33.02 ms | 42.49 ms | 242.31 img/s | 88.64 ms | - W | - | `monitor_off_b8_repeat1` |
| monitor_off_b8_repeat2 | tensorrt | fp16 | 8 | 33.20 ms | 42.87 ms | 240.99 img/s | 90.11 ms | - W | - | `monitor_off_b8_repeat2` |
| monitor_off_b8_repeat3 | tensorrt | fp16 | 8 | 33.04 ms | 42.68 ms | 242.12 img/s | 88.65 ms | - W | - | `monitor_off_b8_repeat3` |
| monitor_on_b8_repeat1 | tensorrt | fp16 | 8 | 33.20 ms | 42.84 ms | 240.95 img/s | 92.32 ms | 32.57 W | 7.398 | `monitor_on_b8_repeat1` |
| monitor_on_b8_repeat2 | tensorrt | fp16 | 8 | 33.24 ms | 42.93 ms | 240.67 img/s | 90.94 ms | 33.12 W | 7.266 | `monitor_on_b8_repeat2` |
| monitor_on_b8_repeat3 | tensorrt | fp16 | 8 | 33.16 ms | 42.77 ms | 241.25 img/s | 91.24 ms | 33.02 W | 7.306 | `monitor_on_b8_repeat3` |

## Repeat Aggregate

| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| monitor_off_b8 | 3 | 33.08 ms | 0.080 ms | 0.24% | 241.81 img/s | 89.13 ms |
| monitor_on_b8 | 3 | 33.20 ms | 0.032 ms | 0.10% | 240.96 img/s | 91.50 ms |

## Measurement Notes

- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.
- Compare only rows with the same image size, dataset source, warmup count, and measured run count.
- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.
