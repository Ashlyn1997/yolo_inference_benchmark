# NVIDIA TensorRT Benchmark Summary

Generated: 2026-11-28 04:59 UTC

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0

## Performance Results

| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_b4_repeat1 | tensorrt | fp16 | 4 | 14.94 ms | 15.52 ms | 267.80 img/s | 41.56 ms | 33.00 W | 8.116 | `fp16_b4_repeat1` |
| fp16_b4_repeat2 | tensorrt | fp16 | 4 | 14.73 ms | 15.34 ms | 271.57 img/s | 38.66 ms | 33.59 W | 8.085 | `fp16_b4_repeat2` |
| fp16_b4_repeat3 | tensorrt | fp16 | 4 | 15.00 ms | 15.50 ms | 266.64 img/s | 42.36 ms | 33.25 W | 8.019 | `fp16_b4_repeat3` |
| fp16_b8_repeat1 | tensorrt | fp16 | 8 | 33.01 ms | 42.43 ms | 242.35 img/s | 88.27 ms | 32.87 W | 7.374 | `fp16_b8_repeat1` |
| fp16_b8_repeat2 | tensorrt | fp16 | 8 | 33.17 ms | 42.80 ms | 241.20 img/s | 89.40 ms | 32.94 W | 7.323 | `fp16_b8_repeat2` |
| fp16_b8_repeat3 | tensorrt | fp16 | 8 | 33.15 ms | 42.75 ms | 241.36 img/s | 90.48 ms | 32.98 W | 7.319 | `fp16_b8_repeat3` |

## Repeat Aggregate

| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16_b4 | 3 | 14.89 ms | 0.116 ms | 0.78% | 268.67 img/s | 40.86 ms |
| fp16_b8 | 3 | 33.11 ms | 0.069 ms | 0.21% | 241.64 img/s | 89.38 ms |

## Measurement Notes

- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.
- Compare only rows with the same image size, dataset source, warmup count, and measured run count.
- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.
