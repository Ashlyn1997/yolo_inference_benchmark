# NVIDIA TensorRT Benchmark Summary

Generated: 2026-11-28 05:53 UTC

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0

## Performance Results

| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_b4_original_repeat1 | tensorrt | fp16 | 4 | 14.96 ms | 15.43 ms | 267.33 img/s | 40.94 ms | - W | - | `fp16_b4_original_repeat1` |
| fp16_b4_original_repeat2 | tensorrt | fp16 | 4 | 14.88 ms | 15.33 ms | 268.74 img/s | 41.91 ms | - W | - | `fp16_b4_original_repeat2` |
| fp16_b4_original_repeat3 | tensorrt | fp16 | 4 | 14.83 ms | 15.23 ms | 269.78 img/s | 40.33 ms | - W | - | `fp16_b4_original_repeat3` |
| fp16_b4_pinned_repeat1 | tensorrt | fp16 | 4 | 13.24 ms | 13.72 ms | 302.19 img/s | 37.57 ms | - W | - | `fp16_b4_pinned_repeat1` |
| fp16_b4_pinned_repeat2 | tensorrt | fp16 | 4 | 13.36 ms | 13.83 ms | 299.30 img/s | 38.08 ms | - W | - | `fp16_b4_pinned_repeat2` |
| fp16_b4_pinned_repeat3 | tensorrt | fp16 | 4 | 13.43 ms | 13.83 ms | 297.88 img/s | 38.47 ms | - W | - | `fp16_b4_pinned_repeat3` |
| fp16_b8_original_repeat1 | tensorrt | fp16 | 8 | 30.75 ms | 36.15 ms | 260.14 img/s | 81.25 ms | - W | - | `fp16_b8_original_repeat1` |
| fp16_b8_original_repeat2 | tensorrt | fp16 | 8 | 30.70 ms | 36.02 ms | 260.58 img/s | 79.97 ms | - W | - | `fp16_b8_original_repeat2` |
| fp16_b8_original_repeat3 | tensorrt | fp16 | 8 | 30.98 ms | 36.27 ms | 258.20 img/s | 82.12 ms | - W | - | `fp16_b8_original_repeat3` |
| fp16_b8_pinned_repeat1 | tensorrt | fp16 | 8 | 24.22 ms | 24.82 ms | 330.33 img/s | 88.44 ms | - W | - | `fp16_b8_pinned_repeat1` |
| fp16_b8_pinned_repeat2 | tensorrt | fp16 | 8 | 24.29 ms | 24.84 ms | 329.34 img/s | 89.63 ms | - W | - | `fp16_b8_pinned_repeat2` |
| fp16_b8_pinned_repeat3 | tensorrt | fp16 | 8 | 24.38 ms | 24.89 ms | 328.19 img/s | 89.04 ms | - W | - | `fp16_b8_pinned_repeat3` |

## Repeat Aggregate

| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16_b4_original | 3 | 14.89 ms | 0.056 ms | 0.37% | 268.62 img/s | 41.06 ms |
| fp16_b4_pinned | 3 | 13.34 ms | 0.080 ms | 0.60% | 299.79 img/s | 38.04 ms |
| fp16_b8_original | 3 | 30.81 ms | 0.123 ms | 0.40% | 259.64 img/s | 81.12 ms |
| fp16_b8_pinned | 3 | 24.30 ms | 0.064 ms | 0.26% | 329.29 img/s | 89.04 ms |

## Measurement Notes

- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.
- Compare only rows with the same image size, dataset source, warmup count, and measured run count.
- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.
