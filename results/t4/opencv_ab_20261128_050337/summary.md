# NVIDIA TensorRT Benchmark Summary

Generated: 2026-11-28 05:08 UTC

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0

## Performance Results

| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv2_t1_b8_repeat1 | tensorrt | fp16 | 8 | 33.17 ms | 42.72 ms | 241.21 img/s | 90.42 ms | 32.86 W | 7.340 | `cv2_t1_b8_repeat1` |
| cv2_t1_b8_repeat2 | tensorrt | fp16 | 8 | 33.17 ms | 42.78 ms | 241.16 img/s | 91.45 ms | 33.30 W | 7.242 | `cv2_t1_b8_repeat2` |
| cv2_t1_b8_repeat3 | tensorrt | fp16 | 8 | 33.03 ms | 42.58 ms | 242.17 img/s | 89.96 ms | 33.23 W | 7.289 | `cv2_t1_b8_repeat3` |
| cv2_t4_b8_repeat1 | tensorrt | fp16 | 8 | 33.05 ms | 42.55 ms | 242.04 img/s | 96.69 ms | 32.78 W | 7.384 | `cv2_t4_b8_repeat1` |
| cv2_t4_b8_repeat2 | tensorrt | fp16 | 8 | 33.21 ms | 42.57 ms | 240.92 img/s | 99.20 ms | 32.01 W | 7.526 | `cv2_t4_b8_repeat2` |
| cv2_t4_b8_repeat3 | tensorrt | fp16 | 8 | 33.36 ms | 43.32 ms | 239.79 img/s | 99.86 ms | 33.09 W | 7.246 | `cv2_t4_b8_repeat3` |

## Repeat Aggregate

| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cv2_t1_b8 | 3 | 33.12 ms | 0.064 ms | 0.19% | 241.51 img/s | 90.61 ms |
| cv2_t4_b8 | 3 | 33.21 ms | 0.127 ms | 0.38% | 240.92 img/s | 98.58 ms |

## Measurement Notes

- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.
- Compare only rows with the same image size, dataset source, warmup count, and measured run count.
- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.
