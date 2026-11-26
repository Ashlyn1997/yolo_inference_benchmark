# NVIDIA TensorRT Benchmark Summary

Generated: 2026-11-26 04:38 UTC

## Environment

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0

## Performance Results

| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t4_ort_fp32_b1_repeat1 | onnxruntime-cuda | fp32 | 1 | 7.53 ms | 7.84 ms | 132.84 img/s | 16.19 ms | 50.63 W | 2.624 | `t4_ort_fp32_b1_repeat1` |
| t4_ort_fp32_b1_repeat2 | onnxruntime-cuda | fp32 | 1 | 7.69 ms | 8.09 ms | 130.10 img/s | 16.83 ms | 50.56 W | 2.573 | `t4_ort_fp32_b1_repeat2` |
| t4_ort_fp32_b1_repeat3 | onnxruntime-cuda | fp32 | 1 | 7.46 ms | 7.70 ms | 134.02 img/s | 16.08 ms | 52.17 W | 2.569 | `t4_ort_fp32_b1_repeat3` |
| t4_pytorch_fp16_b1_repeat1 | pytorch | fp16 | 1 | 15.50 ms | 17.31 ms | 64.52 img/s | 24.11 ms | 31.81 W | 2.028 | `t4_pytorch_fp16_b1_repeat1` |
| t4_pytorch_fp16_b1_repeat2 | pytorch | fp16 | 1 | 15.52 ms | 17.27 ms | 64.42 img/s | 24.17 ms | 32.04 W | 2.011 | `t4_pytorch_fp16_b1_repeat2` |
| t4_pytorch_fp16_b1_repeat3 | pytorch | fp16 | 1 | 15.92 ms | 17.73 ms | 62.80 img/s | 24.69 ms | 32.02 W | 1.961 | `t4_pytorch_fp16_b1_repeat3` |
| t4_pytorch_fp32_b1_repeat1 | pytorch | fp32 | 1 | 12.78 ms | 14.84 ms | 78.24 img/s | 21.09 ms | 36.25 W | 2.158 | `t4_pytorch_fp32_b1_repeat1` |
| t4_pytorch_fp32_b1_repeat2 | pytorch | fp32 | 1 | 12.81 ms | 15.04 ms | 78.08 img/s | 21.58 ms | 36.23 W | 2.155 | `t4_pytorch_fp32_b1_repeat2` |
| t4_pytorch_fp32_b1_repeat3 | pytorch | fp32 | 1 | 12.61 ms | 14.10 ms | 79.28 img/s | 21.26 ms | 37.22 W | 2.130 | `t4_pytorch_fp32_b1_repeat3` |
| t4_trt_fp16_b1_repeat1 | tensorrt | fp16 | 1 | 6.14 ms | 6.96 ms | 162.83 img/s | 14.75 ms | 33.91 W | 4.802 | `t4_trt_fp16_b1_repeat1` |
| t4_trt_fp16_b1_repeat2 | tensorrt | fp16 | 1 | 5.91 ms | 6.97 ms | 169.12 img/s | 14.33 ms | 34.00 W | 4.975 | `t4_trt_fp16_b1_repeat2` |
| t4_trt_fp16_b1_repeat3 | tensorrt | fp16 | 1 | 6.08 ms | 6.95 ms | 164.41 img/s | 14.77 ms | 34.08 W | 4.825 | `t4_trt_fp16_b1_repeat3` |
| t4_trt_fp16_b4_repeat1 | tensorrt | fp16 | 4 | 14.70 ms | 15.08 ms | 272.13 img/s | 42.45 ms | 35.63 W | 7.637 | `t4_trt_fp16_b4_repeat1` |
| t4_trt_fp16_b4_repeat2 | tensorrt | fp16 | 4 | 14.59 ms | 14.94 ms | 274.14 img/s | 41.81 ms | 35.94 W | 7.628 | `t4_trt_fp16_b4_repeat2` |
| t4_trt_fp16_b4_repeat3 | tensorrt | fp16 | 4 | 14.64 ms | 15.02 ms | 273.16 img/s | 41.35 ms | 37.08 W | 7.367 | `t4_trt_fp16_b4_repeat3` |
| t4_trt_fp16_b8_repeat1 | tensorrt | fp16 | 8 | 34.03 ms | 39.48 ms | 235.08 img/s | 100.42 ms | 33.42 W | 7.035 | `t4_trt_fp16_b8_repeat1` |
| t4_trt_fp16_b8_repeat2 | tensorrt | fp16 | 8 | 33.76 ms | 39.25 ms | 236.95 img/s | 100.56 ms | 35.10 W | 6.750 | `t4_trt_fp16_b8_repeat2` |
| t4_trt_fp16_b8_repeat3 | tensorrt | fp16 | 8 | 33.64 ms | 39.37 ms | 237.82 img/s | 99.15 ms | 36.37 W | 6.538 | `t4_trt_fp16_b8_repeat3` |
| t4_trt_fp16_dynamic_b1_repeat1 | tensorrt | fp16 | 1 | 5.96 ms | 6.17 ms | 167.90 img/s | 12.99 ms | 35.68 W | 4.706 | `t4_trt_fp16_dynamic_b1_repeat1` |
| t4_trt_fp16_dynamic_b1_repeat2 | tensorrt | fp16 | 1 | 5.93 ms | 6.15 ms | 168.60 img/s | 12.73 ms | 35.90 W | 4.696 | `t4_trt_fp16_dynamic_b1_repeat2` |
| t4_trt_fp16_dynamic_b1_repeat3 | tensorrt | fp16 | 1 | 5.70 ms | 6.10 ms | 175.49 img/s | 12.45 ms | 36.27 W | 4.839 | `t4_trt_fp16_dynamic_b1_repeat3` |
| t4_trt_fp16_dynamic_b4_repeat1 | tensorrt | fp16 | 4 | 15.12 ms | 15.53 ms | 264.53 img/s | 43.35 ms | 37.17 W | 7.117 | `t4_trt_fp16_dynamic_b4_repeat1` |
| t4_trt_fp16_dynamic_b4_repeat2 | tensorrt | fp16 | 4 | 15.14 ms | 15.51 ms | 264.19 img/s | 43.37 ms | 36.71 W | 7.196 | `t4_trt_fp16_dynamic_b4_repeat2` |
| t4_trt_fp16_dynamic_b4_repeat3 | tensorrt | fp16 | 4 | 15.22 ms | 15.58 ms | 262.83 img/s | 43.33 ms | 36.91 W | 7.120 | `t4_trt_fp16_dynamic_b4_repeat3` |
| t4_trt_fp16_dynamic_b8_repeat1 | tensorrt | fp16 | 8 | 32.06 ms | 39.57 ms | 249.51 img/s | 95.69 ms | 36.73 W | 6.794 | `t4_trt_fp16_dynamic_b8_repeat1` |
| t4_trt_fp16_dynamic_b8_repeat2 | tensorrt | fp16 | 8 | 32.43 ms | 39.71 ms | 246.69 img/s | 94.89 ms | 37.07 W | 6.654 | `t4_trt_fp16_dynamic_b8_repeat2` |
| t4_trt_fp16_dynamic_b8_repeat3 | tensorrt | fp16 | 8 | 32.09 ms | 39.62 ms | 249.31 img/s | 94.79 ms | 36.86 W | 6.763 | `t4_trt_fp16_dynamic_b8_repeat3` |
| t4_trt_fp32_b1_repeat1 | tensorrt | fp32 | 1 | 7.73 ms | 9.05 ms | 129.42 img/s | 15.99 ms | 37.52 W | 3.449 | `t4_trt_fp32_b1_repeat1` |
| t4_trt_fp32_b1_repeat2 | tensorrt | fp32 | 1 | 7.33 ms | 8.97 ms | 136.48 img/s | 16.01 ms | 38.35 W | 3.559 | `t4_trt_fp32_b1_repeat2` |
| t4_trt_fp32_b1_repeat3 | tensorrt | fp32 | 1 | 7.15 ms | 8.38 ms | 139.83 img/s | 15.41 ms | 38.50 W | 3.632 | `t4_trt_fp32_b1_repeat3` |

## Repeat Aggregate

| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t4_ort_fp32_b1 | 3 | 7.56 ms | 0.095 ms | 1.25% | 132.32 img/s | 16.37 ms |
| t4_pytorch_fp16_b1 | 3 | 15.65 ms | 0.194 ms | 1.24% | 63.91 img/s | 24.33 ms |
| t4_pytorch_fp32_b1 | 3 | 12.73 ms | 0.086 ms | 0.67% | 78.53 img/s | 21.31 ms |
| t4_trt_fp16_b1 | 3 | 6.05 ms | 0.097 ms | 1.60% | 165.45 img/s | 14.62 ms |
| t4_trt_fp16_b4 | 3 | 14.64 ms | 0.044 ms | 0.30% | 273.15 img/s | 41.87 ms |
| t4_trt_fp16_b8 | 3 | 33.81 ms | 0.164 ms | 0.48% | 236.62 img/s | 100.04 ms |
| t4_trt_fp16_dynamic_b1 | 3 | 5.86 ms | 0.116 ms | 1.98% | 170.66 img/s | 12.72 ms |
| t4_trt_fp16_dynamic_b4 | 3 | 15.16 ms | 0.042 ms | 0.28% | 263.85 img/s | 43.35 ms |
| t4_trt_fp16_dynamic_b8 | 3 | 32.19 ms | 0.167 ms | 0.52% | 248.51 img/s | 95.12 ms |
| t4_trt_fp32_b1 | 3 | 7.40 ms | 0.241 ms | 3.25% | 135.24 img/s | 15.80 ms |

## Measurement Notes

- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.
- Compare only rows with the same image size, dataset source, warmup count, and measured run count.
- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.
