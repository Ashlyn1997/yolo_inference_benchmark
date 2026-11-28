# T4 output-buffer comparison (V1/V2/V3)

Timings: ms/batch. Throughput: inference images/s (not E2E throughput).

| Configuration | Repeats | Pre | Infer | Post | E2E | Infer img/s | Infer CV | E2E CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b4_original | 1 | 12.930 | 14.805 | 9.853 | 37.681 | 270.18 | 0.00% | 0.00% |
| b4_reuse_pageable | 1 | 12.178 | 13.089 | 10.631 | 35.989 | 305.61 | 0.00% | 0.00% |
| b4_reuse_pinned | 1 | 12.545 | 13.440 | 11.266 | 37.348 | 297.62 | 0.00% | 0.00% |
| b8_original | 1 | 27.726 | 27.008 | 16.541 | 71.393 | 296.21 | 0.00% | 0.00% |
| b8_reuse_pageable | 1 | 29.926 | 28.179 | 21.417 | 79.647 | 283.90 | 0.00% | 0.00% |
| b8_reuse_pinned | 1 | 27.975 | 21.559 | 19.852 | 69.519 | 371.08 | 0.00% | 0.00% |

| Batch | Baseline -> Candidate | Infer reduction | Throughput gain | E2E reduction |
| --- | --- | ---: | ---: | ---: |
| 4 | original -> reuse_pageable | +11.59% | +13.11% | +4.49% |
| 4 | reuse_pageable -> reuse_pinned | -2.68% | -2.61% | -3.78% |
| 4 | original -> reuse_pinned | +9.22% | +10.16% | +0.88% |
| 8 | original -> reuse_pageable | -4.34% | -4.16% | -11.56% |
| 8 | reuse_pageable -> reuse_pinned | +23.49% | +30.71% | +12.72% |
| 8 | original -> reuse_pinned | +20.18% | +25.28% | +2.62% |

V1 original: new pageable CPU output. V2 reuse_pageable: reused pageable CPU output.
V3 reuse_pinned: reused pinned CPU output. V2/V1 tests the reuse path; V3/V2 tests pinning given reuse.
This three-arm experiment does not estimate the full reuse/pinning interaction.
Input handling and GPU output allocation remain unchanged; copies and inference stay synchronous.
V2/V3 outputs are borrowed views, consumed before the next inference (not equivalent ownership to V1).
Controls contain no CUDA-event instrumentation or GPU monitoring. Validation arrays are released before warmup.
Means/stddev/CV are over independent run summaries. Throughput is averaged per run.
P95 values in JSON are averages of per-run P95, not pooled percentiles.
Negative reductions/gains mean regression. No confidence intervals, significance or full-dataset mAP claims.
