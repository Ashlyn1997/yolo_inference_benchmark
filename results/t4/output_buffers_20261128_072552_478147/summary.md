# T4 output-buffer comparison (V1/V2/V3)

Timings: ms/batch. Throughput: inference images/s (not E2E throughput).

| Configuration | Repeats | Pre | Infer | Post | E2E | Infer img/s | Infer CV | E2E CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b4_original | 3 | 12.657 | 15.071 | 11.839 | 39.669 | 265.43 | 0.98% | 1.84% |
| b4_reuse_pageable | 3 | 12.305 | 15.037 | 11.492 | 38.930 | 266.01 | 0.37% | 0.63% |
| b4_reuse_pinned | 3 | 12.445 | 13.439 | 11.913 | 37.898 | 297.66 | 0.70% | 1.67% |
| b8_original | 3 | 29.882 | 29.188 | 19.098 | 78.289 | 275.53 | 7.43% | 5.16% |
| b8_reuse_pageable | 3 | 28.562 | 27.663 | 19.654 | 76.003 | 289.20 | 0.39% | 1.40% |
| b8_reuse_pinned | 3 | 28.665 | 24.337 | 19.558 | 72.683 | 328.73 | 0.46% | 1.65% |

| Batch | Baseline -> Candidate | Infer reduction | Throughput gain | E2E reduction |
| --- | --- | ---: | ---: | ---: |
| 4 | original -> reuse_pageable | +0.23% | +0.22% | +1.86% |
| 4 | reuse_pageable -> reuse_pinned | +10.63% | +11.90% | +2.65% |
| 4 | original -> reuse_pinned | +10.83% | +12.14% | +4.46% |
| 8 | original -> reuse_pageable | +5.22% | +4.96% | +2.92% |
| 8 | reuse_pageable -> reuse_pinned | +12.02% | +13.67% | +4.37% |
| 8 | original -> reuse_pinned | +16.62% | +19.31% | +7.16% |

V1 original: new pageable CPU output. V2 reuse_pageable: reused pageable CPU output.
V3 reuse_pinned: reused pinned CPU output. V2/V1 tests the reuse path; V3/V2 tests pinning given reuse.
This three-arm experiment does not estimate the full reuse/pinning interaction.
Input handling and GPU output allocation remain unchanged; copies and inference stay synchronous.
V2/V3 outputs are borrowed views, consumed before the next inference (not equivalent ownership to V1).
Controls contain no CUDA-event instrumentation or GPU monitoring. Validation arrays are released before warmup.
Means/stddev/CV are over independent run summaries. Throughput is averaged per run.
P95 values in JSON are averages of per-run P95, not pooled percentiles.
Negative reductions/gains mean regression. No confidence intervals, significance or full-dataset mAP claims.
