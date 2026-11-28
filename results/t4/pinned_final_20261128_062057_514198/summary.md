# T4 pinned-output final comparison

All timings are ms/batch; throughput is inference images/s.

| Configuration | Repeats | Pre | Infer | Post | E2E | Infer img/s | Infer CV | E2E CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b4_original | 3 | 12.618 | 15.085 | 11.655 | 39.459 | 265.17 | 0.44% | 0.86% |
| b4_pinned | 3 | 12.472 | 13.450 | 11.961 | 37.986 | 297.39 | 0.33% | 1.40% |
| b8_original | 3 | 29.241 | 28.012 | 19.268 | 76.643 | 285.64 | 1.38% | 2.48% |
| b8_pinned | 3 | 28.212 | 24.195 | 19.063 | 71.591 | 330.65 | 0.52% | 0.06% |

| Batch | Infer latency reduction | Infer throughput gain | E2E latency reduction |
| --- | ---: | ---: | ---: |
| 4 | +10.84% | +12.15% | +3.73% |
| 8 | +13.63% | +15.76% | +6.59% |

Positive reductions/gains mean improvement; negative means regression.
Means and population standard deviations are over independent runs.
Throughput is the mean of per-run throughput, not the reciprocal of mean latency.
P95 fields in comparison.json are averages of per-run P95, not a pooled percentile.
Validation arrays are released before warmup. GPU monitoring and measured profiling are disabled.
Pinned outputs are reused views consumed synchronously; full-dataset mAP is not revalidated here.
No statistical significance or confidence interval is inferred from these repeats.
