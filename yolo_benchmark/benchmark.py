from __future__ import annotations

import csv
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ContextManager

import numpy as np

from .backends import Backend
from .environment import collect_environment
from .pipeline import postprocess, preprocess


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values), percent))


@dataclass
class BenchmarkResult:
    backend: str
    batch_size: int
    image_size: list[int]
    warmup_runs: int
    measured_runs: int
    precision: str
    preprocess_mean_ms: float
    inference_mean_ms: float
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    postprocess_mean_ms: float
    end_to_end_mean_ms: float
    throughput_images_per_second: float
    cuda_memory_allocated_mib: float | None
    inference_stddev_ms: float = 0.0
    end_to_end_p50_ms: float = 0.0
    end_to_end_p95_ms: float = 0.0
    end_to_end_p99_ms: float = 0.0
    end_to_end_stddev_ms: float = 0.0
    images_per_second_per_watt: float | None = None
    gpu_metrics: dict[str, Any] | None = field(default=None)


@dataclass(frozen=True)
class BenchmarkSample:
    backend: str
    iteration: int
    batch_size: int
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    end_to_end_ms: float


def run_benchmark(
    backend: Backend,
    images: list[np.ndarray],
    size: tuple[int, int],
    warmup: int,
    runs: int,
    confidence: float,
    iou: float,
    precision: str,
) -> BenchmarkResult:
    result, _ = run_benchmark_with_samples(
        backend,
        images,
        size,
        warmup,
        runs,
        confidence,
        iou,
        precision,
    )
    return result


def run_benchmark_with_samples(
    backend: Backend,
    images: list[np.ndarray],
    size: tuple[int, int],
    warmup: int,
    runs: int,
    confidence: float,
    iou: float,
    precision: str,
    measurement_context: ContextManager | None = None,
) -> tuple[BenchmarkResult, list[BenchmarkSample]]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be greater than zero")
    batch, metas = preprocess(images, size)
    for _ in range(warmup):
        output = backend.infer(batch)
        backend.synchronize()
        postprocess(output, metas, confidence, iou)

    pre_times: list[float] = []
    infer_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    samples: list[BenchmarkSample] = []
    with (measurement_context if measurement_context is not None else nullcontext()):
        for iteration in range(runs):
            total_start = time.perf_counter()
            start = time.perf_counter()
            batch, metas = preprocess(images, size)
            pre_ms = (time.perf_counter() - start) * 1000

            backend.synchronize()
            start = time.perf_counter()
            output = backend.infer(batch)
            backend.synchronize()
            infer_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            postprocess(output, metas, confidence, iou)
            post_ms = (time.perf_counter() - start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000
            pre_times.append(pre_ms)
            infer_times.append(infer_ms)
            post_times.append(post_ms)
            total_times.append(total_ms)
            samples.append(
                BenchmarkSample(
                    backend=backend.name,
                    iteration=iteration,
                    batch_size=len(images),
                    preprocess_ms=pre_ms,
                    inference_ms=infer_ms,
                    postprocess_ms=post_ms,
                    end_to_end_ms=total_ms,
                )
            )

    mean_infer = statistics.fmean(infer_times)
    result = BenchmarkResult(
        backend=backend.name,
        batch_size=len(images),
        image_size=list(size),
        warmup_runs=warmup,
        measured_runs=runs,
        precision=precision,
        preprocess_mean_ms=statistics.fmean(pre_times),
        inference_mean_ms=mean_infer,
        inference_p50_ms=percentile(infer_times, 50),
        inference_p95_ms=percentile(infer_times, 95),
        inference_p99_ms=percentile(infer_times, 99),
        postprocess_mean_ms=statistics.fmean(post_times),
        end_to_end_mean_ms=statistics.fmean(total_times),
        throughput_images_per_second=len(images) * 1000 / mean_infer,
        cuda_memory_allocated_mib=backend.memory_mib(),
        inference_stddev_ms=statistics.pstdev(infer_times),
        end_to_end_p50_ms=percentile(total_times, 50),
        end_to_end_p95_ms=percentile(total_times, 95),
        end_to_end_p99_ms=percentile(total_times, 99),
        end_to_end_stddev_ms=statistics.pstdev(total_times),
    )
    return result, samples


def write_results(
    results: list[BenchmarkResult],
    output: str,
    *,
    environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "environment": environment if environment is not None else collect_environment(),
        "results": [asdict(result) for result in results],
    }
    if metadata:
        payload["metadata"] = metadata
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_raw_samples(samples: list[BenchmarkSample], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(BenchmarkSample.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(sample) for sample in samples)


def print_results(results: list[BenchmarkResult]) -> None:
    print("\nbackend             batch  infer mean  p95       throughput    e2e mean")
    print("-" * 78)
    for result in results:
        print(
            f"{result.backend:<19} {result.batch_size:>5}  "
            f"{result.inference_mean_ms:>8.2f} ms  {result.inference_p95_ms:>7.2f} ms  "
            f"{result.throughput_images_per_second:>8.2f} img/s  "
            f"{result.end_to_end_mean_ms:>8.2f} ms"
        )
