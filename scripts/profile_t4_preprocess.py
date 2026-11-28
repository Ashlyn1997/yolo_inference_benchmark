#!/usr/bin/env python3
"""Profile CPU preprocessing inside the synchronous TensorRT benchmark.

Run original and pinned modes in separate processes. The pinned implementation
is diagnostic: its NumPy output is overwritten by the next infer call.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path


STAGES = (
    "letterbox_ms",
    "color_ms",
    "layout_normalize_ms",
    "stack_ms",
    "body_total_ms",
)


class ProfiledPreprocess:
    """Keep the production preprocessing operations, adding wall-clock markers."""

    def __init__(self):
        import cv2
        import numpy as np
        from yolo_benchmark.pipeline import letterbox

        self.cv2 = cv2
        self.np = np
        self.letterbox = letterbox
        self.samples: list[dict[str, float]] = []

    def __call__(self, images, size):
        started = time.perf_counter()
        tensors, metas = [], []
        row = {stage: 0.0 for stage in STAGES}
        for image in images:
            tick = time.perf_counter()
            padded, meta = self.letterbox(image, size)
            row["letterbox_ms"] += (time.perf_counter() - tick) * 1000

            tick = time.perf_counter()
            rgb = self.cv2.cvtColor(padded, self.cv2.COLOR_BGR2RGB)
            row["color_ms"] += (time.perf_counter() - tick) * 1000

            tick = time.perf_counter()
            # Keep the original expression and its allocation behavior together.
            tensors.append(
                self.np.ascontiguousarray(
                    rgb.transpose(2, 0, 1), dtype=self.np.float32
                ) / 255.0
            )
            metas.append(meta)
            row["layout_normalize_ms"] += (time.perf_counter() - tick) * 1000

        if not tensors:
            raise ValueError("At least one image is required")
        tick = time.perf_counter()
        batch = self.np.stack(tensors)
        row["stack_ms"] = (time.perf_counter() - tick) * 1000
        row["body_total_ms"] = (time.perf_counter() - started) * 1000
        self.samples.append(row)
        return batch, metas


def summarize_samples(samples, latency_samples, runs):
    """Exclude the initial preprocessing call and align formal iterations."""
    if runs <= 0 or len(latency_samples) != runs or len(samples) != runs + 1:
        raise ValueError("Expected one initial preprocessing call and exactly runs measured calls")
    measured = []
    for iteration, (row, latency) in enumerate(zip(samples[1:], latency_samples)):
        measured.append({
            "iteration": iteration,
            **row,
            "outer_preprocess_ms": latency.preprocess_ms,
            # Includes return-time cleanup, caller assignment, and timing overhead.
            # Do not label it as purely allocation or memory-free time.
            "outside_body_ms": latency.preprocess_ms - row["body_total_ms"],
        })
    means = {
        key: statistics.fmean(row[key] for row in measured)
        for key in (*STAGES, "outer_preprocess_ms", "outside_body_ms")
    }
    return means, measured


def make_pinned_infer(backend):
    import torch

    if not backend.v3:
        raise ValueError("This diagnostic requires TensorRT execute_async_v3")
    cpu_buffer = None

    def infer(batch):
        nonlocal cpu_buffer
        input_key = backend.inputs[0]
        tensor = torch.from_numpy(batch).to(backend.device, non_blocking=True)
        tensor = tensor.to(backend._dtype(input_key)).contiguous()
        backend.context.set_input_shape(input_key, tuple(tensor.shape))
        backend.context.set_tensor_address(input_key, tensor.data_ptr())
        outputs = []
        for key in backend.outputs:
            output = torch.empty(
                tuple(backend.context.get_tensor_shape(key)),
                dtype=backend._dtype(key), device=backend.device,
            )
            backend.context.set_tensor_address(key, output.data_ptr())
            outputs.append(output)
        if not backend.context.execute_async_v3(
            torch.cuda.current_stream(backend.device).cuda_stream
        ):
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.synchronize(backend.device)
        converted = outputs[0].float()
        if cpu_buffer is None:
            cpu_buffer = torch.empty(
                tuple(converted.shape), dtype=converted.dtype,
                device="cpu", pin_memory=True,
            )
        if tuple(cpu_buffer.shape) != tuple(converted.shape):
            raise ValueError("This diagnostic requires a fixed output shape")
        cpu_buffer.copy_(converted, non_blocking=False)
        return cpu_buffer.numpy()

    return infer


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("original", "pinned"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.runs <= 0 or args.warmup < 0:
        parser.error("batch, imgsz, runs must be positive; warmup must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    import torch
    import yolo_benchmark.benchmark as benchmark_module
    from yolo_benchmark.backends import TensorRTBackend
    from yolo_benchmark.environment import collect_environment, file_metadata
    from yolo_benchmark.pipeline import load_images, preprocess

    environment = collect_environment(torch.device(args.device).index or 0)
    size = (args.imgsz, args.imgsz)
    images = load_images(args.source, args.batch, size)
    backend = TensorRTBackend(args.engine, device=args.device)
    test_batch, test_metas = preprocess(images, size)
    reference = backend.infer(test_batch).copy()
    if args.mode == "pinned":
        backend.infer = make_pinned_infer(backend)
    actual = backend.infer(test_batch).copy()
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4, equal_nan=False)

    profiler = ProfiledPreprocess()
    profiled_batch, profiled_metas = profiler(images, size)
    np.testing.assert_array_equal(profiled_batch, test_batch)
    if profiled_metas != test_metas:
        raise AssertionError("Preprocessing metadata changed")
    del actual, reference, profiled_batch, test_batch
    profiler.samples.clear()

    def measure():
        return benchmark_module.run_benchmark_with_samples(
            backend, images, size, args.warmup, args.runs, 0.25, 0.45, "fp16"
        )

    print(f"{args.mode}: [1/2] uninstrumented control", flush=True)
    control, control_raw = measure()
    benchmark_module.write_raw_samples(control_raw, args.output_dir / "control_latency.csv")

    print(f"{args.mode}: [2/2] preprocessing stage timings", flush=True)
    saved_preprocess = benchmark_module.preprocess
    try:
        benchmark_module.preprocess = profiler
        profiled, raw = measure()
    finally:
        benchmark_module.preprocess = saved_preprocess
    means, rows = summarize_samples(profiler.samples, raw, args.runs)
    benchmark_module.write_raw_samples(raw, args.output_dir / "profiled_latency.csv")
    with (args.output_dir / "preprocess_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "mode": args.mode,
        "batch_size": args.batch,
        "environment": environment,
        "engine": file_metadata(args.engine),
        "source": args.source,
        "opencv_threads": cv2.getNumThreads(),
        "torch_threads": torch.get_num_threads(),
        "gpu_monitor_enabled": False,
        "preprocess_equivalence_passed": True,
        "output_allclose_passed": True,
        "control": asdict(control),
        "profiled": asdict(profiled),
        "preprocess_breakdown_mean_ms": means,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.mode}: control pre={control.preprocess_mean_ms:.3f}, "
        f"profiled pre={profiled.preprocess_mean_ms:.3f} ms/batch", flush=True,
    )
    for stage, value in means.items():
        print(f"  {stage}: {value:.3f}", flush=True)
    print(f"Complete: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
