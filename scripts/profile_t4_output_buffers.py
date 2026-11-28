#!/usr/bin/env python3
"""Compare synchronous output paths; optionally diagnose CUDA stream intervals.

This is a separate experiment. Production backends and the historical pinned
worker are unchanged. Reused outputs are borrowed until the next infer call.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path


MODES = ("original", "reuse_pageable", "reuse_pinned")
POLICIES = {
    "original": {"memory": "pageable", "reuse": False, "ownership": "owned"},
    "reuse_pageable": {"memory": "pageable", "reuse": True, "ownership": "borrowed"},
    "reuse_pinned": {"memory": "pinned", "reuse": True, "ownership": "borrowed"},
}
EVENT_INTERVALS = {
    "h2d_event_ms": ("start", "h2d_end"),
    "input_conversion_event_ms": ("h2d_end", "input_conversion_end"),
    "execution_event_ms": ("execution_start", "execution_end"),
    "output_conversion_event_ms": ("output_conversion_start", "output_conversion_end"),
    "d2h_event_ms": ("d2h_start", "d2h_end"),
}


class OutputBufferInfer:
    """Single-context, synchronous diagnostic runner; not safe for concurrent use."""

    def __init__(self, backend, mode, *, cuda_events=False, torch_module=None):
        if mode not in MODES:
            raise ValueError(f"Unknown output mode: {mode}")
        if not backend.v3:
            raise ValueError("This experiment requires TensorRT execute_async_v3")
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.backend = backend
        self.mode = mode
        self.cpu_buffer = None
        self.recording = False
        self.samples = []
        self.events = {}
        if cuda_events:
            # Allocate in the selected device context, before any measured calls.
            with self.torch.cuda.device(backend.device):
                for name in dict.fromkeys(name for pair in EVENT_INTERVALS.values() for name in pair):
                    self.events[name] = self.torch.cuda.Event(enable_timing=True)

    def _host_output(self, converted):
        if self.mode == "original":
            return converted.cpu()
        if self.cpu_buffer is None:
            self.cpu_buffer = self.torch.empty(
                tuple(converted.shape), dtype=converted.dtype, device="cpu",
                pin_memory=self.mode == "reuse_pinned",
            )
        if (tuple(self.cpu_buffer.shape), self.cpu_buffer.dtype) != (tuple(converted.shape), converted.dtype):
            raise ValueError("Reused CPU output requires a fixed output shape and dtype")
        self.cpu_buffer.copy_(converted, non_blocking=False)
        return self.cpu_buffer

    def _prepare_output(self, tensor):
        backend = self.backend
        backend.context.set_input_shape(backend.inputs[0], tuple(tensor.shape))
        backend.context.set_tensor_address(backend.inputs[0], tensor.data_ptr())
        outputs = []
        for key in backend.outputs:
            output = self.torch.empty(
                tuple(backend.context.get_tensor_shape(key)),
                dtype=backend._dtype(key), device=backend.device,
            )
            backend.context.set_tensor_address(key, output.data_ptr())
            outputs.append(output)
        return outputs

    def _enqueue(self, stream):
        if not self.backend.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    def __call__(self, batch):
        if self.recording:
            return self._profiled(batch)
        torch, backend = self.torch, self.backend
        tensor = torch.from_numpy(batch).to(backend.device, non_blocking=True)
        tensor = tensor.to(backend._dtype(backend.inputs[0])).contiguous()
        outputs = self._prepare_output(tensor)
        self._enqueue(torch.cuda.current_stream(backend.device))
        torch.cuda.synchronize(backend.device)
        return self._host_output(outputs[0].float()).numpy()

    def _profiled(self, batch):
        torch, backend = self.torch, self.backend
        stream = torch.cuda.current_stream(backend.device)
        row = {}
        started = previous = time.perf_counter()

        def mark(name):
            nonlocal previous
            now = time.perf_counter()
            row[name] = (now - previous) * 1000
            previous = now

        self.events["start"].record(stream)
        tensor = torch.from_numpy(batch).to(backend.device, non_blocking=True)
        self.events["h2d_end"].record(stream)
        mark("h2d_call_wall_ms")
        tensor = tensor.to(backend._dtype(backend.inputs[0])).contiguous()
        self.events["input_conversion_end"].record(stream)
        mark("input_conversion_call_wall_ms")
        outputs = self._prepare_output(tensor)
        mark("shape_buffers_wall_ms")
        self.events["execution_start"].record(stream)
        self._enqueue(stream)
        self.events["execution_end"].record(stream)
        mark("enqueue_call_wall_ms")
        torch.cuda.synchronize(backend.device)
        mark("device_sync_wait_wall_ms")
        self.events["output_conversion_start"].record(stream)
        converted = outputs[0].float()
        self.events["output_conversion_end"].record(stream)
        mark("output_conversion_call_wall_ms")
        self.events["d2h_start"].record(stream)
        host = self._host_output(converted)
        self.events["d2h_end"].record(stream)
        mark("host_output_call_wall_ms")
        result = host.numpy()
        mark("numpy_view_wall_ms")
        # All events must complete before elapsed_time or reuse on the next call.
        self.events["d2h_end"].synchronize()
        row["backend_wall_before_event_readout_ms"] = (time.perf_counter() - started) * 1000
        for metric, (start, end) in EVENT_INTERVALS.items():
            row[metric] = self.events[start].elapsed_time(self.events[end])
        self.samples.append({"iteration": len(self.samples), **row})
        return result

    @contextmanager
    def measurement(self):
        if not self.events:
            raise ValueError("CUDA events were not enabled")
        self.samples.clear()
        self.recording = True
        try:
            yield
        finally:
            self.recording = False


def summarize_event_samples(rows, runs):
    if len(rows) != runs or not rows or [row["iteration"] for row in rows] != list(range(runs)):
        raise ValueError("Expected exactly runs ordered formal CUDA-event samples")
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0] if key != "iteration"}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--cuda-events", action="store_true",
                        help="After the control, run a separate CUDA-event diagnostic phase")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.batch, args.imgsz, args.runs) <= 0 or args.warmup < 0:
        parser.error("batch, imgsz, runs must be positive; warmup must be non-negative")
    if not Path(args.engine).is_file():
        parser.error(f"Engine does not exist: {args.engine}")
    if args.source != "synthetic" and not Path(args.source).exists():
        parser.error(f"Source does not exist: {args.source}")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    import cv2
    import numpy as np
    import torch
    from yolo_benchmark.backends import TensorRTBackend
    from yolo_benchmark.benchmark import run_benchmark_with_samples, write_raw_samples
    from yolo_benchmark.environment import collect_environment, file_metadata
    from yolo_benchmark.pipeline import load_images, preprocess

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    torch.cuda.set_device(device)
    environment = collect_environment(device.index if device.index is not None else torch.cuda.current_device())
    if "T4" not in environment.get("gpu", ""):
        parser.error("This experiment requires a Tesla T4")
    size = (args.imgsz, args.imgsz)
    images = load_images(args.source, args.batch, size)
    image_hash = hashlib.sha256()
    for image in images:
        image_hash.update(str((image.shape, str(image.dtype))).encode())
        image_hash.update(image.tobytes())
    backend = TensorRTBackend(args.engine, device=args.device)
    if not backend.v3:
        parser.error("This experiment requires TensorRT execute_async_v3")
    original_infer = backend.infer
    experiment = OutputBufferInfer(backend, args.mode)
    test_batch, test_metas = preprocess(images, size)
    reference = original_infer(test_batch).copy()
    actual = experiment(test_batch).copy()
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4, equal_nan=False)
    error = float(np.max(np.abs(actual - reference)))
    # Check a second call, including the reused-buffer path; retain no test arrays.
    np.testing.assert_allclose(experiment(test_batch), reference, rtol=1e-4, atol=1e-4, equal_nan=False)
    del actual, reference, test_batch, test_metas
    # V1 control calls the exact original backend. V2/V3 differ only in CPU output policy.
    backend.infer = original_infer if args.mode == "original" else experiment
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def measure(context=None):
        return run_benchmark_with_samples(
            backend, images, size, args.warmup, args.runs, 0.25, 0.45, "fp16",
            measurement_context=context,
        )

    print(f"{args.mode}: uninstrumented control ({args.warmup} warmup + {args.runs} measured)", flush=True)
    control, raw = measure()
    write_raw_samples(raw, args.output_dir / "control_latency.csv")
    payload = {
        "mode": args.mode, "output_policy": POLICIES[args.mode], "batch_size": args.batch,
        "environment": environment, "engine": file_metadata(args.engine), "source": args.source,
        "input_images_sha256": image_hash.hexdigest(),
        "input_shapes": [list(image.shape) for image in images],
        "opencv_threads": cv2.getNumThreads(), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        "gpu_monitor_enabled": False, "benchmark_only": not args.cuda_events,
        "validation_array_policy": "released_before_warmup", "output_allclose_passed": True,
        "max_abs_output_error": error, "control": asdict(control),
    }
    # Persist the control before starting optional profiling, so failures lose no raw data.
    write_json(args.output_dir / "summary.json", payload)
    print(f"Control complete: infer={control.inference_mean_ms:.3f}, E2E={control.end_to_end_mean_ms:.3f} ms/batch", flush=True)

    if args.cuda_events:
        backend.infer = original_infer
        del experiment  # Do not retain a second CPU output buffer in the diagnostic phase.
        diagnostic = OutputBufferInfer(backend, args.mode, cuda_events=True)
        backend.infer = diagnostic
        print("Separate CUDA-event diagnostic; warmup is not recorded", flush=True)
        profiled, raw = measure(diagnostic.measurement())
        means = summarize_event_samples(diagnostic.samples, args.runs)
        write_raw_samples(raw, args.output_dir / "event_latency.csv")
        with (args.output_dir / "cuda_event_samples.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(diagnostic.samples[0]))
            writer.writeheader()
            writer.writerows(diagnostic.samples)
        write_json(args.output_dir / "cuda_event_summary.json", {
            "profiled": asdict(profiled), "stage_mean_ms": means,
            "notes": [
                "Diagnostic only; do not aggregate with uninstrumented controls.",
                "Events measure selected-stream elapsed intervals, including possible host submission gaps.",
                "Pageable staging/allocation is not isolated; D2H is not a pure DMA bandwidth measurement.",
                "Stage event intervals are not additive to CPU E2E or proof of kernel-only execution time.",
                "Synchronous execution/copies are preserved; this is not an overlap experiment.",
                "backend_wall_before_event_readout_ms excludes event readout/bookkeeping; event_latency.csv includes it.",
            ],
        })
        for name in EVENT_INTERVALS:
            print(f"  {name}: {means[name]:.3f} ms/batch", flush=True)
    print(f"CASE COMPLETE: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
