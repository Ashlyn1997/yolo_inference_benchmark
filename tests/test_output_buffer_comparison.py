from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.profile_t4_output_buffers import (
    EVENT_INTERVALS, MODES, POLICIES, OutputBufferInfer, summarize_event_samples,
)
from scripts.run_t4_output_buffer_comparison import aggregate_results, case_order, main


class FakeTensor:
    """A mutable host view and a pointer registry, without CUDA/numpy dependencies."""

    def __init__(self, torch, values, shape, dtype="float32", device="cpu", pinned=False):
        self.torch, self.values, self.shape = torch, list(values), tuple(shape)
        self.dtype, self.device, self.pinned = dtype, device, pinned
        torch.registry[id(self)] = self

    def to(self, target, **kwargs):
        device, dtype = (target, self.dtype) if str(target).startswith("cuda") else (self.device, target)
        return FakeTensor(self.torch, self.values, self.shape, dtype, device)

    def contiguous(self):
        return self

    def float(self):
        if self.dtype == "float32":
            return self
        return FakeTensor(self.torch, self.values, self.shape, "float32", self.device)

    def cpu(self):
        self.torch.cpu_copies += 1
        return FakeTensor(self.torch, self.values, self.shape, self.dtype)

    def copy_(self, source, *, non_blocking):
        if non_blocking:
            raise AssertionError("The control must preserve synchronous copies")
        self.values[:] = source.values

    def numpy(self):
        return self.values  # Deliberately shared mutable storage for lifetime tests.

    def data_ptr(self):
        return id(self)


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.tick = None

    def record(self, stream):
        if stream is not self.cuda.stream:
            raise AssertionError("Event is on the wrong stream")
        self.cuda.clock += 1
        self.tick = self.cuda.clock

    def synchronize(self):
        self.cuda.completed = self.cuda.clock

    def elapsed_time(self, other):
        if self.tick is None or other.tick is None or other.tick > self.cuda.completed:
            raise AssertionError("Elapsed time read before completion")
        return float(other.tick - self.tick)


class FakeCuda:
    def __init__(self):
        self.stream = SimpleNamespace(cuda_stream=17)
        self.clock = self.completed = self.allocations = 0

    def device(self, device):
        return nullcontext()

    def current_stream(self, device):
        return self.stream

    def Event(self, *, enable_timing):
        if not enable_timing:
            raise AssertionError("Timing must be enabled")
        self.allocations += 1
        return FakeEvent(self)

    def synchronize(self, device):
        self.completed = self.clock


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.registry = {}
        self.host_allocations = []
        self.cpu_copies = 0

    def from_numpy(self, batch):
        return FakeTensor(self, batch, (len(batch),))

    def empty(self, shape, *, dtype, device, pin_memory=False):
        tensor = FakeTensor(self, [0] * shape[0], shape, dtype, device, pin_memory)
        if device == "cpu":
            self.host_allocations.append(tensor)
        return tensor


def fake_backend(torch):
    class Context:
        addresses = None

        def __init__(self):
            self.addresses = {}
            self.shape = (2,)
            self.succeed = True

        def set_input_shape(self, key, shape):
            self.shape = shape

        def set_tensor_address(self, key, address):
            self.addresses[key] = address

        def get_tensor_shape(self, key):
            return self.shape

        def execute_async_v3(self, stream):
            if stream != 17:
                raise AssertionError("Wrong execution stream")
            source = torch.registry[self.addresses["images"]]
            target = torch.registry[self.addresses["output"]]
            target.values[:] = [value * 2 + 1 for value in source.values]
            return self.succeed

    return SimpleNamespace(
        v3=True, device="cuda:0", inputs=["images"], outputs=["output"],
        context=Context(), _dtype=lambda key: "float16",
    )


class OutputBufferInferTest(unittest.TestCase):
    def test_reuse_policy_output_values_and_ownership(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                torch = FakeTorch()
                infer = OutputBufferInfer(fake_backend(torch), mode, torch_module=torch)
                first = infer([1, 2])
                self.assertEqual(first, [3, 5])
                second = infer([4, 5])
                self.assertEqual(second, [9, 11])
                if mode == "original":
                    self.assertIsNot(first, second)
                    self.assertEqual(first, [3, 5])
                    self.assertEqual(torch.cpu_copies, 2)
                    self.assertEqual(torch.host_allocations, [])
                else:
                    self.assertIs(first, second)
                    self.assertEqual(first, [9, 11])
                    self.assertEqual(len(torch.host_allocations), 1)
                    self.assertEqual(torch.host_allocations[0].pinned, mode == "reuse_pinned")
                self.assertEqual(torch.cuda.allocations, 0)
                self.assertEqual(infer.samples, [])

    def test_diagnostic_matches_plain_outputs_and_excludes_warmup(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                torch = FakeTorch()
                infer = OutputBufferInfer(fake_backend(torch), mode, cuda_events=True, torch_module=torch)
                expected = list(infer([1, 2]))
                infer([8, 9])  # Unrecorded warmup.
                self.assertEqual(infer.samples, [])
                with infer.measurement():
                    self.assertEqual(infer([1, 2]), expected)
                    self.assertEqual(infer([3, 4]), [7, 9])
                means = summarize_event_samples(infer.samples, 2)
                self.assertTrue(set(EVENT_INTERVALS) <= means.keys())
                self.assertTrue(all(means[key] == 1 for key in EVENT_INTERVALS))
                self.assertGreater(means["backend_wall_before_event_readout_ms"], 0)
                self.assertFalse(infer.recording)
                infer([5, 6])
                self.assertEqual(len(infer.samples), 2)

    def test_recording_stops_on_exception_and_execution_failure_propagates(self):
        torch = FakeTorch()
        backend = fake_backend(torch)
        infer = OutputBufferInfer(backend, "reuse_pinned", cuda_events=True, torch_module=torch)
        backend.context.succeed = False
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            with infer.measurement():
                infer([1, 2])
        self.assertFalse(infer.recording)
        self.assertEqual(infer.samples, [])

    def test_reused_output_shape_changes_are_rejected(self):
        torch = FakeTorch()
        infer = OutputBufferInfer(fake_backend(torch), "reuse_pageable", torch_module=torch)
        infer([1, 2])
        with self.assertRaisesRegex(ValueError, "fixed output shape"):
            infer([1, 2, 3])

    def test_missing_unordered_or_extra_event_samples_are_rejected(self):
        for rows in ([], [{"iteration": 1}], [{"iteration": 0}, {"iteration": 1}]):
            with self.assertRaisesRegex(ValueError, "ordered formal"):
                summarize_event_samples(rows, 1)


def fixtures(batches=(4,), repeats=2):
    records = []
    for batch in batches:
        for mode, latency, e2e in zip(MODES, (10.0, 8.0, 5.0), (20.0, 18.0, 15.0)):
            for repeat in range(1, repeats + 1):
                records.append({
                    "mode": mode, "batch_size": batch, "repeat": repeat,
                    "output_policy": dict(POLICIES[mode]), "benchmark_only": True,
                    "gpu_monitor_enabled": False, "output_allclose_passed": True,
                    "validation_array_policy": "released_before_warmup", "max_abs_output_error": 0.0,
                    "environment": {"gpu": "Tesla T4", "torch": "test"},
                    "source": "synthetic", "engine": {"sha256": f"engine-b{batch}"},
                    "input_images_sha256": f"images-b{batch}", "input_shapes": [[640, 640, 3]] * batch,
                    "numpy_version": "test", "opencv_version": "test", "opencv_threads": 1,
                    "torch_threads": 2, "torch_interop_threads": 2,
                    "cpu_affinity": [0, 1, 2, 3], "thread_environment": {"OMP_NUM_THREADS": "1"},
                    "control": {
                        "backend": "tensorrt", "precision": "fp16", "batch_size": batch,
                        "gpu_metrics": None, "image_size": [640, 640], "warmup_runs": 50, "measured_runs": 500,
                        "preprocess_mean_ms": 2.0, "inference_mean_ms": latency * repeat,
                        "postprocess_mean_ms": 3.0, "end_to_end_mean_ms": e2e * repeat,
                        "throughput_images_per_second": batch * 1000 / (latency * repeat),
                        "inference_p95_ms": latency * repeat * 1.1,
                        "end_to_end_p95_ms": e2e * repeat * 1.1,
                    },
                })
    return records


class OutputBufferComparisonTest(unittest.TestCase):
    def test_schedule_has_eighteen_unique_cases_and_rotates_first_mode(self):
        cases = list(case_order([4, 8], 3))
        self.assertEqual(len(cases), 18)
        self.assertEqual(len(set(cases)), 18)
        for batch in (4, 8):
            first_modes = []
            for repeat in (1, 2, 3):
                first_modes.append(next(mode for b, mode, r in cases if (b, r) == (batch, repeat)))
            self.assertEqual(first_modes, list(MODES))
        self.assertEqual(cases[6][0], 8)

    def test_means_and_all_three_pairwise_gains(self):
        summary = aggregate_results(fixtures(), [4], 2)
        first = summary["groups"][0]["metrics"]
        self.assertEqual(first["inference_mean_ms"]["mean"], 15.0)
        self.assertEqual(first["inference_mean_ms"]["stddev"], 5.0)
        self.assertEqual(first["throughput_images_per_second"]["mean"], 300.0)
        self.assertEqual(len(summary["comparisons"]), 3)
        self.assertAlmostEqual(summary["comparisons"][0]["throughput_gain_pct"], 25.0)
        self.assertAlmostEqual(summary["comparisons"][1]["throughput_gain_pct"], 60.0)
        self.assertAlmostEqual(summary["comparisons"][2]["inference_latency_reduction_pct"], 50.0)

    def test_changed_conditions_or_instrumented_controls_are_rejected(self):
        changes = (
            ("engine", {"sha256": "changed"}), ("input_images_sha256", "changed"),
            ("benchmark_only", False), ("gpu_monitor_enabled", True),
            ("torch_interop_threads", 4), ("cpu_affinity", [0, 1]),
            ("thread_environment", {}), ("output_policy", POLICIES["original"]),
            ("validation_array_policy", "retained"), ("output_allclose_passed", False),
        )
        for field, value in changes:
            with self.subTest(field=field):
                records = fixtures()
                records[-1][field] = copy.deepcopy(value)
                with self.assertRaises(ValueError):
                    aggregate_results(records, [4], 2)

    def test_duplicates_missing_and_invalid_metrics_are_rejected(self):
        records = fixtures()
        records[-1]["repeat"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate repeats"):
            aggregate_results(records, [4], 2)
        with self.assertRaisesRegex(ValueError, "Missing"):
            aggregate_results(fixtures()[:-1], [4], 2)
        for value in (-1, float("nan"), float("inf")):
            records = fixtures()
            records[-1]["control"]["inference_mean_ms"] = value
            with self.assertRaises(ValueError):
                aggregate_results(records, [4], 2)

    def test_runner_writes_logs_manifest_provenance_and_summary(self):
        commands = []
        records = fixtures()

        def worker(command, **kwargs):
            commands.append(command)
            destination = Path(command[command.index("--output-dir") + 1])
            mode = command[command.index("--mode") + 1]
            repeat = int(destination.name.rsplit("repeat", 1)[1])
            payload = copy.deepcopy(next(row for row in records if (row["mode"], row["repeat"]) == (mode, repeat)))
            import hashlib
            payload["engine"]["sha256"] = hashlib.sha256(b"engine").hexdigest()
            destination.mkdir()
            (destination / "summary.json").write_text(json.dumps(payload))
            kwargs["stdout"].write("CASE COMPLETE\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b4.engine").write_bytes(b"engine")
            output = root / "output"
            argv = [
                "runner", "--source", "synthetic", "--engine-template", str(root / "b{batch}.engine"),
                "--batches", "4", "--repeats", "2", "--output-dir", str(output),
            ]
            captured = io.StringIO()
            with patch("sys.argv", argv), patch(
                "scripts.run_t4_output_buffer_comparison.subprocess.run", side_effect=worker
            ), redirect_stdout(captured):
                main()
            self.assertEqual(len(json.loads((output / "comparison.json").read_text())["runs"]), 6)
            self.assertEqual(len(list((output / "logs").glob("*.log"))), 6)
            self.assertTrue((output / "provenance.json").is_file())
            self.assertTrue((output / "execution_order.json").is_file())
            self.assertIn("V3/V2", (output / "summary.md").read_text())
            self.assertIn("ALL 6 RUNS COMPLETE", captured.getvalue())
        self.assertEqual(len(commands), 6)
        self.assertTrue(all("--cuda-events" not in command for command in commands))

    def test_failed_worker_preserves_log_and_manifest_without_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b4.engine").write_bytes(b"engine")
            output = root / "output"
            argv = ["runner", "--source", "synthetic", "--engine-template", str(root / "b{batch}.engine"),
                    "--batches", "4", "--output-dir", str(output)]
            with patch("sys.argv", argv), patch(
                "scripts.run_t4_output_buffer_comparison.subprocess.run", return_value=SimpleNamespace(returncode=1)
            ), redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "partial results preserved"):
                main()
            self.assertEqual(json.loads((output / "execution_order.json").read_text())[0]["returncode"], 1)
            self.assertTrue(list((output / "logs").glob("*.log")))
            self.assertFalse((output / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
