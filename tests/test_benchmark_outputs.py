from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yolo_benchmark.benchmark import run_benchmark_with_samples, write_raw_samples, write_results


class FakeBackend:
    name = "fake"

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return np.zeros((batch.shape[0], 1, 6), dtype=np.float32)

    def synchronize(self) -> None:
        pass

    def memory_mib(self) -> float:
        return 12.5


class FailingBackend(FakeBackend):
    def infer(self, batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("inference failed")


class RecordingContext:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False


class BenchmarkOutputTest(unittest.TestCase):
    def test_raw_samples_and_additive_summary_fields(self):
        images = [np.zeros((32, 32, 3), dtype=np.uint8)]
        context = RecordingContext()
        result, samples = run_benchmark_with_samples(
            FakeBackend(), images, (32, 32), 1, 3, 0.25, 0.45, "fp16", context
        )

        self.assertTrue(context.entered)
        self.assertTrue(context.exited)
        self.assertEqual(len(samples), 3)
        self.assertEqual(result.measured_runs, 3)
        self.assertGreaterEqual(result.inference_stddev_ms, 0)
        self.assertGreaterEqual(result.end_to_end_p95_ms, result.end_to_end_p50_ms)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.csv"
            json_path = root / "result.json"
            write_raw_samples(samples, raw_path)
            write_results(
                [result],
                str(json_path),
                environment={"gpu": "NVIDIA T4", "gpu_compute_capability": "7.5"},
                metadata={"run_tag": "t4-test"},
            )

            with raw_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["backend"], "fake")
        self.assertEqual(payload["environment"]["gpu"], "NVIDIA T4")
        self.assertEqual(payload["metadata"]["run_tag"], "t4-test")

    def test_invalid_run_counts_are_rejected(self):
        images = [np.zeros((16, 16, 3), dtype=np.uint8)]
        with self.assertRaisesRegex(ValueError, "runs must be greater than zero"):
            run_benchmark_with_samples(
                FakeBackend(), images, (16, 16), 0, 0, 0.25, 0.45, "fp32"
            )

    def test_measurement_context_exits_when_inference_fails(self):
        images = [np.zeros((16, 16, 3), dtype=np.uint8)]
        context = RecordingContext()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            run_benchmark_with_samples(
                FailingBackend(), images, (16, 16), 0, 1, 0.25, 0.45, "fp32", context
            )
        self.assertTrue(context.entered)
        self.assertTrue(context.exited)
