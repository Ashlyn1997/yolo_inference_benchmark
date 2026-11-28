from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_t4_pinned_results import verify_csv


class PinnedResultArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "control_latency.csv"
        self.rows = [
            {"backend": "tensorrt", "iteration": 0, "batch_size": 4,
             "preprocess_ms": 1, "inference_ms": 2, "postprocess_ms": 3, "end_to_end_ms": 6.5},
            {"backend": "tensorrt", "iteration": 1, "batch_size": 4,
             "preprocess_ms": 2, "inference_ms": 4, "postprocess_ms": 4, "end_to_end_ms": 10.5},
        ]
        self.result = {
            "backend": "tensorrt", "batch_size": 4, "measured_runs": 2,
            "preprocess_mean_ms": 1.5, "inference_mean_ms": 3,
            "postprocess_mean_ms": 3.5, "end_to_end_mean_ms": 8.5,
            "inference_stddev_ms": 1, "end_to_end_stddev_ms": 2,
            "inference_p50_ms": 3, "inference_p95_ms": 3.9, "inference_p99_ms": 3.98,
            "end_to_end_p50_ms": 8.5, "end_to_end_p95_ms": 10.3, "end_to_end_p99_ms": 10.46,
            "throughput_images_per_second": 4000 / 3,
        }

    def write_csv(self):
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_raw_rows_reproduce_mean_percentiles_stddev_and_throughput(self):
        self.write_csv()
        self.assertEqual(verify_csv(self.path, self.result), 2)

    def test_modified_timing_is_detected(self):
        self.rows[1]["inference_ms"] = 6
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "inference_mean_ms"):
            verify_csv(self.path, self.result)

    def test_duplicate_iteration_is_detected(self):
        self.rows[1]["iteration"] = 0
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "unordered iteration"):
            verify_csv(self.path, self.result)

    def test_truncated_csv_is_detected(self):
        self.rows.pop()
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "row count"):
            verify_csv(self.path, self.result)

    def test_nonfinite_timing_is_detected(self):
        self.rows[1]["inference_ms"] = float("nan")
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "invalid timing"):
            verify_csv(self.path, self.result)
