from __future__ import annotations

import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_t4_pinned_comparison import aggregate_results, case_order, main


def example_records():
    records = []
    for mode, latencies, e2es in (
        ("original", (10.0, 20.0), (20.0, 40.0)),
        ("pinned", (5.0, 10.0), (15.0, 25.0)),
    ):
        for repeat, (latency, e2e) in enumerate(zip(latencies, e2es), 1):
            records.append({
                "mode": mode, "batch_size": 4, "repeat": repeat,
                "benchmark_only": True,
                "validation_array_policy": "released_before_warmup",
                "gpu_monitor_enabled": False,
                "output_allclose_passed": True,
                "preprocess_equivalence_passed": True,
                "environment": {"gpu": "Tesla T4", "torch": "test"},
                "source": "fixed-images", "opencv_threads": 1, "torch_threads": 2,
                "engine": {"sha256": "same-engine"},
                "control": {
                    "batch_size": 4, "image_size": [640, 640],
                    "warmup_runs": 50, "measured_runs": 500,
                    "preprocess_mean_ms": 2.0,
                    "inference_mean_ms": latency,
                    "postprocess_mean_ms": 3.0,
                    "end_to_end_mean_ms": e2e,
                    "throughput_images_per_second": 4000 / latency,
                    "inference_p95_ms": latency * 1.1,
                    "end_to_end_p95_ms": e2e * 1.1,
                },
            })
    return records


class PinnedComparisonTest(unittest.TestCase):
    def test_schedule_covers_all_twelve_cases_and_alternates_order(self):
        cases = list(case_order([4, 8], 3))
        self.assertEqual(len(cases), 12)
        self.assertEqual(len(set(cases)), 12)
        self.assertEqual(cases[0], (4, "original", 1))
        self.assertEqual(cases[4], (8, "pinned", 2))
        for batch in (4, 8):
            for mode in ("original", "pinned"):
                self.assertEqual(sum((b, m) == (batch, mode) for b, m, _ in cases), 3)

    def test_aggregate_uses_mean_of_run_throughputs_and_separate_e2e_gain(self):
        result = aggregate_results(example_records(), [4], 2)
        original = next(group for group in result["groups"] if group["mode"] == "original")
        self.assertEqual(original["metrics"]["inference_mean_ms"]["mean"], 15.0)
        self.assertEqual(original["metrics"]["throughput_images_per_second"]["mean"], 300.0)
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["throughput_gain_pct"], 100.0)
        self.assertEqual(comparison["inference_latency_reduction_pct"], 50.0)
        self.assertAlmostEqual(comparison["e2e_latency_reduction_pct"], 100 / 3)

    def test_engine_changes_are_rejected(self):
        records = example_records()
        records[-1]["engine"]["sha256"] = "another-engine"
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            aggregate_results(records, [4], 2)

    def test_duplicate_repeat_numbers_are_rejected(self):
        records = example_records()
        records[-1]["repeat"] = 1
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            aggregate_results(records, [4], 2)

    def test_memory_policy_changes_are_rejected(self):
        records = example_records()
        records[-1]["validation_array_policy"] = "retained"
        with self.assertRaisesRegex(ValueError, "policy differs"):
            aggregate_results(records, [4], 2)

    def test_thread_changes_are_rejected(self):
        records = example_records()
        records[-1]["opencv_threads"] = 4
        with self.assertRaisesRegex(ValueError, "thread settings changed"):
            aggregate_results(records, [4], 2)

    def test_runner_writes_complete_report_using_isolated_worker_commands(self):
        commands = []
        fixtures = example_records()

        def fake_worker(command, **kwargs):
            commands.append(command)
            destination = Path(command[command.index("--output-dir") + 1])
            mode = command[command.index("--mode") + 1]
            repeat = int(destination.name.rsplit("repeat", 1)[1])
            payload = next(row for row in fixtures if row["mode"] == mode and row["repeat"] == repeat)
            destination.mkdir(parents=True)
            (destination / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "b4.engine").write_bytes(b"mock engine")
            output = root / "output"
            argv = [
                "run_t4_pinned_comparison.py", "--source", str(root / "images"),
                "--engine-template", str(root / "b{batch}.engine"),
                "--batches", "4", "--repeats", "2", "--output-dir", str(output),
            ]
            captured = io.StringIO()
            with patch("sys.argv", argv), patch(
                "scripts.run_t4_pinned_comparison.subprocess.run", side_effect=fake_worker
            ), redirect_stdout(captured):
                main()
            summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["runs"]), 4)
            self.assertTrue((output / "summary.md").is_file())
            self.assertTrue((output / "execution_order.json").is_file())
            self.assertIn("ALL 4 RUNS COMPLETE", captured.getvalue())
        self.assertEqual(len(commands), 4)
        self.assertTrue(all("--benchmark-only" in command for command in commands))
