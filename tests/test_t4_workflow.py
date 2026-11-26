from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from yolo_benchmark import cli
from yolo_benchmark.validation import validate_model


class T4WorkflowTest(unittest.TestCase):
    def test_t4_cli_arguments(self):
        args = cli.build_parser().parse_args(
            [
                "benchmark",
                "--engine",
                "model.engine",
                "--expected-gpu",
                "T4",
                "--gpu-metrics-dir",
                "results/t4/gpu",
                "--raw-output",
                "results/t4/raw.csv",
                "--run-tag",
                "t4-fp16-b1",
            ]
        )
        self.assertEqual(args.expected_gpu, "T4")
        self.assertEqual(args.gpu_monitor_interval_ms, 200)
        self.assertEqual(args.run_tag, "t4-fp16-b1")

    def test_t4_scripts_use_isolated_directories(self):
        project = Path(__file__).resolve().parents[1]
        build_script = (project / "scripts/build_t4_engines.sh").read_text(encoding="utf-8")
        run_script = (project / "scripts/run_t4_matrix.sh").read_text(encoding="utf-8")
        validate_script = (project / "scripts/validate_t4_models.sh").read_text(encoding="utf-8")
        self.assertIn("artifacts/t4", build_script)
        self.assertIn("results/t4", run_script)
        self.assertIn("runs/t4", validate_script)
        self.assertIn("REPEATS=\"${REPEATS:-3}\"", run_script)
        self.assertIn("--expected-gpu", run_script)

    def test_validation_writes_structured_accuracy(self):
        class FakeBox:
            map = 0.4
            map50 = 0.6
            map75 = 0.45
            mp = 0.7
            mr = 0.8
            maps = [0.3, 0.5]

        class FakeMetrics:
            box = FakeBox()
            names = {0: "cat", 1: "dog"}
            speed = {"preprocess": 1.0, "inference": 2.0}
            save_dir = "runs/t4/test"

        class FakeYOLO:
            names = {0: "cat", 1: "dog"}

            def __init__(self, model):
                self.model = model

            def val(self, **kwargs):
                return FakeMetrics()

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.engine"
            model.write_bytes(b"engine")
            output = root / "validation.json"
            with patch.dict("sys.modules", {"ultralytics": fake_ultralytics}), patch(
                "yolo_benchmark.validation.collect_environment", return_value={"gpu": "NVIDIA T4"}
            ):
                payload = validate_model(
                    str(model), "coco128.yaml", str(output), precision="fp16", plots=False
                )
            output_exists = output.exists()

        self.assertEqual(payload["validation"]["map50_95"], 0.4)
        self.assertEqual(payload["validation"]["per_class_map50_95"]["dog"], 0.5)
        self.assertEqual(payload["model"]["precision"], "fp16")
        self.assertTrue(output_exists)
