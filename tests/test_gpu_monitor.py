from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from yolo_benchmark.gpu_monitor import GPUMonitor, summarize_gpu_metrics


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.wait_calls: list[int] = []

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        return 0


class GPUMonitorTest(unittest.TestCase):
    def test_start_stop_and_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            process = FakeProcess()
            with patch("yolo_benchmark.gpu_monitor.subprocess.Popen", return_value=process) as popen:
                with GPUMonitor(path, interval_ms=250):
                    pass

            self.assertIn("--loop-ms=250", popen.call_args.args[0])
            self.assertTrue(process.terminated)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("timestamp,"))

    def test_missing_nvidia_smi_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "yolo_benchmark.gpu_monitor.subprocess.Popen",
                side_effect=FileNotFoundError("nvidia-smi"),
            ):
                monitor = GPUMonitor(Path(directory) / "gpu.csv")
                monitor.start()
                monitor.stop()
        self.assertIsNotNone(monitor.warning)
        self.assertTrue(caught)

    def test_summary_filters_gpu_and_calculates_power(self):
        header = (
            "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                header
                + "t0,0,10,20,100,1000,50,60,70,500,4000\n"
                + "t0,1,99,99,999,1000,99,99,100,999,9999\n"
                + "t1,0,30,40,300,1000,70,70,70,700,5000\n",
                encoding="utf-8",
            )
            result = summarize_gpu_metrics(path, 0)

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["gpu_utilization_pct"]["average"], 20)
        self.assertEqual(result["memory_used_mib"]["maximum"], 300)
        self.assertEqual(result["power_w"]["average"], 65)
        self.assertEqual(result["sm_clock_mhz"]["maximum"], 700)
