from __future__ import annotations

import csv
import subprocess
import warnings
from pathlib import Path
from typing import TextIO


GPU_METRIC_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
)


class GPUMonitor:
    """Best-effort nvidia-smi sampler; monitoring failures never fail inference."""

    def __init__(self, output_path: Path, interval_ms: int = 200):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be greater than zero")
        self.output_path = Path(output_path)
        self.interval_ms = interval_ms
        self.warning: str | None = None
        self._process: subprocess.Popen | None = None
        self._output_file: TextIO | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = self.output_path.open("w", newline="", encoding="utf-8")
        csv.writer(self._output_file).writerow(GPU_METRIC_FIELDS)
        self._output_file.flush()
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_METRIC_FIELDS)}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._output_file,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            self.warning = f"GPU monitoring unavailable; benchmark continues without it: {exc}"
            warnings.warn(self.warning, RuntimeWarning, stacklevel=2)
            self._output_file.close()
            self._output_file = None

    def stop(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except OSError as exc:
            warnings.warn(f"Failed to stop GPU monitor cleanly: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            if self._output_file is not None:
                self._output_file.close()
                self._output_file = None

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return False


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "average": sum(values) / len(values) if values else None,
        "maximum": max(values) if values else None,
    }


def summarize_gpu_metrics(path: Path, gpu_index: int = 0) -> dict[str, object]:
    columns = {
        "gpu_utilization_pct": "utilization.gpu",
        "memory_utilization_pct": "utilization.memory",
        "memory_used_mib": "memory.used",
        "temperature_c": "temperature.gpu",
        "power_w": "power.draw",
        "sm_clock_mhz": "clocks.sm",
        "memory_clock_mhz": "clocks.mem",
    }
    values: dict[str, list[float]] = {key: [] for key in columns}
    sample_count = 0
    csv_path = Path(path)
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("index", "").strip() != str(gpu_index):
                    continue
                sample_count += 1
                for key, column in columns.items():
                    parsed = _as_float(row.get(column))
                    if parsed is not None:
                        values[key].append(parsed)
    result: dict[str, object] = {"index": gpu_index, "sample_count": sample_count}
    result.update({key: _summary(items) for key, items in values.items()})
    return result
