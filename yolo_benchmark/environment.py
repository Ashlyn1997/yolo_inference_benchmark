from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:  # Environment diagnostics must remain best-effort.
        return f"unavailable: {exc}"


def _nvidia_smi(index: int) -> dict[str, Any]:
    fields = (
        "name",
        "driver_version",
        "memory.total",
        "power.limit",
        "compute_cap",
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if len(values) != len(fields):
            raise ValueError(f"unexpected nvidia-smi output: {completed.stdout!r}")
        return {
            "gpu": values[0],
            "driver_version": values[1],
            "gpu_memory_total_mib": float(values[2]),
            "gpu_power_limit_w": float(values[3]),
            "gpu_compute_capability": values[4],
            "nvidia_smi": completed.stdout.strip(),
        }
    except Exception as exc:
        # Some older nvidia-smi versions do not expose compute_cap. Keep the
        # remaining identity fields and let PyTorch report compute capability.
        fallback_fields = fields[:-1]
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    f"--query-gpu={','.join(fallback_fields)}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            values = [value.strip() for value in completed.stdout.strip().split(",")]
            if len(values) != len(fallback_fields):
                raise ValueError(f"unexpected nvidia-smi output: {completed.stdout!r}")
            return {
                "gpu": values[0],
                "driver_version": values[1],
                "gpu_memory_total_mib": float(values[2]),
                "gpu_power_limit_w": float(values[3]),
                "nvidia_smi": completed.stdout.strip(),
            }
        except Exception:
            return {"nvidia_smi": f"unavailable: {exc}"}


def collect_environment(gpu_index: int = 0) -> dict[str, Any]:
    """Collect enough software and hardware identity to compare benchmark runs."""
    import torch

    cuda_available = torch.cuda.is_available()
    details: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_index": gpu_index,
        "gpu": torch.cuda.get_device_name(gpu_index) if cuda_available else None,
        "gpu_compute_capability": (
            ".".join(str(value) for value in torch.cuda.get_device_capability(gpu_index))
            if cuda_available
            else None
        ),
        "ultralytics": _module_version("ultralytics"),
        "onnx": _module_version("onnx"),
        "onnxruntime": _module_version("onnxruntime"),
        "tensorrt": _module_version("tensorrt"),
    }
    smi = _nvidia_smi(gpu_index)
    for key, value in smi.items():
        if value is not None or details.get(key) is None:
            details[key] = value
    return details


def file_metadata(path: str | Path) -> dict[str, Any]:
    """Return immutable identity fields for a model, ONNX file, or engine."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }
