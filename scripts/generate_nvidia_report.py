#!/usr/bin/env python3
"""Generate a compact Markdown report from TensorRT benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


DISPLAY_NAMES = {
    "nvidia_trt_fp32_b1": "TensorRT FP32 / Batch 1",
    "nvidia_trt_fp16_b1": "TensorRT FP16 / Batch 1",
    "nvidia_trt_fp16_b4": "TensorRT FP16 / Batch 4",
    "nvidia_trt_fp16_b8": "TensorRT FP16 / Batch 8",
    "nvidia_trt_int8_b4": "TensorRT INT8 / Batch 4",
    "coco128_batch1_fp32": "Backend comparison / Batch 1 FP32",
    "coco128_batch1_trt_fp16": "TensorRT FP16 / Batch 1",
    "coco128_trt_fp16_fixed_batch4": "TensorRT FP16 / Batch 4",
    "coco128_trt_fp16_fixed_batch8": "TensorRT FP16 / Batch 8",
    "coco128_trt_int8_fixed_batch4": "TensorRT INT8 / Batch 4",
}


def load_rows(results_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    environment: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        environment = environment or dict(payload.get("environment") or {})
        for result in payload["results"]:
            if isinstance(result, dict):
                rows.append({"source": path.stem, **result})
    return environment, rows


def value(row: dict[str, object], key: str, digits: int = 2) -> str:
    raw = row.get(key)
    return f"{float(raw):.{digits}f}" if isinstance(raw, (float, int)) else "-"


def nested_value(row: dict[str, object], *keys: str, digits: int = 2) -> str:
    raw: object = row
    for key in keys:
        if not isinstance(raw, dict):
            return "-"
        raw = raw.get(key)
    return f"{float(raw):.{digits}f}" if isinstance(raw, (float, int)) else "-"


def repeat_group(source: str) -> str:
    return re.sub(r"_repeat\d+$", "", source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="reports/nvidia_summary.md")
    args = parser.parse_args()

    environment, rows = load_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No benchmark JSON files found under {args.results_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NVIDIA TensorRT Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Environment",
        "",
        f"- GPU: {environment.get('gpu', '-')}",
        f"- PyTorch: {environment.get('torch', '-')}",
        f"- CUDA: {environment.get('torch_cuda', '-')}",
        "",
        "## Performance Results",
        "",
        "| Configuration | Backend | Precision | Batch | Infer Mean | P95 | Throughput | E2E Mean | Avg Power | img/s/W | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        source = str(row["source"])
        label = DISPLAY_NAMES.get(source, source)
        lines.append(
            "| {label} | {backend} | {precision} | {batch} | {infer} ms | {p95} ms | {throughput} img/s | {e2e} ms | {power} W | {efficiency} | `{source}` |".format(
                label=label,
                backend=row.get("backend", "-"),
                precision=row.get("precision", "-"),
                batch=row.get("batch_size", "-"),
                infer=value(row, "inference_mean_ms"),
                p95=value(row, "inference_p95_ms"),
                throughput=value(row, "throughput_images_per_second"),
                e2e=value(row, "end_to_end_mean_ms"),
                power=nested_value(row, "gpu_metrics", "power_w", "average"),
                efficiency=value(row, "images_per_second_per_watt", 3),
                source=source,
            )
        )
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(repeat_group(str(row["source"])), []).append(row)
    if any(len(items) > 1 for items in groups.values()):
        lines.extend(
            [
                "",
                "## Repeat Aggregate",
                "",
                "| Configuration | Repeats | Infer Mean | Between-run Stddev | CV | Throughput Mean | E2E Mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, items in sorted(groups.items()):
            inference = [float(item["inference_mean_ms"]) for item in items]
            throughput = [float(item["throughput_images_per_second"]) for item in items]
            e2e = [float(item["end_to_end_mean_ms"]) for item in items]
            infer_mean = statistics.fmean(inference)
            infer_stddev = statistics.pstdev(inference)
            cv = infer_stddev / infer_mean * 100 if infer_mean > 0 else 0.0
            lines.append(
                f"| {label} | {len(items)} | {infer_mean:.2f} ms | {infer_stddev:.3f} ms | "
                f"{cv:.2f}% | {statistics.fmean(throughput):.2f} img/s | "
                f"{statistics.fmean(e2e):.2f} ms |"
            )
    lines.extend([
        "",
        "## Measurement Notes",
        "",
        "- Benchmark JSON measures synchronized backend execution and shared preprocessing/postprocessing.",
        "- Compare only rows with the same image size, dataset source, warmup count, and measured run count.",
        "- Accuracy metrics must be collected separately with Ultralytics validation and paired with this report.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
