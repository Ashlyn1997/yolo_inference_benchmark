#!/usr/bin/env python3
"""Run the final original/pinned output comparison in isolated Python processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    "preprocess_mean_ms", "inference_mean_ms", "postprocess_mean_ms",
    "end_to_end_mean_ms", "throughput_images_per_second",
    "inference_p95_ms", "end_to_end_p95_ms",
)


def case_order(batches, repeats):
    for repeat in range(1, repeats + 1):
        batch_order = batches if repeat % 2 else list(reversed(batches))
        modes = ("original", "pinned") if repeat % 2 else ("pinned", "original")
        for batch in batch_order:
            for mode in modes:
                yield batch, mode, repeat


def aggregate_results(records, batches, repeats):
    expected = {(batch, mode) for batch in batches for mode in ("original", "pinned")}
    grouped = {key: [] for key in expected}
    hashes = {}
    conditions = None
    for payload in records:
        key = (payload["batch_size"], payload["mode"])
        if key not in grouped:
            raise ValueError(f"Unexpected configuration: {key}")
        if (
            not payload.get("benchmark_only")
            or payload.get("validation_array_policy") != "released_before_warmup"
            or payload.get("gpu_monitor_enabled") is not False
            or payload.get("output_allclose_passed") is not True
            or payload.get("preprocess_equivalence_passed") is not True
        ):
            raise ValueError(f"Measurement/validation policy differs: {key}")
        environment = payload["environment"]
        result = payload["control"]
        if result["batch_size"] != key[0]:
            raise ValueError(f"Batch size mismatch: {key}")
        signature = (
            tuple(environment.get(field) for field in (
                "gpu", "gpu_index", "gpu_compute_capability", "driver_version",
                "python", "torch", "torch_cuda", "tensorrt", "ultralytics",
                "onnx", "onnxruntime",
            )),
            payload["source"], payload["opencv_threads"], payload["torch_threads"],
            payload.get("numpy_version"), payload.get("opencv_version"),
            tuple(result["image_size"]), result["warmup_runs"], result["measured_runs"],
        )
        if conditions is not None and signature != conditions:
            raise ValueError("Environment, input configuration, or thread settings changed between runs")
        conditions = signature
        digest = payload["engine"]["sha256"]
        if key[0] in hashes and hashes[key[0]] != digest:
            raise ValueError(f"Engine SHA256 changed for batch {key[0]}")
        hashes[key[0]] = digest
        for metric in METRICS:
            value = float(result[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metric {metric}: {value}")
        if min(result["inference_mean_ms"], result["end_to_end_mean_ms"],
               result["throughput_images_per_second"]) <= 0:
            raise ValueError("Latency and throughput must be positive")
        grouped[key].append(payload)

    groups = []
    for (batch, mode), items in sorted(grouped.items()):
        if len(items) != repeats or {item["repeat"] for item in items} != set(range(1, repeats + 1)):
            raise ValueError(f"Missing or duplicate repeats: batch={batch}, mode={mode}")
        metrics = {}
        for metric in METRICS:
            values = [float(item["control"][metric]) for item in items]
            mean, stddev = statistics.fmean(values), statistics.pstdev(values)
            metrics[metric] = {
                "mean": mean, "stddev": stddev,
                "cv_pct": stddev / mean * 100 if mean else None,
            }
        groups.append({"batch_size": batch, "mode": mode, "repeats": repeats, "metrics": metrics})

    lookup = {(group["batch_size"], group["mode"]): group["metrics"] for group in groups}
    comparisons = []
    for batch in batches:
        original, pinned = lookup[batch, "original"], lookup[batch, "pinned"]
        comparisons.append({
            "batch_size": batch,
            "inference_latency_reduction_pct": (
                1 - pinned["inference_mean_ms"]["mean"] / original["inference_mean_ms"]["mean"]
            ) * 100,
            "throughput_gain_pct": (
                pinned["throughput_images_per_second"]["mean"]
                / original["throughput_images_per_second"]["mean"] - 1
            ) * 100,
            "e2e_latency_reduction_pct": (
                1 - pinned["end_to_end_mean_ms"]["mean"] / original["end_to_end_mean_ms"]["mean"]
            ) * 100,
        })
    return {"groups": groups, "comparisons": comparisons}


def format_report(summary):
    lines = [
        "# T4 pinned-output final comparison", "",
        "All timings are ms/batch; throughput is inference images/s.", "",
        "| Configuration | Repeats | Pre | Infer | Post | E2E | Infer img/s | Infer CV | E2E CV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        m = group["metrics"]
        lines.append(
            f"| b{group['batch_size']}_{group['mode']} | {group['repeats']} | "
            f"{m['preprocess_mean_ms']['mean']:.3f} | {m['inference_mean_ms']['mean']:.3f} | "
            f"{m['postprocess_mean_ms']['mean']:.3f} | {m['end_to_end_mean_ms']['mean']:.3f} | "
            f"{m['throughput_images_per_second']['mean']:.2f} | "
            f"{m['inference_mean_ms']['cv_pct']:.2f}% | {m['end_to_end_mean_ms']['cv_pct']:.2f}% |"
        )
    lines.extend([
        "", "| Batch | Infer latency reduction | Infer throughput gain | E2E latency reduction |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['batch_size']} | {row['inference_latency_reduction_pct']:+.2f}% | "
            f"{row['throughput_gain_pct']:+.2f}% | {row['e2e_latency_reduction_pct']:+.2f}% |"
        )
    lines.extend([
        "", "Positive reductions/gains mean improvement; negative means regression.",
        "Means and population standard deviations are over independent runs.",
        "Throughput is the mean of per-run throughput, not the reciprocal of mean latency.",
        "P95 fields in comparison.json are averages of per-run P95, not a pooled percentile.",
        "Validation arrays are released before warmup. GPU monitoring and measured profiling are disabled.",
        "Pinned outputs are reused views consumed synchronously; full-dataset mAP is not revalidated here.",
        "No statistical significance or confidence interval is inferred from these repeats.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--engine-template", default="artifacts/t4/yolo11n_t4_fp16_b{batch}.engine")
    parser.add_argument("--batches", default="4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        batches = [int(value.strip()) for value in args.batches.split(",")]
        engines = {batch: str(Path(args.engine_template.format(batch=batch)).resolve()) for batch in batches}
    except (ValueError, KeyError, IndexError) as exc:
        parser.error(str(exc))
    if not batches or len(set(batches)) != len(batches) or min(batches) <= 0:
        parser.error("batches must be distinct positive integers")
    if args.repeats <= 0 or args.runs <= 0 or args.warmup < 0:
        parser.error("repeats/runs must be positive; warmup must be non-negative")
    if args.source != "synthetic" and not Path(args.source).exists():
        parser.error(f"Source does not exist: {args.source}")
    for engine in engines.values():
        if not Path(engine).is_file():
            parser.error(f"Engine does not exist: {engine}")
    worker = Path(__file__).resolve().with_name("profile_t4_preprocess.py")
    output = args.output_dir or Path("results/t4") / (
        "pinned_final_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    cases = list(case_order(batches, args.repeats))
    records, execution_order = [], []
    for index, (batch, mode, repeat) in enumerate(cases, 1):
        label = f"b{batch}_{mode}_repeat{repeat}"
        case_dir = output / label
        command = [
            sys.executable, str(worker), "--benchmark-only",
            "--engine", engines[batch], "--source", args.source,
            "--batch", str(batch), "--mode", mode,
            "--device", args.device, "--warmup", str(args.warmup), "--runs", str(args.runs),
            "--output-dir", str(case_dir),
        ]
        print(f"[{index}/{len(cases)}] {label}", flush=True)
        log_path = output / "logs" / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        execution_order.append({"label": label, "command": command, "returncode": process.returncode})
        (output / "execution_order.json").write_text(json.dumps(execution_order, indent=2), encoding="utf-8")
        if process.returncode:
            print(log_path.read_text(errors="replace")[-8000:], flush=True)
            raise SystemExit(f"Failed: {label}; log: {log_path}")
        payload = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        records.append({**payload, "repeat": repeat})
        result = payload["control"]
        print(
            f"  infer={result['inference_mean_ms']:.3f} ms, "
            f"E2E={result['end_to_end_mean_ms']:.3f} ms", flush=True,
        )
    summary = aggregate_results(records, batches, args.repeats)
    summary["worker_sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
    summary["runs"] = records
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = format_report(summary)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"\nALL {len(cases)} RUNS COMPLETE: {output}\n", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
