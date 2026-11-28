#!/usr/bin/env python3
"""Run V1/V2/V3 output-buffer controls in isolated processes (18 cases by default)."""
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

if __package__:
    from .profile_t4_output_buffers import MODES, POLICIES, write_json
else:
    from profile_t4_output_buffers import MODES, POLICIES, write_json


METRICS = (
    "preprocess_mean_ms", "inference_mean_ms", "postprocess_mean_ms", "end_to_end_mean_ms",
    "throughput_images_per_second", "inference_p95_ms", "end_to_end_p95_ms",
)


def case_order(batches, repeats):
    for repeat in range(1, repeats + 1):
        shift = (repeat - 1) % len(MODES)
        modes = MODES[shift:] + MODES[:shift]
        for batch in batches if repeat % 2 else reversed(batches):
            for mode in modes:
                yield batch, mode, repeat


def aggregate_results(records, batches, repeats):
    grouped = {(batch, mode): [] for batch in batches for mode in MODES}
    per_batch = {}
    common = None
    for payload in records:
        key = (payload["batch_size"], payload["mode"])
        if key not in grouped:
            raise ValueError(f"Unexpected configuration: {key}")
        if (
            payload.get("benchmark_only") is not True
            or payload.get("gpu_monitor_enabled") is not False
            or payload.get("validation_array_policy") != "released_before_warmup"
            or payload.get("output_allclose_passed") is not True
            or payload.get("output_policy") != POLICIES[key[1]]
        ):
            raise ValueError(f"Measurement/output policy differs: {key}")
        error = payload["max_abs_output_error"]
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid output error")
        result = payload["control"]
        if result["batch_size"] != key[0] or result["gpu_metrics"] is not None:
            raise ValueError(f"Batch or monitor mismatch: {key}")
        conditions = {
            "environment": payload["environment"], "source": payload["source"],
            "numpy": payload["numpy_version"], "opencv": payload["opencv_version"],
            "opencv_threads": payload["opencv_threads"], "torch_threads": payload["torch_threads"],
            "torch_interop_threads": payload["torch_interop_threads"],
            "cpu_affinity": payload["cpu_affinity"], "thread_environment": payload["thread_environment"],
            "image_size": result["image_size"], "warmup": result["warmup_runs"],
            "runs": result["measured_runs"], "precision": result["precision"], "backend": result["backend"],
        }
        if common is not None and common != conditions:
            raise ValueError("Environment, threads, or measurement conditions changed")
        common = conditions
        identity = {
            "engine": payload["engine"]["sha256"], "images": payload["input_images_sha256"],
            "input_shapes": payload["input_shapes"],
        }
        if key[0] in per_batch and per_batch[key[0]] != identity:
            raise ValueError(f"Engine SHA256 or actual input images changed for batch {key[0]}")
        per_batch[key[0]] = identity
        for metric in METRICS:
            if not math.isfinite(result[metric]) or result[metric] < 0:
                raise ValueError(f"Invalid metric: {metric}")
        if min(result[name] for name in ("inference_mean_ms", "end_to_end_mean_ms", "throughput_images_per_second")) <= 0:
            raise ValueError("Latency and throughput must be positive")
        expected_tps = key[0] * 1000 / result["inference_mean_ms"]
        if not math.isclose(result["throughput_images_per_second"], expected_tps, rel_tol=1e-8):
            raise ValueError("Throughput does not match the per-run inference mean")
        grouped[key].append(payload)

    groups = []
    for batch in sorted(batches):
        for mode in MODES:
            items = grouped[batch, mode]
            if len(items) != repeats or {item["repeat"] for item in items} != set(range(1, repeats + 1)):
                raise ValueError(f"Missing or duplicate repeats: {(batch, mode)}")
            metrics = {}
            for metric in METRICS:
                values = [item["control"][metric] for item in items]
                mean, stddev = statistics.fmean(values), statistics.pstdev(values)
                metrics[metric] = {"mean": mean, "stddev": stddev, "cv_pct": stddev / mean * 100 if mean else None}
            groups.append({"batch_size": batch, "mode": mode, "repeats": repeats, "metrics": metrics})
    lookup = {(group["batch_size"], group["mode"]): group["metrics"] for group in groups}
    comparisons = []
    for batch in sorted(batches):
        for baseline, candidate in ((MODES[0], MODES[1]), (MODES[1], MODES[2]), (MODES[0], MODES[2])):
            before, after = lookup[batch, baseline], lookup[batch, candidate]
            comparisons.append({
                "batch_size": batch, "baseline": baseline, "candidate": candidate,
                "inference_latency_reduction_pct": (1 - after["inference_mean_ms"]["mean"] / before["inference_mean_ms"]["mean"]) * 100,
                "throughput_gain_pct": (after["throughput_images_per_second"]["mean"] / before["throughput_images_per_second"]["mean"] - 1) * 100,
                "e2e_latency_reduction_pct": (1 - after["end_to_end_mean_ms"]["mean"] / before["end_to_end_mean_ms"]["mean"]) * 100,
            })
    return {"groups": groups, "comparisons": comparisons}


def format_report(summary):
    lines = [
        "# T4 output-buffer comparison (V1/V2/V3)", "",
        "Timings: ms/batch. Throughput: inference images/s (not E2E throughput).", "",
        "| Configuration | Repeats | Pre | Infer | Post | E2E | Infer img/s | Infer CV | E2E CV |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        m = group["metrics"]
        values = " | ".join(f"{m[name]['mean']:.3f}" for name in METRICS[:4])
        lines.append(
            f"| b{group['batch_size']}_{group['mode']} | {group['repeats']} | {values} | "
            f"{m['throughput_images_per_second']['mean']:.2f} | "
            f"{m['inference_mean_ms']['cv_pct']:.2f}% | {m['end_to_end_mean_ms']['cv_pct']:.2f}% |"
        )
    lines += ["", "| Batch | Baseline -> Candidate | Infer reduction | Throughput gain | E2E reduction |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['batch_size']} | {row['baseline']} -> {row['candidate']} | "
            f"{row['inference_latency_reduction_pct']:+.2f}% | {row['throughput_gain_pct']:+.2f}% | "
            f"{row['e2e_latency_reduction_pct']:+.2f}% |"
        )
    lines += [
        "", "V1 original: new pageable CPU output. V2 reuse_pageable: reused pageable CPU output.",
        "V3 reuse_pinned: reused pinned CPU output. V2/V1 tests the reuse path; V3/V2 tests pinning given reuse.",
        "This three-arm experiment does not estimate the full reuse/pinning interaction.",
        "Input handling and GPU output allocation remain unchanged; copies and inference stay synchronous.",
        "V2/V3 outputs are borrowed views, consumed before the next inference (not equivalent ownership to V1).",
        "Controls contain no CUDA-event instrumentation or GPU monitoring. Validation arrays are released before warmup.",
        "Means/stddev/CV are over independent run summaries. Throughput is averaged per run.",
        "P95 values in JSON are averages of per-run P95, not pooled percentiles.",
        "Negative reductions/gains mean regression. No confidence intervals, significance or full-dataset mAP claims.", "",
    ]
    return "\n".join(lines)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--engine-template", default="artifacts/t4/yolo11n_t4_fp16_b{batch}.engine")
    parser.add_argument("--batches", default="4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        batches = [int(value.strip()) for value in args.batches.split(",")]
        if not batches or len(set(batches)) != len(batches) or min(batches) <= 0:
            raise ValueError("batches must be distinct positive integers")
        engines = {batch: Path(args.engine_template.format(batch=batch)).resolve() for batch in batches}
    except (ValueError, KeyError, IndexError) as exc:
        parser.error(str(exc))
    if min(args.repeats, args.runs, args.imgsz) <= 0 or args.warmup < 0:
        parser.error("repeats, runs, imgsz must be positive; warmup must be non-negative")
    if args.source != "synthetic" and not Path(args.source).exists():
        parser.error(f"Source does not exist: {args.source}")
    for path in engines.values():
        if not path.is_file():
            parser.error(f"Engine does not exist: {path}")
    worker = Path(__file__).resolve().with_name("profile_t4_output_buffers.py")
    project = worker.parent.parent
    sources = [worker, Path(__file__).resolve(), *(project / "yolo_benchmark" / name for name in (
        "backends.py", "benchmark.py", "pipeline.py", "environment.py",
    ))]
    source_hashes = {str(path.relative_to(project)): sha256(path) for path in sources}
    engine_hashes = {batch: sha256(path) for batch, path in engines.items()}
    output = args.output_dir or Path("results/t4") / (
        "output_buffers_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    write_json(output / "provenance.json", {"source_sha256": source_hashes, "engine_sha256": engine_hashes})
    cases = list(case_order(batches, args.repeats))
    records, execution = [], []
    for index, (batch, mode, repeat) in enumerate(cases, 1):
        label = f"b{batch}_{mode}_repeat{repeat}"
        destination = output / label
        command = [
            sys.executable, str(worker), "--engine", str(engines[batch]), "--source", args.source,
            "--batch", str(batch), "--mode", mode, "--device", args.device,
            "--warmup", str(args.warmup), "--runs", str(args.runs), "--imgsz", str(args.imgsz),
            "--output-dir", str(destination),
        ]
        print(f"[{index}/{len(cases)}] {label} (log: {output / 'logs' / (label + '.log')})", flush=True)
        with (output / "logs" / f"{label}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        execution.append({"label": label, "command": command, "returncode": process.returncode})
        write_json(output / "execution_order.json", execution)
        if process.returncode:
            raise SystemExit(f"Failed: {label}. See {output / 'logs' / (label + '.log')}; partial results preserved.")
        payload = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        if (payload["mode"], payload["batch_size"]) != (mode, batch):
            raise ValueError(f"Worker metadata does not match scheduled case: {label}")
        if payload["engine"]["sha256"] != engine_hashes[batch]:
            raise ValueError(f"Engine changed after scheduling: {label}")
        records.append({**payload, "repeat": repeat})
        result = payload["control"]
        print(f"  infer={result['inference_mean_ms']:.3f}, E2E={result['end_to_end_mean_ms']:.3f} ms/batch", flush=True)
    if any(sha256(path) != source_hashes[str(path.relative_to(project))] for path in sources):
        raise ValueError("Experiment source files changed during the run")
    if any(sha256(path) != engine_hashes[batch] for batch, path in engines.items()):
        raise ValueError("Engine files changed during the run")
    summary = aggregate_results(records, batches, args.repeats)
    summary.update({"source_sha256": source_hashes, "runs": records})
    write_json(output / "comparison.json", summary)
    report = format_report(summary)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(f"\nALL {len(cases)} RUNS COMPLETE: {output}\n\n{report}", flush=True)


if __name__ == "__main__":
    main()
