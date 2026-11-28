#!/usr/bin/env python3
"""Verify archived pinned-output results against raw CSV, then plot PNG figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

if __package__:
    from .run_t4_pinned_comparison import aggregate_results
else:
    from run_t4_pinned_comparison import aggregate_results


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare_values(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{label}: JSON fields differ")
        for key in expected:
            compare_values(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: JSON list length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_values(left, right, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            not isinstance(actual, (int, float)) or isinstance(actual, bool)
            or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-7)
        ):
            raise ValueError(f"{label}: stored={actual!r}, recomputed={expected!r}")
    elif actual != expected:
        raise ValueError(f"{label}: stored={actual!r}, expected={expected!r}")


def verify_csv(path, result):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != result["measured_runs"] or not rows:
        raise ValueError(f"{path}: row count does not match measured_runs")
    timings = {key: [] for key in ("preprocess", "inference", "postprocess", "end_to_end")}
    for iteration, row in enumerate(rows):
        if int(row["iteration"]) != iteration:
            raise ValueError(f"{path}: missing, duplicated, or unordered iteration {iteration}")
        if int(row["batch_size"]) != result["batch_size"] or row["backend"] != result["backend"]:
            raise ValueError(f"{path}: backend or batch size differs from summary")
        for stage in timings:
            value = float(row[f"{stage}_ms"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid timing in iteration {iteration}")
            timings[stage].append(value)
    calculated = {
        f"{stage}_mean_ms": statistics.fmean(values)
        for stage, values in timings.items()
    }
    for stage in ("inference", "end_to_end"):
        calculated[f"{stage}_stddev_ms"] = statistics.pstdev(timings[stage])
        for p in (50, 95, 99):
            calculated[f"{stage}_p{p}_ms"] = percentile(timings[stage], p)
    calculated["throughput_images_per_second"] = (
        result["batch_size"] * 1000 / calculated["inference_mean_ms"]
    )
    for metric, value in calculated.items():
        compare_values(result[metric], value, f"{path}:{metric}")
    return len(rows)


def verify_archive(run_dir):
    combined = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    stored_runs = {}
    for item in combined["runs"]:
        key = (item["batch_size"], item["mode"], item["repeat"])
        if key in stored_runs:
            raise ValueError(f"Duplicate run in comparison.json: {key}")
        stored_runs[key] = item
    records, labels, csv_count = [], set(), 0
    for case_dir in sorted(run_dir.iterdir()):
        match = re.fullmatch(r"b(\d+)_(original|pinned)_repeat(\d+)", case_dir.name)
        if not case_dir.is_dir() or not match:
            continue
        batch, mode, repeat = int(match[1]), match[2], int(match[3])
        payload = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        record = {**payload, "repeat": repeat}
        if (payload["batch_size"], payload["mode"]) != (batch, mode):
            raise ValueError(f"Case directory and metadata disagree: {case_dir}")
        compare_values(stored_runs.get((batch, mode, repeat)), record, case_dir.name)
        csv_count += verify_csv(case_dir / "control_latency.csv", payload["control"])
        records.append(record)
        labels.add(case_dir.name)
    if len(records) != len(stored_runs) or not records:
        raise ValueError("Missing or extra per-run directories")
    batches = sorted({item["batch_size"] for item in records})
    repeats = max(item["repeat"] for item in records)
    recomputed = aggregate_results(records, batches, repeats)
    for key in ("groups", "comparisons"):
        compare_values(combined[key], recomputed[key], f"comparison.json:{key}")
    execution = json.loads((run_dir / "execution_order.json").read_text(encoding="utf-8"))
    if len(execution) != len(records) or {item["label"] for item in execution} != labels:
        raise ValueError("Execution-order manifest does not match the archived cases")
    if any(item["returncode"] != 0 or "--benchmark-only" not in item["command"] for item in execution):
        raise ValueError("Execution manifest includes a failure or a profiled run")
    worker = Path(__file__).resolve().with_name("profile_t4_preprocess.py")
    local_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    audit = {
        "run_directory": str(run_dir.resolve()),
        "verified_cases": len(records),
        "verified_csv_rows": csv_count,
        "verified_csv_metrics": "stage means, standard deviations, P50/P95/P99, inference throughput",
        "aggregate_matches_raw": True,
        "worker_sha256": combined.get("worker_sha256"),
        "local_worker_sha256": local_hash,
        "worker_matches_local": local_hash == combined.get("worker_sha256"),
        "maximum_recorded_output_error": max(item["max_abs_output_error"] for item in records),
        "missing_logs": sorted(label for label in labels if not (run_dir / "logs" / f"{label}.log").is_file()),
        "environment": records[0]["environment"],
        "numpy_version": records[0].get("numpy_version"),
        "opencv_version": records[0].get("opencv_version"),
        "engine_sha256_by_batch": {
            str(batch): next(item["engine"]["sha256"] for item in records if item["batch_size"] == batch)
            for batch in batches
        },
        "notes": [
            "CSV verification does not re-execute GPU inference or reproduce mAP.",
            "Error bars in figures show population stddev of run means, not confidence intervals.",
        ],
    }
    return recomputed, audit


def plot_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "savefig.dpi": 180})
    batches = sorted({group["batch_size"] for group in summary["groups"]})
    groups = {(group["batch_size"], group["mode"]): group for group in summary["groups"]}
    colors = {"original": "#53769b", "pinned": "#168d83"}
    files = []
    for metric, filename, title, ylabel in (
        ("throughput_images_per_second", "output_buffer_throughput.png", "Pinned Output: Inference Throughput", "Inference Throughput (images/s)"),
        ("inference_mean_ms", "output_buffer_latency.png", "Pinned Output: Backend Call Latency", "Backend Call Latency (ms/batch)"),
        ("end_to_end_mean_ms", "output_buffer_e2e.png", "Pinned Output: Pipeline Latency", "Pipeline Latency (ms/batch)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for mode, offset, label in (("original", -0.19, "Original"), ("pinned", 0.19, "Reused pinned output")):
            positions = [index + offset for index in range(len(batches))]
            stats = [groups[batch, mode]["metrics"][metric] for batch in batches]
            heights = [item["mean"] for item in stats]
            errors = [item["stddev"] for item in stats]
            bars = ax.bar(positions, heights, width=0.36, yerr=errors, capsize=4, color=colors[mode], label=label)
            for bar, value, error in zip(bars, heights, errors):
                ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value + error),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
        ax.set_xticks(range(len(batches)), [str(batch) for batch in batches])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=15)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.text(0.5, 0.015, "3 independent runs per configuration; error bars = population standard deviation of run means", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(output / filename)
        plt.close(fig)
        files.append(filename)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = [groups[batch, mode] for batch in batches for mode in ("original", "pinned")]
    bottoms = [0.0] * len(ordered)
    for metric, label, color in (
        ("preprocess_mean_ms", "Preprocess", "#90aeca"),
        ("inference_mean_ms", "Backend call", "#168d83"),
        ("postprocess_mean_ms", "Postprocess", "#e9b568"),
        (None, "Other", "#a6a6a6"),
    ):
        if metric:
            values = [group["metrics"][metric]["mean"] for group in ordered]
        else:
            values = [max(0, group["metrics"]["end_to_end_mean_ms"]["mean"] - bottom)
                      for group, bottom in zip(ordered, bottoms)]
        ax.bar(range(len(ordered)), values, bottom=bottoms, color=color, width=0.6, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    for index, total in enumerate(bottoms):
        ax.annotate(f"{total:.2f}", (index, total), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(len(ordered)), [f"b{group['batch_size']}\n{group['mode'].capitalize()}" for group in ordered])
    ax.set_ylabel("Pipeline Latency (ms/batch)")
    ax.set_title("Pinned Output: Pipeline Stage Breakdown", pad=15)
    ax.set_ylim(0, max(bottoms) * 1.25)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.text(0.5, 0.015, "Arithmetic means of 3 independent runs; inference includes transfers and synchronization", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    filename = "output_buffer_breakdown.png"
    fig.savefig(output / filename)
    plt.close(fig)
    return files + [filename]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        summary, audit = verify_archive(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    audit["figures"] = [] if args.verify_only else plot_figures(summary, output)
    (output / "verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {audit['verified_cases']} cases / {audit['verified_csv_rows']} raw rows")
    print(f"Worker source matches local: {audit['worker_matches_local']}")
    print(f"Missing per-run logs: {len(audit['missing_logs'])}")
    for filename in audit["figures"]:
        print(output / filename)
    print(output / "verification.json")


if __name__ == "__main__":
    main()
