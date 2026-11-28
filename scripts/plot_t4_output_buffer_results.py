#!/usr/bin/env python3
"""Recompute V1/V2/V3 and Event archives, then optionally render CPU-only figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

if __package__:
    from .plot_t4_pinned_results import compare_values, verify_csv
    from .profile_t4_output_buffers import EVENT_INTERVALS, MODES, POLICIES, write_json
    from .run_t4_output_buffer_comparison import aggregate_results, sha256
else:
    from plot_t4_pinned_results import compare_values, verify_csv
    from profile_t4_output_buffers import EVENT_INTERVALS, MODES, POLICIES, write_json
    from run_t4_output_buffer_comparison import aggregate_results, sha256


WALL_FIELDS = (
    "h2d_call_wall_ms", "input_conversion_call_wall_ms", "shape_buffers_wall_ms",
    "enqueue_call_wall_ms", "device_sync_wait_wall_ms", "output_conversion_call_wall_ms",
    "host_output_call_wall_ms", "numpy_view_wall_ms", "backend_wall_before_event_readout_ms",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_event_csv(path, runs, stored_means):
    fields = {*WALL_FIELDS, *EVENT_INTERVALS}
    if set(stored_means) != fields:
        raise ValueError(f"{path}: missing or unexpected Event summary fields")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != {"iteration", *fields}:
            raise ValueError(f"{path}: missing or unexpected Event CSV columns")
        rows = list(reader)
    if len(rows) != runs or not rows:
        raise ValueError(f"{path}: Event row count differs from measured_runs")
    values = {field: [] for field in fields}
    for iteration, row in enumerate(rows):
        if int(row["iteration"]) != iteration:
            raise ValueError(f"{path}: missing or unordered Event iteration")
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid Event timing in {field}")
            values[field].append(value)
    means = {field: statistics.fmean(samples) for field, samples in values.items()}
    compare_values(stored_means, means, str(path))
    return len(rows)


def verify_archives(run_dir, event_dir):
    combined = read_json(run_dir / "comparison.json")
    provenance = read_json(run_dir / "provenance.json")
    compare_values(combined["source_sha256"], provenance["source_sha256"], "source provenance")
    records, stored, labels = [], {}, set()
    formal_rows = 0
    for item in combined["runs"]:
        key = (item["batch_size"], item["mode"], item["repeat"])
        if key in stored:
            raise ValueError(f"Duplicate stored formal case: {key}")
        stored[key] = item
    pattern = r"b(\d+)_(original|reuse_pageable|reuse_pinned)_repeat(\d+)"
    for directory in sorted(run_dir.iterdir()):
        match = re.fullmatch(pattern, directory.name)
        if not directory.is_dir() or not match:
            continue
        batch, mode, repeat = int(match[1]), match[2], int(match[3])
        payload = read_json(directory / "summary.json")
        if (payload["batch_size"], payload["mode"]) != (batch, mode):
            raise ValueError(f"Case directory and metadata disagree: {directory}")
        record = {**payload, "repeat": repeat}
        compare_values(stored.get((batch, mode, repeat)), record, directory.name)
        compare_values(payload["engine"]["sha256"], provenance["engine_sha256"][str(batch)], directory.name)
        formal_rows += verify_csv(directory / "control_latency.csv", payload["control"])
        records.append(record)
        labels.add(directory.name)
    if not records or len(records) != len(stored):
        raise ValueError("Missing or extra formal cases")
    batches = sorted({item["batch_size"] for item in records})
    repeats = max(item["repeat"] for item in records)
    summary = aggregate_results(records, batches, repeats)
    for field in ("groups", "comparisons"):
        compare_values(combined[field], summary[field], f"comparison.json:{field}")
    execution = read_json(run_dir / "execution_order.json")
    if len(execution) != len(records) or {item["label"] for item in execution} != labels:
        raise ValueError("Execution manifest does not match the formal cases")
    by_label = {f"b{item['batch_size']}_{item['mode']}_repeat{item['repeat']}": item for item in records}
    for item in execution:
        command = item["command"]
        if item["returncode"] != 0 or "--cuda-events" in command:
            raise ValueError("Failed or instrumented formal case in execution manifest")
        payload = by_label[item["label"]]
        arguments = {
            "--batch": str(payload["batch_size"]), "--mode": payload["mode"],
            "--source": payload["source"], "--engine": payload["engine"]["path"],
            "--warmup": str(payload["control"]["warmup_runs"]),
            "--runs": str(payload["control"]["measured_runs"]),
            "--imgsz": str(payload["control"]["image_size"][0]),
        }
        for flag, expected in arguments.items():
            if flag not in command or command[command.index(flag) + 1] != expected:
                raise ValueError(f"Manifest argument differs: {item['label']} {flag}")

    project = Path(__file__).resolve().parent.parent
    source_matches = {}
    for name, digest in provenance["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source provenance must use project-relative paths")
        path = project / relative
        source_matches[name] = path.is_file() and sha256(path) == digest

    expected_cases = {(batch, mode) for batch in batches for mode in MODES}
    event_groups, event_cases, event_errors = [], set(), []
    control_rows = profiled_rows = event_rows = 0
    for directory in sorted(event_dir.iterdir()):
        match = re.fullmatch(r"b(\d+)_(original|reuse_pageable|reuse_pinned)", directory.name)
        if not directory.is_dir() or not match:
            continue
        batch, mode = int(match[1]), match[2]
        if (batch, mode) not in expected_cases:
            raise ValueError(f"Unexpected Event case: {directory}")
        payload = read_json(directory / "summary.json")
        if (payload["batch_size"], payload["mode"]) != (batch, mode):
            raise ValueError(f"Event directory and metadata disagree: {directory}")
        if (payload["benchmark_only"] is not False or payload["output_policy"] != POLICIES[mode]
                or payload["gpu_monitor_enabled"] is not False
                or payload["output_allclose_passed"] is not True
                or payload["validation_array_policy"] != "released_before_warmup"):
            raise ValueError(f"Event phase validation policy differs: {directory}")
        reference = next(item for item in records if (item["batch_size"], item["mode"]) == (batch, mode))
        for key in (
            "environment", "source", "input_images_sha256", "input_shapes", "opencv_threads",
            "torch_threads", "torch_interop_threads", "cpu_affinity", "thread_environment",
            "numpy_version", "opencv_version",
        ):
            compare_values(payload[key], reference[key], f"{directory.name}:{key}")
        compare_values(payload["engine"]["sha256"], reference["engine"]["sha256"], f"{directory.name}:engine")
        event_summary = read_json(directory / "cuda_event_summary.json")
        for key in ("backend", "batch_size", "image_size", "precision", "warmup_runs", "measured_runs"):
            compare_values(event_summary["profiled"][key], payload["control"][key], f"{directory.name}:{key}")
        for key in ("backend", "batch_size", "image_size", "precision"):
            compare_values(payload["control"][key], reference["control"][key], f"{directory.name}:{key}")
        control_rows += verify_csv(directory / "control_latency.csv", payload["control"])
        profiled_rows += verify_csv(directory / "event_latency.csv", event_summary["profiled"])
        event_rows += verify_event_csv(directory / "cuda_event_samples.csv",
                                       event_summary["profiled"]["measured_runs"], event_summary["stage_mean_ms"])
        if not math.isfinite(payload["max_abs_output_error"]) or payload["max_abs_output_error"] < 0:
            raise ValueError("Invalid Event output error")
        event_errors.append(payload["max_abs_output_error"])
        event_groups.append({"batch_size": batch, "mode": mode, **event_summary})
        event_cases.add((batch, mode))
    if event_cases != expected_cases:
        raise ValueError("Missing Event cases")
    slow_runs = []
    for record in records:
        if record["mode"] != "original":
            continue
        label = f"b{record['batch_size']}_original_repeat{record['repeat']}"
        with (run_dir / label / "control_latency.csv").open(newline="") as file:
            times = [float(row["inference_ms"]) for row in csv.DictReader(file)]
        slow_runs.append({
            "case": label, "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times),
            "samples_over_35_ms": sum(value > 35 for value in times),
            "threshold_note": "35 ms is descriptive only; no samples are excluded",
        })
    audit = {
        "formal_directory": str(run_dir.resolve()), "event_directory": str(event_dir.resolve()),
        "verified_formal_cases": len(records), "formal_latency_rows": formal_rows,
        "verified_event_cases": len(event_groups), "diagnostic_control_latency_rows": control_rows,
        "diagnostic_profiled_latency_rows": profiled_rows, "cuda_event_rows": event_rows,
        "total_csv_rows": formal_rows + control_rows + profiled_rows + event_rows,
        "csv_and_aggregate_match": True, "source_matches_local": source_matches,
        "engine_sha256_by_batch": provenance["engine_sha256"],
        "formal_max_recorded_output_error": max(item["max_abs_output_error"] for item in records),
        "diagnostic_max_recorded_output_error": max(event_errors),
        "missing_formal_logs": sorted(label for label in labels if not (run_dir / "logs" / f"{label}.log").is_file()),
        "original_run_diagnostics": slow_runs,
        "notes": [
            "All latency CSV means/stddev/P50/P95/P99/throughput and all Event CSV means are recomputed.",
            "Event archives contain no independent source hash or execution manifest; matched metadata is not source proof.",
            "Engine digests are compared across JSON records; Engine binaries and original images are not rehashed here.",
            "Reported output error is the saved fixed-batch check, not a new inference or full-dataset mAP test.",
        ],
    }
    return summary, event_groups, audit


def plot_figures(summary, events, run_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#53769b", "#d59a42", "#168d83")
    labels = ("V1 Original", "V2 Reuse pageable", "V3 Reuse pinned")
    batches = sorted({item["batch_size"] for item in summary["groups"]})
    lookup = {(item["batch_size"], item["mode"]): item for item in summary["groups"]}
    event_lookup = {(item["batch_size"], item["mode"]): item for item in events}
    files = []
    for filename, metric, title, ylabel in (
        ("three_mode_throughput.png", "throughput_images_per_second", "Output Buffers: Inference Throughput", "Inference Throughput (images/s)"),
        ("three_mode_latency.png", "inference_mean_ms", "Output Buffers: Backend Call Latency", "Backend Call Latency (ms/batch)"),
        ("three_mode_e2e.png", "end_to_end_mean_ms", "Output Buffers: Pipeline Latency", "Pipeline Latency (ms/batch)"),
        ("three_mode_d2h_event.png", None, "Output Buffers: D2H Event Interval", "Stream Interval (ms/batch)"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for index, (mode, color, label) in enumerate(zip(MODES, colors, labels)):
            stats = [lookup[batch, mode]["metrics"][metric] for batch in batches] if metric else None
            values = [item["mean"] for item in stats] if metric else [event_lookup[batch, mode]["stage_mean_ms"]["d2h_event_ms"] for batch in batches]
            errors = [item["stddev"] for item in stats] if metric else [0] * len(batches)
            bars = ax.bar([i + (index - 1) * .25 for i in range(len(batches))], values, width=.24,
                          color=color, label=label, yerr=errors if metric else None, capsize=3)
            for bar, value, error in zip(bars, values, errors):
                ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value + error),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(batches)), [str(batch) for batch in batches])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=14)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
        note = ("Formal controls; error bars = population stddev of run means (not confidence intervals)" if metric else
                "Separate diagnostic phase; stream intervals include possible host gaps (not pure DMA time)")
        fig.text(.5, .015, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        files.append(filename)

    fig, ax = plt.subplots(figsize=(9, 5))
    for directory in sorted(run_dir.glob("b8_original_repeat*")):
        with (directory / "control_latency.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        ax.plot([int(row["iteration"]) for row in rows], [float(row["inference_ms"]) for row in rows],
                label=directory.name.rsplit("_", 1)[-1], linewidth=.8, alpha=.8)
    ax.set_xlabel("Measured Iteration (warmup excluded)")
    ax.set_ylabel("Backend Call Latency (ms/batch)")
    ax.set_title("Batch 8 Original: All Formal Samples")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5, .015, "All samples retained; elevated run is shown without claiming its underlying cause", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    name = "b8_original_samples.png"
    fig.savefig(output / name, dpi=180)
    plt.close(fig)
    return files + [name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        summary, events, audit = verify_archives(args.run_dir, args.events_dir)
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    audit["figures"] = [] if args.verify_only else plot_figures(summary, events, args.run_dir, output)
    write_json(output / "verification.json", audit)
    print(f"Verified {audit['verified_formal_cases']} formal cases + {audit['verified_event_cases']} Event cases")
    print(f"CSV rows: {audit['total_csv_rows']} (formal {audit['formal_latency_rows']}, "
          f"diagnostic control/profiled/Event {audit['diagnostic_control_latency_rows']}/"
          f"{audit['diagnostic_profiled_latency_rows']}/{audit['cuda_event_rows']})")
    print(f"All recorded source hashes match local: {all(audit['source_matches_local'].values())}")
    print(f"Missing formal logs: {len(audit['missing_formal_logs'])}")
    for filename in [*audit["figures"], "verification.json"]:
        print(output / filename)


if __name__ == "__main__":
    main()
