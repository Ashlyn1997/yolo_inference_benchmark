from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .backends import create_backend
from .benchmark import print_results, run_benchmark_with_samples, write_raw_samples, write_results
from .environment import collect_environment, file_metadata
from .gpu_monitor import GPUMonitor, summarize_gpu_metrics
from .pipeline import load_images, postprocess, preprocess
from .validation import validate_model


def parse_size(values: list[int]) -> tuple[int, int]:
    return (values[0], values[0]) if len(values) == 1 else (values[0], values[1])


def environment(args: argparse.Namespace) -> None:
    print(json.dumps(collect_environment(args.gpu_index), indent=2, ensure_ascii=False))


def download(args: argparse.Namespace) -> None:
    """Download an official Ultralytics checkpoint into the artifacts directory."""
    from ultralytics import YOLO

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(args.model).name
    if destination.is_file() and not args.force:
        print(f"Model already exists: {destination}")
        return

    # A bare official model name makes Ultralytics resolve and download the
    # checkpoint from its official release assets when it is not cached.
    model = YOLO(Path(args.model).name)
    downloaded = Path(model.ckpt_path).resolve()
    if not downloaded.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce a checkpoint at {downloaded}")
    if downloaded != destination:
        if destination.exists():
            destination.unlink()
        shutil.move(str(downloaded), str(destination))
    print(f"Downloaded model: {destination}")


def export(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    if args.half and args.int8:
        raise SystemExit("--half and --int8 are mutually exclusive")
    if args.int8 and not args.data:
        raise SystemExit("--int8 requires --data with a representative calibration dataset YAML")
    if args.output_name and len(args.formats) != 1:
        raise SystemExit("--output-name requires exactly one export format")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    for export_format in args.formats:
        # Ultralytics validates arguments per export format. In particular,
        # workspace and calibration data are TensorRT-only and must not be
        # passed (even with an inactive value) to the ONNX exporter.
        export_options = {
            "format": export_format,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "dynamic": args.dynamic,
            "device": args.device,
            "simplify": True,
        }
        if export_format == "engine":
            export_options["workspace"] = args.workspace
            if args.int8:
                export_options["int8"] = True
                export_options["data"] = args.data
            elif args.half:
                export_options["half"] = True
        exported = Path(model.export(**export_options))
        destination = output_dir / (args.output_name or exported.name)
        if exported.resolve() != destination:
            exported.replace(destination)
        print(f"Exported {export_format}: {destination}")


def benchmark(args: argparse.Namespace) -> None:
    size = parse_size(args.imgsz)
    images = load_images(args.source, args.batch, size)
    results = []
    all_samples = []
    environment_details = collect_environment(args.gpu_index)
    if args.expected_gpu and args.expected_gpu.lower() not in str(environment_details.get("gpu", "")).lower():
        raise SystemExit(
            f"Expected GPU containing {args.expected_gpu!r}, found {environment_details.get('gpu')!r}"
        )
    definitions = [
        ("pytorch", args.pt, args.half),
        ("onnxruntime", args.onnx, False),
        ("tensorrt", args.engine, False),
    ]
    if not any(path for _, path, _ in definitions):
        raise SystemExit("Provide at least one of --pt, --onnx or --engine")
    for kind, path, use_half in definitions:
        if not path:
            continue
        print(f"Loading {kind}: {path}")
        backend = create_backend(kind, path, args.device, half=use_half)
        monitor = None
        monitor_path = None
        if args.gpu_metrics_dir:
            tag = args.run_tag or Path(args.output).stem
            monitor_path = Path(args.gpu_metrics_dir) / f"{tag}_{kind}.csv"
            monitor = GPUMonitor(monitor_path, interval_ms=args.gpu_monitor_interval_ms)
        result, samples = run_benchmark_with_samples(
            backend,
            images,
            size,
            args.warmup,
            args.runs,
            args.confidence,
            args.iou,
            args.precision or ("fp16" if (kind == "pytorch" and use_half) else "engine-defined"),
            measurement_context=monitor,
        )
        if monitor_path is not None:
            result.gpu_metrics = summarize_gpu_metrics(monitor_path, args.gpu_index)
            power = result.gpu_metrics["power_w"]
            average_power = power.get("average") if isinstance(power, dict) else None
            if isinstance(average_power, (int, float)) and average_power > 0:
                result.images_per_second_per_watt = (
                    result.throughput_images_per_second / average_power
                )
        results.append(result)
        all_samples.extend(samples)
        del backend
    print_results(results)
    artifacts = [
        {"backend": kind, **file_metadata(path)}
        for kind, path, _ in definitions
        if path
    ]
    metadata = {
        "run_tag": args.run_tag,
        "command": sys.argv,
        "source": args.source,
        "artifacts": artifacts,
    }
    write_results(results, args.output, environment=environment_details, metadata=metadata)
    if args.raw_output:
        write_raw_samples(all_samples, args.raw_output)
        print(f"Raw latency samples: {Path(args.raw_output).resolve()}")
    print(f"\nJSON report: {Path(args.output).resolve()}")


def validate(args: argparse.Namespace) -> None:
    payload = validate_model(
        args.model,
        args.data,
        args.output,
        device=args.device,
        image_size=args.imgsz,
        batch=args.batch,
        split=args.split,
        precision=args.precision,
        project=args.project,
        name=args.name,
        plots=not args.no_plots,
        save_json=args.save_json,
    )
    validation = payload["validation"]
    print(
        f"mAP50-95={validation['map50_95']} mAP50={validation['map50']} "
        f"precision={validation['mean_precision']} recall={validation['mean_recall']}"
    )
    print(f"Validation JSON: {Path(args.output).resolve()}")


def infer(args: argparse.Namespace) -> None:
    import cv2

    size = parse_size(args.imgsz)
    images = load_images(args.source, 1, size)
    batch, metas = preprocess(images, size)
    backend = create_backend(args.backend, args.model, args.device, half=args.half)
    detections = postprocess(backend.infer(batch), metas, args.confidence, args.iou)[0]
    image = images[0].copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{detection.class_id}:{detection.score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)
    print(f"Saved {len(detections)} detections to {destination.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO backend export, inference and fair benchmarking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Print CUDA and inference-library versions")
    env_parser.add_argument("--gpu-index", type=int, default=0)

    download_parser = subparsers.add_parser("download", help="Download an official Ultralytics checkpoint")
    download_parser.add_argument("--model", default="yolo11n.pt", help="Official model filename")
    download_parser.add_argument("--output-dir", default="artifacts")
    download_parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint")

    export_parser = subparsers.add_parser("export", help="Export an Ultralytics YOLO checkpoint")
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--formats", nargs="+", choices=("onnx", "engine"), default=("onnx", "engine"))
    export_parser.add_argument("--output-dir", default="artifacts")
    export_parser.add_argument("--imgsz", type=int, default=640)
    export_parser.add_argument("--batch", type=int, default=1)
    export_parser.add_argument("--device", default="0")
    export_parser.add_argument("--dynamic", action="store_true")
    export_parser.add_argument("--half", action="store_true")
    export_parser.add_argument("--int8", action="store_true")
    export_parser.add_argument("--data", help="Dataset YAML required for representative INT8 calibration")
    export_parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GiB")
    export_parser.add_argument("--output-name", help="Rename a single exported artifact")

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare one or more inference backends")
    benchmark_parser.add_argument("--pt")
    benchmark_parser.add_argument("--onnx")
    benchmark_parser.add_argument("--engine")
    benchmark_parser.add_argument("--source", default="synthetic", help="Image, image directory, or synthetic")
    benchmark_parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    benchmark_parser.add_argument("--batch", type=int, default=1)
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--half", action="store_true", help="Use FP16 for the PyTorch baseline")
    benchmark_parser.add_argument("--warmup", type=int, default=20)
    benchmark_parser.add_argument("--runs", type=int, default=100)
    benchmark_parser.add_argument("--confidence", type=float, default=0.25)
    benchmark_parser.add_argument("--iou", type=float, default=0.45)
    benchmark_parser.add_argument("--output", default="results/benchmark.json")
    benchmark_parser.add_argument("--raw-output", help="Optional per-iteration latency CSV")
    benchmark_parser.add_argument("--run-tag", help="Stable label used in metadata and GPU CSV names")
    benchmark_parser.add_argument("--precision", help="Explicit precision label for this run")
    benchmark_parser.add_argument("--gpu-index", type=int, default=0)
    benchmark_parser.add_argument("--gpu-metrics-dir", help="Directory for 200 ms nvidia-smi CSV files")
    benchmark_parser.add_argument("--gpu-monitor-interval-ms", type=int, default=200)
    benchmark_parser.add_argument(
        "--expected-gpu",
        help="Fail early unless the detected GPU name contains this value, e.g. T4",
    )

    validate_parser = subparsers.add_parser("validate", help="Save structured detection accuracy metrics")
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument("--data", required=True, help="Dataset YAML or Ultralytics dataset name")
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--device", default="0")
    validate_parser.add_argument("--imgsz", type=int, default=640)
    validate_parser.add_argument("--batch", type=int, default=1)
    validate_parser.add_argument("--split", default="val")
    validate_parser.add_argument("--precision", default="engine-defined")
    validate_parser.add_argument("--project", default="runs/validation")
    validate_parser.add_argument("--name", default="validation")
    validate_parser.add_argument("--no-plots", action="store_true")
    validate_parser.add_argument("--save-json", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run one image and save visualized detections")
    infer_parser.add_argument("--backend", required=True, choices=("pytorch", "onnxruntime", "tensorrt"))
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--source", required=True)
    infer_parser.add_argument("--output", default="results/prediction.jpg")
    infer_parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    infer_parser.add_argument("--device", default="cuda:0")
    infer_parser.add_argument("--half", action="store_true")
    infer_parser.add_argument("--confidence", type=float, default=0.25)
    infer_parser.add_argument("--iou", type=float, default=0.45)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "env":
        environment(args)
    else:
        {
            "download": download,
            "export": export,
            "benchmark": benchmark,
            "validate": validate,
            "infer": infer,
        }[args.command](args)


if __name__ == "__main__":
    main()
