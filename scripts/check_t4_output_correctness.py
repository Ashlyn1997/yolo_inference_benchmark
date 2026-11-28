#!/usr/bin/env python3
"""Check sequential real-image outputs and borrowed/owned lifetimes on a T4.

Uses the historical pinned diagnostic without modifying it. This small acceptance
check is neither a performance benchmark nor a COCO mAP evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .profile_t4_output_buffers import OutputBufferInfer, write_json
else:
    from profile_t4_output_buffers import OutputBufferInfer, write_json


class OwnedOutput:
    """Copy a synchronous borrowed output before returning it to its caller.

    Does not make the wrapped runner thread-safe. Performance, including this
    copy, would need a separate measurement before production integration.
    """

    def __init__(self, infer):
        self.infer = infer

    def __call__(self, batch):
        return self.infer(batch).copy()


def image_groups(paths, batch_size):
    """Use each real path once; repeat only the final image to fill a static batch."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(paths), batch_size):
        real = paths[start:start + batch_size]
        yield real + [real[-1]] * (batch_size - len(real)), len(real)


def select_images(source, count):
    if not source.is_dir():
        raise ValueError(f"Source must be an image directory: {source}")
    paths = sorted(path for path in source.iterdir() if path.is_file()
                   and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if count <= 0 or len(paths) < count:
        raise ValueError(f"Requested {count} distinct image paths, found {len(paths)} at {source}")
    return paths[:count]


def encode_detections(detections):
    return [[{"box": [float(value) for value in detection.box],
              "score": float(detection.score), "class_id": int(detection.class_id)}
             for detection in image] for image in detections]


def compare_detections(actual, expected, np, *, rtol, atol):
    if len(actual) != len(expected):
        raise AssertionError("Decoded output image count differs")
    for image, reference in zip(actual, expected):
        if len(image) != len(reference):
            raise AssertionError("Detection count differs")
        for detection, baseline in zip(image, reference):
            if detection["class_id"] != baseline["class_id"]:
                raise AssertionError("Detection class/order differs")
            np.testing.assert_allclose(detection["box"], baseline["box"], rtol=rtol, atol=atol, equal_nan=False)
            np.testing.assert_allclose(detection["score"], baseline["score"], rtol=rtol, atol=atol, equal_nan=False)


def check_sequence(reference_infer, borrowed_infer, batches, decode, *, rtol=1e-4,
                   atol=1e-4, on_batch=None):
    """Compare outputs immediately; retain first views to test later overwrites.

    Each batch is (input_array, full_metas, real_image_metadata). decode is the
    existing CPU postprocess, consumed synchronously before the next inference.
    """
    import numpy as np

    owned_infer = OwnedOutput(borrowed_infer)
    first_borrowed = first_owned = first_reference = None
    borrowed_snapshot = owned_snapshot = reference_snapshot = None
    seen_ids, seen_hashes = set(), set()
    trace, images = [], []
    observed_change = False
    for iteration, (batch, metas, real_images) in enumerate(batches):
        try:
            count = len(real_images)
            if not 0 < count <= len(batch) or len(metas) != len(batch):
                raise AssertionError("Invalid real-image count or preprocessing metadata")
            ids = [image["image_id"] for image in real_images]
            if len(set(ids)) != count or seen_ids.intersection(ids):
                raise AssertionError("Duplicate image ID in the real sequence")
            seen_ids.update(ids)
            seen_hashes.update(image["decoded_sha256"] for image in real_images)
            reference = reference_infer(batch)
            if reference.shape[0] != len(batch) or not np.isfinite(reference).all():
                raise AssertionError("Invalid reference output batch or nonfinite values")
            reference_detections = encode_detections(decode(reference[:count], metas[:count]))

            borrowed = borrowed_infer(batch)
            if borrowed.shape != reference.shape or not np.isfinite(borrowed).all():
                raise AssertionError("Invalid borrowed output shape or values")
            np.testing.assert_allclose(borrowed, reference, rtol=rtol, atol=atol, equal_nan=False)
            borrowed_error = float(np.max(np.abs(borrowed - reference)))
            # Consume before owned_infer invokes the shared underlying runner again.
            borrowed_detections = encode_detections(decode(borrowed[:count], metas[:count]))
            compare_detections(borrowed_detections, reference_detections, np, rtol=rtol, atol=atol)

            owned = owned_infer(batch)
            if owned.shape != reference.shape or not np.isfinite(owned).all():
                raise AssertionError("Invalid owned output shape or values")
            np.testing.assert_allclose(owned, reference, rtol=rtol, atol=atol, equal_nan=False)
            if np.shares_memory(owned, borrowed):
                raise AssertionError("Owned output aliases the borrowed buffer")
            owned_error = float(np.max(np.abs(owned - reference)))
            owned_detections = encode_detections(decode(owned[:count], metas[:count]))
            compare_detections(owned_detections, reference_detections, np, rtol=rtol, atol=atol)
            if first_borrowed is None:
                first_borrowed, first_owned, first_reference = borrowed, owned, reference
                borrowed_snapshot = borrowed.copy()
                owned_snapshot, reference_snapshot = owned.copy(), reference.copy()
            else:
                if not np.shares_memory(first_borrowed, borrowed):
                    raise AssertionError("Borrowed output did not reuse the same buffer")
                if np.shares_memory(first_owned, owned) or np.shares_memory(first_reference, reference):
                    raise AssertionError("Independent outputs alias earlier results")
            # The first borrowed view must now contain this call's output. Equality
            # with its original contents is not required to change for every image.
            np.testing.assert_array_equal(first_borrowed, borrowed)
            np.testing.assert_array_equal(first_owned, owned_snapshot)
            np.testing.assert_array_equal(first_reference, reference_snapshot)
            observed_change = observed_change or not np.array_equal(first_borrowed, borrowed_snapshot)
            row = {
                "iteration": iteration, "real_images": count, "padded_images": len(batch) - count,
                "borrowed_max_abs_error": borrowed_error, "owned_max_abs_error": owned_error,
                "borrowed_shared_buffer": True, "owned_and_reference_stable": True,
                "first_borrowed_contents_changed": observed_change,
            }
            image_rows = [{
                **metadata, "iteration": iteration, "batch_slot": slot,
                "reference": reference_detections[slot], "borrowed": borrowed_detections[slot],
                "owned": owned_detections[slot],
            } for slot, metadata in enumerate(real_images)]
            trace.append(row)
            images.extend(image_rows)
            if on_batch:
                on_batch(row, image_rows)
        except Exception as exc:
            raise AssertionError(f"Batch {iteration} correctness check failed: {exc}") from exc
    if len(trace) < 2 or len(seen_hashes) < 2:
        raise AssertionError("Need at least two calls and two different decoded image contents")
    return {
        "passed": True, "batches_checked": len(trace), "images_checked": len(images),
        "distinct_decoded_images": len(seen_hashes),
        "padded_images": sum(row["padded_images"] for row in trace),
        "borrowed_max_abs_error": max(row["borrowed_max_abs_error"] for row in trace),
        "owned_max_abs_error": max(row["owned_max_abs_error"] for row in trace),
        "borrowed_shared_buffer_verified": True,
        "first_owned_and_reference_survived_later_calls": True,
        "borrowed_contents_changed_observed": observed_change,
        "notes": [
            "Borrowed results are consumed synchronously; retain .copy() if needed after the next call.",
            "Identical predictions for different images do not alone invalidate the lifetime check.",
            "Uses the same Engine and preprocessing/postprocessing for all paths; not a full mAP test.",
            "OwnedOutput adds a copy; no latency or throughput claim is made.",
            "Neither borrowed nor owned wrapper supports concurrent use of one execution context.",
        ],
    }, trace, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--images", type=int, default=33)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.images <= args.batch:
        parser.error("batch/imgsz must be positive; images must exceed batch to test later calls")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("rtol and atol must be finite and non-negative")
    if not args.engine.is_file():
        parser.error(f"Engine does not exist: {args.engine}")
    try:
        paths = select_images(args.source, args.images)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir or Path("results/t4") / (
        f"output_correctness_b{args.batch}_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty: {output}")

    import cv2
    import numpy as np
    import torch
    from yolo_benchmark.backends import TensorRTBackend
    from yolo_benchmark.environment import collect_environment, file_metadata
    from yolo_benchmark.pipeline import postprocess, preprocess

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    torch.cuda.set_device(device)
    environment = collect_environment(device.index if device.index is not None else torch.cuda.current_device())
    if "T4" not in environment.get("gpu", ""):
        parser.error("This acceptance check requires a Tesla T4")
    backend = TensorRTBackend(str(args.engine), device=args.device)
    borrowed = OutputBufferInfer(backend, "reuse_pinned")
    project = Path(__file__).resolve().parent.parent
    source_files = [Path(__file__).resolve(), project / "scripts/profile_t4_output_buffers.py",
                    *(project / "yolo_benchmark" / name for name in ("backends.py", "pipeline.py", "environment.py"))]
    hashes = {str(path.relative_to(project)): file_metadata(path)["sha256"] for path in source_files}
    metadata = {
        "purpose": "sequential_correctness_not_performance_or_map", "environment": environment,
        "engine": file_metadata(args.engine), "source_sha256": hashes,
        "source": str(args.source.resolve()), "selected_images": [str(path.resolve()) for path in paths],
        "batch_size": args.batch, "requested_images": args.images, "image_size": [args.imgsz, args.imgsz],
        "rtol": args.rtol, "atol": args.atol, "confidence": 0.25, "iou": 0.45,
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(), "torch_threads": torch.get_num_threads(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "provenance.json", metadata)

    def batches():
        for padded_paths, real_count in image_groups(paths, args.batch):
            images, identities = [], []
            for slot, path in enumerate(padded_paths):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"OpenCV failed to read {path}")
                images.append(image)
                if slot < real_count:
                    identities.append({"image_id": path.name, "path": str(path.resolve()),
                                       "decoded_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                       "original_shape": list(image.shape)})
            batch, metas = preprocess(images, (args.imgsz, args.imgsz))
            yield batch, metas, identities

    completed = []
    with (output / "batch_checks.csv").open("w", newline="", encoding="utf-8") as csv_file, \
            (output / "image_results.jsonl").open("w", encoding="utf-8") as image_file:
        writer = None

        def save_batch(row, image_rows):
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            for image_row in image_rows:
                image_file.write(json.dumps(image_row, allow_nan=False) + "\n")
            image_file.flush()
            completed.append(row)
            print(f"[{len(completed)}/{math.ceil(args.images / args.batch)}] "
                  f"images={row['real_images']}, padded={row['padded_images']}, "
                  f"borrowed/owned max error={row['borrowed_max_abs_error']:.6g}/{row['owned_max_abs_error']:.6g}", flush=True)

        try:
            summary, _, image_rows = check_sequence(
                backend.infer, borrowed, batches(),
                lambda prediction, metas: postprocess(prediction, metas, confidence=0.25, iou=0.45),
                rtol=args.rtol, atol=args.atol, on_batch=save_batch,
            )
            if len(image_rows) != args.images or [row["image_id"] for row in image_rows] != [path.name for path in paths]:
                raise AssertionError("Real image coverage/order differs from the selected input list")
            if file_metadata(args.engine)["sha256"] != metadata["engine"]["sha256"]:
                raise AssertionError("Engine changed during correctness check")
            if any(file_metadata(path)["sha256"] != hashes[str(path.relative_to(project))] for path in source_files):
                raise AssertionError("Source code changed during correctness check")
        except Exception as exc:
            write_json(output / "summary.json", {
                "passed": False, "error": str(exc), "batches_completed": len(completed),
                "images_completed": sum(row["real_images"] for row in completed),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise SystemExit(f"CORRECTNESS FAILED: {output}; {exc}") from exc
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print(f"CORRECTNESS PASSED: {output} ({summary['images_checked']} images)", flush=True)


if __name__ == "__main__":
    main()
