from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .environment import collect_environment, file_metadata


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _class_names(metrics: Any, model: Any) -> dict[int, str]:
    names = getattr(metrics, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def validate_model(
    model_path: str,
    data: str,
    output: str,
    *,
    device: str = "0",
    image_size: int = 640,
    batch: int = 1,
    split: str = "val",
    precision: str = "engine-defined",
    project: str = "runs/validation",
    name: str = "validation",
    plots: bool = True,
    save_json: bool = False,
) -> dict[str, Any]:
    """Run Ultralytics detection validation and persist machine-readable accuracy evidence."""
    from ultralytics import YOLO

    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_file}")

    model = YOLO(str(model_file))
    metrics = model.val(
        data=data,
        imgsz=image_size,
        batch=batch,
        device=device,
        split=split,
        project=project,
        name=name,
        plots=plots,
        save_json=save_json,
    )
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validation did not return detection box metrics")

    names = _class_names(metrics, model)
    raw_maps = getattr(box, "maps", None)
    maps = list(raw_maps) if raw_maps is not None else []
    per_class_map50_95 = {
        names.get(index, str(index)): _number(value) for index, value in enumerate(maps)
    }
    dataset_path = Path(data)
    payload: dict[str, Any] = {
        "environment": collect_environment(),
        "model": {"precision": precision, **file_metadata(model_file)},
        "dataset": (
            file_metadata(dataset_path)
            if dataset_path.is_file()
            else {"reference": data, "sha256": None}
        ),
        "validation": {
            "split": split,
            "image_size": image_size,
            "batch_size": batch,
            "map50_95": _number(getattr(box, "map", None)),
            "map50": _number(getattr(box, "map50", None)),
            "map75": _number(getattr(box, "map75", None)),
            "mean_precision": _number(getattr(box, "mp", None)),
            "mean_recall": _number(getattr(box, "mr", None)),
            "per_class_map50_95": per_class_map50_95,
            "speed_ms_per_image": {
                str(key): _number(value) for key, value in dict(getattr(metrics, "speed", {}) or {}).items()
            },
            "save_dir": str(getattr(metrics, "save_dir", "")),
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
