from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ImageMeta:
    original_height: int
    original_width: int
    ratio: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, ImageMeta]:
    """Resize with unchanged aspect ratio and symmetric padding."""
    target_h, target_w = size
    src_h, src_w = image.shape[:2]
    ratio = min(target_h / src_h, target_w / src_w)
    resized_w, resized_h = round(src_w * ratio), round(src_h * ratio)
    pad_w, pad_h = target_w - resized_w, target_h - resized_h
    pad_x, pad_y = pad_w / 2, pad_h / 2

    if (resized_w, resized_h) != (src_w, src_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image, ImageMeta(src_h, src_w, ratio, pad_x, pad_y)


def preprocess(images: Iterable[np.ndarray], size: tuple[int, int]) -> tuple[np.ndarray, list[ImageMeta]]:
    tensors: list[np.ndarray] = []
    metas: list[ImageMeta] = []
    for image in images:
        padded, meta = letterbox(image, size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensors.append(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        metas.append(meta)
    if not tensors:
        raise ValueError("At least one image is required")
    return np.stack(tensors), metas


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    result = boxes.clone()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Pure PyTorch NMS to avoid a binary dependency on torchvision C++ ops."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        inter_x1 = torch.maximum(x1[current], x1[remaining])
        inter_y1 = torch.maximum(y1[current], y1[remaining])
        inter_x2 = torch.minimum(x2[current], x2[remaining])
        inter_y2 = torch.minimum(y2[current], y2[remaining])
        intersection = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = areas[current] + areas[remaining] - intersection
        overlap = intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)
        order = remaining[overlap <= iou_threshold]
    return torch.stack(kept)


def _restore_box(box: torch.Tensor, meta: ImageMeta) -> tuple[float, float, float, float]:
    box = box.clone()
    box[[0, 2]] = (box[[0, 2]] - meta.pad_x) / meta.ratio
    box[[1, 3]] = (box[[1, 3]] - meta.pad_y) / meta.ratio
    box[[0, 2]].clamp_(0, meta.original_width)
    box[[1, 3]].clamp_(0, meta.original_height)
    return tuple(float(value) for value in box)


def postprocess(
    output: np.ndarray,
    metas: list[ImageMeta],
    confidence: float = 0.25,
    iou: float = 0.45,
    max_detections: int = 300,
) -> list[list[Detection]]:
    """Decode common Ultralytics detection outputs and apply class-aware NMS.

    Raw outputs are expected as [B, 4 + classes, anchors] or
    [B, anchors, 4 + classes]. End-to-end [B, N, 6] output is also accepted.
    """
    prediction = torch.from_numpy(np.asarray(output, dtype=np.float32))
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 3:
        raise ValueError(f"Expected a 3-D detection tensor, got {tuple(prediction.shape)}")
    if prediction.shape[0] != len(metas):
        raise ValueError("Output batch size does not match preprocessing metadata")

    decoded: list[list[Detection]] = []
    for sample, meta in zip(prediction, metas):
        # End-to-end exports commonly emit xyxy, confidence, class_id.
        if sample.shape[-1] == 6:
            rows = sample[sample[:, 4] >= confidence]
            rows = rows[:max_detections]
            decoded.append(
                [Detection(_restore_box(row[:4], meta), float(row[4]), int(row[5])) for row in rows]
            )
            continue

        # The feature axis contains at least xywh + one class. It is normally
        # much smaller than the anchor axis; the second clause also handles
        # deliberately tiny tensors used in tests.
        if sample.shape[0] >= 5 and (sample.shape[1] < 5 or sample.shape[0] < sample.shape[1]):
            sample = sample.transpose(0, 1)
        if sample.shape[1] <= 4:
            raise ValueError(f"Cannot decode output shape {tuple(sample.shape)}")
        boxes = _xywh_to_xyxy(sample[:, :4])
        scores, class_ids = sample[:, 4:].max(dim=1)
        keep_mask = scores >= confidence
        boxes, scores, class_ids = boxes[keep_mask], scores[keep_mask], class_ids[keep_mask]
        if not boxes.numel():
            decoded.append([])
            continue

        # Offset boxes by class so boxes from different classes do not suppress each other.
        offsets = class_ids.to(boxes) * 8192
        keep = nms(boxes + offsets[:, None], scores, iou)[:max_detections]
        decoded.append(
            [
                Detection(_restore_box(boxes[index], meta), float(scores[index]), int(class_ids[index]))
                for index in keep
            ]
        )
    return decoded


def load_images(source: str | None, batch: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Load one image repeatedly, a directory batch, or deterministic synthetic input."""
    if source is None or source == "synthetic":
        rng = np.random.default_rng(0)
        return [rng.integers(0, 256, (*size, 3), dtype=np.uint8) for _ in range(batch)]

    path = Path(source)
    candidates = [path] if path.is_file() else sorted(
        item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not candidates:
        raise FileNotFoundError(f"No image files found at {path}")
    images: list[np.ndarray] = []
    for index in range(batch):
        image = cv2.imread(str(candidates[index % len(candidates)]))
        if image is None:
            raise ValueError(f"OpenCV failed to read {candidates[index % len(candidates)]}")
        images.append(image)
    return images
