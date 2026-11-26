import numpy as np

import torch

from yolo_benchmark.backends import TensorRTBackend
from yolo_benchmark.pipeline import ImageMeta, letterbox, nms, postprocess, preprocess


def test_letterbox_preserves_target_shape():
    image = np.zeros((480, 800, 3), dtype=np.uint8)
    result, meta = letterbox(image, (640, 640))
    assert result.shape == (640, 640, 3)
    assert meta.ratio == 0.8
    assert meta.pad_y == 128


def test_preprocess_returns_normalized_nchw():
    batch, metas = preprocess([np.full((100, 200, 3), 255, dtype=np.uint8)], (640, 640))
    assert batch.shape == (1, 3, 640, 640)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() <= batch.max() <= 1
    assert len(metas) == 1


def test_postprocess_decodes_raw_yolo_output():
    # Shape [batch, 4 + 2 classes, anchors]
    output = np.zeros((1, 6, 2), dtype=np.float32)
    output[0, :, 0] = [320, 320, 100, 200, 0.9, 0.1]
    output[0, :, 1] = [100, 100, 20, 20, 0.1, 0.2]
    result = postprocess(output, [ImageMeta(640, 640, 1.0, 0, 0)], confidence=0.25)
    assert len(result[0]) == 1
    assert result[0][0].class_id == 0
    assert result[0][0].box == (270.0, 220.0, 370.0, 420.0)


def test_pure_torch_nms_suppresses_overlapping_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_ultralytics_engine_metadata_is_unwrapped():
    metadata = b'{"imgsz": [640, 640], "batch": 1}'
    native_engine = b"native-tensorrt-engine"
    wrapped = len(metadata).to_bytes(4, byteorder="little", signed=True) + metadata + native_engine
    result, parsed = TensorRTBackend._unwrap_ultralytics_engine(wrapped)
    assert result == native_engine
    assert parsed["batch"] == 1


def test_native_engine_is_left_unchanged():
    native_engine = b"native-tensorrt-engine"
    result, metadata = TensorRTBackend._unwrap_ultralytics_engine(native_engine)
    assert result == native_engine
    assert metadata == {}
