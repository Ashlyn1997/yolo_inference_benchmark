from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np
import torch


def _first_detection_tensor(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, dict):
        value = next(iter(value.values()))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(value)!r}")
    return value


class Backend(ABC):
    name: str

    @abstractmethod
    def infer(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def memory_mib(self) -> float | None:
        return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else None


class PyTorchBackend(Backend):
    name = "pytorch"

    def __init__(self, model_path: str, device: str = "cuda:0", half: bool = False):
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self.model = YOLO(model_path).model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()

    @torch.inference_mode()
    def infer(self, batch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(batch).to(self.device, non_blocking=True)
        tensor = tensor.half() if self.half else tensor.float()
        output = _first_detection_tensor(self.model(tensor))
        return output.float().cpu().numpy()


class OnnxRuntimeBackend(Backend):
    name = "onnxruntime-cuda"

    def __init__(self, model_path: str, device_id: int = 0):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu and check CUDA libraries. "
                f"Available providers: {available}"
            )
        # ORT >= 1.21 can discover CUDA/cuDNN wheels under site-packages.
        # This is important in Conda/pip environments without a system CUDA
        # toolkit on LD_LIBRARY_PATH.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        self.session = ort.InferenceSession(
            model_path,
            providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
        )
        active = self.session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "ONNX Runtime failed to activate CUDAExecutionProvider and would fall back to CPU. "
                f"Active providers: {active}. Check CUDA/cuDNN libraries or call `onnxruntime.print_debug_info()`."
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: batch})[0]


class TensorRTBackend(Backend):
    name = "tensorrt"

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires a CUDA device")
        self.trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as file:
            engine_bytes = file.read()
        engine_bytes, self.metadata = self._unwrap_ultralytics_engine(engine_bytes)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.v3 = hasattr(self.engine, "num_io_tensors")
        self._inspect_io()

    @staticmethod
    def _unwrap_ultralytics_engine(data: bytes) -> tuple[bytes, dict]:
        """Remove the metadata prefix used by Ultralytics `.engine` exports.

        Ultralytics stores a signed little-endian 4-byte JSON length, followed
        by the JSON document and then the native TensorRT serialized engine.
        Plain TensorRT plan files are returned unchanged.
        """
        if len(data) < 5:
            return data, {}
        metadata_length = int.from_bytes(data[:4], byteorder="little", signed=True)
        if metadata_length <= 0 or metadata_length > len(data) - 4:
            return data, {}
        metadata_bytes = data[4 : 4 + metadata_length]
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return data, {}
        if not isinstance(metadata, dict):
            return data, {}
        return data[4 + metadata_length :], metadata

    def _inspect_io(self) -> None:
        trt = self.trt
        self.inputs: list[str | int] = []
        self.outputs: list[str | int] = []
        if self.v3:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
        else:
            for index in range(self.engine.num_bindings):
                (self.inputs if self.engine.binding_is_input(index) else self.outputs).append(index)
        if len(self.inputs) != 1 or not self.outputs:
            raise ValueError("This example expects one input and at least one TensorRT output")

    def _dtype(self, key: str | int) -> torch.dtype:
        dtype = self.engine.get_tensor_dtype(key) if self.v3 else self.engine.get_binding_dtype(key)
        mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.bool_): torch.bool,
        }
        numpy_dtype = np.dtype(self.trt.nptype(dtype))
        if numpy_dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")
        return mapping[numpy_dtype]

    def infer(self, batch: np.ndarray) -> np.ndarray:
        input_key = self.inputs[0]
        input_tensor = torch.from_numpy(batch).to(self.device, non_blocking=True)
        input_tensor = input_tensor.to(self._dtype(input_key)).contiguous()
        shape = tuple(input_tensor.shape)

        if self.v3:
            self.context.set_input_shape(input_key, shape)
            self.context.set_tensor_address(input_key, input_tensor.data_ptr())
            output_tensors = []
            for key in self.outputs:
                output = torch.empty(
                    tuple(self.context.get_tensor_shape(key)), dtype=self._dtype(key), device=self.device
                )
                self.context.set_tensor_address(key, output.data_ptr())
                output_tensors.append(output)
            ok = self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        else:
            input_index = int(input_key)
            self.context.set_binding_shape(input_index, shape)
            bindings = [0] * self.engine.num_bindings
            bindings[input_index] = input_tensor.data_ptr()
            output_tensors = []
            for key in self.outputs:
                index = int(key)
                output = torch.empty(
                    tuple(self.context.get_binding_shape(index)), dtype=self._dtype(index), device=self.device
                )
                bindings[index] = output.data_ptr()
                output_tensors.append(output)
            ok = self.context.execute_async_v2(bindings, torch.cuda.current_stream(self.device).cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        torch.cuda.synchronize(self.device)
        return output_tensors[0].float().cpu().numpy()


def create_backend(kind: str, model_path: str, device: str, half: bool = False) -> Backend:
    if not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    if kind == "pytorch":
        return PyTorchBackend(model_path, device=device, half=half)
    if kind == "onnxruntime":
        device_id = int(device.split(":")[-1]) if ":" in device else 0
        return OnnxRuntimeBackend(model_path, device_id=device_id)
    if kind == "tensorrt":
        return TensorRTBackend(model_path, device=device)
    raise ValueError(f"Unknown backend: {kind}")
