# Build with an NGC TensorRT image matched to the target NVIDIA driver.
# Example: docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:<compatible-tag> -f docker/Dockerfile -t yolo-nvidia-benchmark .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace/yolo-inference-benchmark
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY docker/requirements-nvidia.txt /tmp/requirements-nvidia.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements-nvidia.txt && \
    python3 -c "import torch" || \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1

COPY pyproject.toml README.md ./
COPY yolo_benchmark ./yolo_benchmark
RUN python3 -m pip install --no-deps -e .

COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["python3", "-m", "yolo_benchmark.cli"]
CMD ["env"]
