# Multi-arch GPU base with PyTorch preinstalled.
# Using NVIDIA NGC PyTorch container for consistent CUDA support across platforms (linux/amd64 and linux/arm64).
# See: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        ninja-build \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT: For NVIDIA Arm platforms (e.g., GB10 / DX Spark / Grace Hopper),
# the CUDA-enabled PyTorch build is shipped with the NGC base image.
# We install torchaudio (no-deps) and rely on soundfile-backed I/O shims
# (provided by app.utils.torchaudio_compat) to maintain compatibility with
# NGC's customized PyTorch build.
RUN python -m pip install --no-cache-dir packaging ninja && \
    python -m pip install --no-cache-dir --no-deps torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /usr/local/lib/python3.12/dist-packages/torchaudio/lib

COPY requirements.txt /app/requirements.txt

# Install application requirements
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Sanity check: ensure we are still on a CUDA-enabled torch build.
RUN python - <<"PY"
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("ERROR: torch.version.cuda is None (CPU-only torch installed); refusing to build")
PY

# Sanity check: transformers, pyannote, and soundfile import cleanly
RUN python - <<"PY"
import transformers
print("transformers", transformers.__version__)
import soundfile
print("soundfile", soundfile.__version__)
PY

COPY . /app

# Hugging Face & Torch cache locations (mount volumes here in Docker/K8s)
ENV HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_CACHE_DIR=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    TORCH_HOME=/data/torch

RUN mkdir -p /data/hf /data/torch /app/storage/incoming /app/storage/jobs /app/storage/exports && \
    chmod -R 0777 /data /app/storage

EXPOSE 8000

# Single worker to avoid duplicating the Whisper model in GPU memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
