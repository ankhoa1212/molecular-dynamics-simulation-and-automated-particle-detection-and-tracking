# CUDA 13.0 wheels are used (see pyproject.toml: pytorch-cu130 index).
# Requires NVIDIA driver >= 570 on the cluster node.
# If your cluster has an older driver, change cu130 -> cu124 in pyproject.toml
# and regenerate the lock file with: uv lock
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl ca-certificates \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv — handles Python 3.11 + all dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Install deps first (separate layer — rebuilt only when lock file changes)
COPY pyproject.toml uv.lock ./
RUN uv python install 3.11 && uv sync --frozen

# Activate venv for the CMD
ENV PATH="/workspace/.venv/bin:$PATH"

# Copy source — separate layer so code changes don't bust the deps cache
COPY train.py dataset.py mlflow_utils.py evaluate.py ./

# Pretrained backbone weights — must be present for the K8s job (not downloaded at runtime)
COPY rf-detr-large-2026.pth ./

CMD ["python", "train.py", "--config", "/config/config.yaml"]
