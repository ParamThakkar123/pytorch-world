# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCHWM_EXTRAS=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCHWM_HOME=/data/torchwm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py README.md ./
COPY torchwm ./torchwm
COPY tools ./tools

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio && \
    if [ -n "${TORCHWM_EXTRAS}" ]; then \
        python -m pip install --editable ".[${TORCHWM_EXTRAS}]"; \
    else \
        python -m pip install --editable .; \
    fi && \
    torchwm version && \
    mkdir -p "${TORCHWM_HOME}"

VOLUME ["/data/torchwm"]

ENTRYPOINT ["torchwm"]
CMD ["--help"]
