# syntax=docker/dockerfile:1
#
# Production image for the core-heartbeat FastAPI gateway + LangGraph orchestrator.
# The requirements are pinned to CPython 3.14 wheels (see requirements.txt), so the
# base image tracks that exact minor to reuse those wheels.

FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# docling pulls torch; install the CPU-only build first (this box has no GPU) so the
# image doesn't carry the multi-GB CUDA wheels. requirements.txt then finds torch
# already satisfied and won't pull the default CUDA build.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

# Install pinned runtime deps first so this layer caches across source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# docling downloads its layout/OCR models on first use; point its (HuggingFace)
# cache at a path we mount as a volume in compose so models persist across
# container recreates instead of re-downloading (~1GB) each time.
ENV HF_HOME=/models/hf DOCLING_CACHE_DIR=/models/docling

# Application modules (main.py, orchestrator.py, router.py, models.py).
COPY . .

# Uvicorn serves the ASGI app here.
EXPOSE 8000

# Bind to all interfaces so the published port is reachable on the host/tailnet.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
