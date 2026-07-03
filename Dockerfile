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

# Install pinned runtime deps first so this layer caches across source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application modules (main.py, orchestrator.py, router.py, models.py).
COPY . .

# Uvicorn serves the ASGI app here.
EXPOSE 8000

# Bind to all interfaces so the published port is reachable on the host/tailnet.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
