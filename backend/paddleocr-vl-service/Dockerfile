# syntax=docker/dockerfile:1
#
# paddleocr-vl-service — standalone PaddleOCR-VL inference service, see
# Documentation/system-design/02-ingestion-pipeline.md §4.0 (Opsi 3:
# PaddleOCR-VL always runs as a separate container/service, called by
# backend-worker-gpu via PADDLE_OCR_VL_BACKEND=remote_api, even in dev).
#
# Deliberately its OWN image/venv/pyproject.toml/uv.lock, entirely separate
# from vrf-chat/backend/ — this is the whole point of the Opsi 3 decision:
# torch (Docling, backend/) and paddlepaddle-gpu (this service) must never
# resolve into the same Python environment (see README.md "Why a separate
# service" for the real conflict this avoids).
#
# No special CUDA base image needed (verified live during I1.10, same
# finding as backend/Dockerfile's worker-gpu stage for Docling/torch):
# paddlepaddle-gpu's own pip wheel bundles the CUDA runtime as `nvidia-*`
# dependencies. What IS required (host-level, not image-level) is the
# NVIDIA Container Toolkit + driver on the Docker host, already verified
# working from WSL (see backend/docs/i1.6-vram-benchmark-report.md §1.2).

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

RUN groupadd --system app && useradd --system --gid app --home /app app \
    && chown -R app:app /app
USER app

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health').read()" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
