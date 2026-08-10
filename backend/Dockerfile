# syntax=docker/dockerfile:1
#
# Multi-stage Dockerfile for the backend — one image, three targets sharing
# the same dependency layer (`base`): `api` (FastAPI), `worker` (default
# Celery queue), `worker-gpu` (GPU-bound ingestion queue). See
# Documentation/system-design/01-architecture-overview.md §6 and
# Documentation/system-design/04-provider-abstractions.md.
#
# All dependency installation goes through `uv` (never bare pip/poetry), per
# CLAUDE.md.
#
# NOTE (worker-gpu, I1.4 update): GPU-heavy dependencies (torch, docling,
# paddleocr-vl) are now added (see pyproject.toml) — the B0.5 TODO above
# ("likely need to switch to an nvidia/cuda-enabled base image") was
# investigated and turned out to be UNNECESSARY: `uv.lock` is a universal
# (multi-platform) lock, and torch's Linux-marker dependencies already
# include `cuda-bindings`/`cuda-toolkit[cublas,cudart,cufft,...]` (see
# `uv.lock`, `name = "torch"` entry, `marker = "sys_platform == 'linux'"`) —
# i.e. `uv sync` inside a Linux container resolves a CUDA-capable torch
# wheel with the CUDA runtime bundled as pip dependencies, no system-level
# CUDA toolkit/base image required. What IS required (host-level, not
# image-level) is the NVIDIA Container Toolkit + driver on the Docker host —
# verified working from WSL: `docker run --rm --gpus all
# nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` correctly reports the RTX
# 3060 (6144MiB) from inside a plain container. `worker-gpu` therefore stays
# on the same `base` (python:3.12-slim) stage as api/worker.
#
# NOT YET verified: an actual `uv sync --frozen` build of THIS Dockerfile
# inside WSL Docker, confirming `torch.cuda.is_available() is True` inside
# the built worker-gpu container (as opposed to the ad-hoc nvidia/cuda
# smoke-test image above) — deferred to the I1.6 WSL GPU benchmark, which
# builds/runs this image for real anyway. On the Windows-native dev venv
# used for I1.1-I1.4 development, `uv add docling` resolved a CPU-only
# torch wheel (`2.13.0+cpu`) — expected/correct for Windows dev (no CUDA
# toolkit resolution attempted there), not a sign of a lock-file problem.

FROM python:3.12-slim AS base

# uv, installed via the official distroless image (no pip bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching — only re-runs when
# pyproject.toml/uv.lock change, not on every source change).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the actual application source and install the project itself.
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Run as non-root — celery/uvicorn have no reason to run as uid 0. Revisit
# only if Fase 1 GPU device passthrough (worker-gpu) turns out to need
# specific host GPU group membership not satisfiable by this generic user.
RUN groupadd --system app && useradd --system --gid app --home /app app \
    && chown -R app:app /app
USER app

# ---- API service --------------------------------------------------------
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Default Celery worker (lightweight/CPU-bound tasks) ----------------
FROM base AS worker

CMD ["uv", "run", "--no-sync", "celery", "-A", "app.workers.celery_app", "worker", \
     "--loglevel=INFO", "-Q", "default"]

# ---- GPU Celery worker (Docling GPU + PaddleOCR-VL cascade ingestion) ---
# concurrency=1 is deliberate — see 02-ingestion-pipeline.md §4 (RTX 3060
# 6GB VRAM constraint: Docling and PaddleOCR-VL must run sequentially with
# explicit VRAM unload between stages, never concurrently).
FROM base AS worker-gpu

CMD ["uv", "run", "--no-sync", "celery", "-A", "app.workers.celery_app", "worker", \
     "--loglevel=INFO", "--concurrency=1", "-Q", "gpu"]
