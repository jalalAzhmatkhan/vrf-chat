"""Settings for `paddleocr-vl-service`, see
`Documentation/system-design/02-ingestion-pipeline.md` §4.0 env block.

Deliberately a small, standalone settings module (not shared with
`vrf-chat/backend/app/core/config.py`) — this is an intentionally separate
project/venv/deployable, per the Opsi 3 architecture decision (§4.0): no
shared code with `backend/` beyond the HTTP JSON contract itself
(`vrf-chat/backend/app/ingestion/paddleocr_vl_cascade.py`
`RemoteAPIPaddleOCRVLClient`), so its dependency resolution (torch/
paddlepaddle) never mixes with `backend/`'s (torch/docling).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    PADDLE_OCR_VL_SERVICE_DEVICE: Literal["cuda", "cpu"] = "cuda"
    PADDLE_OCR_VL_SERVICE_DTYPE: Literal["fp16", "bf16", "fp32"] = "fp16"
    # WAJIB 1 in dev (RTX 3060 6GB) — see live VRAM finding in README.md
    # (~5.98GB peak observed for a SINGLE image; batch>1 not evaluated and
    # not safe to assume fits).
    PADDLE_OCR_VL_SERVICE_BATCH_SIZE: int = 1
    PADDLE_OCR_VL_SERVICE_PORT: int = 8100
    # Auto-unload the model from VRAM after this many seconds idle — VRAM
    # mitigation layer 2 (02-ingestion-pipeline.md §4.0), reduces the window
    # where this service and backend-worker-gpu's Docling could collide on
    # the same physical GPU.
    PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
