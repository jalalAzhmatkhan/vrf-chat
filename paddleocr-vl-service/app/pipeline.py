"""PaddleOCR-VL pipeline wrapper with idle-unload, see
`Documentation/system-design/02-ingestion-pipeline.md` §4.0 mitigation
layer 2 ("Idle-unload di paddleocr-vl-service... WAJIB melepas model dari
VRAM secara otomatis setelah periode idle").

Verified live (I1.10) in this exact venv/dependency combination (no
`torch` — see README.md): pipeline construction + real single-image
inference both succeed, peak VRAM ~5.98GB/6.14GB on the RTX 3060 6GB dev
GPU (far tighter than the design doc's original ~2.5-4GB estimate).
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings

logger = logging.getLogger("paddleocr_vl_service")


@dataclass(slots=True)
class VisualDescriptionResponse:
    description: str | None
    figure_type: str | None = None
    components: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TableReparseResponse:
    markdown: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class OCRPageResponse:
    text: str


def _build_pipeline(settings: Settings) -> Any:  # pragma: no cover
    # Real model construction — see README.md for the live-verified
    # transcript; not meaningfully unit-testable (real GPU/model download).
    from paddleocr import PaddleOCRVL

    device = "gpu:0" if settings.PADDLE_OCR_VL_SERVICE_DEVICE == "cuda" else "cpu"
    return PaddleOCRVL(
        vl_rec_backend="native",
        device=device,
        precision=settings.PADDLE_OCR_VL_SERVICE_DTYPE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=True,
        use_seal_recognition=False,
    )


def _predict_kwargs(settings: Settings) -> dict[str, Any]:
    """`max_new_tokens` bounds the VL model's generation length — this is
    the single change with the best joint payoff for the two I1.10 E2E
    findings (README.md "VRAM" + "why some requests hang"): it directly
    caps peak KV-cache memory during generation (a real, if partial, VRAM
    mitigation — see README.md "VRAM reduction investigation" for what was
    and wasn't tried) AND bounds worst-case per-request latency (a request
    can no longer generate an unbounded-length description that runs past
    any timeout no matter how patient the caller is)."""
    return {"max_new_tokens": settings.PADDLE_OCR_VL_SERVICE_MAX_NEW_TOKENS}


def _decode_image(image_base64: str) -> Any:  # pragma: no cover - thin cv2 decode, real-model-only
    import cv2
    import numpy as np

    raw = base64.b64decode(image_base64)
    array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _first_prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Same best-effort mapping as `backend/app/ingestion/paddleocr_vl_cascade.py`
    `_first_prediction_to_dict` — kept independent (no shared code between
    the two projects, see config.py docstring) but algorithmically
    identical, and now confirmed correct against real output (see
    README.md): real `PaddleOCRVL.predict()` output exposes
    `result.json == {"markdown": {"markdown_texts": "...", ...}, ...}`."""
    if prediction is None:
        return {}
    as_json = getattr(prediction, "json", None)
    if isinstance(as_json, dict):
        return as_json
    markdown = getattr(prediction, "markdown", None)
    if markdown is not None:
        return {"markdown": markdown}
    if isinstance(prediction, dict):
        return prediction
    return {}


def _extract_markdown_text(result_dict: dict[str, Any]) -> str | None:
    markdown = result_dict.get("markdown")
    if isinstance(markdown, str):
        return markdown
    if isinstance(markdown, dict):
        text = markdown.get("markdown_texts") or markdown.get("text")
        if isinstance(text, str):
            return text
    text = result_dict.get("text")
    return text if isinstance(text, str) else None


class PaddleOCRVLPipeline:
    """Owns the PaddleOCR-VL pipeline, with idle-unload (background thread)
    per the design doc's mitigation layer 2.

    **[I1.10 fix]** `self._lock` (an `RLock`, not a plain `Lock` — see
    `_predict` docstring for why reentrancy is required) now covers the
    ENTIRE `_predict` call (model load + decode + inference), not just
    `_ensure_loaded()` as an earlier version did. This is what makes
    `PADDLE_OCR_VL_SERVICE_BATCH_SIZE=1`/"single-flight GPU usage" an
    actually-enforced guarantee rather than an assumption that happened to
    hold only as long as the FastAPI route handlers were `async def`
    (blocking the whole event loop serialized everything "by accident").
    Now that the route handlers are synchronous `def` (dispatched to
    Starlette's threadpool, see `app/main.py`), FastAPI CAN serve multiple
    requests concurrently on different threads — this lock is what
    prevents two of them from calling `pipeline.predict()` on the GPU at
    the same time (a real VRAM/correctness risk on a 6GB card, not just a
    style preference), while still letting cheap endpoints like `/health`
    (which never touches this lock) respond immediately even while a
    request is queued waiting for its turn here."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any | None = None
        self._lock = threading.RLock()
        self._last_used_at: float = time.monotonic()
        self._stop_idle_watcher = threading.Event()
        self._idle_watcher = threading.Thread(target=self._watch_idle, daemon=True)
        self._idle_watcher.start()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _watch_idle(self) -> None:
        while not self._stop_idle_watcher.wait(timeout=10):
            self._maybe_unload_if_idle()

    def _maybe_unload_if_idle(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            idle_seconds = time.monotonic() - self._last_used_at
            if idle_seconds >= self._settings.PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS:
                logger.info(
                    "paddleocr_vl_service.idle_unload",
                    extra={"idle_seconds": idle_seconds},
                )
                self._unload_locked()

    def _unload_locked(self) -> None:
        self._pipeline = None
        _release_gpu_memory()

    def shutdown(self) -> None:
        self._stop_idle_watcher.set()

    def _ensure_loaded(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                logger.info("paddleocr_vl_service.loading_pipeline")
                self._pipeline = _build_pipeline(self._settings)
            self._last_used_at = time.monotonic()
            return self._pipeline

    def _predict(self, image_base64: str) -> dict[str, Any]:  # pragma: no cover
        """`self._lock` wraps the WHOLE call (not just `_ensure_loaded()`)
        so at most one GPU inference runs at a time regardless of how many
        concurrent HTTP requests arrive — see class docstring."""
        with self._lock:
            pipeline = self._ensure_loaded()
            image = _decode_image(image_base64)
            results = pipeline.predict(image, **_predict_kwargs(self._settings))
            first = results[0] if results else None
            return _first_prediction_to_dict(first)

    def describe_figure(self, image_base64: str) -> VisualDescriptionResponse:
        result = self._predict(image_base64)
        return VisualDescriptionResponse(description=_extract_markdown_text(result))

    def reparse_table(self, image_base64: str) -> TableReparseResponse:
        result = self._predict(image_base64)
        return TableReparseResponse(markdown=_extract_markdown_text(result))

    def ocr_page(self, image_base64: str) -> OCRPageResponse:
        result = self._predict(image_base64)
        return OCRPageResponse(text=_extract_markdown_text(result) or "")


def _release_gpu_memory() -> None:  # pragma: no cover - no GPU in unit test env
    import gc

    gc.collect()
    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
