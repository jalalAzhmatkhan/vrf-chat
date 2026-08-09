"""Stage 4 — PaddleOCR-VL cascade, see
`Documentation/system-design/02-ingestion-pipeline.md` §3-4 (I1.4).

PaddleOCR-VL is **never run for all 2.581 pages** — only for the elements/
pages queued by Stage 3 (`app/ingestion/cascade_trigger.py`, I1.3):
table re-parse (`table_reparse`), figure/icon semantic understanding
(`visual_description`), and full-page OCR fallback (`full_page_ocr`).

WAJIB (RTX 3060 6GB dev, `02-ingestion-pipeline.md` §4): Docling (Stage 2)
and PaddleOCR-VL must never be GPU-resident simultaneously.
`require_docling_unloaded_before_paddle_stage()` below is an enforced guard
for the (now largely hypothetical, see next paragraph) same-process case.

**[REVISI 2026-08-09, I1.6 finding + §4.0 keputusan Opsi 3]** `backend-worker-gpu`
now **only** runs Docling — `PADDLE_OCR_VL_BACKEND=local` is **not used** by
`backend-worker-gpu` (`remote_api` is the only backend it's configured with,
pointing at the separate `paddleocr-vl-service`, see
`vrf-chat/paddleocr-vl-service/`). Root cause: `paddlepaddle-gpu` (needed
by `LocalPaddleOCRVLClient`'s layout submodel) and `torch` (used by
Docling) vendor incompatible `nvidia-nccl-cu12` pip packages into the same
venv, breaking `import torch` entirely — a packaging conflict, not a VRAM
one (see `docs/i1.6-vram-benchmark-report.md` §6). `LocalPaddleOCRVLClient`
is **kept in this module** (still implements the `PaddleOCRVLClient`
Protocol, still fully unit-tested) purely for interface completeness/
documentation and as the reference implementation `paddleocr-vl-service`'s
own FastAPI app is modeled on — `paddleocr` is **no longer a `backend/`
dependency** (removed from `pyproject.toml`; `_build_local_pipeline`'s
`from paddleocr import PaddleOCRVL` import is lazy/pragma-excluded, so this
module still imports/type-checks fine without it installed). Constructing
`LocalPaddleOCRVLClient` for real from within `backend/` would now fail
with `ModuleNotFoundError` — this is intentional, not a latent bug.

**Separately confirmed important finding** (I1.10 live verification,
`paddleocr-vl-service`'s own isolated venv, `torch` genuinely absent —
PaddleOCR-VL's core pipeline runs entirely on PaddlePaddle's own inference
engine, `torch` was never actually a hard requirement despite earlier
assumption): real single-image inference peaked at **~5.98GB VRAM out of
6.14GB** (RTX 3060 6GB) — see `vrf-chat/paddleocr-vl-service/README.md` for
full detail. This is far higher than the design doc §4's "~2.5-4GB"
estimate and leaves only ~150MB headroom, a critical operational
consideration flagged separately.

Two backends, both implemented from the start (per design doc §4
"Portabilitas ke cloud GPU"/DIRECT MESSAGE to Backend Engineer):
- `local` — in-process inference via the `paddleocr` package's `PaddleOCRVL`
  pipeline (`vl_rec_backend="native"`), `PADDLE_OCR_VL_BATCH_SIZE=1` and
  `PADDLE_OCR_VL_DTYPE=fp16` wajib in dev. **Not used by `backend-worker-gpu`
  in this project's actual deployment** (see revision note above) — kept as
  a general-purpose, backend-agnostic class other deployments could use.
- `remote_api` — a generic HTTP JSON contract this project defines itself
  (POST base64 PNG to `{PADDLE_OCR_VL_API_URL}/{describe_figure,
  reparse_table,ocr_page}`, see `RemoteAPIPaddleOCRVLClient`), deliberately
  NOT coupled to PaddleOCR-VL's own `vllm-server`/`sglang-server`/etc.
  serving protocols. **This is what `backend-worker-gpu` actually uses**,
  pointed at `paddleocr-vl-service` (a container-boundary "remote", not
  necessarily a network-distant one).

Coverage/testing trade-off (documented per task instructions): actually
constructing the local `PaddleOCRVL` pipeline and running real inference
downloads/executes a multi-GB vision-language model (ERNIE4.5-0.9B based) —
genuinely GPU-appropriate work, not meaningfully unit-testable from
`backend/` (which, per the revision above, no longer even has `paddleocr`
installed). Those two boundaries (`_build_local_pipeline`,
`LocalPaddleOCRVLClient._run_predict`) are `# pragma: no cover` with this
rationale; everything around them (orchestration, response mapping, config
validation, the remote HTTP client via `httpx.MockTransport`) is fully
unit-tested. Real local-backend execution (construction + actual inference,
including the markdown output schema `_extract_markdown_text` maps) WAS
verified live during I1.10, in `paddleocr-vl-service`'s own isolated venv —
see that service's README for the full transcript/findings.
"""

from __future__ import annotations

import base64
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import pymupdf
import torch

from app.core.config import Settings
from app.core.observability import get_logger
from app.ingestion.cascade_trigger import (
    TASK_FULL_PAGE_OCR,
    TASK_TABLE_REPARSE,
    TASK_VISUAL_DESCRIPTION,
    CascadePlan,
    CascadeTask,
)
from app.ingestion.docling_parser import DoclingParser, ElementDraft

logger = get_logger(__name__)

SUPPORTED_BACKENDS = ("local", "remote_api")


class PaddleOCRVLConfigError(ValueError):
    """Raised (fail-fast) for an invalid `PADDLE_OCR_VL_*` config combination,
    or when the Docling-unload-before-Stage-4 VRAM safety check fails."""


class PaddleOCRVLRemoteError(RuntimeError):
    """Raised when the `remote_api` backend returns a non-2xx response or the
    request otherwise fails — a genuine external-service failure the caller
    (ingestion orchestrator) must surface (e.g. mark `ingestion_jobs` failed),
    not silently swallow."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VisualDescription:
    description: str | None
    figure_type: str | None = None
    components: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TableReparseResult:
    markdown: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRPageResult:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CascadeResult:
    task: CascadeTask
    visual_description: VisualDescription | None = None
    table_reparse: TableReparseResult | None = None
    ocr_page: OCRPageResult | None = None
    extraction_method: str = "paddle_vlm"


class PaddleOCRVLClient(Protocol):
    def describe_figure(self, image_png: bytes) -> VisualDescription: ...  # pragma: no cover
    def reparse_table(self, image_png: bytes) -> TableReparseResult: ...  # pragma: no cover
    def ocr_page(self, image_png: bytes) -> OCRPageResult: ...  # pragma: no cover
    def unload(self) -> None: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Bbox rendering (shared with I1.5 canonical store for the crop it persists)
# ---------------------------------------------------------------------------


def render_bbox_crop(
    pdf_path: str | Path,
    page_number: int,
    bbox: dict[str, Any] | None = None,
    *,
    dpi: int = 200,
) -> bytes:
    """Render a page (or a bbox region within it) to PNG bytes via PyMuPDF.

    `bbox` follows the `ElementDraft.bbox`/Docling convention: `{"l","t",
    "r","b"}` in `BOTTOMLEFT` coordinate origin (`t` measured from the
    bottom of the page). PyMuPDF's `Rect` uses TOP-LEFT origin, hence the
    flip below. `bbox=None` renders the full page (used for
    `full_page_ocr`).
    """
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        clip = None
        if bbox is not None:
            page_height = page.rect.height
            top = page_height - bbox["t"]
            bottom = page_height - bbox["b"]
            clip = pymupdf.Rect(bbox["l"], top, bbox["r"], bottom)
        pixmap = page.get_pixmap(matrix=matrix, clip=clip)
        return pixmap.tobytes("png")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


def _build_local_pipeline(settings: Settings) -> Any:  # pragma: no cover
    # See module docstring "Coverage/testing trade-off" — constructing this
    # pipeline resolves/downloads real PaddleOCR-VL model weights.
    from paddleocr import PaddleOCRVL

    device = "gpu:0" if settings.PADDLE_OCR_VL_DEVICE == "cuda" else "cpu"
    return PaddleOCRVL(
        vl_rec_backend="native",
        device=device,
        precision=settings.PADDLE_OCR_VL_DTYPE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=True,
        use_seal_recognition=False,
    )


def _decode_png_to_array(image_png: bytes) -> Any:  # pragma: no cover
    # Thin cv2 decode step, exercised only together with real local
    # inference — see module docstring.
    import cv2
    import numpy as np

    array = np.frombuffer(image_png, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _first_prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Best-effort, defensive extraction of a PaddleX `Result`-like object
    into a plain dict, kept separate from `_run_predict` so it IS unit
    tested (with fakes) even though real prediction objects aren't
    available in this dev environment (see module docstring). Uses
    `getattr` throughout because the exact PaddleX `Result` schema for this
    pipeline has not been confirmed against real output yet — to be
    corrected during the I1.6 live run if it differs.
    """
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


class LocalPaddleOCRVLClient:
    """In-process PaddleOCR-VL inference. Owns the pipeline (and therefore
    whatever GPU/CPU memory it allocates) — call `unload()` when the Stage 4
    batch for a document/batch is finished."""

    def __init__(self, settings: Settings) -> None:
        if settings.PADDLE_OCR_VL_DEVICE not in ("cuda", "cpu"):
            raise PaddleOCRVLConfigError(
                "PADDLE_OCR_VL_DEVICE must be 'cuda' or 'cpu', got "
                f"'{settings.PADDLE_OCR_VL_DEVICE}'"
            )
        if settings.PADDLE_OCR_VL_BATCH_SIZE != 1:
            logger.warning(
                "paddleocr_vl_cascade.batch_size_above_dev_recommendation",
                extra={"batch_size": settings.PADDLE_OCR_VL_BATCH_SIZE},
            )
        if settings.PADDLE_OCR_VL_DTYPE != "fp16" and settings.PADDLE_OCR_VL_DEVICE == "cuda":
            logger.warning(
                "paddleocr_vl_cascade.dtype_not_fp16_on_cuda",
                extra={"dtype": settings.PADDLE_OCR_VL_DTYPE},
            )
        self._settings = settings
        self._device = settings.PADDLE_OCR_VL_DEVICE
        self._pipeline: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _ensure_loaded(self) -> Any:
        if self._pipeline is None:
            logger.info(
                "paddleocr_vl_cascade.loading_local_pipeline",
                extra={"device": self._device},
            )
            self._pipeline = _build_local_pipeline(self._settings)
        return self._pipeline

    def _run_predict(self, image_png: bytes) -> dict[str, Any]:  # pragma: no cover
        # See module docstring "Coverage/testing trade-off".
        pipeline = self._ensure_loaded()
        array = _decode_png_to_array(image_png)
        results = pipeline.predict(array)
        first = results[0] if results else None
        return _first_prediction_to_dict(first)

    def describe_figure(self, image_png: bytes) -> VisualDescription:
        result = self._run_predict(image_png)
        return VisualDescription(description=_extract_markdown_text(result), raw=result)

    def reparse_table(self, image_png: bytes) -> TableReparseResult:
        result = self._run_predict(image_png)
        return TableReparseResult(markdown=_extract_markdown_text(result), raw=result)

    def ocr_page(self, image_png: bytes) -> OCRPageResult:
        result = self._run_predict(image_png)
        return OCRPageResult(text=_extract_markdown_text(result) or "", raw=result)

    def unload(self) -> None:
        """Explicitly release the pipeline and, on CUDA, empty the CUDA
        cache — mirrors `DoclingParser.unload()`."""
        self._pipeline = None
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Remote API backend
# ---------------------------------------------------------------------------


class RemoteAPIPaddleOCRVLClient:
    """HTTP client for a self-hosted/third-party PaddleOCR-VL-compatible
    service. Project-defined contract (see module docstring) — NOT coupled
    to PaddleOCR-VL's own server backends. No real endpoint is available to
    exercise in this dev environment; unit-tested via `httpx.MockTransport`
    (see tests/unit/test_ingestion_paddleocr_vl_cascade.py)."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.PADDLE_OCR_VL_API_URL:
            raise PaddleOCRVLConfigError(
                "PADDLE_OCR_VL_API_URL is required when PADDLE_OCR_VL_BACKEND=remote_api."
            )
        headers = {}
        if settings.PADDLE_OCR_VL_API_KEY:
            headers["Authorization"] = f"Bearer {settings.PADDLE_OCR_VL_API_KEY}"
        self._client = httpx.Client(
            base_url=settings.PADDLE_OCR_VL_API_URL,
            headers=headers,
            timeout=settings.PADDLE_OCR_VL_TIMEOUT_SECONDS,
            transport=transport,
        )
        self._max_retries = settings.PADDLE_OCR_VL_MAX_RETRIES

    def _post(self, endpoint: str, image_png: bytes) -> dict[str, Any]:
        # [I1.10 live finding, 286-page Zeggo VRV IV REYQ E2E run] Real
        # single-image inference latency is highly variable — a request
        # landing right as paddleocr-vl-service's idle-unload kicks in pays
        # a "cold start" (model reload + inference) that can exceed even a
        # generous timeout, while the *same* image on a warm model is much
        # faster. A bounded retry (not just a larger timeout alone) is the
        # right mitigation: one retry after a timeout gives the service a
        # chance to finish warming up rather than repeating the same
        # request-vs-model-load race.
        payload = {"image_base64": base64.b64encode(image_png).decode("ascii")}
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(f"/{endpoint}", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "paddleocr_vl_cascade.remote_request_failed",
                    extra={"endpoint": endpoint, "attempt": attempt, "error": str(exc)},
                )
                continue
            if response.status_code >= 400:
                raise PaddleOCRVLRemoteError(
                    f"{endpoint} returned HTTP {response.status_code}: {response.text[:500]}"
                )
            return dict(response.json())
        raise PaddleOCRVLRemoteError(
            f"request to {endpoint} failed after {self._max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def describe_figure(self, image_png: bytes) -> VisualDescription:
        body = self._post("describe_figure", image_png)
        return VisualDescription(
            description=body.get("description"),
            figure_type=body.get("figure_type"),
            components=list(body.get("components") or []),
            connections=list(body.get("connections") or []),
            raw=body,
        )

    def reparse_table(self, image_png: bytes) -> TableReparseResult:
        body = self._post("reparse_table", image_png)
        return TableReparseResult(
            markdown=body.get("markdown"),
            rows=list(body.get("rows") or []),
            raw=body,
        )

    def ocr_page(self, image_png: bytes) -> OCRPageResult:
        body = self._post("ocr_page", image_png)
        return OCRPageResult(text=body.get("text") or "", raw=body)

    def unload(self) -> None:
        """No local VRAM to release — closes the HTTP connection pool."""
        self._client.close()


# ---------------------------------------------------------------------------
# Factory + VRAM coordination guard + orchestration
# ---------------------------------------------------------------------------


def build_paddleocr_vl_client(settings: Settings) -> PaddleOCRVLClient:
    """Fail-fast factory — call `validate_paddleocr_vl_config_or_raise`
    first at ingestion-task startup so an invalid config is caught before
    any GPU/network work happens."""
    if settings.PADDLE_OCR_VL_BACKEND == "local":
        return LocalPaddleOCRVLClient(settings)
    if settings.PADDLE_OCR_VL_BACKEND == "remote_api":
        return RemoteAPIPaddleOCRVLClient(settings)
    raise PaddleOCRVLConfigError(
        f"PADDLE_OCR_VL_BACKEND: unknown backend '{settings.PADDLE_OCR_VL_BACKEND}', "
        f"must be one of {SUPPORTED_BACKENDS}"
    )


def validate_paddleocr_vl_config_or_raise(settings: Settings) -> None:
    """Fail-fast validation of the configured backend's requirements, without
    actually constructing a client (no model load / no HTTP connection)."""
    if settings.PADDLE_OCR_VL_BACKEND not in SUPPORTED_BACKENDS:
        raise PaddleOCRVLConfigError(
            f"PADDLE_OCR_VL_BACKEND: unknown backend '{settings.PADDLE_OCR_VL_BACKEND}', "
            f"must be one of {SUPPORTED_BACKENDS}"
        )
    if settings.PADDLE_OCR_VL_BACKEND == "remote_api" and not settings.PADDLE_OCR_VL_API_URL:
        raise PaddleOCRVLConfigError(
            "PADDLE_OCR_VL_API_URL is required when PADDLE_OCR_VL_BACKEND=remote_api."
        )


def require_docling_unloaded_before_paddle_stage(
    docling_parser: DoclingParser | None, settings: Settings
) -> None:
    """Enforced VRAM-coordination guard consumed by the ingestion
    orchestrator (I1.10) before constructing a *local* PaddleOCR-VL client —
    see `02-ingestion-pipeline.md` §4, `DOCLING_UNLOAD_BEFORE_PADDLE_STAGE`.
    A no-op for the `remote_api` backend (no local VRAM contention)."""
    if settings.PADDLE_OCR_VL_BACKEND != "local":
        return
    if not settings.DOCLING_UNLOAD_BEFORE_PADDLE_STAGE:
        return
    if docling_parser is not None and docling_parser.is_loaded:
        raise PaddleOCRVLConfigError(
            "DoclingParser is still loaded (VRAM not released) — call "
            "DoclingParser.unload() before starting the local PaddleOCR-VL "
            "stage. See 02-ingestion-pipeline.md §4."
        )


def run_cascade_stage(
    plan: CascadePlan,
    pdf_path: str | Path,
    client: PaddleOCRVLClient,
    *,
    elements_by_local_id: dict[int, ElementDraft] | None = None,
) -> list[CascadeResult]:
    """Execute every task in `plan` against `client`, rendering the relevant
    page/bbox crop for each via `render_bbox_crop`. Fully unit-testable with
    a fake `client` (no real model/network needed) — see
    `tests/unit/test_ingestion_paddleocr_vl_cascade.py`.
    """
    elements_by_local_id = elements_by_local_id or {}
    results: list[CascadeResult] = []

    for task in plan.tasks:
        bbox = None
        if task.element_local_id is not None:
            element = elements_by_local_id.get(task.element_local_id)
            bbox = element.bbox if element is not None else None
        image = render_bbox_crop(pdf_path, task.page_number, bbox)

        if task.task_type == TASK_TABLE_REPARSE:
            results.append(CascadeResult(task=task, table_reparse=client.reparse_table(image)))
        elif task.task_type == TASK_VISUAL_DESCRIPTION:
            results.append(
                CascadeResult(task=task, visual_description=client.describe_figure(image))
            )
        elif task.task_type == TASK_FULL_PAGE_OCR:
            results.append(CascadeResult(task=task, ocr_page=client.ocr_page(image)))

    return results
