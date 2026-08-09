"""Stage 2 — Docling structural parser, see
`Documentation/system-design/02-ingestion-pipeline.md` §3 (I1.2).

Docling is the **canonical structural parser for 100% of pages** (not a
fallback) — see design doc §2. This module owns a single `DocumentConverter`
instance per `DoclingParser`, produces `ElementDraft` records (heading /
paragraph / list / table / figure / figure_caption / icon), and exposes an
explicit `unload()` used by the ingestion Celery task to release VRAM before
Stage 4 (PaddleOCR-VL, `app/ingestion/paddleocr_vl_cascade.py`) begins on the
RTX 3060 6GB dev box — see `02-ingestion-pipeline.md` §4
(`DOCLING_UNLOAD_BEFORE_PADDLE_STAGE`).

Design notes:
- `DOCLING_DEVICE` (`cuda` default in dev, `cpu` fallback) is validated
  fail-fast in `DoclingParser.__init__`, not at FastAPI app startup like the
  LLM/object-storage/DB providers (`04-provider-abstractions.md` DIRECT
  MESSAGE point 2) — Docling is only ever instantiated by the GPU ingestion
  Celery task (`backend-worker-gpu`), never by the lightweight API process,
  so validating at API `create_app()` time would force torch/docling's heavy
  import cost onto a process that never uses them.
- The element↔Postgres-id mapping is deferred to I1.5 (`elements` insert):
  `ElementDraft.local_id`/`parent_local_id` are sequential ids scoped to one
  `parse()` call, resolved to real `elements.id` once rows exist (parent
  inserted before child, per the `elements.parent_id` FK).
- Confidence is Docling's own `ConfidenceReport` (per-page `parse_score` /
  `layout_score` / `table_score` / `ocr_score` / `mean_score`), NOT a
  heuristic — verified against a real conversion (see STATUS REPORT). This is
  what Stage 3 (`app/ingestion/cascade_trigger.py`, I1.3) compares against
  `THRESHOLD_TABLE`/`THRESHOLD_TEXT`.
- Inline icon association (`CLAUDE.md` §4, most critical requirement):
  `picture` items that are small relative to the page (see
  `_is_icon_sized`/`DOCLING_ICON_*`) are re-tagged `icon` (rather than
  `figure`, e.g. a full electrical schematic) and `parent_local_id` is set to
  the most recently seen paragraph/list/heading element on the SAME page in
  linear reading order (`doc.iterate_items()` — the same traversal that
  backs Docling's own `export_to_markdown()`, so it reflects true reading
  order, not raw insertion/detection order).
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.core.observability import get_logger

logger = get_logger(__name__)

SUPPORTED_DEVICES = ("cuda", "cpu")

ELEMENT_TYPE_HEADING = "heading"
ELEMENT_TYPE_PARAGRAPH = "paragraph"
ELEMENT_TYPE_LIST = "list"
ELEMENT_TYPE_TABLE = "table"
ELEMENT_TYPE_FIGURE = "figure"
ELEMENT_TYPE_FIGURE_CAPTION = "figure_caption"
ELEMENT_TYPE_ICON = "icon"

# Docling `DocItemLabel` value -> our `elements.element_type`
# (`06-data-schema.md` §1 ELEMENT_TYPES). `picture` maps to `figure` by
# default, re-tagged to `icon` for small inline images (see module docstring).
_LABEL_TO_ELEMENT_TYPE: dict[str, str] = {
    "title": ELEMENT_TYPE_HEADING,
    "section_header": ELEMENT_TYPE_HEADING,
    "text": ELEMENT_TYPE_PARAGRAPH,
    "paragraph": ELEMENT_TYPE_PARAGRAPH,
    "footnote": ELEMENT_TYPE_PARAGRAPH,
    "page_header": ELEMENT_TYPE_PARAGRAPH,
    "page_footer": ELEMENT_TYPE_PARAGRAPH,
    "formula": ELEMENT_TYPE_PARAGRAPH,
    "code": ELEMENT_TYPE_PARAGRAPH,
    "list_item": ELEMENT_TYPE_LIST,
    "table": ELEMENT_TYPE_TABLE,
    "picture": ELEMENT_TYPE_FIGURE,
    "chart": ELEMENT_TYPE_FIGURE,
    "caption": ELEMENT_TYPE_FIGURE_CAPTION,
}

# Icon-vs-figure heuristic starting point (mirrored as `DOCLING_ICON_*` in
# app/core/config.py) — calibrated further by I1.6 benchmark.
DEFAULT_ICON_MAX_AREA_RATIO = 0.02
DEFAULT_ICON_MAX_DIM_PT = 80.0


class DoclingConfigError(ValueError):
    """Raised (fail-fast) for an invalid `DOCLING_DEVICE`/config combination."""


@dataclass(slots=True)
class BBox:
    left: float
    top: float
    right: float
    bottom: float
    page_number: int
    coord_origin: str = "BOTTOMLEFT"

    def as_dict(self) -> dict[str, Any]:
        return {
            "l": self.left,
            "t": self.top,
            "r": self.right,
            "b": self.bottom,
            "page_number": self.page_number,
            "coord_origin": self.coord_origin,
        }

    @property
    def width(self) -> float:
        return abs(self.right - self.left)

    @property
    def height(self) -> float:
        return abs(self.top - self.bottom)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(slots=True)
class PageConfidence:
    page_number: int
    parse_score: float | None
    layout_score: float | None
    table_score: float | None
    ocr_score: float | None
    mean_score: float | None
    mean_grade: str | None


@dataclass(slots=True)
class ElementDraft:
    """One canonical element, pre-Postgres-insert. See module docstring for
    `local_id`/`parent_local_id` semantics."""

    local_id: int
    element_type: str
    text: str | None
    bbox: dict[str, Any] | None
    page_number: int
    parent_local_id: int | None
    section_path: list[str]
    extraction_method: str
    extraction_confidence: float | None


@dataclass(slots=True)
class DoclingParseResult:
    elements: list[ElementDraft]
    page_confidence: dict[int, PageConfidence]
    page_count: int


def _clean_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _page_confidence_from_report(
    confidence_report: Any, page_count: int
) -> dict[int, PageConfidence]:
    result: dict[int, PageConfidence] = {}
    pages = getattr(confidence_report, "pages", None) or {}
    for page_no in range(1, page_count + 1):
        raw = pages.get(page_no)
        if raw is None:
            result[page_no] = PageConfidence(page_no, None, None, None, None, None, None)
            continue
        mean_grade = getattr(raw, "mean_grade", None)
        result[page_no] = PageConfidence(
            page_number=page_no,
            parse_score=_clean_score(getattr(raw, "parse_score", None)),
            layout_score=_clean_score(getattr(raw, "layout_score", None)),
            table_score=_clean_score(getattr(raw, "table_score", None)),
            ocr_score=_clean_score(getattr(raw, "ocr_score", None)),
            mean_score=_clean_score(getattr(raw, "mean_score", None)),
            mean_grade=str(mean_grade) if mean_grade is not None else None,
        )
    return result


def _is_icon_sized(
    bbox: BBox,
    page_size: tuple[float, float],
    max_area_ratio: float,
    max_dim_pt: float,
) -> bool:
    page_w, page_h = page_size
    page_area = max(page_w * page_h, 1.0)
    area_ratio = bbox.area / page_area
    return area_ratio < max_area_ratio and bbox.width < max_dim_pt and bbox.height < max_dim_pt


def _extract_bbox(item: Any) -> BBox | None:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    provenance = prov[0]
    box = provenance.bbox
    return BBox(
        left=float(box.l),
        top=float(box.t),
        right=float(box.r),
        bottom=float(box.b),
        page_number=int(provenance.page_no),
        coord_origin=str(getattr(box, "coord_origin", "BOTTOMLEFT")),
    )


def _confidence_for(element_type: str, page_conf: PageConfidence | None) -> float | None:
    if page_conf is None:
        return None
    if element_type == ELEMENT_TYPE_TABLE:
        return page_conf.table_score if page_conf.table_score is not None else page_conf.mean_score
    if element_type in (ELEMENT_TYPE_FIGURE, ELEMENT_TYPE_ICON, ELEMENT_TYPE_FIGURE_CAPTION):
        return (
            page_conf.layout_score if page_conf.layout_score is not None else page_conf.mean_score
        )
    return page_conf.parse_score if page_conf.parse_score is not None else page_conf.mean_score


def _safe_export_markdown(item: Any, doc: Any) -> str | None:
    export = getattr(item, "export_to_markdown", None)
    if export is None:
        return None
    try:
        return export(doc=doc)
    except Exception:  # pragma: no cover - defensive against docling internal errors
        logger.warning("docling_parser.table_markdown_export_failed", exc_info=True)
        return None


def map_document_to_elements(
    doc: Any,
    confidence_report: Any,
    page_count: int,
    *,
    icon_max_area_ratio: float = DEFAULT_ICON_MAX_AREA_RATIO,
    icon_max_dim_pt: float = DEFAULT_ICON_MAX_DIM_PT,
) -> DoclingParseResult:
    """Pure mapping: `DoclingDocument` -> `DoclingParseResult`.

    Decoupled from model inference on purpose, so this is fully unit-testable
    with fake duck-typed objects (no GPU/model download required) — see
    `tests/unit/test_ingestion_docling_parser.py`.
    """
    page_confidence = _page_confidence_from_report(confidence_report, page_count)

    page_sizes: dict[int, tuple[float, float]] = {}
    for page_no, page_item in (getattr(doc, "pages", None) or {}).items():
        size = getattr(page_item, "size", None)
        if size is not None:
            page_sizes[int(page_no)] = (float(size.width), float(size.height))

    elements: list[ElementDraft] = []
    ref_to_local_id: dict[str, int] = {}
    last_text_local_id_by_page: dict[int, int] = {}
    section_path: list[str] = []
    pending_caption_owner_by_ref: dict[str, str] = {}
    next_id = 1

    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "") or "")
        self_ref = getattr(item, "self_ref", None)
        bbox = _extract_bbox(item)
        page_number = bbox.page_number if bbox is not None else 1
        text = getattr(item, "text", None)

        element_type = _LABEL_TO_ELEMENT_TYPE.get(label, ELEMENT_TYPE_PARAGRAPH)
        parent_local_id: int | None = None

        if element_type == ELEMENT_TYPE_HEADING and text:
            level = getattr(item, "level", 1) or 1
            section_path = section_path[: max(level - 1, 0)]
            section_path = [*section_path, text]

        if element_type == ELEMENT_TYPE_FIGURE and bbox is not None:
            page_size = page_sizes.get(page_number)
            if page_size is not None and _is_icon_sized(
                bbox, page_size, icon_max_area_ratio, icon_max_dim_pt
            ):
                element_type = ELEMENT_TYPE_ICON
                parent_local_id = last_text_local_id_by_page.get(page_number)

        if element_type == ELEMENT_TYPE_TABLE:
            text = _safe_export_markdown(item, doc)

        draft = ElementDraft(
            local_id=next_id,
            element_type=element_type,
            text=text,
            bbox=bbox.as_dict() if bbox is not None else None,
            page_number=page_number,
            parent_local_id=parent_local_id,
            section_path=list(section_path),
            extraction_method="docling",
            extraction_confidence=_confidence_for(element_type, page_confidence.get(page_number)),
        )
        elements.append(draft)

        if self_ref:
            ref_to_local_id[self_ref] = next_id
        if element_type in (ELEMENT_TYPE_PARAGRAPH, ELEMENT_TYPE_LIST, ELEMENT_TYPE_HEADING):
            last_text_local_id_by_page[page_number] = next_id

        for cap_ref in getattr(item, "captions", None) or []:
            cref = getattr(cap_ref, "cref", None)
            if cref and self_ref:
                pending_caption_owner_by_ref[cref] = self_ref

        next_id += 1

    # Second pass: re-tag caption text elements + point them at the
    # figure/table they caption (`elements.parent_id` chain in 06-data-schema.md).
    if pending_caption_owner_by_ref:
        local_id_to_ref = {v: k for k, v in ref_to_local_id.items()}
        for element in elements:
            ref = local_id_to_ref.get(element.local_id)
            if ref is not None and ref in pending_caption_owner_by_ref:
                owner_ref = pending_caption_owner_by_ref[ref]
                element.element_type = ELEMENT_TYPE_FIGURE_CAPTION
                element.parent_local_id = ref_to_local_id.get(owner_ref)

    return DoclingParseResult(
        elements=elements, page_confidence=page_confidence, page_count=page_count
    )


def _build_converter(device: str) -> Any:  # pragma: no cover
    # Excluded from unit coverage deliberately: constructing a real
    # `DocumentConverter` initializes/downloads Docling's actual layout+table
    # models (network + several seconds, first run), which is exactly the
    # kind of "genuinely needs a real model" code the project's coverage
    # policy allows mocking around (see backend STATUS REPORT trade-off
    # note). `DoclingParser.parse()`'s orchestration around this call IS
    # covered, via monkeypatching this function — see
    # tests/unit/test_ingestion_docling_parser.py. A real conversion against
    # a `source-documents/` sample was run manually as a live smoke test
    # (both this function's CPU path and the mapping logic it feeds) — see
    # the I1.2 STATUS REPORT.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption

    accelerator_device = AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU
    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(device=accelerator_device)
    format_options: dict[InputFormat, FormatOption] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    }
    return DocumentConverter(format_options=format_options)


class DoclingParser:
    """Owns a single Docling `DocumentConverter` (and therefore whatever
    GPU/CPU memory Docling's pipeline models allocate). Call `unload()`
    before Stage 4 begins — see module docstring.
    """

    def __init__(self, device: str = "cuda") -> None:
        if device not in SUPPORTED_DEVICES:
            raise DoclingConfigError(
                f"DOCLING_DEVICE: unknown device '{device}', must be one of {SUPPORTED_DEVICES}"
            )
        self._device = device
        self._converter: Any | None = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._converter is not None

    def _ensure_loaded(self) -> Any:
        if self._converter is None:
            logger.info("docling_parser.loading_model", extra={"device": self._device})
            self._converter = _build_converter(self._device)
        return self._converter

    def parse(self, pdf_path: str | Path) -> DoclingParseResult:
        converter = self._ensure_loaded()
        result = converter.convert(str(pdf_path))
        doc = result.document
        page_count = len(getattr(doc, "pages", None) or {})
        return map_document_to_elements(doc, result.confidence, page_count)

    def unload(self) -> None:
        """Explicitly release the converter/pipeline models and, on CUDA,
        empty the CUDA cache — WAJIB before Stage 4 (PaddleOCR-VL) begins on
        the RTX 3060 6GB dev box (`DOCLING_UNLOAD_BEFORE_PADDLE_STAGE`,
        `02-ingestion-pipeline.md` §4). Merely finishing Stage 2 in code
        order is NOT sufficient — Docling's pipeline models stay resident in
        VRAM until references to the converter are dropped."""
        self._converter = None
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
