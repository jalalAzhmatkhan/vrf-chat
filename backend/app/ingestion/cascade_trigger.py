"""Stage 3 — deterministic cascade trigger rules, see
`Documentation/system-design/02-ingestion-pipeline.md` §3 (I1.3).

Pure, deterministic rule evaluation (NOT an LLM call — see design doc §3 and
`01-architecture-overview.md` §4 "agent tidak menentukan strategi retrieval
sendiri", the same "deterministic tools, not model judgment" principle
applied to the ingestion side) over the outputs of Stage 1
(`app/ingestion/native_probe.py`) and Stage 2
(`app/ingestion/docling_parser.py`), producing the work queue Stage 4
(`app/ingestion/paddleocr_vl_cascade.py`, I1.4) consumes:

  if element_type == table and table_confidence < THRESHOLD_TABLE:
      queue table_reparse
  if element_type in (figure, icon) and page region flagged
     vector_diagram/raster_image (Stage 1):
      queue visual_description
  if page_text_confidence < THRESHOLD_TEXT:
      queue full_page_ocr

`THRESHOLD_TABLE`/`THRESHOLD_TEXT`, configurable via env — `THRESHOLD_TABLE`
is **calibrated final** (`0.90`, `SA1.1 2026-08-09`, see
`02-ingestion-pipeline.md` §3): the I1.6 50-page benchmark found all 27
real table `table_score` observations above the original `0.75`
starting-point default, meaning it never actually triggered on real
content. `THRESHOLD_TEXT` remains a starting point (`0.6`, unchanged by
SA1.1).

Missing/unresolvable confidence (`None`) is treated conservatively as "queue
it" (fail-safe) rather than silently skipped — matches the design doc's
"jaga-jaga jika ada halaman scan tersisip" framing for Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.docling_parser import (
    ELEMENT_TYPE_FIGURE,
    ELEMENT_TYPE_ICON,
    ELEMENT_TYPE_TABLE,
    DoclingParseResult,
)
from app.ingestion.native_probe import (
    PAGE_FLAG_RASTER_IMAGE,
    PAGE_FLAG_VECTOR_DIAGRAM,
    DocumentProbe,
)

DEFAULT_THRESHOLD_TABLE = 0.90  # [SA1.1 KALIBRASI FINAL, 2026-08-09] was 0.75
DEFAULT_THRESHOLD_TEXT = 0.6

TASK_TABLE_REPARSE = "table_reparse"
TASK_VISUAL_DESCRIPTION = "visual_description"
TASK_FULL_PAGE_OCR = "full_page_ocr"

TASK_TYPES = (TASK_TABLE_REPARSE, TASK_VISUAL_DESCRIPTION, TASK_FULL_PAGE_OCR)


@dataclass(slots=True, frozen=True)
class CascadeTask:
    """One unit of Stage 4 work. `element_local_id` is `None` for
    page-level tasks (`full_page_ocr`) — see `ElementDraft.local_id`
    (`app/ingestion/docling_parser.py`) for the id scoping convention."""

    task_type: str
    page_number: int
    element_local_id: int | None
    reason: str


@dataclass(slots=True)
class CascadePlan:
    tasks: list[CascadeTask] = field(default_factory=list)
    threshold_table: float = DEFAULT_THRESHOLD_TABLE
    threshold_text: float = DEFAULT_THRESHOLD_TEXT

    def tasks_of_type(self, task_type: str) -> list[CascadeTask]:
        return [t for t in self.tasks if t.task_type == task_type]

    def task_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(TASK_TYPES, 0)
        for task in self.tasks:
            counts[task.task_type] += 1
        return counts


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def _table_reason(confidence: float | None, threshold: float) -> str:
    return f"table_confidence {_fmt(confidence)} < THRESHOLD_TABLE {threshold:.4f}"


def _text_reason(confidence: float | None, threshold: float) -> str:
    return f"page_text_confidence {_fmt(confidence)} < THRESHOLD_TEXT {threshold:.4f}"


def _region_reason(flags: list[str]) -> str:
    flagged = [f for f in (PAGE_FLAG_VECTOR_DIAGRAM, PAGE_FLAG_RASTER_IMAGE) if f in flags]
    return f"page region flagged: {', '.join(flagged)}"


def build_cascade_plan(
    parse_result: DoclingParseResult,
    probe: DocumentProbe,
    *,
    threshold_table: float = DEFAULT_THRESHOLD_TABLE,
    threshold_text: float = DEFAULT_THRESHOLD_TEXT,
) -> CascadePlan:
    """Evaluate the Stage 3 rules over one document's Stage 1 + Stage 2 output."""
    tasks: list[CascadeTask] = []

    for element in parse_result.elements:
        if element.element_type == ELEMENT_TYPE_TABLE:
            confidence = element.extraction_confidence
            if confidence is None or confidence < threshold_table:
                tasks.append(
                    CascadeTask(
                        task_type=TASK_TABLE_REPARSE,
                        page_number=element.page_number,
                        element_local_id=element.local_id,
                        reason=_table_reason(confidence, threshold_table),
                    )
                )
        elif element.element_type in (ELEMENT_TYPE_FIGURE, ELEMENT_TYPE_ICON):
            try:
                page_probe = probe.page(element.page_number)
            except KeyError:
                continue
            if page_probe.has_flag(PAGE_FLAG_VECTOR_DIAGRAM) or page_probe.has_flag(
                PAGE_FLAG_RASTER_IMAGE
            ):
                tasks.append(
                    CascadeTask(
                        task_type=TASK_VISUAL_DESCRIPTION,
                        page_number=element.page_number,
                        element_local_id=element.local_id,
                        reason=_region_reason(page_probe.flags),
                    )
                )

    for page_number in sorted(parse_result.page_confidence):
        page_confidence = parse_result.page_confidence[page_number]
        text_confidence = page_confidence.parse_score
        if text_confidence is None or text_confidence < threshold_text:
            tasks.append(
                CascadeTask(
                    task_type=TASK_FULL_PAGE_OCR,
                    page_number=page_number,
                    element_local_id=None,
                    reason=_text_reason(text_confidence, threshold_text),
                )
            )

    return CascadePlan(tasks=tasks, threshold_table=threshold_table, threshold_text=threshold_text)
