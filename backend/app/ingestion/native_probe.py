"""Stage 1 — native PDF probe (PyMuPDF), see
`Documentation/system-design/02-ingestion-pipeline.md` §3.

Fast, GPU-free pass over every page of a source PDF using PyMuPDF's native
text/drawing/image APIs. Flags each page as one/more of
``text_normal | vector_diagram | raster_image | low_text_confidence`` *before*
the expensive Docling (Stage 2) and PaddleOCR-VL (Stage 4) stages run.

Granularity is deliberately **page-level**, matching what System Analyst
actually validated by sampling in `02-ingestion-pipeline.md` §1 (e.g. "6.931
vector drawing primitives pada satu halaman... 0 embedded image" is reported
per-page, not per sub-page-region). Stage 3 (`app/ingestion/cascade_trigger.py`)
treats "region flagged vector_diagram/raster_image" as "the page a given
Docling figure/table element lives on carries that flag" — this is the
region-flag consumption point required by the design.

``text_normal``/``low_text_confidence`` are mutually exclusive per page (a
page either reads as clean digital narrative text or it doesn't).
``vector_diagram``/``raster_image`` are independent, additive flags that can
co-occur with either text flag — this matters because inline icons/screenshots
embedded in otherwise-normal narrative text is an explicit requirement
(`CLAUDE.md` §4, `02-ingestion-pipeline.md` §6): a page with normal paragraph
text AND one small inline icon image must carry both ``text_normal`` and
``raster_image``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PAGE_FLAG_TEXT_NORMAL = "text_normal"
PAGE_FLAG_VECTOR_DIAGRAM = "vector_diagram"
PAGE_FLAG_RASTER_IMAGE = "raster_image"
PAGE_FLAG_LOW_TEXT_CONFIDENCE = "low_text_confidence"

PAGE_FLAGS: tuple[str, ...] = (
    PAGE_FLAG_TEXT_NORMAL,
    PAGE_FLAG_VECTOR_DIAGRAM,
    PAGE_FLAG_RASTER_IMAGE,
    PAGE_FLAG_LOW_TEXT_CONFIDENCE,
)

# Starting-point thresholds (also mirrored as `NATIVE_PROBE_*` in
# app/core/config.py so they are configurable via env without a code change).
# These are Stage 1 heuristics only — NOT the `THRESHOLD_TABLE`/`THRESHOLD_TEXT`
# Stage 3 thresholds (those apply to Docling's own `extraction_confidence`,
# see app/ingestion/cascade_trigger.py). Calibrated further by I1.6 benchmark,
# see backend/docs/i1.6-vram-benchmark-report.md.
DEFAULT_MIN_TEXT_CHARS_PER_PAGE = 40
DEFAULT_VECTOR_DRAWING_THRESHOLD = 50
# Chars per square point of page area. Calibrated against a 15-page random
# sample of `source-documents/Zeggo VRV IV Service Manual REYQ.pdf`, which
# ranged 0.0017-0.0063 for ordinary narrative/procedural pages (Letter-ish
# page size ~609x793pt) — 0.0005 sits well below that range so only pages
# with near-zero real text (blank/scanned) are flagged, not merely sparse
# ones. Revisit with I1.6 benchmark data across all 7 documents.
DEFAULT_TEXT_DENSITY_MIN = 0.0005


@dataclass(slots=True)
class PageProbe:
    """Stage 1 result for a single page."""

    page_number: int  # 1-indexed, matches `pages.page_number` (06-data-schema.md)
    char_count: int
    text_density: float
    vector_drawing_count: int
    embedded_image_count: int
    page_width: float
    page_height: float
    flags: list[str] = field(default_factory=list)

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags


@dataclass(slots=True)
class DocumentProbe:
    """Stage 1 result for an entire document."""

    page_count: int
    pages: list[PageProbe]

    def flag_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(PAGE_FLAGS, 0)
        for page in self.pages:
            for flag in page.flags:
                counts[flag] += 1
        return counts

    def page(self, page_number: int) -> PageProbe:
        """Look up a page's probe result by 1-indexed `page_number`.

        Used by Stage 3 cascade trigger rules to check whether a Docling
        element's page was flagged `vector_diagram`/`raster_image`.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        raise KeyError(f"page_number {page_number} not found in probe result")


def probe_page(
    page: pymupdf.Page,
    *,
    vector_drawing_threshold: int = DEFAULT_VECTOR_DRAWING_THRESHOLD,
    text_density_min: float = DEFAULT_TEXT_DENSITY_MIN,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
) -> PageProbe:
    """Stage 1 probe for a single already-open PyMuPDF page."""
    text = page.get_text()
    char_count = len(text.strip())
    drawings = page.get_drawings()
    images = page.get_images(full=False)
    rect = page.rect
    area = max(rect.width * rect.height, 1.0)
    text_density = char_count / area

    flags: list[str] = []
    if char_count < min_text_chars or text_density < text_density_min:
        flags.append(PAGE_FLAG_LOW_TEXT_CONFIDENCE)
    else:
        flags.append(PAGE_FLAG_TEXT_NORMAL)

    if len(drawings) >= vector_drawing_threshold:
        flags.append(PAGE_FLAG_VECTOR_DIAGRAM)
    if len(images) > 0:
        flags.append(PAGE_FLAG_RASTER_IMAGE)

    return PageProbe(
        page_number=page.number + 1,
        char_count=char_count,
        text_density=text_density,
        vector_drawing_count=len(drawings),
        embedded_image_count=len(images),
        page_width=rect.width,
        page_height=rect.height,
        flags=flags,
    )


def probe_document(
    pdf_path: str | Path,
    *,
    vector_drawing_threshold: int = DEFAULT_VECTOR_DRAWING_THRESHOLD,
    text_density_min: float = DEFAULT_TEXT_DENSITY_MIN,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS_PER_PAGE,
) -> DocumentProbe:
    """Stage 1 probe for an entire PDF, opened fresh from `pdf_path`."""
    doc = pymupdf.open(pdf_path)
    try:
        pages = [
            probe_page(
                doc[i],
                vector_drawing_threshold=vector_drawing_threshold,
                text_density_min=text_density_min,
                min_text_chars=min_text_chars,
            )
            for i in range(doc.page_count)
        ]
        return DocumentProbe(page_count=doc.page_count, pages=pages)
    finally:
        doc.close()
