"""I1.6 benchmark — select 50 representative pages across the 7
`source-documents/` PDFs (10 text_normal, 10 table, 10 electrical diagram,
10 screenshot/UI, 10 troubleshooting), per
`Documentation/system-design/02-ingestion-pipeline.md` §3 and
`Documentation/project-milestones/02-phase-1-ingestion.md` I1.6.

`source-documents/` is READ-ONLY (CLAUDE.md §11) — this script only reads
those PDFs and writes its outputs under `backend/docs/` (manifest JSON +
combined sample PDF used as the actual benchmark input, so Stage 2/4 only
process these 50 pages, not full ~2.581-page documents).

Sampling heuristics (all native, PyMuPDF-only, no GPU/model needed):
- text_normal: Stage 1 native probe flags `text_normal` only (no
  vector_diagram/raster_image/low_text_confidence), moderate text length.
- table: `page.find_tables()` (PyMuPDF's native table detector) finds >= 1
  table AND Stage 1 doesn't flag `vector_diagram` (keeps this category
  distinct from electrical diagrams, which also often contain grid-like
  vector art that `find_tables()` can false-positive on).
- electrical_diagram: Stage 1 `vector_diagram` flag with a high
  `vector_drawing_count` (top of the observed distribution) — matches the
  design doc §1 finding that schematics are vector-drawn, not raster.
- screenshot_ui: Stage 1 `raster_image` flag (embedded raster image —
  design doc §1: "kemungkinan gambar layar remote controller/ikon tombol").
- troubleshooting: page text contains troubleshooting-domain keywords
  (case-insensitive): "troubleshooting", "malfunction", "error code",
  "check code", "symptom", "abnormal".

A page can match multiple heuristics; each page is assigned to exactly one
category (priority order above) and each category is capped at 10, with a
per-document cap to encourage spread across all 7 manuals rather than
clustering in one.

Usage:
    uv run python scripts/i1_6_sample_pages.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from app.ingestion.native_probe import (  # noqa: E402
    PAGE_FLAG_LOW_TEXT_CONFIDENCE,
    PAGE_FLAG_RASTER_IMAGE,
    PAGE_FLAG_TEXT_NORMAL,
    PAGE_FLAG_VECTOR_DIAGRAM,
    probe_document,
)

SOURCE_DOCS_DIR = Path(r"D:\Training\vrf-chatbot\source-documents")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs"
MANIFEST_PATH = OUTPUT_DIR / "i1.6-sample-manifest.json"
SAMPLE_PDF_PATH = OUTPUT_DIR / "i1.6-sample-50pages.pdf"

CATEGORY_TARGET_COUNT = 10
PER_DOCUMENT_CAP_PER_CATEGORY = 3  # encourage spread across all 7 manuals
VECTOR_DIAGRAM_MIN_DRAWINGS = 300  # top of observed distribution, see report

TROUBLESHOOTING_KEYWORDS = re.compile(
    r"troubleshooting|malfunction|error code|check code|symptom|abnormal",
    re.IGNORECASE,
)
# Stronger signal, restricted to pages past likely front-matter/safety-notice
# content (which frequently mentions "abnormal"/"malfunction" in generic
# safety warnings without being genuine troubleshooting procedure pages —
# see I1.6 report "Sampling methodology" note): require a procedure-shaped
# co-occurrence (cause/remedy/countermeasure/check alongside the keyword)
# AND page_number past this floor.
TROUBLESHOOTING_MIN_PAGE_NUMBER = 50
TROUBLESHOOTING_PROCEDURE_KEYWORDS = re.compile(
    r"cause|remedy|countermeasure|corrective action", re.IGNORECASE
)

CATEGORIES = (
    "text_normal",
    "table",
    "electrical_diagram",
    "screenshot_ui",
    "troubleshooting",
)


def _classify_page(
    page: pymupdf.Page, probe_flags: list[str], vector_count: int, *, strict: bool
) -> str | None:
    text = page.get_text()

    is_troubleshooting = TROUBLESHOOTING_KEYWORDS.search(text) is not None
    if strict:
        is_troubleshooting = (
            is_troubleshooting
            and page.number + 1 >= TROUBLESHOOTING_MIN_PAGE_NUMBER
            and TROUBLESHOOTING_PROCEDURE_KEYWORDS.search(text) is not None
        )
    if is_troubleshooting:
        return "troubleshooting"

    if PAGE_FLAG_VECTOR_DIAGRAM in probe_flags and vector_count >= VECTOR_DIAGRAM_MIN_DRAWINGS:
        return "electrical_diagram"

    if PAGE_FLAG_RASTER_IMAGE in probe_flags:
        return "screenshot_ui"

    if (
        PAGE_FLAG_TEXT_NORMAL in probe_flags
        and PAGE_FLAG_VECTOR_DIAGRAM not in probe_flags
        and PAGE_FLAG_RASTER_IMAGE not in probe_flags
        and PAGE_FLAG_LOW_TEXT_CONFIDENCE not in probe_flags
    ):
        tables = page.find_tables()
        if len(tables.tables) >= 1:
            return "table"
        return "text_normal"

    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(SOURCE_DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {SOURCE_DOCS_DIR}")

    selected: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    per_doc_category_count: dict[tuple[str, str], int] = {}
    seen_pages: set[tuple[str, int]] = set()

    # Cache Stage 1 probes (pure PyMuPDF, cheap) so the two passes below
    # don't re-probe every document twice.
    probes = {p: probe_document(p) for p in pdf_paths}

    def _sweep(strict: bool) -> None:
        for pdf_path in pdf_paths:
            if all(len(v) >= CATEGORY_TARGET_COUNT for v in selected.values()):
                return
            doc_probe = probes[pdf_path]
            doc = pymupdf.open(pdf_path)
            try:
                for page_probe in doc_probe.pages:
                    if all(len(v) >= CATEGORY_TARGET_COUNT for v in selected.values()):
                        break
                    page_key = (pdf_path.name, page_probe.page_number)
                    if page_key in seen_pages:
                        continue
                    page = doc[page_probe.page_number - 1]
                    category = _classify_page(
                        page, page_probe.flags, page_probe.vector_drawing_count, strict=strict
                    )
                    if category is None or len(selected[category]) >= CATEGORY_TARGET_COUNT:
                        continue
                    key = (pdf_path.name, category)
                    if per_doc_category_count.get(key, 0) >= PER_DOCUMENT_CAP_PER_CATEGORY:
                        continue
                    selected[category].append(
                        {
                            "document": pdf_path.name,
                            "page_number": page_probe.page_number,
                            "flags": page_probe.flags,
                            "vector_drawing_count": page_probe.vector_drawing_count,
                            "embedded_image_count": page_probe.embedded_image_count,
                        }
                    )
                    per_doc_category_count[key] = per_doc_category_count.get(key, 0) + 1
                    seen_pages.add(page_key)
            finally:
                doc.close()

    # Pass 1: strict troubleshooting signal (see TROUBLESHOOTING_* constants
    # above) — other categories are unaffected by `strict`.
    _sweep(strict=True)
    # Pass 2: relaxed fallback, only fills categories still short after pass 1
    # (in practice, this project's 7 manuals had enough strict matches that
    # pass 2 rarely contributes — see i1.6-sample-manifest.json for the
    # actual outcome).
    _sweep(strict=False)

    manifest = {
        "categories": selected,
        "counts": {c: len(v) for c, v in selected.items()},
        "total": sum(len(v) for v in selected.values()),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(json.dumps(manifest["counts"], indent=2))

    # Build the combined 50-page sample PDF, in manifest order, page-tagged
    # via a text watermark identifying source document + original page
    # number (so mismatched local page numbers are never ambiguous when
    # inspecting the sample PDF directly).
    out_doc = pymupdf.open()
    opened_docs: dict[str, pymupdf.Document] = {}
    try:
        for category in CATEGORIES:
            for entry in selected[category]:
                doc_name = entry["document"]
                if doc_name not in opened_docs:
                    opened_docs[doc_name] = pymupdf.open(SOURCE_DOCS_DIR / doc_name)
                src_doc = opened_docs[doc_name]
                page_index = entry["page_number"] - 1
                out_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
                new_page = out_doc[-1]
                new_page.insert_text(
                    (10, 15),
                    f"[{category}] {doc_name} p.{entry['page_number']}",
                    fontsize=7,
                    color=(1, 0, 0),
                )
        out_doc.save(str(SAMPLE_PDF_PATH))
    finally:
        out_doc.close()
        for d in opened_docs.values():
            d.close()

    print(f"Sample PDF written to {SAMPLE_PDF_PATH} ({manifest['total']} pages)")


if __name__ == "__main__":
    main()
