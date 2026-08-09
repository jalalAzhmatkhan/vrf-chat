"""Canonical document representation store, see
`Documentation/system-design/02-ingestion-pipeline.md` §5 (I1.5).

Persists Stage 2 (Docling, `app/ingestion/docling_parser.py`) + optional
Stage 4 (PaddleOCR-VL cascade, `app/ingestion/paddleocr_vl_cascade.py`)
output as `documents`/`pages`/`elements` rows (`06-data-schema.md` §1),
uploading page renders + figure/icon crops to object storage
(`ObjectStorageClient`, `04-provider-abstractions.md` Bagian B) via the
shared `render_bbox_crop` utility from `paddleocr_vl_cascade.py`.

Idempotency (design doc §5): `documents.source_hash` (document_hash,
already present from Fase 0) / `pages.page_hash` (I1.5, this module) /
`elements.source_hash` (element_hash, already present). This module
implements the **page-level** skip: if an existing `Page` row's
`page_hash` matches the freshly-computed hash for that page, the page (and
therefore its elements) is left untouched — no re-render, no re-upload, no
re-insert. `page_hash` is computed from **source-derived, deterministic**
facts about the page (extracted text + vector-drawing/embedded-image
counts) — NOT from any ML model's output — so re-ingesting an unchanged PDF
always yields the same hash regardless of Docling/PaddleOCR-VL run-to-run
nondeterminism (see `compute_page_hash`).

Known simplification (documented, not silently skipped): this module makes
STORING idempotent (skip DB writes when `page_hash` matches), but does not
itself skip the *expensive upstream* Stage 2/4 computation for unchanged
pages — that "skip before even running Docling" optimization would need
Stage 1 (cheap) to compute page hashes and compare against the DB *before*
invoking Stage 2, which is an orchestration concern for I1.10, not
implemented here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.ingestion.docling_parser import (
    ELEMENT_TYPE_FIGURE,
    ELEMENT_TYPE_ICON,
    DoclingParseResult,
    ElementDraft,
)
from app.ingestion.kg_candidate_extractor import (
    ElementKGCandidates,
    entities_to_jsonb,
    relations_to_jsonb,
)
from app.ingestion.paddleocr_vl_cascade import CascadeResult, render_bbox_crop
from app.storage.base import ObjectStorageClient


def compute_document_hash(pdf_bytes: bytes) -> str:
    """`documents.source_hash` — hash of the raw PDF file bytes."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def compute_page_hash(page: pymupdf.Page) -> str:
    """`pages.page_hash` — see module docstring. Deterministic, source-derived
    (not ML-output-derived): extracted text + vector drawing/embedded image
    counts, matching the same PyMuPDF facts Stage 1 (`native_probe.py`)
    already reads."""
    text = page.get_text()
    drawing_count = len(page.get_drawings())
    image_count = len(page.get_images(full=False))
    payload = f"{text}|{drawing_count}|{image_count}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_element_hash(draft: ElementDraft) -> str:
    """`elements.source_hash` — hash of the element's own content-defining
    fields (not identity fields like `local_id`), so an unchanged element
    hashes identically across re-parses of the same source page."""
    payload = json.dumps(
        {
            "element_type": draft.element_type,
            "text": draft.text,
            "bbox": draft.bbox,
            "page_number": draft.page_number,
            "section_path": draft.section_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CanonicalStoreSummary:
    document_id: int
    pages_stored: int = 0
    pages_skipped: int = 0
    elements_stored: int = 0
    figure_crops_uploaded: int = 0


@dataclass(slots=True)
class DocumentMetadata:
    title: str
    manufacturer: str | None
    model_family: str | None
    filename: str
    source_hash: str
    page_count: int


def get_or_create_document(db: Session, metadata: DocumentMetadata) -> tuple[Document, bool]:
    """Idempotent document upsert by `source_hash`. Returns `(document,
    created)` — `created=False` means an identical document (same PDF bytes)
    already exists, and the caller should treat the whole ingestion as a
    no-op re-run (per design doc §5) rather than re-processing pages.

    Does not implement `version` bump for a *changed* document reusing the
    same filename/title — that scenario naturally gets a different
    `source_hash` (different content -> different hash) and therefore a new
    `Document` row via this same function; superseding the old row's
    `version` is a documented simplification/gap, not handled here.
    """
    existing = db.execute(
        select(Document).where(Document.source_hash == metadata.source_hash)
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    document = Document(
        title=metadata.title,
        manufacturer=metadata.manufacturer,
        model_family=metadata.model_family,
        filename=metadata.filename,
        source_hash=metadata.source_hash,
        page_count=metadata.page_count,
    )
    db.add(document)
    db.flush()
    return document, True


def _build_visual_description_jsonb(cascade_result: CascadeResult) -> dict[str, Any] | None:
    if cascade_result.visual_description is None:
        return None
    vd = cascade_result.visual_description
    return {
        "figure_type": vd.figure_type,
        "components": vd.components,
        "connections": vd.connections,
        "description": vd.description,
    }


def _object_key(document_id: int, page_number: int, suffix: str) -> str:
    return f"documents/{document_id}/pages/{page_number}/{suffix}"


def store_pages_and_elements(
    db: Session,
    storage: ObjectStorageClient,
    *,
    document: Document,
    pdf_path: str | Path,
    parse_result: DoclingParseResult,
    cascade_results: list[CascadeResult] | None = None,
    kg_candidates: dict[int, ElementKGCandidates] | None = None,
) -> CanonicalStoreSummary:
    """Persist Stage 2 (+ optional Stage 4, + optional I1.9 KG candidates)
    output for `document`.

    `parse_result.elements` MUST be in the order produced by
    `map_document_to_elements` (parent elements before their children,
    `local_id` ascending) — this is relied upon to resolve `parent_id`
    within a single pass, per-page.

    `kg_candidates` (from `app.ingestion.kg_candidate_extractor
    .extract_kg_candidates`, I1.9) is keyed by the same `ElementDraft
    .local_id` used throughout this function — entries are written to
    `elements.kg_candidate_entities`/`kg_candidate_relations` (default `[]`
    when absent, per `06-data-schema.md` §1).
    """
    kg_candidates = kg_candidates or {}
    summary = CanonicalStoreSummary(document_id=document.id)
    cascade_by_local_id = {
        r.task.element_local_id: r
        for r in (cascade_results or [])
        if r.task.element_local_id is not None
    }
    local_id_to_db_id: dict[int, int] = {}

    elements_by_page: dict[int, list[ElementDraft]] = {}
    for draft in parse_result.elements:
        elements_by_page.setdefault(draft.page_number, []).append(draft)

    pdf_doc = pymupdf.open(pdf_path)
    try:
        for page_number in range(1, parse_result.page_count + 1):
            pymupdf_page = pdf_doc[page_number - 1]
            page_hash = compute_page_hash(pymupdf_page)

            existing_page = db.execute(
                select(Page).where(
                    Page.document_id == document.id, Page.page_number == page_number
                )
            ).scalar_one_or_none()

            if existing_page is not None and existing_page.page_hash == page_hash:
                summary.pages_skipped += 1
                continue

            if existing_page is not None:
                # Explicit child delete rather than relying on the DB-level
                # `ON DELETE CASCADE` FK — portable across dialects/test
                # setups (e.g. plain SQLite does not enforce FK actions
                # unless `PRAGMA foreign_keys=ON` is explicitly set) and
                # keeps this module's idempotency behavior independent of
                # that configuration.
                db.execute(delete(Element).where(Element.page_id == existing_page.id))
                db.delete(existing_page)
                db.flush()

            text_layer_present = len(pymupdf_page.get_text().strip()) > 0
            page_image_bytes = render_bbox_crop(pdf_path, page_number, bbox=None)
            page_image_uri = storage.put_object(
                _object_key(document.id, page_number, "page.png"), page_image_bytes, "image/png"
            )

            new_page = Page(
                document_id=document.id,
                page_number=page_number,
                page_image_uri=page_image_uri,
                text_layer_present=text_layer_present,
                extraction_method="docling",
                page_hash=page_hash,
            )
            db.add(new_page)
            db.flush()

            for draft in elements_by_page.get(page_number, []):
                parent_db_id = (
                    local_id_to_db_id.get(draft.parent_local_id)
                    if draft.parent_local_id is not None
                    else None
                )
                element_hash = compute_element_hash(draft)

                text = draft.text
                extraction_method = draft.extraction_method
                visual_description: dict[str, Any] | None = None
                image_uri: str | None = None

                cascade_result = cascade_by_local_id.get(draft.local_id)
                if cascade_result is not None:
                    if cascade_result.table_reparse is not None:
                        text = cascade_result.table_reparse.markdown or text
                        extraction_method = "docling+paddle_table"
                    if cascade_result.visual_description is not None:
                        visual_description = _build_visual_description_jsonb(cascade_result)
                        extraction_method = "paddle_vlm"

                if draft.element_type in (ELEMENT_TYPE_FIGURE, ELEMENT_TYPE_ICON) and draft.bbox:
                    crop_bytes = render_bbox_crop(pdf_path, page_number, draft.bbox)
                    image_uri = storage.put_object(
                        _object_key(document.id, page_number, f"elements/{draft.local_id}.png"),
                        crop_bytes,
                        "image/png",
                    )
                    summary.figure_crops_uploaded += 1

                element_candidates = kg_candidates.get(draft.local_id)
                kg_entities = (
                    entities_to_jsonb(element_candidates.entities) if element_candidates else []
                )
                kg_relations = (
                    relations_to_jsonb(element_candidates.relations) if element_candidates else []
                )

                element_row = Element(
                    document_id=document.id,
                    page_id=new_page.id,
                    element_type=draft.element_type,
                    text=text,
                    bbox=draft.bbox,
                    parent_id=parent_db_id,
                    section_path=draft.section_path,
                    image_uri=image_uri,
                    source_hash=element_hash,
                    extraction_method=extraction_method,
                    extraction_confidence=draft.extraction_confidence,
                    visual_description=visual_description,
                    kg_candidate_entities=kg_entities,
                    kg_candidate_relations=kg_relations,
                )
                db.add(element_row)
                db.flush()
                local_id_to_db_id[draft.local_id] = element_row.id
                summary.elements_stored += 1

            summary.pages_stored += 1

        db.commit()
    finally:
        pdf_doc.close()

    return summary
