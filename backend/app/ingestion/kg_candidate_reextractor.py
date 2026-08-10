"""KG candidate re-extraction — read `elements`/`pages`/`documents
.model_family` straight from Postgres for an already-ingested document and
re-run `kg_candidate_extractor.extract_kg_candidates` against them, writing
the results back to `elements.kg_candidate_entities`/`kg_candidate_relations`
— **NOT** a re-ingest.

See `Documentation/system-design/09-kg-extraction-strategy.md` §4 for the
cost argument this module exists to realize: `extract_kg_candidates()` only
needs `text`/`page_number`/`element_type`/`section_path`/`parent_id`/
`visual_description` per element — every one of those is already persisted
permanently by Stage 1-4 (`elements`/`pages` columns, `06-data-schema.md`
§1), so re-running KG candidate extraction after a strategy/rule change
(e.g. every fix in `KG-W1.1`-`KG-W1.4`/`KG-W1.7`) costs **~seconds per
document**, not the ~82 minutes/document a full re-ingest (Docling + the
PaddleOCR-VL cascade) would cost. Real measurement backing this: the
`kg_candidate` orchestrator stage itself was 0.3s out of 4033.4s total for
the real 286-page `document_id=3` run (`09-...md` §4).

**Deliberately independent of `app/ingestion/orchestrator.py`** (Wave 1 DoD:
"seluruh perubahan hidup di modul KG candidate... menjaga sifat 'murah untuk
diulang'"). This module owns its OWN read query and its OWN write-back —
it does **not** import or modify `app/ingestion/canonical_store.py` (kept
untouched per an explicit coordination note: a parallel Fase 2 branch is
actively changing that file for an unrelated reason, `pages.page_width_pt`/
`page_height_pt`; touching it here would risk a real merge conflict).

Every candidate's `element_id` in the re-extraction result is ALREADY the
real `elements.id` — unlike the original ingestion flow
(`app/ingestion/docling_parser.py`'s pre-insert `ElementDraft.local_id`,
resolved to a real id only after `canonical_store.store_pages_and_elements`
inserts the row), this module constructs each `ElementDraft.local_id`
directly FROM the real, already-existing `elements.id` — there is no
local-id-to-db-id remapping step needed here (nothing to remap: the ids are
already final).

**Full replacement, not merge-in**: every element scanned for a document
gets its `kg_candidate_entities`/`kg_candidate_relations` columns
OVERWRITTEN with the fresh extraction result (`[]` when the element now has
no candidates) — this is what makes it a true RE-extraction (old, possibly
now-wrong candidates from a stale rule version are fully replaced), not an
accumulation of old-plus-new results across repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.ingestion.cascade_trigger import TASK_VISUAL_DESCRIPTION, CascadeTask
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft
from app.ingestion.kg_candidate_extractor import (
    entities_to_jsonb,
    extract_kg_candidates,
    relations_to_jsonb,
)
from app.ingestion.paddleocr_vl_cascade import CascadeResult, VisualDescription


class DocumentNotFoundError(ValueError):
    """Raised when `document_id` doesn't exist — see
    `reextract_kg_candidates_for_document`."""


@dataclass(slots=True)
class ReextractionSummary:
    document_id: int
    elements_scanned: int
    elements_with_candidates: int
    entities_written: int
    relations_written: int


def _load_element_drafts_and_cascade_results(
    db: Session, document_id: int
) -> tuple[list[ElementDraft], list[CascadeResult], dict[int, Element]]:
    """Read every `elements` row for `document_id` (joined to `pages` for
    `page_number`) and rebuild the `ElementDraft`/`CascadeResult` shapes
    `extract_kg_candidates` expects — `ElementDraft.local_id`/
    `.parent_local_id` are set directly to the real `elements.id`/
    `.parent_id` (see module docstring: no remapping needed here).

    `ORDER BY page_number, elements.id` matches the same ordering
    `canonical_store.store_pages_and_elements` produces on first insert
    (parent rows always precede their children — `.parent_id` is a real,
    already-resolved FK, unlike ingestion-time `parent_local_id`) — kept
    here only for deterministic/stable iteration order, not because this
    module's own logic requires parent-before-child ordering (it doesn't;
    `parent_id` is already fully resolved for every row read from
    Postgres).
    """
    rows = db.execute(
        select(Element, Page.page_number)
        .join(Page, Element.page_id == Page.id)
        .where(Element.document_id == document_id)
        .order_by(Page.page_number, Element.id)
    ).all()

    element_drafts: list[ElementDraft] = []
    cascade_results: list[CascadeResult] = []
    element_rows_by_id: dict[int, Element] = {}

    for element, page_number in rows:
        element_rows_by_id[element.id] = element
        element_drafts.append(
            ElementDraft(
                local_id=element.id,
                element_type=element.element_type,
                text=element.text,
                bbox=None,
                page_number=page_number,
                parent_local_id=element.parent_id,
                section_path=list(element.section_path or []),
                extraction_method=element.extraction_method or "docling",
                extraction_confidence=element.extraction_confidence,
            )
        )

        vd_raw = element.visual_description
        if vd_raw:
            cascade_results.append(
                CascadeResult(
                    task=CascadeTask(
                        task_type=TASK_VISUAL_DESCRIPTION,
                        page_number=page_number,
                        element_local_id=element.id,
                        reason="kg_candidate_reextraction",
                    ),
                    visual_description=VisualDescription(
                        description=vd_raw.get("description"),
                        figure_type=vd_raw.get("figure_type"),
                        components=list(vd_raw.get("components") or []),
                        connections=list(vd_raw.get("connections") or []),
                    ),
                )
            )

    return element_drafts, cascade_results, element_rows_by_id


def reextract_kg_candidates_for_document(db: Session, document_id: int) -> ReextractionSummary:
    """Re-run `kg_candidate_extractor.extract_kg_candidates` for an
    already-ingested `document_id`, reading straight from Postgres, and
    write the (full-replacement, see module docstring) result back to
    `elements.kg_candidate_entities`/`kg_candidate_relations`.

    Raises `DocumentNotFoundError` if `document_id` doesn't exist. Commits
    once at the end (all-or-nothing for this document's re-extraction) —
    the caller is responsible for the `Session`'s lifecycle otherwise (same
    convention as `app/ingestion/canonical_store.py`'s
    `store_pages_and_elements`)."""
    document = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document {document_id} not found")

    element_drafts, cascade_results, element_rows_by_id = _load_element_drafts_and_cascade_results(
        db, document_id
    )

    parse_result = DoclingParseResult(
        elements=element_drafts,
        page_confidence={},
        page_count=document.page_count or 0,
    )

    candidates_by_id = extract_kg_candidates(
        parse_result,
        document.filename,
        cascade_results=cascade_results,
        model_family=document.model_family,
    )

    entities_written = 0
    relations_written = 0
    for element_id, element_row in element_rows_by_id.items():
        candidates = candidates_by_id.get(element_id)
        if candidates is None:
            element_row.kg_candidate_entities = []
            element_row.kg_candidate_relations = []
            continue
        element_row.kg_candidate_entities = entities_to_jsonb(candidates.entities)
        element_row.kg_candidate_relations = relations_to_jsonb(candidates.relations)
        entities_written += len(candidates.entities)
        relations_written += len(candidates.relations)

    db.commit()

    return ReextractionSummary(
        document_id=document_id,
        elements_scanned=len(element_rows_by_id),
        elements_with_candidates=len(candidates_by_id),
        entities_written=entities_written,
        relations_written=relations_written,
    )
