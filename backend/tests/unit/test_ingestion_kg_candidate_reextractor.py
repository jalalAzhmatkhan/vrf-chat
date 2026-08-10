"""Unit tests for `app/ingestion/kg_candidate_reextractor.py` (Wave 1 DoD
re-extractor).

In-memory SQLite session (same pattern as `test_ingestion_canonical_store.py`)
— no GPU/model/real-Postgres dependency, `canonical_store.py` itself never
imported (this module deliberately doesn't depend on it — see module
docstring).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.ingestion import kg_candidate_reextractor as reextractor


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_document(
    db: Session, *, page_count: int = 1, model_family: str | None = "REYQ"
) -> Document:
    document = Document(
        title="Manual",
        manufacturer="Zeggo",
        model_family=model_family,
        filename="manual.pdf",
        source_hash="hash-1",
        page_count=page_count,
    )
    db.add(document)
    db.flush()
    return document


def _make_page(db: Session, document: Document, page_number: int) -> Page:
    page = Page(document_id=document.id, page_number=page_number, extraction_method="docling")
    db.add(page)
    db.flush()
    return page


def _make_element(
    db: Session,
    document: Document,
    page: Page,
    *,
    element_type: str,
    text: str | None = None,
    parent_id: int | None = None,
    section_path: list[str] | None = None,
    visual_description: dict | None = None,
) -> Element:
    element = Element(
        document_id=document.id,
        page_id=page.id,
        element_type=element_type,
        text=text,
        parent_id=parent_id,
        section_path=section_path or [],
        source_hash="elem-hash",
        extraction_method="docling",
        visual_description=visual_description,
        kg_candidate_entities=[],
        kg_candidate_relations=[],
    )
    db.add(element)
    db.flush()
    return element


# ---------------------------------------------------------------------------
# reextract_kg_candidates_for_document
# ---------------------------------------------------------------------------


def test_reextract_unknown_document_raises() -> None:
    db = _make_session()
    with pytest.raises(reextractor.DocumentNotFoundError):
        reextractor.reextract_kg_candidates_for_document(db, 999)


def test_reextract_writes_candidates_to_elements() -> None:
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 1)
    element = _make_element(
        db, document, page, element_type="paragraph", text="Check TH3 near the compressor."
    )
    db.commit()

    summary = reextractor.reextract_kg_candidates_for_document(db, document.id)

    assert summary.document_id == document.id
    assert summary.elements_scanned == 1
    assert summary.elements_with_candidates == 1
    assert summary.entities_written == 2  # TH3 (Sensor) + compressor (Component)
    assert summary.relations_written == 0

    db.refresh(element)
    names = {e["name"] for e in element.kg_candidate_entities}
    assert names == {"TH3", "compressor"}
    # [K2] model_family threaded from documents.model_family
    assert all(e["model_family"] == "REYQ" for e in element.kg_candidate_entities)
    # element_id in the jsonb is already the REAL elements.id, no remapping
    # needed (see module docstring)
    assert all(e["element_id"] == element.id for e in element.kg_candidate_entities)


def test_reextract_element_id_already_real_no_remapping_needed() -> None:
    """[Module docstring] Unlike ingestion-time local_id, this module's
    ElementDraft.local_id IS the real elements.id already — confirms no
    off-by-something remapping bug by checking against a case where the
    element's real id is deliberately NOT 1 (auto-increment offset)."""
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 1)
    _make_element(db, document, page, element_type="paragraph", text="Nothing here.")
    element = _make_element(
        db, document, page, element_type="paragraph", text="Check CN105 wiring."
    )
    db.commit()
    assert element.id != 1  # sanity: this element's real id is NOT 1

    reextractor.reextract_kg_candidates_for_document(db, document.id)
    db.refresh(element)
    assert element.kg_candidate_entities[0]["element_id"] == element.id


def test_reextract_full_replacement_clears_stale_candidates() -> None:
    """[Module docstring "full replacement, not merge-in"] An element that
    previously had (now-stale) candidates from an old rule version gets
    them fully overwritten, including cleared to `[]` if the new rules no
    longer match anything."""
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 1)
    element = _make_element(db, document, page, element_type="paragraph", text="Nothing relevant.")
    element.kg_candidate_entities = [{"name": "stale", "entity_type": "Component"}]
    db.commit()

    reextractor.reextract_kg_candidates_for_document(db, document.id)

    db.refresh(element)
    assert element.kg_candidate_entities == []
    assert element.kg_candidate_relations == []


def test_reextract_uses_visual_description_for_vlm_candidates() -> None:
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 3)
    element = _make_element(
        db,
        document,
        page,
        element_type="figure",
        text=None,
        visual_description={
            "description": "Schematic showing the compressor.",
            "figure_type": None,
            "components": ["fan motor"],
            "connections": [],
        },
    )
    db.commit()

    reextractor.reextract_kg_candidates_for_document(db, document.id)

    db.refresh(element)
    names = {e["name"] for e in element.kg_candidate_entities}
    assert "fan motor" in names  # from structured components[]
    assert "compressor" in names  # from description text (KG-W1.1)


def test_reextract_parent_id_preserved_from_real_fk() -> None:
    """`ElementDraft.parent_local_id` is set from the real `elements
    .parent_id` — confirms section anchor logic (which reads
    `section_path`, not `parent_id`) still works correctly when an
    element has a real parent (e.g. an icon linked to its paragraph)."""
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 1)
    parent = _make_element(
        db,
        document,
        page,
        element_type="paragraph",
        text="If error code P8 appears, check the flow sensor.",
        section_path=["Troubleshooting"],
    )
    _make_element(
        db, document, page, element_type="icon", text=None, parent_id=parent.id
    )
    db.commit()

    summary = reextractor.reextract_kg_candidates_for_document(db, document.id)
    # parent paragraph produces the ErrorCode candidate; icon has no text ->
    # no candidates of its own, doesn't crash on the parent_id reference.
    assert summary.entities_written == 1
    db.refresh(parent)
    assert parent.kg_candidate_entities[0]["entity_type"] == "ErrorCode"
    assert parent.kg_candidate_entities[0]["context_anchor_matched"] is True


def test_reextract_no_model_family_stays_none() -> None:
    db = _make_session()
    document = _make_document(db, model_family=None)
    page = _make_page(db, document, 1)
    _make_element(db, document, page, element_type="paragraph", text="Check the compressor.")
    db.commit()

    reextractor.reextract_kg_candidates_for_document(db, document.id)

    element = db.get(Element, 1)
    assert element is not None
    assert element.kg_candidate_entities[0]["model_family"] is None


def test_reextract_cross_source_agreement_applied_across_elements() -> None:
    """[KG-W1.7] The re-extractor calls extract_kg_candidates() as a whole
    (not per-element), so cross-source agreement (page-level, across
    elements) still fires correctly here too."""
    db = _make_session()
    document = _make_document(db)
    page = _make_page(db, document, 7)
    figure = _make_element(
        db,
        document,
        page,
        element_type="figure",
        text=None,
        visual_description={
            "description": "Schematic showing the compressor.",
            "figure_type": None,
            "components": [],
            "connections": [],
        },
    )
    text_element = _make_element(
        db, document, page, element_type="paragraph", text="Check the compressor for leaks."
    )
    db.commit()

    reextractor.reextract_kg_candidates_for_document(db, document.id)

    db.refresh(figure)
    db.refresh(text_element)
    figure_entity = next(e for e in figure.kg_candidate_entities if e["name"] == "compressor")
    text_entity = next(e for e in text_element.kg_candidate_entities if e["name"] == "compressor")
    assert figure_entity["cross_source_corroborated"] is True
    assert text_entity["cross_source_corroborated"] is True


def test_reextract_empty_document_no_elements() -> None:
    db = _make_session()
    document = _make_document(db, page_count=2)
    db.commit()

    summary = reextractor.reextract_kg_candidates_for_document(db, document.id)

    assert summary.elements_scanned == 0
    assert summary.elements_with_candidates == 0
    assert summary.entities_written == 0
    assert summary.relations_written == 0
