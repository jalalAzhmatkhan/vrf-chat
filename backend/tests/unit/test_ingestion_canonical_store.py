"""Unit tests for `app/ingestion/canonical_store.py` (I1.5).

Uses an in-memory SQLite session (same pattern as `tests/unit/test_models.py`)
and a fake in-memory `ObjectStorageClient`, plus small synthetic PDFs built
via PyMuPDF's own authoring API — no GPU/model dependency, matching the
approach used for I1.1-I1.4.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.ingestion import canonical_store as cs
from app.ingestion.cascade_trigger import TASK_TABLE_REPARSE, TASK_VISUAL_DESCRIPTION, CascadeTask
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft, PageConfidence
from app.ingestion.kg_candidate_extractor import (
    ElementKGCandidates,
    KGCandidateEntity,
    KGCandidateRelation,
)
from app.ingestion.paddleocr_vl_cascade import (
    CascadeResult,
    TableReparseResult,
    VisualDescription,
)


class FakeObjectStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[str] = []

    def put_object(self, key: str, data: bytes, content_type: str) -> str:
        self.objects[key] = data
        self.put_calls.append(key)
        return f"fake://{key}"

    def get_object(self, uri: str) -> bytes:
        key = uri.removeprefix("fake://")
        return self.objects[key]

    def get_presigned_url(self, uri: str, expires_in_seconds: int = 3600) -> str:
        return uri

    def delete_object(self, uri: str) -> None:  # pragma: no cover - unused by canonical_store
        key = uri.removeprefix("fake://")
        self.objects.pop(key, None)

    def exists(self, uri: str) -> bool:  # pragma: no cover - unused by canonical_store
        return uri.removeprefix("fake://") in self.objects


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page(width=600, height=800)
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
    doc.save(str(path))
    doc.close()


def _page_confidence(page_number: int) -> PageConfidence:
    return PageConfidence(
        page_number=page_number,
        parse_score=1.0,
        layout_score=0.95,
        table_score=None,
        ocr_score=None,
        mean_score=0.97,
        mean_grade="excellent",
    )


# ---------------------------------------------------------------------------
# Pure hash functions
# ---------------------------------------------------------------------------


def test_compute_document_hash_deterministic() -> None:
    assert cs.compute_document_hash(b"abc") == cs.compute_document_hash(b"abc")
    assert cs.compute_document_hash(b"abc") != cs.compute_document_hash(b"xyz")


def test_compute_page_hash_same_content_same_hash(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    _write_pdf(path, ["Hello world " * 20])
    doc1 = pymupdf.open(path)
    doc2 = pymupdf.open(path)
    try:
        assert cs.compute_page_hash(doc1[0]) == cs.compute_page_hash(doc2[0])
    finally:
        doc1.close()
        doc2.close()


def test_compute_page_hash_different_content_different_hash(tmp_path: Path) -> None:
    path_a = tmp_path / "a.pdf"
    path_b = tmp_path / "b.pdf"
    _write_pdf(path_a, ["Hello world " * 20])
    _write_pdf(path_b, ["Something totally different " * 20])
    doc_a = pymupdf.open(path_a)
    doc_b = pymupdf.open(path_b)
    try:
        assert cs.compute_page_hash(doc_a[0]) != cs.compute_page_hash(doc_b[0])
    finally:
        doc_a.close()
        doc_b.close()


def _element(**overrides: object) -> ElementDraft:
    base = dict(
        local_id=1,
        element_type="paragraph",
        text="Hello",
        bbox=None,
        page_number=1,
        parent_local_id=None,
        section_path=[],
        extraction_method="docling",
        extraction_confidence=0.9,
    )
    base.update(overrides)
    return ElementDraft(**base)  # type: ignore[arg-type]


def test_compute_element_hash_ignores_identity_fields() -> None:
    a = _element(local_id=1, extraction_confidence=0.9, extraction_method="docling")
    b = _element(local_id=99, extraction_confidence=0.1, extraction_method="paddle_vlm")
    assert cs.compute_element_hash(a) == cs.compute_element_hash(b)


def test_compute_element_hash_changes_with_text() -> None:
    a = _element(text="Hello")
    b = _element(text="Goodbye")
    assert cs.compute_element_hash(a) != cs.compute_element_hash(b)


def test_build_visual_description_jsonb_none() -> None:
    result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="x"
        ),
        visual_description=None,
    )
    assert cs._build_visual_description_jsonb(result) is None


# ---------------------------------------------------------------------------
# get_or_create_document
# ---------------------------------------------------------------------------


def test_get_or_create_document_creates_new() -> None:
    db = _make_session()
    metadata = cs.DocumentMetadata(
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        filename="manual.pdf",
        source_hash="hash-1",
        page_count=1,
    )
    document, created = cs.get_or_create_document(db, metadata)
    db.commit()

    assert created is True
    assert document.id is not None
    assert document.source_hash == "hash-1"


def test_get_or_create_document_idempotent_on_matching_hash() -> None:
    db = _make_session()
    metadata = cs.DocumentMetadata(
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        filename="manual.pdf",
        source_hash="hash-1",
        page_count=1,
    )
    document1, created1 = cs.get_or_create_document(db, metadata)
    db.commit()
    document2, created2 = cs.get_or_create_document(db, metadata)
    db.commit()

    assert created1 is True
    assert created2 is False
    assert document1.id == document2.id
    assert db.execute(select(Document)).scalars().all() == [document1]


# ---------------------------------------------------------------------------
# store_pages_and_elements
# ---------------------------------------------------------------------------


def _basic_document(db: Session, page_count: int) -> Document:
    document, _ = cs.get_or_create_document(
        db,
        cs.DocumentMetadata(
            title="Manual",
            manufacturer="Zeggo",
            model_family="REYQ",
            filename="manual.pdf",
            source_hash="doc-hash",
            page_count=page_count,
        ),
    )
    db.commit()
    return document


def test_store_pages_and_elements_basic_insert(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Press the button to reset."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    parse_result = DoclingParseResult(
        elements=[_element(local_id=1, element_type="paragraph", text="Press the button.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )

    summary = cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    assert summary.pages_stored == 1
    assert summary.pages_skipped == 0
    assert summary.elements_stored == 1

    pages = db.execute(select(Page)).scalars().all()
    elements = db.execute(select(Element)).scalars().all()
    assert len(pages) == 1
    assert pages[0].page_hash is not None
    assert pages[0].page_image_uri == f"fake://documents/{document.id}/pages/1/page.png"
    assert len(elements) == 1
    assert elements[0].text == "Press the button."


def test_store_pages_and_elements_parent_child_resolution(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Press the button icon to reset LOSSNAY."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    paragraph = _element(local_id=1, element_type="paragraph", text="Press the button")
    icon = _element(
        local_id=2,
        element_type="icon",
        text=None,
        bbox={"l": 10.0, "t": 100.0, "r": 30.0, "b": 80.0},
        parent_local_id=1,
    )
    parse_result = DoclingParseResult(
        elements=[paragraph, icon],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )

    cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    rows = db.execute(select(Element).order_by(Element.id)).scalars().all()
    paragraph_row, icon_row = rows
    assert icon_row.parent_id == paragraph_row.id
    assert icon_row.image_uri is not None
    assert storage.put_calls[-1] == f"documents/{document.id}/pages/1/elements/2.png"


def test_store_pages_and_elements_idempotent_skip_when_unchanged(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Unchanged content."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()
    parse_result = DoclingParseResult(
        elements=[_element(local_id=1, text="Unchanged content.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )

    summary1 = cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )
    summary2 = cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    assert summary1.pages_stored == 1
    assert summary2.pages_stored == 0
    assert summary2.pages_skipped == 1
    # No duplicate rows.
    assert len(db.execute(select(Page)).scalars().all()) == 1
    assert len(db.execute(select(Element)).scalars().all()) == 1
    # No re-upload on skip.
    assert storage.put_calls.count(f"documents/{document.id}/pages/1/page.png") == 1


def test_store_pages_and_elements_reparses_changed_page(tmp_path: Path) -> None:
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    pdf_path_v1 = tmp_path / "v1.pdf"
    _write_pdf(pdf_path_v1, ["Version one content."])
    parse_result_v1 = DoclingParseResult(
        elements=[_element(local_id=1, text="Version one content.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path_v1, parse_result=parse_result_v1
    )
    first_page_hash = db.execute(select(Page)).scalar_one().page_hash

    pdf_path_v2 = tmp_path / "v2.pdf"
    _write_pdf(pdf_path_v2, ["Version two, totally different content."])
    parse_result_v2 = DoclingParseResult(
        elements=[_element(local_id=1, text="Version two, totally different content.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    summary2 = cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path_v2, parse_result=parse_result_v2
    )

    assert summary2.pages_stored == 1
    assert summary2.pages_skipped == 0
    pages = db.execute(select(Page)).scalars().all()
    assert len(pages) == 1
    assert pages[0].page_hash != first_page_hash  # stale page row replaced, hash updated
    elements = db.execute(select(Element)).scalars().all()
    assert len(elements) == 1
    assert elements[0].text == "Version two, totally different content."


def test_store_pages_and_elements_applies_table_reparse_cascade_result(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Spec table page."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    table_element = _element(local_id=1, element_type="table", text="| bad markdown |")
    parse_result = DoclingParseResult(
        elements=[table_element],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_TABLE_REPARSE, page_number=1, element_local_id=1, reason="low confidence"
        ),
        table_reparse=TableReparseResult(markdown="| corrected | markdown |"),
    )

    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parse_result,
        cascade_results=[cascade_result],
    )

    element = db.execute(select(Element)).scalar_one()
    assert element.text == "| corrected | markdown |"
    assert element.extraction_method == "docling+paddle_table"


def test_store_pages_and_elements_applies_visual_description_cascade_result(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Diagram page."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    figure_element = _element(
        local_id=1,
        element_type="figure",
        text=None,
        bbox={"l": 0.0, "t": 700.0, "r": 500.0, "b": 100.0},
    )
    parse_result = DoclingParseResult(
        elements=[figure_element],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="diagram"
        ),
        visual_description=VisualDescription(
            description="Electrical schematic",
            figure_type="schematic",
            components=["compressor", "TH3"],
            connections=["compressor->TH3"],
        ),
    )

    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parse_result,
        cascade_results=[cascade_result],
    )

    element = db.execute(select(Element)).scalar_one()
    assert element.extraction_method == "paddle_vlm"
    assert element.visual_description == {
        "figure_type": "schematic",
        "components": ["compressor", "TH3"],
        "connections": ["compressor->TH3"],
        "description": "Electrical schematic",
    }
    assert element.image_uri is not None


def test_store_pages_and_elements_multi_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Page one text.", "Page two text."])
    db = _make_session()
    document = _basic_document(db, page_count=2)
    storage = FakeObjectStorageClient()

    parse_result = DoclingParseResult(
        elements=[
            _element(local_id=1, page_number=1, text="Page one text."),
            _element(local_id=2, page_number=2, text="Page two text."),
        ],
        page_confidence={1: _page_confidence(1), 2: _page_confidence(2)},
        page_count=2,
    )

    summary = cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    assert summary.pages_stored == 2
    assert summary.elements_stored == 2
    pages = db.execute(select(Page).order_by(Page.page_number)).scalars().all()
    assert [p.page_number for p in pages] == [1, 2]


def test_store_pages_and_elements_applies_kg_candidates(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Check TH3 near the compressor."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    # local_id=7 deliberately differs from the DB id this element will get
    # (1, the only row) — verifies `element_id` inside the stored jsonb is
    # rewritten to the real `elements.id`, not left as the ingestion-time
    # local_id (see canonical_store.py comment above the rewrite).
    parse_result = DoclingParseResult(
        elements=[_element(local_id=7, text="Check TH3 near the compressor.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    kg_candidates = {
        7: ElementKGCandidates(
            entities=[
                KGCandidateEntity(
                    name="TH3",
                    entity_type="Sensor",
                    confidence=0.7,
                    source_document="manual.pdf",
                    page=1,
                    element_id=7,
                )
            ],
            relations=[
                KGCandidateRelation(
                    subject="compressor",
                    predicate="CONNECTED_TO",
                    object="TH3",
                    confidence=0.5,
                    source_document="manual.pdf",
                    page=1,
                    element_id=7,
                )
            ],
        )
    }

    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parse_result,
        kg_candidates=kg_candidates,
    )

    element = db.execute(select(Element)).scalar_one()
    assert element.id != 7  # sanity: local_id and real db id genuinely differ here
    # [KG-W1.2] KGCandidateEntity/KGCandidateRelation gained extraction_method
    # /canonical_name/model_family/justification_span (K1) — the fixtures
    # above don't set them, so they serialize at their defaults.
    assert element.kg_candidate_entities == [
        {
            "name": "TH3",
            "entity_type": "Sensor",
            "confidence": 0.7,
            "source_document": "manual.pdf",
            "page": 1,
            "element_id": element.id,
            "extraction_method": "",
            "canonical_name": None,
            "model_family": None,
            "justification_span": None,
            "context_anchor_matched": None,
            "cross_source_corroborated": False,
            "corroboration_count": 0,
        }
    ]
    assert element.kg_candidate_relations == [
        {
            "subject": "compressor",
            "predicate": "CONNECTED_TO",
            "object": "TH3",
            "confidence": 0.5,
            "source_document": "manual.pdf",
            "page": 1,
            "element_id": element.id,
            "extraction_method": "",
            "canonical_name": None,
            "model_family": None,
            "justification_span": None,
            "context_anchor_matched": None,
            "cross_source_corroborated": False,
            "corroboration_count": 0,
            "type_constraint_violated": None,
        }
    ]


def test_store_pages_and_elements_kg_candidates_default_empty(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["No candidates here."])
    db = _make_session()
    document = _basic_document(db, page_count=1)
    storage = FakeObjectStorageClient()

    parse_result = DoclingParseResult(
        elements=[_element(local_id=1, text="No candidates here.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )

    cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    element = db.execute(select(Element)).scalar_one()
    assert element.kg_candidate_entities == []
    assert element.kg_candidate_relations == []


# ---------------------------------------------------------------------------
# page-range batching (2026-08-11, round 2): `page_range`/`local_id_to_db_id`
# — see module docstring "round 2" note for the full rationale (real OOM
# still observed after round 1's Docling-only batching; the fix here is
# calling this function once per page-range batch instead of once per
# document, with `local_id_to_db_id` carried across those calls).
# ---------------------------------------------------------------------------


def test_store_pages_and_elements_page_range_restricts_processed_pages(tmp_path: Path) -> None:
    """Only pages inside `page_range` are processed — pages outside it are
    left completely untouched (not even queried/skipped-counted), matching
    what a caller doing per-batch calls needs (each call only "owns" its
    own batch's page numbers)."""
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Page one.", "Page two.", "Page three."])
    db = _make_session()
    document = _basic_document(db, page_count=3)
    storage = FakeObjectStorageClient()
    parse_result = DoclingParseResult(
        elements=[
            _element(local_id=1, page_number=1, text="Page one."),
            _element(local_id=2, page_number=2, text="Page two."),
            _element(local_id=3, page_number=3, text="Page three."),
        ],
        page_confidence={
            1: _page_confidence(1),
            2: _page_confidence(2),
            3: _page_confidence(3),
        },
        page_count=3,
    )

    summary = cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parse_result,
        page_range=(2, 2),
    )

    assert summary.pages_stored == 1
    assert summary.elements_stored == 1
    pages = db.execute(select(Page)).scalars().all()
    assert len(pages) == 1
    assert pages[0].page_number == 2
    element = db.execute(select(Element)).scalar_one()
    assert element.text == "Page two."


def test_store_pages_and_elements_local_id_to_db_id_carries_parent_across_calls(
    tmp_path: Path,
) -> None:
    """The exact cross-batch scenario `_run_batched_pipeline` relies on: a
    child element's `parent_local_id` refers to an element stored in an
    EARLIER call — passing the SAME `local_id_to_db_id` dict into both calls
    (as the orchestrator does across batches) must resolve `parent_id`
    correctly; using two SEPARATE (unshared) dicts must NOT (regression
    guard proving the carry is actually load-bearing, not incidental)."""
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Press the button icon.", "Continued icon legend."])
    db = _make_session()
    document = _basic_document(db, page_count=2)
    storage = FakeObjectStorageClient()

    parent_only = DoclingParseResult(
        elements=[_element(local_id=1, page_number=1, text="Press the button icon.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    # `parent_local_id=1` refers to local_id 1 from the FIRST call's
    # DoclingParseResult, not this one's (its own elements start at
    # local_id 2 — matching how `ParseCarryState.next_local_id` keeps
    # local_ids globally unique/monotonic across real batches).
    child_only = DoclingParseResult(
        elements=[
            _element(
                local_id=2, page_number=2, element_type="icon", text=None, parent_local_id=1
            )
        ],
        page_confidence={2: _page_confidence(2)},
        page_count=2,
    )

    shared_local_id_to_db_id: dict[int, int] = {}
    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parent_only,
        page_range=(1, 1),
        local_id_to_db_id=shared_local_id_to_db_id,
    )
    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=child_only,
        page_range=(2, 2),
        local_id_to_db_id=shared_local_id_to_db_id,
    )

    rows = db.execute(select(Element).order_by(Element.id)).scalars().all()
    parent_row, child_row = rows
    assert child_row.parent_id == parent_row.id  # correctly resolved cross-call


def test_store_pages_and_elements_without_shared_dict_orphans_cross_call_parent(
    tmp_path: Path,
) -> None:
    """Negative counterpart to the test above: WITHOUT carrying
    `local_id_to_db_id` across the two calls (a fresh, unshared dict each
    time — i.e. what happens if a caller forgets to carry it, or the
    pre-2026-08-11 behavior for a caller not passing it at all), the
    cross-call parent link canNOT be resolved and `parent_id` is `NULL`
    instead — this is what proves the carry in the test above is load-
    bearing, not a coincidence."""
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Press the button icon.", "Continued icon legend."])
    db = _make_session()
    document = _basic_document(db, page_count=2)
    storage = FakeObjectStorageClient()

    parent_only = DoclingParseResult(
        elements=[_element(local_id=1, page_number=1, text="Press the button icon.")],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    child_only = DoclingParseResult(
        elements=[
            _element(
                local_id=2, page_number=2, element_type="icon", text=None, parent_local_id=1
            )
        ],
        page_confidence={2: _page_confidence(2)},
        page_count=2,
    )

    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=parent_only,
        page_range=(1, 1),
        # no local_id_to_db_id -> fresh, discarded dict (old default behavior)
    )
    cs.store_pages_and_elements(
        db,
        storage,
        document=document,
        pdf_path=pdf_path,
        parse_result=child_only,
        page_range=(2, 2),
        # no local_id_to_db_id -> fresh dict again, has no memory of local_id 1
    )

    rows = db.execute(select(Element).order_by(Element.id)).scalars().all()
    _parent_row, child_row = rows
    assert child_row.parent_id is None


def test_store_pages_and_elements_expunges_created_objects_from_session(tmp_path: Path) -> None:
    """[2026-08-11, round 2 memory fix] Regression guard for the actual
    fix: `Page`/`Element` ORM objects created by this call must not remain
    resident in the session's identity map afterward — this is what makes
    peak memory O(batch) instead of O(document) under this project's
    `expire_on_commit=False` session config (`app/db/engine.py`; a plain
    `db.commit()` alone would NOT release them, see module docstring). Only
    `document` (never expunged — the caller keeps using it for the rest of
    the ingestion run) should remain in the identity map afterward."""
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, ["Press the button icon.", "Continued icon legend."])
    db = _make_session()
    document = _basic_document(db, page_count=2)
    storage = FakeObjectStorageClient()
    parse_result = DoclingParseResult(
        elements=[
            _element(local_id=1, page_number=1, text="Press the button icon."),
            _element(
                local_id=2,
                page_number=2,
                element_type="icon",
                text=None,
                parent_local_id=1,
            ),
        ],
        page_confidence={1: _page_confidence(1), 2: _page_confidence(2)},
        page_count=2,
    )

    cs.store_pages_and_elements(
        db, storage, document=document, pdf_path=pdf_path, parse_result=parse_result
    )

    identity_map_classes = {type(state.object) for state in db.identity_map.all_states()}
    assert identity_map_classes == {Document}
    assert document in db  # still attached — this function must not expunge it
