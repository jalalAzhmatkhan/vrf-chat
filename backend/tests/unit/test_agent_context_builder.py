"""Unit tests for `app/agent/context_builder.py` (C2.3) — the deterministic
marker-insertion Context Builder, §5.1 layer 1 of
`Documentation/system-design/05-streaming-and-api-contract.md`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent import context_builder as cb
from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.retrieval.hybrid_search import RetrievedChunk


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_document_and_page(db: Session, page_number: int = 238) -> tuple[Document, Page]:
    document = Document(title="Manual", filename="m.pdf", source_hash=f"hash-{page_number}")
    db.add(document)
    db.commit()
    page = Page(document_id=document.id, page_number=page_number)
    db.add(page)
    db.commit()
    return document, page


def _make_element(db: Session, page: Page, document: Document, **overrides: Any) -> Element:
    base = dict(
        document_id=document.id,
        page_id=page.id,
        element_type="paragraph",
        text="Some text.",
        source_hash="hash",
    )
    base.update(overrides)
    element = Element(**base)
    db.add(element)
    db.commit()
    return element


def _chunk(
    chunk_id: int,
    *,
    document_id: int,
    chunk_type: str,
    element_ids: list[int],
    content_text: str = "fallback text",
    page_start: int | None = 238,
    page_end: int | None = 238,
    section_path: list[str] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_type=chunk_type,
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        element_ids=element_ids,
        content_text=content_text,
        content_structured=None,
        model_family="REYQ",
        score=0.9,
        rank=1,
    )


# ---------------------------------------------------------------------------
# build_marker
# ---------------------------------------------------------------------------


def test_build_marker_format() -> None:
    assert cb.build_marker(4961) == "{{el:4961}}"


# ---------------------------------------------------------------------------
# fetch_context_elements
# ---------------------------------------------------------------------------


def test_fetch_context_elements_empty_list_returns_empty_dict() -> None:
    db = _make_session()
    assert cb.fetch_context_elements(db, []) == {}


def test_fetch_context_elements_maps_page_number() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db, page_number=42)
    element = _make_element(db, page, document, text="hello")

    result = cb.fetch_context_elements(db, [element.id])

    assert result[element.id].page == 42
    assert result[element.id].text == "hello"
    assert result[element.id].document_id == document.id


# ---------------------------------------------------------------------------
# build_context — text chunk with inline icon
# ---------------------------------------------------------------------------


def test_build_context_text_chunk_inserts_marker_for_icon() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    para = _make_element(db, page, document, element_type="paragraph", text="Press the button")
    icon = _make_element(db, page, document, element_type="icon", text=None)

    chunk = _chunk(
        1, document_id=document.id, chunk_type="text", element_ids=[para.id, icon.id]
    )
    context = cb.build_context(db, [chunk])

    assert len(context.chunks) == 1
    annotated = context.chunks[0].annotated_text
    assert "Press the button" in annotated
    assert cb.build_marker(icon.id) in annotated
    assert icon.id in context.elements_by_id
    assert para.id in context.elements_by_id


def test_build_context_icon_with_caption_text_included_after_marker() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    figure_caption = _make_element(
        db, page, document, element_type="figure_caption", text="Fig. 1 Display panel"
    )

    chunk = _chunk(2, document_id=document.id, chunk_type="text", element_ids=[figure_caption.id])
    context = cb.build_context(db, [chunk])

    annotated = context.chunks[0].annotated_text
    assert cb.build_marker(figure_caption.id) in annotated
    assert "Fig. 1 Display panel" in annotated


# ---------------------------------------------------------------------------
# build_context — chunk-type scoping
# ---------------------------------------------------------------------------


def test_build_context_figure_chunk_gets_single_marker() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    figure = _make_element(db, page, document, element_type="figure", text="Wiring diagram")

    chunk = _chunk(3, document_id=document.id, chunk_type="figure", element_ids=[figure.id])
    context = cb.build_context(db, [chunk])

    annotated = context.chunks[0].annotated_text
    assert annotated == f"{cb.build_marker(figure.id)}\nWiring diagram"


def test_build_context_table_chunk_not_annotated() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    table = _make_element(db, page, document, element_type="table", text="| a | b |")

    chunk = _chunk(
        4,
        document_id=document.id,
        chunk_type="table",
        element_ids=[table.id],
        content_text="| a | b |",
    )
    context = cb.build_context(db, [chunk])

    assert context.chunks[0].annotated_text == "| a | b |"
    assert cb.build_marker(table.id) not in context.chunks[0].annotated_text


def test_build_context_entity_chunk_not_annotated() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="paragraph", text="TH3 mentioned")

    chunk = _chunk(
        5,
        document_id=document.id,
        chunk_type="entity",
        element_ids=[element.id],
        content_text="TH3 (Sensor) — referenced on page(s) 238.",
    )
    context = cb.build_context(db, [chunk])

    assert context.chunks[0].annotated_text == "TH3 (Sensor) — referenced on page(s) 238."


# ---------------------------------------------------------------------------
# build_context — missing elements / empty fallback
# ---------------------------------------------------------------------------


def test_build_context_missing_element_skipped_falls_back_to_content_text() -> None:
    db = _make_session()
    document, _page = _seed_document_and_page(db)
    chunk = _chunk(
        6,
        document_id=document.id,
        chunk_type="text",
        element_ids=[99999],  # no such element row
        content_text="original content",
    )
    context = cb.build_context(db, [chunk])

    assert context.chunks[0].annotated_text == "original content"
    assert context.elements_by_id == {}


# ---------------------------------------------------------------------------
# build_context — max_chunks trimming
# ---------------------------------------------------------------------------


def test_build_context_trims_to_max_chunks() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    e1 = _make_element(db, page, document, text="one")
    e2 = _make_element(db, page, document, text="two")
    e3 = _make_element(db, page, document, text="three")
    chunks = [
        _chunk(1, document_id=document.id, chunk_type="text", element_ids=[e1.id]),
        _chunk(2, document_id=document.id, chunk_type="text", element_ids=[e2.id]),
        _chunk(3, document_id=document.id, chunk_type="text", element_ids=[e3.id]),
    ]

    context = cb.build_context(db, chunks, max_chunks=2)

    assert len(context.chunks) == 2
    assert e3.id not in context.elements_by_id


# ---------------------------------------------------------------------------
# build_context — header formatting
# ---------------------------------------------------------------------------


def test_build_context_header_single_page_and_section_path() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db, page_number=10)
    element = _make_element(db, page, document, text="hi")
    chunk = _chunk(
        1,
        document_id=document.id,
        chunk_type="text",
        element_ids=[element.id],
        page_start=10,
        page_end=10,
        section_path=["Chapter 7", "7.3 Troubleshooting"],
    )

    context = cb.build_context(db, [chunk])

    assert 'page=10 section="Chapter 7 > 7.3 Troubleshooting"' in context.context_text
    assert f"document_id={document.id}" in context.context_text


def test_build_context_header_multi_page_range() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db, page_number=10)
    element = _make_element(db, page, document, text="hi")
    chunk = _chunk(
        1,
        document_id=document.id,
        chunk_type="text",
        element_ids=[element.id],
        page_start=10,
        page_end=12,
        section_path=None,
    )

    context = cb.build_context(db, [chunk])

    assert "page=10-12" in context.context_text
    assert 'section=""' in context.context_text


def test_build_context_multiple_chunks_joined_with_separator() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    e1 = _make_element(db, page, document, text="one")
    e2 = _make_element(db, page, document, text="two")
    chunks = [
        _chunk(1, document_id=document.id, chunk_type="text", element_ids=[e1.id]),
        _chunk(2, document_id=document.id, chunk_type="text", element_ids=[e2.id]),
    ]

    context = cb.build_context(db, chunks)

    assert "---" in context.context_text
    assert context.context_text.count("document_id=") == 2


def test_build_context_empty_chunks_list() -> None:
    db = _make_session()
    context = cb.build_context(db, [])
    assert context.chunks == []
    assert context.context_text == ""
    assert context.elements_by_id == {}
