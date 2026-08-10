"""Unit tests for `app/agent/tools.py` (C2.3) — deterministic agent tools.

`app.agent.tools.search_documents` (the retrieval primitive, already fully
unit-tested in `tests/unit/test_retrieval_hybrid_search.py`) is monkeypatched
here with a fake that records call kwargs and returns a scripted
`RetrievalResult` — these tests are about tool-level orchestration
(argument wiring, context merging, fallback behavior, message selection),
not retrieval/fusion correctness itself.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent import tools as t
from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.db.models.error_codes import ErrorCode
from app.retrieval.hybrid_search import RetrievalResult, RetrievedChunk


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
    chunk_id: int, *, document_id: int, element_ids: list[int], chunk_type: str = "text"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_type=chunk_type,
        section_path=["Ch1"],
        page_start=238,
        page_end=238,
        element_ids=element_ids,
        content_text="fallback",
        content_structured=None,
        model_family="REYQ",
        score=0.9,
        rank=1,
    )


class FakeSearchDocuments:
    def __init__(self, responses: list[RetrievalResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        db: Any,
        qdrant_client: Any,
        dense_model: Any,
        sparse_model: Any,
        *,
        query: str,
        collection: str,
        model_family: str | None = None,
        error_code: str | None = None,
        component: str | None = None,
        chunk_type: str | None = None,
        top_k: int = 20,
        circuit_breaker_seconds: float = 8.0,
    ) -> RetrievalResult:
        self.calls.append(
            dict(
                query=query,
                collection=collection,
                model_family=model_family,
                error_code=error_code,
                component=component,
                chunk_type=chunk_type,
                top_k=top_k,
                circuit_breaker_seconds=circuit_breaker_seconds,
            )
        )
        return self.responses.pop(0)


def _deps(db: Session, *, model_family: str | None = None) -> t.AgentDeps:
    return t.AgentDeps(
        db=db,
        qdrant_client=object(),  # never dereferenced — search_documents is monkeypatched
        dense_model=object(),
        sparse_model=object(),
        collection="vrf_chunks",
        model_family=model_family,
    )


def _empty_result(query: str = "x") -> RetrievalResult:
    return RetrievalResult(query=query, effective_query=query, chunks=[], elapsed_ms=1)


def _circuit_breaker_result(query: str = "x") -> RetrievalResult:
    return RetrievalResult(
        query=query, effective_query=query, chunks=[], elapsed_ms=1, circuit_breaker_triggered=True
    )


# ---------------------------------------------------------------------------
# tool_search_documents
# ---------------------------------------------------------------------------


def test_tool_search_documents_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, text="Compressor info.")
    chunk = _chunk(1, document_id=document.id, element_ids=[element.id])
    fake = FakeSearchDocuments([RetrievalResult(query="q", effective_query="q", chunks=[chunk])])
    monkeypatch.setattr(t, "search_documents", fake)

    deps = _deps(db)
    output = t.tool_search_documents(deps, "compressor issue")

    assert "Compressor info." in output
    assert deps.tool_call_count == 1
    assert deps.any_chunks_retrieved is True
    assert element.id in deps.context_elements
    # query expansion detects "compressor" as a known component keyword and
    # folds it into the expanded query text (app/domain/query_expansion.py)
    assert fake.calls[0]["query"].startswith("compressor issue")


def test_tool_search_documents_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_documents(_deps(db), "unrelated question")

    assert output == t.NO_RESULTS_MESSAGE


def test_tool_search_documents_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_circuit_breaker_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_documents(_deps(db), "question")

    assert output == t.CIRCUIT_BREAKER_MESSAGE


def test_tool_search_documents_uses_deps_model_family_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    t.tool_search_documents(_deps(db, model_family="REYQ"), "question")

    assert fake.calls[0]["model_family"] == "REYQ"


def test_tool_search_documents_explicit_model_family_overrides_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    t.tool_search_documents(_deps(db, model_family="REYQ"), "question", model_family="OTHER")

    assert fake.calls[0]["model_family"] == "OTHER"


def test_tool_search_documents_folds_error_code_into_query(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    t.tool_search_documents(_deps(db), "what does it mean", error_code="P8")

    # error_code is passed through to search_documents (which folds it into
    # the effective query internally — see test_retrieval_hybrid_search.py);
    # tools.py itself doesn't need to duplicate that folding.
    assert fake.calls[0]["error_code"] == "P8"


# ---------------------------------------------------------------------------
# tool_search_error_code / _exact_error_code_lookup
# ---------------------------------------------------------------------------


def test_exact_error_code_lookup_no_rows_returns_empty_string() -> None:
    db = _make_session()
    deps = _deps(db)
    assert t._exact_error_code_lookup(deps, "P8", None) == ""


def test_exact_error_code_lookup_found_with_element(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="paragraph", text="Check the fan.")
    db.add(
        ErrorCode(
            document_id=document.id, code="P8", description="Fan error", element_id=element.id
        )
    )
    db.commit()

    deps = _deps(db)
    result = t._exact_error_code_lookup(deps, "P8", None)

    assert "Fan error" in result
    assert t.build_marker(element.id) not in result  # non-visual element type -> no marker
    assert deps.any_chunks_retrieved is True
    assert element.id in deps.context_elements


def test_exact_error_code_lookup_visual_element_gets_marker() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="figure", text="Diagram")
    db.add(ErrorCode(document_id=document.id, code="P8", description=None, element_id=element.id))
    db.commit()

    result = t._exact_error_code_lookup(_deps(db), "P8", None)

    assert t.build_marker(element.id) in result
    assert "Diagram" in result


def test_exact_error_code_lookup_filters_by_model_family() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document)
    db.add(
        ErrorCode(document_id=document.id, code="P8", model_family="OTHER", element_id=element.id)
    )
    db.commit()

    result = t._exact_error_code_lookup(_deps(db), "P8", "REYQ")
    assert result == ""


def test_exact_error_code_lookup_row_without_element_id_skipped_from_fetch_but_still_rendered() -> (
    None
):
    db = _make_session()
    document, _page = _seed_document_and_page(db)
    other_page = Page(document_id=document.id, page_number=1)
    db.add(other_page)
    db.commit()
    element = Element(
        document_id=document.id,
        page_id=other_page.id,
        element_type="paragraph",
        text="x",
        source_hash="h",
    )
    db.add(element)
    db.commit()
    db.add_all(
        [
            ErrorCode(
                document_id=document.id,
                code="P8",
                description="With element",
                element_id=element.id,
            ),
            ErrorCode(
                document_id=document.id, code="P8", description="No element", element_id=None
            ),
        ]
    )
    db.commit()

    result = t._exact_error_code_lookup(_deps(db), "P8", None)

    assert "With element" in result
    assert "No element" in result


def test_tool_search_error_code_combines_exact_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    exact_element = _make_element(db, page, document, element_type="paragraph", text="Exact hit.")
    db.add(
        ErrorCode(
            document_id=document.id, code="P8", description="desc", element_id=exact_element.id
        )
    )
    db.commit()

    search_element = _make_element(db, page, document, text="Search hit.")
    chunk = _chunk(1, document_id=document.id, element_ids=[search_element.id])
    fake = FakeSearchDocuments([RetrievalResult(query="q", effective_query="q", chunks=[chunk])])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_error_code(_deps(db), "P8")

    assert "Exact hit." in output or "desc" in output
    assert "Search hit." in output
    assert "---" in output


def test_tool_search_error_code_exact_only_when_search_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document)
    db.add(
        ErrorCode(
            document_id=document.id, code="P8", description="only exact", element_id=element.id
        )
    )
    db.commit()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_error_code(_deps(db), "P8")

    assert output.strip().endswith("only exact") or "only exact" in output
    assert "---" not in output


def test_tool_search_error_code_search_only_when_no_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, text="Search only.")
    chunk = _chunk(1, document_id=document.id, element_ids=[element.id])
    fake = FakeSearchDocuments([RetrievalResult(query="q", effective_query="q", chunks=[chunk])])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_error_code(_deps(db), "U4")

    assert "Search only." in output
    assert "---" not in output


def test_exact_error_code_lookup_dangling_element_id_not_in_db() -> None:
    db = _make_session()
    document, _page = _seed_document_and_page(db)
    db.add(ErrorCode(document_id=document.id, code="P8", description="dangling", element_id=99999))
    db.commit()

    result = t._exact_error_code_lookup(_deps(db), "P8", None)

    assert "dangling" in result
    assert t.build_marker(99999) not in result


def test_tool_search_error_code_neither_found(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_search_error_code(_deps(db), "U4")

    assert output == t.NO_RESULTS_MESSAGE


# ---------------------------------------------------------------------------
# tool_find_component
# ---------------------------------------------------------------------------


def test_tool_find_component_passes_component_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    t.tool_find_component(_deps(db), "compressor")

    assert fake.calls[0]["component"] == "compressor"


# ---------------------------------------------------------------------------
# tool_find_troubleshooting_procedure
# ---------------------------------------------------------------------------


def test_tool_find_troubleshooting_procedure_found(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, text="Step 1: check compressor.")
    chunk = _chunk(1, document_id=document.id, element_ids=[element.id], chunk_type="procedure")
    fake = FakeSearchDocuments([RetrievalResult(query="q", effective_query="q", chunks=[chunk])])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_troubleshooting_procedure(_deps(db), "outdoor unit mati")

    assert "Step 1" in output
    assert fake.calls[0]["chunk_type"] == "procedure"
    assert len(fake.calls) == 1  # no fallback needed


def test_tool_find_troubleshooting_procedure_falls_back_to_general_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, text="General guidance.")
    chunk = _chunk(1, document_id=document.id, element_ids=[element.id], chunk_type="text")
    fake = FakeSearchDocuments(
        [
            _empty_result("procedure search"),
            RetrievalResult(query="q", effective_query="q", chunks=[chunk]),
        ]
    )
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_troubleshooting_procedure(_deps(db), "outdoor unit mati")

    assert "General guidance." in output
    assert len(fake.calls) == 2
    assert fake.calls[0]["chunk_type"] == "procedure"
    assert fake.calls[1]["chunk_type"] is None


def test_tool_find_troubleshooting_procedure_circuit_breaker_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_circuit_breaker_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_troubleshooting_procedure(_deps(db), "outdoor unit mati")

    assert output == t.CIRCUIT_BREAKER_MESSAGE
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# tool_find_wiring_diagram
# ---------------------------------------------------------------------------


def test_tool_find_wiring_diagram_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="figure", text="Wiring schematic.")
    chunk = _chunk(1, document_id=document.id, element_ids=[element.id], chunk_type="figure")
    fake = FakeSearchDocuments([RetrievalResult(query="q", effective_query="q", chunks=[chunk])])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_wiring_diagram(_deps(db), "compressor")

    assert "Wiring schematic." in output
    assert fake.calls[0]["chunk_type"] == "figure"
    assert fake.calls[0]["component"] == "compressor"


def test_tool_find_wiring_diagram_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_empty_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_wiring_diagram(_deps(db), "compressor")
    assert output == t.NO_RESULTS_MESSAGE


def test_tool_find_wiring_diagram_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    fake = FakeSearchDocuments([_circuit_breaker_result()])
    monkeypatch.setattr(t, "search_documents", fake)

    output = t.tool_find_wiring_diagram(_deps(db), "compressor")
    assert output == t.CIRCUIT_BREAKER_MESSAGE


# ---------------------------------------------------------------------------
# tool_get_document_page
# ---------------------------------------------------------------------------


def test_tool_get_document_page_not_found() -> None:
    db = _make_session()
    output = t.tool_get_document_page(_deps(db), 999, 1)
    assert output == t.NOT_FOUND_PAGE_MESSAGE


def test_tool_get_document_page_found_with_icon() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db, page_number=5)
    para = _make_element(db, page, document, element_type="paragraph", text="Press start")
    icon = _make_element(db, page, document, element_type="icon", text=None)

    deps = _deps(db)
    output = t.tool_get_document_page(deps, document.id, 5)

    assert "Press start" in output
    assert t.build_marker(icon.id) in output
    assert deps.any_chunks_retrieved is True
    assert para.id in deps.context_elements
    assert icon.id in deps.context_elements


def test_tool_get_document_page_skips_element_missing_from_fetch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive branch: an element_id queried from `elements` but absent
    from `fetch_context_elements`'s result (e.g. deleted between the two
    queries) is skipped, not treated as a crash."""
    db = _make_session()
    document, page = _seed_document_and_page(db, page_number=5)
    missing_element = _make_element(db, page, document, text="Press start")
    present_element = _make_element(db, page, document, text="Second element")

    original_fetch = t.fetch_context_elements

    def _fetch_missing_one(db_arg: Any, element_ids: list[int]) -> dict[int, Any]:
        result = original_fetch(db_arg, element_ids)
        result.pop(missing_element.id, None)
        return result

    monkeypatch.setattr(t, "fetch_context_elements", _fetch_missing_one)

    output = t.tool_get_document_page(_deps(db), document.id, 5)

    assert "Press start" not in output
    assert "Second element" in output
    assert present_element.id is not None


def test_tool_get_document_page_no_elements_on_page() -> None:
    db = _make_session()
    document, _page = _seed_document_and_page(db, page_number=7)
    output = t.tool_get_document_page(_deps(db), document.id, 7)
    assert output == t.NOT_FOUND_PAGE_MESSAGE


# ---------------------------------------------------------------------------
# tool_get_figure
# ---------------------------------------------------------------------------


def test_tool_get_figure_not_found() -> None:
    db = _make_session()
    output = t.tool_get_figure(_deps(db), 999)
    assert output == t.NOT_FOUND_FIGURE_MESSAGE


def test_tool_get_figure_found_with_text() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="figure", text="A wiring diagram.")

    deps = _deps(db)
    output = t.tool_get_figure(deps, element.id)

    assert t.build_marker(element.id) in output
    assert "A wiring diagram." in output
    assert deps.any_chunks_retrieved is True


def test_tool_get_figure_uses_visual_description_when_no_text() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(
        db,
        page,
        document,
        element_type="figure",
        text=None,
        visual_description={"description": "A high-pressure switch diagram."},
    )

    output = t.tool_get_figure(_deps(db), element.id)

    assert "A high-pressure switch diagram." in output


def test_tool_get_figure_no_text_no_description() -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    element = _make_element(db, page, document, element_type="figure", text=None)

    output = t.tool_get_figure(_deps(db), element.id)

    assert output == t.build_marker(element.id)


# ---------------------------------------------------------------------------
# tool_search_knowledge_graph
# ---------------------------------------------------------------------------


def test_tool_search_knowledge_graph_stub() -> None:
    db = _make_session()
    deps = _deps(db)

    output = t.tool_search_knowledge_graph(deps, "TH3", relation="CONNECTED_TO")

    assert output == t.KG_STUB_MESSAGE
    assert deps.tool_call_count == 1
