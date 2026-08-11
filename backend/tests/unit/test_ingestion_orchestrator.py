"""Unit tests for `app/ingestion/orchestrator.py` (I1.10).

`run_ingestion_pipeline` is fully unit-tested with EVERY heavy/GPU
component (DoclingParser, PaddleOCR-VL client, Qdrant client, embedding
models) injected as a fake — no GPU/model/network dependency, same pattern
established throughout I1.1-I1.9. A small real synthetic PDF (PyMuPDF
authoring API) provides a real `pdf_path` for `render_bbox_crop`/
`compute_page_hash` calls inside the real (non-faked) canonical_store/
chunker/cascade_trigger logic this orchestrates.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pymupdf
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.db.models.ingestion_jobs import IngestionJob
from app.ingestion import docling_parser as dp
from app.ingestion import orchestrator as orch
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft, PageConfidence
from app.ingestion.paddleocr_vl_cascade import (
    OCRPageResult,
    TableReparseResult,
    VisualDescription,
)
from tests.unit.test_ingestion_docling_parser import (
    FakeDoc,
    _confidence_report,
    _item,
    _page_scores,
)


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


class FakeObjectStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, key: str, data: bytes, content_type: str) -> str:
        uri = f"fake://{key}"
        self.objects[uri] = data
        return uri

    def get_object(self, uri: str) -> bytes:
        return self.objects[uri]

    def get_presigned_url(self, uri: str, expires_in_seconds: int = 3600) -> str:
        return uri

    def delete_object(self, uri: str) -> None:  # pragma: no cover - unused
        self.objects.pop(uri, None)

    def exists(self, uri: str) -> bool:  # pragma: no cover - unused
        return uri in self.objects


def _write_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "Check the compressor for TH3.")
    doc.save(str(path))
    doc.close()


def _write_multi_page_pdf(path: Path, num_pages: int) -> None:
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page(width=600, height=800)
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), f"Page {i + 1} content.")
    doc.save(str(path))
    doc.close()


def _page_confidence(page_number: int = 1, parse_score: float | None = 1.0) -> PageConfidence:
    return PageConfidence(
        page_number=page_number,
        parse_score=parse_score,
        layout_score=0.95,
        table_score=0.5,  # low -> triggers table_reparse cascade task
        ocr_score=None,
        mean_score=0.9,
        mean_grade="excellent",
    )


def _result_for_page(page_number: int) -> DoclingParseResult:
    """A minimal one-element `DoclingParseResult`, used by
    `_run_docling_in_batches` tests to fake each batch returning a distinct
    per-batch result (so the merge logic is actually exercised, not just
    the same canned object N times)."""
    return DoclingParseResult(
        elements=[
            ElementDraft(
                local_id=1,
                element_type="paragraph",
                text=f"page {page_number}",
                bbox=None,
                page_number=page_number,
                parent_local_id=None,
                section_path=[],
                extraction_method="docling",
                extraction_confidence=1.0,
            )
        ],
        page_confidence={page_number: _page_confidence(page_number)},
        page_count=page_number,
    )


class FakeDoclingParser:
    """[2026-08-11 page-range batching] `parse()` now accepts `page_range`/
    `carry_state` kwargs (see `app/ingestion/orchestrator.py`
    `_run_docling_in_batches`) — this fake records every `page_range` it
    was called with (`parse_page_ranges`) so orchestration wiring (correct
    number of batches, correct page windows) can be asserted without a real
    Docling model."""

    def __init__(self, parse_result: DoclingParseResult) -> None:
        self._parse_result = parse_result
        self.is_loaded = True
        self.parse_calls = 0
        self.unload_calls = 0
        self.parse_page_ranges: list[tuple[int, int] | None] = []
        self.parse_carry_states: list[Any] = []

    def parse(
        self,
        pdf_path: Any,
        *,
        page_range: tuple[int, int] | None = None,
        carry_state: Any = None,
    ) -> DoclingParseResult:
        self.parse_calls += 1
        self.parse_page_ranges.append(page_range)
        self.parse_carry_states.append(carry_state)
        return self._parse_result

    def unload(self) -> None:
        self.unload_calls += 1
        self.is_loaded = False


class FakePaddleClient:
    def __init__(self) -> None:
        self.unload_calls = 0

    def describe_figure(self, image_png: bytes) -> VisualDescription:
        return VisualDescription(description="A figure")

    def reparse_table(self, image_png: bytes) -> TableReparseResult:
        return TableReparseResult(markdown="| corrected |")

    def ocr_page(self, image_png: bytes) -> OCRPageResult:  # pragma: no cover - not triggered here
        return OCRPageResult(text="ocr")

    def unload(self) -> None:
        self.unload_calls += 1


class FakeQdrantClient:
    def __init__(self) -> None:
        self._exists = False
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def collection_exists(self, collection: str) -> bool:
        return self._exists

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)
        self._exists = True

    def upsert(self, collection_name: str, points: Any) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})

    def delete(self, collection_name: str, points_selector: Any) -> None:
        self.delete_calls.append(
            {"collection_name": collection_name, "points_selector": points_selector}
        )


class FakeDenseModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeSparseModel:
    def embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        from app.ingestion.embedder import SparseVectorData

        return [SparseVectorData(indices=[1], values=[1.0]) for _ in texts]


def _settings(**overrides: Any):  # type: ignore[no-untyped-def]
    from app.core.config import Settings

    return Settings(_env_file=None, **overrides)


def _parse_result_for(pdf_path: Path) -> DoclingParseResult:
    return DoclingParseResult(
        elements=[
            ElementDraft(
                local_id=1,
                element_type="paragraph",
                text="Check the compressor for TH3.",
                bbox=None,
                page_number=1,
                parent_local_id=None,
                section_path=["Ch1"],
                extraction_method="docling",
                extraction_confidence=1.0,
            ),
            ElementDraft(
                local_id=2,
                element_type="table",
                text="| a | b |",
                bbox={"l": 10.0, "t": 700.0, "r": 100.0, "b": 650.0},
                page_number=1,
                parent_local_id=None,
                section_path=["Ch1"],
                extraction_method="docling",
                extraction_confidence=0.5,
            ),
        ],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )


@pytest.fixture
def pipeline_env(tmp_path: Path):  # type: ignore[no-untyped-def]
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path)
    db = _make_session()
    storage = FakeObjectStorageClient()

    document, created = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_path.read_bytes(),
        filename="manual.pdf",
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        document_hash="hash-1",
        page_count=1,
    )
    assert created is True

    parse_result = _parse_result_for(pdf_path)
    docling_parser = FakeDoclingParser(parse_result)
    paddle_client = FakePaddleClient()
    qdrant_client = FakeQdrantClient()

    return {
        "db": db,
        "storage": storage,
        "document": document,
        "pdf_path": pdf_path,
        "docling_parser": docling_parser,
        "paddle_client": paddle_client,
        "qdrant_client": qdrant_client,
    }


# ---------------------------------------------------------------------------
# prepare_document_for_ingestion / materialize_source_pdf
# ---------------------------------------------------------------------------


def test_prepare_document_for_ingestion_uploads_pdf_and_sets_uri(tmp_path: Path) -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    document, created = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=b"%PDF-1.4 fake",
        filename="manual.pdf",
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        document_hash="hash-x",
        page_count=10,
    )
    assert created is True
    assert document.source_pdf_uri == f"fake://documents/{document.id}/source.pdf"
    assert storage.objects[document.source_pdf_uri] == b"%PDF-1.4 fake"


def test_prepare_document_for_ingestion_idempotent() -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    kwargs = dict(
        pdf_bytes=b"data",
        filename="manual.pdf",
        title="Manual",
        manufacturer=None,
        model_family=None,
        document_hash="hash-y",
        page_count=1,
    )
    doc1, created1 = orch.prepare_document_for_ingestion(db, storage, **kwargs)  # type: ignore[arg-type]
    doc2, created2 = orch.prepare_document_for_ingestion(db, storage, **kwargs)  # type: ignore[arg-type]

    assert created1 is True
    assert created2 is False
    assert doc1.id == doc2.id
    assert len(storage.objects) == 1  # not re-uploaded


def test_materialize_source_pdf_writes_temp_file() -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    document, _ = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=b"pdf-content",
        filename="manual.pdf",
        title="Manual",
        manufacturer=None,
        model_family=None,
        document_hash="hash-z",
        page_count=1,
    )

    path = orch.materialize_source_pdf(db, storage, document.id)
    try:
        assert path.read_bytes() == b"pdf-content"
    finally:
        path.unlink()


def test_materialize_source_pdf_missing_document_raises() -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    with pytest.raises(ValueError, match="not found|source_pdf_uri"):
        orch.materialize_source_pdf(db, storage, 999)


def test_materialize_source_pdf_no_uri_raises() -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    document = Document(
        title="No URI", filename="x.pdf", source_hash="h", page_count=1, status="queued"
    )
    db.add(document)
    db.commit()

    with pytest.raises(ValueError, match="source_pdf_uri"):
        orch.materialize_source_pdf(db, storage, document.id)


# ---------------------------------------------------------------------------
# run_ingestion_pipeline — full orchestration
# ---------------------------------------------------------------------------


def test_run_ingestion_pipeline_full_success(pipeline_env: dict[str, Any]) -> None:
    result = orch.run_ingestion_pipeline(
        pipeline_env["db"],
        pipeline_env["storage"],
        _settings(),
        document_id=pipeline_env["document"].id,
        pdf_path=pipeline_env["pdf_path"],
        celery_task_id="task-123",
        docling_parser=pipeline_env["docling_parser"],
        paddle_client=pipeline_env["paddle_client"],
        qdrant_client=pipeline_env["qdrant_client"],
        dense_model=FakeDenseModel(),
        sparse_model=FakeSparseModel(),
    )

    assert result.document_id == pipeline_env["document"].id
    assert result.pages_stored == 1
    assert result.elements_stored == 2
    assert result.chunks_stored >= 1
    assert result.chunks_embedded == result.chunks_stored
    assert result.cascade_task_count == 1  # the low-confidence table

    # [2026-08-11 page-range batching] a 1-page document fits in a single
    # batch (default INGESTION_PAGE_BATCH_SIZE=100) — Docling is still
    # called exactly once, with an explicit page_range derived from Stage
    # 1's probe, not the old no-page_range call shape.
    assert pipeline_env["docling_parser"].parse_calls == 1
    assert pipeline_env["docling_parser"].parse_page_ranges == [(1, 1)]

    db = pipeline_env["db"]
    document = db.get(Document, pipeline_env["document"].id)
    assert document.status == "ready"

    jobs = db.execute(select(IngestionJob).order_by(IngestionJob.id)).scalars().all()
    stages = [job.stage for job in jobs]
    assert stages == [
        orch.STAGE_NATIVE_PROBE,
        orch.STAGE_DOCLING,
        orch.STAGE_PADDLE_CASCADE,
        orch.STAGE_KG_CANDIDATE,
        orch.STAGE_CHUNKING,
        orch.STAGE_EMBEDDING,
    ]
    assert all(job.status == "done" for job in jobs)
    assert all(job.celery_task_id == "task-123" for job in jobs)
    assert all(job.stage_metrics is not None for job in jobs)

    # Docling and PaddleOCR-VL both explicitly unloaded.
    assert pipeline_env["docling_parser"].unload_calls == 1
    assert pipeline_env["paddle_client"].unload_calls == 1

    # Table's markdown was corrected via the (faked) cascade result.
    assert pipeline_env["qdrant_client"].upsert_calls  # embedding actually ran

    # [2026-08-10 operational-habit correction] Every embedding run does a
    # document-scoped Qdrant cleanup FIRST (a no-op here, first-time
    # ingest) — never a full collection drop. See
    # app/ingestion/embedder.py delete_chunks_by_document docstring.
    assert len(pipeline_env["qdrant_client"].delete_calls) == 1
    delete_call = pipeline_env["qdrant_client"].delete_calls[0]
    assert delete_call["collection_name"] == "vrf_chunks"
    selector = delete_call["points_selector"]
    assert selector.filter.must[0].key == "document_id"
    assert selector.filter.must[0].match.value == pipeline_env["document"].id


def test_run_ingestion_pipeline_document_not_found_raises() -> None:
    db = _make_session()
    storage = FakeObjectStorageClient()
    with pytest.raises(ValueError, match="not found"):
        orch.run_ingestion_pipeline(
            db,
            storage,
            _settings(),
            document_id=999,
            pdf_path="dummy.pdf",
        )


def test_run_ingestion_pipeline_stage_failure_marks_job_and_document_failed(
    pipeline_env: dict[str, Any],
) -> None:
    class BrokenDoclingParser(FakeDoclingParser):
        def parse(
            self,
            pdf_path: Any,
            *,
            page_range: tuple[int, int] | None = None,
            carry_state: Any = None,
        ) -> DoclingParseResult:
            raise RuntimeError("docling exploded")

    broken_parser = BrokenDoclingParser(pipeline_env["docling_parser"]._parse_result)

    with pytest.raises(RuntimeError, match="docling exploded"):
        orch.run_ingestion_pipeline(
            pipeline_env["db"],
            pipeline_env["storage"],
            _settings(),
            document_id=pipeline_env["document"].id,
            pdf_path=pipeline_env["pdf_path"],
            docling_parser=broken_parser,
            paddle_client=pipeline_env["paddle_client"],
            qdrant_client=pipeline_env["qdrant_client"],
            dense_model=FakeDenseModel(),
            sparse_model=FakeSparseModel(),
        )

    db = pipeline_env["db"]
    document = db.get(Document, pipeline_env["document"].id)
    assert document.status == "failed"

    jobs = db.execute(select(IngestionJob).order_by(IngestionJob.id)).scalars().all()
    assert jobs[-1].stage == orch.STAGE_DOCLING
    assert jobs[-1].status == "failed"
    assert "docling exploded" in jobs[-1].error_message


# ---------------------------------------------------------------------------
# _run_batched_pipeline (2026-08-11, round 2 — parse -> cascade -> store ->
# release PER BATCH, replacing round 1's Docling-only batching that still
# OOM'd on the real 567-page document; see orchestrator.py module docstring)
# ---------------------------------------------------------------------------


def test_run_ingestion_pipeline_rejects_non_positive_batch_size(
    pipeline_env: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="page_batch_size"):
        orch.run_ingestion_pipeline(
            pipeline_env["db"],
            pipeline_env["storage"],
            _settings(INGESTION_PAGE_BATCH_SIZE=0),
            document_id=pipeline_env["document"].id,
            pdf_path=pipeline_env["pdf_path"],
            docling_parser=pipeline_env["docling_parser"],
            paddle_client=pipeline_env["paddle_client"],
            qdrant_client=pipeline_env["qdrant_client"],
            dense_model=FakeDenseModel(),
            sparse_model=FakeSparseModel(),
        )


def test_run_batched_pipeline_rejects_non_positive_total_pages(tmp_path: Path) -> None:
    """`total_pages` (from Stage 1's probe) is validated too, not just
    `page_batch_size` — exercised directly against `_run_batched_pipeline`
    with a stub `probe`, since a real `probe_document()` on a real PDF can
    never itself report 0 pages."""
    pdf_path = tmp_path / "x.pdf"
    _write_pdf(pdf_path)
    db = _make_session()
    storage = FakeObjectStorageClient()
    document, _ = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_path.read_bytes(),
        filename="x.pdf",
        title="X",
        manufacturer=None,
        model_family=None,
        document_hash="hash-zero-pages",
        page_count=0,
    )
    stages = orch._StageRunner(db, document.id, None)
    probe = SimpleNamespace(page_count=0)
    parser = FakeDoclingParser(_parse_result_for(pdf_path))

    with pytest.raises(ValueError, match="total_pages"):
        orch._run_batched_pipeline(
            db,
            storage,
            _settings(),
            document=document,
            pdf_path=pdf_path,
            probe=probe,
            parser=parser,
            stages=stages,
            paddle_client=FakePaddleClient(),
        )


def test_run_ingestion_pipeline_builds_paddle_client_lazily_when_not_injected(
    tmp_path: Path,
) -> None:
    """When `paddle_client` is not injected, `_run_batched_pipeline` builds
    one lazily via the real (unmocked) `build_paddleocr_vl_client(settings)`
    — safe to exercise for real here because construction itself is cheap/
    lazy (`LocalPaddleOCRVLClient.__init__` only validates config, no GPU/
    model load) AND this test's content has no table/figure/low-confidence
    page, so the resulting cascade plan is empty and no actual inference
    method is ever called on the real client."""
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path)
    db = _make_session()
    storage = FakeObjectStorageClient()
    document, _ = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_path.read_bytes(),
        filename="manual.pdf",
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        document_hash="hash-lazy-client",
        page_count=1,
    )
    items_by_page = {1: [_item("text", "#/texts/0", text="Check the compressor.", page_no=1)]}
    page_sizes = {1: (609.0, 793.0)}
    parser = _RealishDoclingParser(items_by_page, page_sizes)

    result = orch.run_ingestion_pipeline(
        db,
        storage,
        _settings(),
        document_id=document.id,
        pdf_path=pdf_path,
        docling_parser=parser,
        # paddle_client intentionally omitted -> forces the real lazy
        # build_paddleocr_vl_client(settings) path.
        qdrant_client=FakeQdrantClient(),
        dense_model=FakeDenseModel(),
        sparse_model=FakeSparseModel(),
    )

    assert result.cascade_task_count == 0


def test_run_ingestion_pipeline_uses_configured_page_batch_size(
    pipeline_env: dict[str, Any],
) -> None:
    """`INGESTION_PAGE_BATCH_SIZE` is read from `settings`, not hardcoded —
    a batch size of 1 against the 1-page fixture still yields exactly one
    (1, 1) batch (same as the default-settings case), proving the value is
    actually threaded from `Settings` into `_run_batched_pipeline`."""
    orch.run_ingestion_pipeline(
        pipeline_env["db"],
        pipeline_env["storage"],
        _settings(INGESTION_PAGE_BATCH_SIZE=1),
        document_id=pipeline_env["document"].id,
        pdf_path=pipeline_env["pdf_path"],
        docling_parser=pipeline_env["docling_parser"],
        paddle_client=pipeline_env["paddle_client"],
        qdrant_client=pipeline_env["qdrant_client"],
        dense_model=FakeDenseModel(),
        sparse_model=FakeSparseModel(),
    )

    assert pipeline_env["docling_parser"].parse_page_ranges == [(1, 1)]


class _RealishDoclingParser:
    """Wraps the REAL `docling_parser.map_document_to_elements` (imported
    as `dp`) over a fixed whole-document item list (`items_by_page`),
    filtering down to whichever `page_range` a given call requests — unlike
    `FakeDoclingParser` above (which always returns one canned result
    regardless of `page_range`), this exercises the ACTUAL production
    mapping/`ParseCarryState` logic through the full `run_ingestion_
    pipeline` (real DB persistence), giving the strongest possible
    batched-vs-unbatched equivalence proof at this layer — see
    `test_ingestion_docling_parser.py`'s round-1
    `test_batched_parse_matches_single_pass_*` for the analogous proof at
    the mapping-logic-only layer this builds on."""

    def __init__(
        self,
        items_by_page: dict[int, list[Any]],
        page_sizes: dict[int, tuple[float, float]],
    ) -> None:
        self._items_by_page = items_by_page
        self._page_sizes = page_sizes
        self.unload_calls = 0
        self.is_loaded = True
        self.parse_calls = 0
        self.parse_page_ranges: list[tuple[int, int] | None] = []

    def parse(
        self,
        pdf_path: Any,
        *,
        page_range: tuple[int, int] | None = None,
        carry_state: Any = None,
    ) -> DoclingParseResult:
        self.parse_calls += 1
        self.parse_page_ranges.append(page_range)
        start, end = page_range if page_range is not None else (1, max(self._items_by_page))
        items: list[Any] = []
        for page_no in range(start, end + 1):
            items.extend(self._items_by_page.get(page_no, []))
        page_sizes = {
            p: self._page_sizes[p] for p in range(start, end + 1) if p in self._page_sizes
        }
        doc = FakeDoc(items, page_sizes=page_sizes)
        confidence = _confidence_report({p: _page_scores() for p in range(start, end + 1)})
        return dp.map_document_to_elements(
            doc, confidence, end, page_range=(start, end), carry_state=carry_state
        )

    def unload(self) -> None:
        self.unload_calls += 1
        self.is_loaded = False


def _icon_fallback_items_and_sizes() -> tuple[dict[int, list[Any]], dict[int, tuple[float, float]]]:
    """The exact SA1.2 scenario (`docling_parser.py`/`test_ingestion_
    docling_parser.py` round-1 tests): page 1 has a heading + paragraph,
    page 2 has ONLY icons (no text/heading of its own) that must fall back
    to the page-1 paragraph as their parent."""
    items_by_page = {
        1: [
            _item("section_header", "#/texts/0", text="Pictograms", page_no=1, level=1),
            _item("text", "#/texts/1", text="See legend below.", page_no=1),
        ],
        2: [
            _item("picture", "#/pictures/0", page_no=2, bbox=(10, 780, 30, 765)),
            _item("picture", "#/pictures/1", page_no=2, bbox=(40, 780, 60, 765)),
        ],
    }
    page_sizes = {1: (609.0, 793.0), 2: (609.0, 793.0)}
    return items_by_page, page_sizes


def test_run_ingestion_pipeline_multi_batch_job_counts_and_order(tmp_path: Path) -> None:
    """2 batches (batch size 1 against a 2-page document) -> `docling`/
    `paddle_cascade`/`kg_candidate` each get 2 `ingestion_jobs` rows (one
    per batch), interleaved per batch, NOT grouped by stage — `native_probe`
    /`chunking`/`embedding` stay at exactly 1 row each (unchanged, whole-
    document stages). Docling is `unload()`ed once per batch (2 total);
    the PaddleOCR-VL client is still `unload()`ed exactly once overall
    (unchanged from round 1/pre-batching — see `_run_batched_pipeline`
    docstring for why its lifecycle didn't need to change)."""
    pdf_path = tmp_path / "manual.pdf"
    _write_multi_page_pdf(pdf_path, 2)
    db = _make_session()
    storage = FakeObjectStorageClient()
    document, _ = orch.prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_path.read_bytes(),
        filename="manual.pdf",
        title="Manual",
        manufacturer="Zeggo",
        model_family="REYQ",
        document_hash="hash-multi",
        page_count=2,
    )
    items_by_page, page_sizes = _icon_fallback_items_and_sizes()
    parser = _RealishDoclingParser(items_by_page, page_sizes)
    paddle_client = FakePaddleClient()

    orch.run_ingestion_pipeline(
        db,
        storage,
        _settings(INGESTION_PAGE_BATCH_SIZE=1),
        document_id=document.id,
        pdf_path=pdf_path,
        docling_parser=parser,
        paddle_client=paddle_client,
        qdrant_client=FakeQdrantClient(),
        dense_model=FakeDenseModel(),
        sparse_model=FakeSparseModel(),
    )

    jobs = db.execute(select(IngestionJob).order_by(IngestionJob.id)).scalars().all()
    stages = [job.stage for job in jobs]
    assert stages == [
        orch.STAGE_NATIVE_PROBE,
        orch.STAGE_DOCLING,
        orch.STAGE_PADDLE_CASCADE,
        orch.STAGE_KG_CANDIDATE,
        orch.STAGE_DOCLING,
        orch.STAGE_PADDLE_CASCADE,
        orch.STAGE_KG_CANDIDATE,
        orch.STAGE_CHUNKING,
        orch.STAGE_EMBEDDING,
    ]
    assert all(job.status == "done" for job in jobs)

    assert parser.parse_calls == 2
    assert parser.parse_page_ranges == [(1, 1), (2, 2)]
    assert parser.unload_calls == 2  # once per batch (round 2 change from round 1's once-total)
    assert paddle_client.unload_calls == 1  # unchanged — see docstring


def test_run_ingestion_pipeline_batched_matches_unbatched_icon_parent_across_batches(
    tmp_path: Path,
) -> None:
    """End-to-end equivalence proof (round 2, `CLAUDE.md` §4's most
    critical requirement): the same SA1.2 icon-fallback-across-pages
    scenario run TWICE — once with a batch size that puts the heading/
    paragraph and the icon-only page in DIFFERENT batches (size 1 -> 2
    batches), once with a batch size that puts everything in ONE batch
    (size 100) — and asserts the PERSISTED `elements` rows (real DB query,
    not just in-memory dataclasses like the round-1 docling_parser-level
    equivalence tests) are structurally identical either way, including the
    icons' `parent_id` correctly resolving to the page-1 paragraph even
    when that paragraph was stored in an EARLIER BATCH (proving
    `local_id_to_db_id` carries correctly through `store_pages_and_elements`
    across batch boundaries, not just `ParseCarryState` through parsing)."""
    items_by_page, page_sizes = _icon_fallback_items_and_sizes()

    def _run(batch_size: int) -> tuple[list[tuple[str, str | None, int]], list[int | None]]:
        pdf_path = tmp_path / f"doc_{batch_size}.pdf"
        _write_multi_page_pdf(pdf_path, 2)
        db = _make_session()
        storage = FakeObjectStorageClient()
        document, _ = orch.prepare_document_for_ingestion(
            db,
            storage,
            pdf_bytes=pdf_path.read_bytes(),
            filename="manual.pdf",
            title="Manual",
            manufacturer="Zeggo",
            model_family="REYQ",
            document_hash=f"hash-{batch_size}",
            page_count=2,
        )
        parser = _RealishDoclingParser(items_by_page, page_sizes)
        orch.run_ingestion_pipeline(
            db,
            storage,
            _settings(INGESTION_PAGE_BATCH_SIZE=batch_size),
            document_id=document.id,
            pdf_path=pdf_path,
            docling_parser=parser,
            paddle_client=FakePaddleClient(),
            qdrant_client=FakeQdrantClient(),
            dense_model=FakeDenseModel(),
            sparse_model=FakeSparseModel(),
        )
        rows = db.execute(
            select(Element, Page.page_number)
            .join(Page, Element.page_id == Page.id)
            .where(Element.document_id == document.id)
            .order_by(Page.page_number, Element.id)
        ).all()
        by_id = {element.id: element for element, _ in rows}
        structure = [(element.element_type, element.text, page_no) for element, page_no in rows]
        # Resolve each icon's parent to a STRUCTURAL position (the parent's
        # own text), not a raw db id — ids are meaningless to compare
        # across two separate DB sessions/documents.
        icon_parent_texts = [
            by_id[element.parent_id].text if element.parent_id is not None else None
            for element, _ in rows
            if element.element_type == "icon"
        ]
        return structure, icon_parent_texts

    batched_structure, batched_icon_parents = _run(batch_size=1)  # 2 batches: boundary is mid-doc
    unbatched_structure, unbatched_icon_parents = _run(batch_size=100)  # 1 batch

    assert batched_structure == unbatched_structure
    assert batched_icon_parents == unbatched_icon_parents
    # The actual critical assertion: both icons resolve to the page-1
    # paragraph's text, NOT None, in BOTH runs — i.e. batching did not
    # silently orphan the cross-batch icon parent link.
    assert batched_icon_parents == ["See legend below.", "See legend below."]
