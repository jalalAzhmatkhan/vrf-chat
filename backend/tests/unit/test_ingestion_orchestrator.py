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
from typing import Any

import pymupdf
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document
from app.db.models.ingestion_jobs import IngestionJob
from app.ingestion import orchestrator as orch
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft, PageConfidence
from app.ingestion.paddleocr_vl_cascade import (
    OCRPageResult,
    TableReparseResult,
    VisualDescription,
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


class FakeDoclingParser:
    def __init__(self, parse_result: DoclingParseResult) -> None:
        self._parse_result = parse_result
        self.is_loaded = True
        self.parse_calls = 0
        self.unload_calls = 0

    def parse(self, pdf_path: Any) -> DoclingParseResult:
        self.parse_calls += 1
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
        def parse(self, pdf_path: Any) -> DoclingParseResult:
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
