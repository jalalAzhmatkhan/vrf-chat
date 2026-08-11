"""Ingestion pipeline orchestration (I1.10), see
`Documentation/project-milestones/02-phase-1-ingestion.md` I1.10 and
`Documentation/system-design/06-data-schema.md` §1 (`ingestion_jobs`).

Wires the full pipeline built across I1.1-I1.9 for a single document:

    Stage 1 native_probe -> Stage 2 docling -> Stage 3 cascade_trigger
    -> Stage 4 paddle_cascade -> canonical_store (+ kg_candidate)
    -> chunking -> embedding

Each of the 6 `INGESTION_STAGES` (`app/db/models/ingestion_jobs.py`) gets
its own `ingestion_jobs` row (queued -> running -> done/failed,
`stage_metrics.duration_ms`), independent of Celery task boundaries, so
progress/failure is visible per-stage in Postgres even mid-run.

All heavy/GPU components (`DoclingParser`, the PaddleOCR-VL client, the
Qdrant client, the embedding models) are **dependency-injected** with
real-from-settings defaults — this is what makes `run_ingestion_pipeline`
itself fully unit-testable (fakes injected for every heavy component) while
still being the actual code path a real Celery task
(`app/workers/tasks.py`) calls with real settings-built components.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.observability import get_logger
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.db.models.ingestion_jobs import IngestionJob
from app.ingestion.canonical_store import (
    DocumentMetadata,
    get_or_create_document,
    store_pages_and_elements,
)
from app.ingestion.cascade_trigger import build_cascade_plan
from app.ingestion.chunker import ChunkableElement, build_chunks, build_entity_chunks, store_chunks
from app.ingestion.docling_parser import (
    DoclingParser,
    DoclingParseResult,
    ElementDraft,
    PageConfidence,
    ParseCarryState,
)
from app.ingestion.embedder import (
    DenseEmbeddingModel,
    FastEmbedDenseModel,
    FastEmbedSparseModel,
    SparseEmbeddingModel,
    delete_chunks_by_document,
    embed_and_upsert_chunks,
    ensure_collection,
)
from app.ingestion.kg_candidate_extractor import extract_kg_candidates
from app.ingestion.native_probe import probe_document
from app.ingestion.paddleocr_vl_cascade import (
    PaddleOCRVLClient,
    build_paddleocr_vl_client,
    require_docling_unloaded_before_paddle_stage,
    run_cascade_stage,
    validate_paddleocr_vl_config_or_raise,
)
from app.retrieval.vector_store import build_qdrant_client
from app.storage.base import ObjectStorageClient

logger = get_logger(__name__)

STAGE_NATIVE_PROBE = "native_probe"
STAGE_DOCLING = "docling"
STAGE_PADDLE_CASCADE = "paddle_cascade"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_KG_CANDIDATE = "kg_candidate"


@dataclass(slots=True)
class OrchestratorResult:
    document_id: int
    pages_stored: int
    elements_stored: int
    chunks_stored: int
    chunks_embedded: int
    cascade_task_count: int


def _document_source_key(document_id: int) -> str:
    return f"documents/{document_id}/source.pdf"


def _run_docling_in_batches(
    parser: DoclingParser,
    pdf_path: str | Path,
    *,
    page_batch_size: int,
    total_pages: int,
) -> DoclingParseResult:
    """Stage 2 (Docling), driven in `page_batch_size`-page windows via
    Docling's own `page_range` support, instead of one `parser.parse()`
    call over the whole document — see `docling_parser.py` module
    docstring/`ParseCarryState` for the full rationale.

    Added 2026-08-11 after a real WSL kernel oom-killer event (~12.5GB
    resident on a 13GB WSL VM) ingesting the 567-page "Zeggo VRV III"
    document as a single `converter.convert()` call, 3 failures in a row,
    always within 1-2 minutes of starting — i.e. during this stage, not a
    later one. 5 other documents (286-403 pages) processed unbatched
    without incident, so this bounds Docling's OWN per-conversion memory to
    O(batch_size pages) instead of O(total_pages), which is the specific
    thing that scaled with the failing document's page count.

    `total_pages` is expected to come from Stage 1's native probe
    (`probe_document(pdf_path).page_count`, already computed cheaply via
    PyMuPDF before this stage runs in `run_ingestion_pipeline` below) rather
    than re-derived from Docling itself, since the whole point is to never
    ask Docling to look at the full document in one call.

    Returns ONE merged `DoclingParseResult` that is semantically identical
    to what a single unbatched `parser.parse(pdf_path)` call would have
    produced (same `local_id` sequence, `section_path` breadcrumbs, and
    icon->text parent fallback across the batch boundary — proven by
    `tests/unit/test_ingestion_docling_parser.py
    test_batched_parse_matches_single_pass_*`) — every stage downstream of
    this function (cascade_trigger, canonical_store, kg_candidate_extractor,
    chunker) is unmodified by this change and receives that merged result
    exactly as it always has.
    """
    if page_batch_size <= 0:
        raise ValueError(f"page_batch_size must be positive, got {page_batch_size}")
    if total_pages <= 0:
        raise ValueError(f"total_pages must be positive, got {total_pages}")

    batch_starts = list(range(1, total_pages + 1, page_batch_size))
    total_batches = len(batch_starts)

    all_elements: list[ElementDraft] = []
    page_confidence: dict[int, PageConfidence] = {}
    carry_state: ParseCarryState | None = None

    for batch_index, start in enumerate(batch_starts, start=1):
        end = min(start + page_batch_size - 1, total_pages)
        logger.info(
            "orchestrator.docling_batch_start",
            extra={
                "page_range_start": start,
                "page_range_end": end,
                "batch_index": batch_index,
                "total_batches": total_batches,
            },
        )
        batch_result = parser.parse(pdf_path, page_range=(start, end), carry_state=carry_state)
        all_elements.extend(batch_result.elements)
        page_confidence.update(batch_result.page_confidence)
        carry_state = batch_result.carry_state
        logger.info(
            "orchestrator.docling_batch_done",
            extra={
                "page_range_start": start,
                "page_range_end": end,
                "batch_index": batch_index,
                "total_batches": total_batches,
                "elements_so_far": len(all_elements),
            },
        )

    return DoclingParseResult(
        elements=all_elements, page_confidence=page_confidence, page_count=total_pages
    )


class _StageRunner:
    """Brackets a stage callable with an `ingestion_jobs` row — see module
    docstring. A thin, deliberately simple helper (not a generic
    "pipeline framework") kept private to this module."""

    def __init__(self, db: Session, document_id: int, celery_task_id: str | None) -> None:
        self._db = db
        self._document_id = document_id
        self._celery_task_id = celery_task_id

    def run(self, stage: str, fn: Callable[[], Any]) -> Any:
        job = IngestionJob(
            document_id=self._document_id,
            celery_task_id=self._celery_task_id,
            stage=stage,
            status="running",
            started_at=datetime.now(UTC),
        )
        self._db.add(job)
        self._db.commit()

        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            job.status = "failed"
            job.finished_at = datetime.now(UTC)
            job.error_message = str(exc)
            job.stage_metrics = {"duration_ms": int((time.perf_counter() - start) * 1000)}
            self._db.commit()
            raise

        job.status = "done"
        job.finished_at = datetime.now(UTC)
        job.stage_metrics = {"duration_ms": int((time.perf_counter() - start) * 1000)}
        self._db.commit()
        return result


def run_ingestion_pipeline(
    db: Session,
    storage: ObjectStorageClient,
    settings: Settings,
    *,
    document_id: int,
    pdf_path: str | Path,
    celery_task_id: str | None = None,
    docling_parser: DoclingParser | None = None,
    paddle_client: PaddleOCRVLClient | None = None,
    qdrant_client: Any | None = None,
    dense_model: DenseEmbeddingModel | None = None,
    sparse_model: SparseEmbeddingModel | None = None,
) -> OrchestratorResult:
    """Run the full ingestion pipeline for an already-created `Document` row
    (`document_id`) whose source PDF is at `pdf_path` (a local filesystem
    path — the caller, typically `app/workers/tasks.py`, is responsible for
    materializing it, e.g. downloading from object storage to a temp file
    for a real distributed worker).
    """
    document = db.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document {document_id} not found")

    stages = _StageRunner(db, document_id, celery_task_id)
    document.status = "processing"
    db.commit()

    try:
        probe = stages.run(STAGE_NATIVE_PROBE, lambda: probe_document(pdf_path))

        parser = docling_parser or DoclingParser(device=settings.DOCLING_DEVICE)

        def _run_docling() -> Any:
            # [2026-08-11 OOM fix] page-range batching, see
            # `_run_docling_in_batches` docstring above — `parser.unload()`
            # still happens exactly once, after ALL batches finish (same
            # contract as before: `DOCLING_UNLOAD_BEFORE_PADDLE_STAGE`,
            # `02-ingestion-pipeline.md` §4). The converter/its models stay
            # loaded and are reused across batches (no per-batch reload
            # cost) — only the per-batch `ConversionResult`/`DoclingDocument`
            # object graph is released between batches, inside
            # `DoclingParser.parse()` itself.
            result = _run_docling_in_batches(
                parser,
                pdf_path,
                page_batch_size=settings.INGESTION_PAGE_BATCH_SIZE,
                total_pages=probe.page_count,
            )
            parser.unload()
            return result

        parse_result = stages.run(STAGE_DOCLING, _run_docling)

        plan = build_cascade_plan(
            parse_result,
            probe,
            threshold_table=settings.THRESHOLD_TABLE,
            threshold_text=settings.THRESHOLD_TEXT,
        )

        def _run_paddle_cascade() -> Any:
            validate_paddleocr_vl_config_or_raise(settings)
            require_docling_unloaded_before_paddle_stage(parser, settings)
            client = paddle_client or build_paddleocr_vl_client(settings)
            elements_by_local_id = {e.local_id: e for e in parse_result.elements}
            results = run_cascade_stage(
                plan, pdf_path, client, elements_by_local_id=elements_by_local_id
            )
            client.unload()
            return results

        cascade_results = stages.run(STAGE_PADDLE_CASCADE, _run_paddle_cascade)

        def _run_kg_candidates() -> Any:
            return extract_kg_candidates(
                parse_result, document.filename, cascade_results=cascade_results
            )

        kg_candidates = stages.run(STAGE_KG_CANDIDATE, _run_kg_candidates)

        store_summary = store_pages_and_elements(
            db,
            storage,
            document=document,
            pdf_path=pdf_path,
            parse_result=parse_result,
            cascade_results=cascade_results,
            kg_candidates=kg_candidates,
        )

        def _run_chunking() -> Any:
            elements = _load_chunkable_elements(db, document_id)
            drafts = build_chunks(elements) + build_entity_chunks(elements)
            return store_chunks(db, document_id, drafts)

        chunk_rows = stages.run(STAGE_CHUNKING, _run_chunking)

        def _run_embedding() -> int:
            client = qdrant_client or build_qdrant_client(settings)
            ensure_collection(
                client, settings.VECTOR_STORE_COLLECTION, settings.VECTOR_STORE_DENSE_DIM
            )
            # [2026-08-10, operational-habit correction — see
            # app/ingestion/embedder.py delete_chunks_by_document docstring]
            # `store_chunks` (STAGE_CHUNKING above) always deletes+recreates
            # this document's `chunks` rows with fresh ids on every
            # ingestion run (e.g. a re-ingest after a bugfix) — without this
            # scoped cleanup, the OLD Qdrant points (upserted under the OLD
            # chunk ids) would be silently orphaned forever. A no-op, safe
            # to call unconditionally, on a first-time ingest.
            delete_chunks_by_document(client, settings.VECTOR_STORE_COLLECTION, document_id)
            dense = dense_model or FastEmbedDenseModel(settings.EMBEDDER_DENSE_MODEL)
            sparse = sparse_model or FastEmbedSparseModel(settings.EMBEDDER_SPARSE_MODEL)
            return embed_and_upsert_chunks(
                db,
                client,
                document_id=document_id,
                collection=settings.VECTOR_STORE_COLLECTION,
                dense_model=dense,
                sparse_model=sparse,
                model_family=document.model_family,
            )

        embedded_count = stages.run(STAGE_EMBEDDING, _run_embedding)

    except Exception:
        document.status = "failed"
        db.commit()
        raise

    document.status = "ready"
    db.commit()

    return OrchestratorResult(
        document_id=document_id,
        pages_stored=store_summary.pages_stored,
        elements_stored=store_summary.elements_stored,
        chunks_stored=len(chunk_rows),
        chunks_embedded=embedded_count,
        cascade_task_count=len(cascade_results),
    )


def _load_chunkable_elements(db: Session, document_id: int) -> list[ChunkableElement]:
    rows = db.execute(
        select(Element, Page.page_number)
        .join(Page, Element.page_id == Page.id)
        .where(Element.document_id == document_id)
        .order_by(Page.page_number, Element.id)
    ).all()
    return [
        ChunkableElement(
            element_id=element.id,
            element_type=element.element_type,
            text=element.text,
            page_number=page_number,
            parent_id=element.parent_id,
            section_path=element.section_path or [],
            image_uri=element.image_uri,
            visual_description=element.visual_description,
            kg_candidate_entities=element.kg_candidate_entities or [],
        )
        for element, page_number in rows
    ]


def prepare_document_for_ingestion(
    db: Session,
    storage: ObjectStorageClient,
    *,
    pdf_bytes: bytes,
    filename: str,
    title: str,
    manufacturer: str | None,
    model_family: str | None,
    document_hash: str,
    page_count: int,
) -> tuple[Document, bool]:
    """Idempotent document creation + source PDF upload — called by the
    trigger endpoint (`app/api/v1/documents.py`) *before* enqueueing the
    Celery task, so the task itself only ever needs a `document_id`."""
    document, created = get_or_create_document(
        db,
        DocumentMetadata(
            title=title,
            manufacturer=manufacturer,
            model_family=model_family,
            filename=filename,
            source_hash=document_hash,
            page_count=page_count,
        ),
    )
    if created:
        uri = storage.put_object(_document_source_key(document.id), pdf_bytes, "application/pdf")
        document.source_pdf_uri = uri
        db.commit()
    return document, created


def materialize_source_pdf(db: Session, storage: ObjectStorageClient, document_id: int) -> Path:
    """Download the stored source PDF to a local temp file — real Celery
    workers run in a different container/process than the API that
    uploaded it, so a real filesystem path (needed by PyMuPDF/Docling) must
    be re-materialized from object storage, not assumed to still exist
    locally. Uses `Document.source_pdf_uri` (the exact URI
    `ObjectStorageClient.put_object()` returned at upload time) rather than
    reconstructing a URI from a hardcoded scheme guess, which would break
    for non-`local` backends (`s3://`/MinIO don't use `file://`)."""
    document = db.get(Document, document_id)
    if document is None or not document.source_pdf_uri:
        raise ValueError(f"Document {document_id} has no source_pdf_uri to materialize")
    data = storage.get_object(document.source_pdf_uri)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)
