"""Ingestion pipeline orchestration (I1.10), see
`Documentation/project-milestones/02-phase-1-ingestion.md` I1.10 and
`Documentation/system-design/06-data-schema.md` §1 (`ingestion_jobs`).

Wires the full pipeline built across I1.1-I1.9 for a single document:

    Stage 1 native_probe -> [per page-range batch: Stage 2 docling ->
    Stage 3 cascade_trigger -> Stage 4 paddle_cascade -> canonical_store
    (+ kg_candidate)] -> chunking -> embedding

Each of the 6 `INGESTION_STAGES` (`app/db/models/ingestion_jobs.py`) gets
its own `ingestion_jobs` row per invocation (queued -> running ->
done/failed, `stage_metrics.duration_ms`), independent of Celery task
boundaries, so progress/failure is visible per-stage in Postgres even
mid-run. **[2026-08-11, page-range batching round 2]** `docling`/
`paddle_cascade`/`kg_candidate` now run ONCE PER BATCH (see
`_run_batched_pipeline` below) rather than once per document — for an
N-batch document there are N rows for each of those three stages, not one;
`native_probe`/`chunking`/`embedding` remain once-per-document (unchanged).

All heavy/GPU components (`DoclingParser`, the PaddleOCR-VL client, the
Qdrant client, the embedding models) are **dependency-injected** with
real-from-settings defaults — this is what makes `run_ingestion_pipeline`
itself fully unit-testable (fakes injected for every heavy component) while
still being the actual code path a real Celery task
(`app/workers/tasks.py`) calls with real settings-built components.

**[2026-08-11, page-range batching round 2 — real OOM still observed]**
Round 1 (`ParseCarryState`, `DoclingParser.parse(page_range=...)`) bounded
ONLY Docling's own per-`convert()` memory — a real run against the actual
567-page "Zeggo VRV III" document still OOM'd (~12.7GB resident, virtually
unchanged from the ~12.5GB pre-round-1 baseline), because
`_run_docling_in_batches` (round 1, now removed) still accumulated every
batch's `ElementDraft`s into ONE merged `DoclingParseResult` and called
`store_pages_and_elements` exactly ONCE at the end for the whole document —
so peak memory was still O(total_pages), just with the specific "Docling's
internal per-conversion state" contributor now bounded. The dominant
remaining contributor, per this project's `Session` being configured with
`expire_on_commit=False` (`app/db/engine.py`), was almost certainly
thousands of fully-loaded `Element`/`Page` ORM objects accumulating in the
session's identity map for the entire ingestion run (a `commit()` alone
does not release them under `expire_on_commit=False`) — see
`canonical_store.py` module docstring for the fix (`db.expunge()` right
after each object is no longer needed) and
`docs/ingestion-page-range-batching.md` §2 (round 2) for the full
investigation/evidence.

Round 2 restructures the pipeline so **parse -> cascade -> store -> release**
happens PER BATCH (`_run_batched_pipeline`), so peak memory is genuinely
O(batch_size pages), not O(total_pages), for both Docling's own state AND
the SQLAlchemy session's identity map. `chunking`/`embedding` remain
whole-document (unchanged, not implicated by the evidence — see
`docs/ingestion-page-range-batching.md` §2 for why).
"""

from __future__ import annotations

import gc
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
from app.ingestion.docling_parser import DoclingParser, ParseCarryState
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
from app.ingestion.native_probe import DocumentProbe, probe_document
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


@dataclass(slots=True)
class _BatchPipelineResult:
    pages_stored: int
    elements_stored: int
    cascade_task_count: int


def _run_batched_pipeline(
    db: Session,
    storage: ObjectStorageClient,
    settings: Settings,
    *,
    document: Document,
    pdf_path: str | Path,
    probe: DocumentProbe,
    parser: DoclingParser,
    stages: _StageRunner,
    paddle_client: PaddleOCRVLClient | None,
) -> _BatchPipelineResult:
    """Stages 2-4 + canonical store + KG candidates, driven in
    `settings.INGESTION_PAGE_BATCH_SIZE`-page windows: **parse -> cascade ->
    store -> release**, per batch, so peak memory is O(batch_size pages),
    not O(total_pages) — see module docstring (2026-08-11, round 2) for the
    full rationale/evidence this replaced round 1's Docling-only batching.

    `ParseCarryState` (`docling_parser.py`) and `local_id_to_db_id`
    (`canonical_store.py`) are both threaded across every batch call in this
    loop so the final persisted result is identical to what a single
    unbatched run would have produced — same `local_id` sequence/
    `section_path`/icon-parent fallback chain (`ParseCarryState`), AND
    correct `elements.parent_id` FKs even when a child element's parent was
    stored in an EARLIER batch (`local_id_to_db_id`). See
    `tests/unit/test_ingestion_orchestrator.py
    test_run_ingestion_pipeline_batched_matches_unbatched_*` for the
    end-to-end equivalence proof (analogous to, but at a higher level than,
    `test_ingestion_docling_parser.py`'s round-1 equivalence tests).

    Docling is `unload()`ed after EVERY batch (not just once at the end,
    unlike round 1) — this is what makes `require_docling_unloaded_before_
    paddle_stage` (`02-ingestion-pipeline.md` §4, WAJIB for the `local`
    PaddleOCR-VL backend) hold at every batch's cascade step, not just once,
    and is itself part of the O(batch) memory guarantee for Docling's own
    state. The PaddleOCR-VL client's lifecycle is UNCHANGED (built lazily
    once, `unload()`ed once, after the whole loop) — it is a thin HTTP
    client wrapper around a remote/separate service in the supported
    `remote_api` configuration (`02-ingestion-pipeline.md` §4.0), not a
    locally-memory-heavy resource in this worker process, so there is no
    O(document) memory concern to fix there.
    """
    page_batch_size = settings.INGESTION_PAGE_BATCH_SIZE
    if page_batch_size <= 0:
        raise ValueError(f"page_batch_size must be positive, got {page_batch_size}")
    total_pages = probe.page_count
    if total_pages <= 0:
        raise ValueError(f"total_pages must be positive, got {total_pages}")

    batch_starts = list(range(1, total_pages + 1, page_batch_size))
    total_batches = len(batch_starts)

    carry_state: ParseCarryState | None = None
    local_id_to_db_id: dict[int, int] = {}
    paddle_client_instance = paddle_client
    pages_stored = 0
    elements_stored = 0
    cascade_task_count = 0

    for batch_index, start in enumerate(batch_starts, start=1):
        end = min(start + page_batch_size - 1, total_pages)
        batch_log_extra = {
            "page_range_start": start,
            "page_range_end": end,
            "batch_index": batch_index,
            "total_batches": total_batches,
        }
        logger.info("orchestrator.batch_start", extra=batch_log_extra)

        def _parse_batch(
            start: int = start, end: int = end, carry_state: ParseCarryState | None = carry_state
        ) -> Any:
            result = parser.parse(pdf_path, page_range=(start, end), carry_state=carry_state)
            # [2026-08-11, round 2] unload EVERY batch (not just once at the
            # end, unlike round 1) — see function docstring.
            parser.unload()
            return result

        batch_parse_result = stages.run(STAGE_DOCLING, _parse_batch)
        carry_state = batch_parse_result.carry_state

        batch_plan = build_cascade_plan(
            batch_parse_result,
            probe,
            threshold_table=settings.THRESHOLD_TABLE,
            threshold_text=settings.THRESHOLD_TEXT,
        )

        def _run_batch_cascade(
            batch_plan: Any = batch_plan, batch_parse_result: Any = batch_parse_result
        ) -> Any:
            nonlocal paddle_client_instance
            validate_paddleocr_vl_config_or_raise(settings)
            require_docling_unloaded_before_paddle_stage(parser, settings)
            if paddle_client_instance is None:
                paddle_client_instance = build_paddleocr_vl_client(settings)
            elements_by_local_id = {e.local_id: e for e in batch_parse_result.elements}
            return run_cascade_stage(
                batch_plan,
                pdf_path,
                paddle_client_instance,
                elements_by_local_id=elements_by_local_id,
            )

        batch_cascade_results = stages.run(STAGE_PADDLE_CASCADE, _run_batch_cascade)
        cascade_task_count += len(batch_cascade_results)

        def _run_batch_kg(
            batch_parse_result: Any = batch_parse_result,
            batch_cascade_results: Any = batch_cascade_results,
        ) -> Any:
            return extract_kg_candidates(
                batch_parse_result, document.filename, cascade_results=batch_cascade_results
            )

        batch_kg_candidates = stages.run(STAGE_KG_CANDIDATE, _run_batch_kg)

        batch_summary = store_pages_and_elements(
            db,
            storage,
            document=document,
            pdf_path=pdf_path,
            parse_result=batch_parse_result,
            cascade_results=batch_cascade_results,
            kg_candidates=batch_kg_candidates,
            page_range=(start, end),
            local_id_to_db_id=local_id_to_db_id,
        )
        pages_stored += batch_summary.pages_stored
        elements_stored += batch_summary.elements_stored

        logger.info(
            "orchestrator.batch_done",
            extra={**batch_log_extra, "pages_stored_so_far": pages_stored},
        )

        # [2026-08-11, round 2] explicit release of this batch's transient
        # Python objects, matching the "parse -> cascade -> store -> release"
        # instruction literally — `store_pages_and_elements` already
        # `db.expunge()`s every ORM object it created (the dominant real
        # fix, see canonical_store.py module docstring); this `gc.collect()`
        # is the cheap, harmless complement for everything else (cascade
        # results holding VLM description strings, the batch's `ElementDraft`
        # list, etc.), same reasoning as `DoclingParser.parse()`'s own
        # unconditional release.
        del batch_parse_result, batch_plan
        del batch_cascade_results, batch_kg_candidates, batch_summary
        gc.collect()

    # `batch_starts` always has >=1 entry (`total_pages`/`page_batch_size`
    # are both validated positive above), so the loop above always ran at
    # least once — and every iteration's `_run_batch_cascade` either uses
    # the injected client or lazily builds one — so `paddle_client_instance`
    # is guaranteed non-None here. `cast`, not an `if`/`assert`, expresses
    # that as a real invariant rather than a runtime branch that could never
    # actually be taken (an untestable, and therefore misleading, branch).
    cast(PaddleOCRVLClient, paddle_client_instance).unload()

    return _BatchPipelineResult(
        pages_stored=pages_stored,
        elements_stored=elements_stored,
        cascade_task_count=cascade_task_count,
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

        # [2026-08-11, page-range batching round 2] Stages 2-4 + canonical
        # store + KG candidates now run per page-range batch (parse ->
        # cascade -> store -> release) via `_run_batched_pipeline`, instead
        # of once for the whole document — see module docstring for why
        # round 1 (Docling-only batching) still OOM'd on the real 567-page
        # document, and `_run_batched_pipeline`'s own docstring for the
        # full per-batch contract this replaces.
        batch_result = _run_batched_pipeline(
            db,
            storage,
            settings,
            document=document,
            pdf_path=pdf_path,
            probe=probe,
            parser=parser,
            stages=stages,
            paddle_client=paddle_client,
        )

        def _run_chunking() -> Any:
            # [2026-08-11, round 3 memory fix] `_load_chunkable_elements`
            # streams `Element` rows (`yield_per`) instead of materializing
            # every row's full ORM state at once — see that function's
            # docstring. `store_chunks` returns a count, not
            # `list[Chunk]` — see chunker.py module docstring "round 3" for
            # why the OLD `list[Chunk]` return was itself O(document)
            # accumulation under this project's `expire_on_commit=False`.
            elements = _load_chunkable_elements(db, document_id)
            drafts = build_chunks(elements) + build_entity_chunks(elements)
            return store_chunks(db, document_id, drafts)

        chunks_stored = stages.run(STAGE_CHUNKING, _run_chunking)

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
            # [2026-08-11, round 3 memory fix] batched internally — see
            # embedder.py module docstring "round 3".
            return embed_and_upsert_chunks(
                db,
                client,
                document_id=document_id,
                collection=settings.VECTOR_STORE_COLLECTION,
                dense_model=dense,
                sparse_model=sparse,
                model_family=document.model_family,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
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
        pages_stored=batch_result.pages_stored,
        elements_stored=batch_result.elements_stored,
        chunks_stored=chunks_stored,
        chunks_embedded=embedded_count,
        cascade_task_count=batch_result.cascade_task_count,
    )


# [2026-08-11, round 3 memory fix] Purely an internal DB-cursor-fetch batch
# size for `_load_chunkable_elements` below (not an OOM-critical tunable
# like `INGESTION_PAGE_BATCH_SIZE`, so not exposed via `Settings`/env — see
# that function's docstring for what this does and doesn't fix).
_CHUNKABLE_ELEMENTS_YIELD_PER = 500


def _load_chunkable_elements(db: Session, document_id: int) -> list[ChunkableElement]:
    """[2026-08-11, round 3 memory fix] Streams `Element` rows via
    `execution_options(yield_per=...)` instead of `.all()`, and
    `db.expunge()`s each `Element` ORM row right after mapping it to a
    lightweight `ChunkableElement` — avoids holding every row's full ORM
    state (jsonb `visual_description`/`kg_candidate_entities`/`bbox`, etc.)
    simultaneously with the mapped dataclasses during the load itself. See
    `docs/ingestion-page-range-batching.md` §3 (round 3) and the
    module-level docstring of `chunker.py` for the fuller picture — the
    RETURNED `list[ChunkableElement]` here is still held in full (needed by
    `build_chunks`, which is deliberately NOT restructured into page-range
    batches this round — see that docstring for the full risk analysis);
    this streaming load only avoids DOUBLE-holding raw ORM rows alongside
    the (lighter) mapped dataclasses during the query itself.
    """
    result = db.execute(
        select(Element, Page.page_number)
        .join(Page, Element.page_id == Page.id)
        .where(Element.document_id == document_id)
        .order_by(Page.page_number, Element.id)
        .execution_options(yield_per=_CHUNKABLE_ELEMENTS_YIELD_PER)
    )
    elements: list[ChunkableElement] = []
    for element, page_number in result:
        elements.append(
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
        )
        db.expunge(element)
    return elements


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
