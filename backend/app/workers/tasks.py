"""Celery task registration (I1.10) — thin wrapper around
`app/ingestion/orchestrator.py`'s `run_ingestion_pipeline`, which holds the
actual pipeline logic and is independently unit-tested with injected fakes.
This module's only real job is: build real settings-derived
DB session/storage client, materialize the source PDF, call the
orchestrator, clean up the temp file.

Routed to the `gpu` queue (`backend-worker-gpu`, `concurrency=1`) — see
`Documentation/system-design/01-architecture-overview.md` §6 and
`backend/Dockerfile`'s `worker-gpu` stage. Never the `default` queue, which
would let this GPU-bound task queue-block lightweight jobs.
"""

from __future__ import annotations

import os

from app.core.config import get_settings
from app.core.observability import get_logger
from app.db.engine import get_session_factory
from app.ingestion.orchestrator import materialize_source_pdf, run_ingestion_pipeline
from app.storage.factory import build_storage_client
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="ingestion.run_pipeline", queue="gpu")
def run_ingestion_task(self, document_id: int) -> dict[str, int]:
    """Celery entrypoint — see module docstring. `self.request.id` becomes
    `ingestion_jobs.celery_task_id` for every stage row this run produces."""
    settings = get_settings()
    session_factory = get_session_factory()
    storage = build_storage_client(settings)

    db = session_factory()
    pdf_path = None
    try:
        pdf_path = materialize_source_pdf(db, storage, document_id)
        logger.info(
            "ingestion_task.started",
            extra={"document_id": document_id, "celery_task_id": self.request.id},
        )
        result = run_ingestion_pipeline(
            db,
            storage,
            settings,
            document_id=document_id,
            pdf_path=pdf_path,
            celery_task_id=self.request.id,
        )
        logger.info(
            "ingestion_task.finished",
            extra={"document_id": document_id, "chunks_embedded": result.chunks_embedded},
        )
        return {
            "document_id": result.document_id,
            "pages_stored": result.pages_stored,
            "elements_stored": result.elements_stored,
            "chunks_stored": result.chunks_stored,
            "chunks_embedded": result.chunks_embedded,
            "cascade_task_count": result.cascade_task_count,
        }
    finally:
        db.close()
        if pdf_path is not None and os.path.exists(pdf_path):
            os.remove(pdf_path)
