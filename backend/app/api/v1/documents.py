"""`POST /documents` — ingestion trigger endpoint (I1.10), see
`Documentation/system-design/08-authentication-rbac.md` §3.3 (`documents:write`
scope, path noted there as "belum ada path final" at design time).

**Path decision** (Backend Engineer, coordinated with orchestrator/System
Analyst per `CLAUDE.md` §9 — non-blocking, flagged for awareness):
`POST /api/v1/documents`, multipart file upload, `documents:write` scope,
returns `202 Accepted` + `document_id`/`ingestion_jobs`-trackable status
rather than a synchronous response — ingestion is a long-running
(potentially tens of minutes for a 400-page manual) async pipeline, never
run inline on the request (`01-architecture-overview.md` §2: "Tidak
melakukan ingestion berat secara sinkron").

Idempotent: uploading byte-identical PDF content again returns the existing
`document_id` with `status="already_ingested"` (per `document_hash`
idempotency, `02-ingestion-pipeline.md` §5) rather than re-enqueueing.
"""

from __future__ import annotations

import pymupdf
from fastapi import APIRouter, Depends, Security, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models.ingestion_jobs import IngestionJob
from app.ingestion.canonical_store import compute_document_hash
from app.ingestion.orchestrator import prepare_document_for_ingestion
from app.storage.factory import build_storage_client
from app.workers.tasks import run_ingestion_task

router = APIRouter(prefix="/documents", tags=["documents"])

WRITE = ["documents:write"]
READ = ["documents:read"]


class DocumentIngestResponse(BaseModel):
    document_id: int
    status: str  # "queued" | "already_ingested"
    celery_task_id: str | None


class IngestionJobResponse(BaseModel):
    id: int
    stage: str
    status: str
    started_at: str | None
    finished_at: str | None
    error_message: str | None
    stage_metrics: dict | None


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


@router.post("", response_model=DocumentIngestResponse, status_code=202)
async def trigger_document_ingestion(
    file: UploadFile,
    title: str,
    manufacturer: str | None = None,
    model_family: str | None = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> DocumentIngestResponse:
    pdf_bytes = await file.read()
    document_hash = compute_document_hash(pdf_bytes)
    page_count = _count_pdf_pages(pdf_bytes)
    storage = build_storage_client(settings)

    document, created = prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_bytes,
        filename=file.filename or "upload.pdf",
        title=title,
        manufacturer=manufacturer,
        model_family=model_family,
        document_hash=document_hash,
        page_count=page_count,
    )

    if not created:
        return DocumentIngestResponse(
            document_id=document.id, status="already_ingested", celery_task_id=None
        )

    task = run_ingestion_task.delay(document.id)
    return DocumentIngestResponse(document_id=document.id, status="queued", celery_task_id=task.id)


@router.get("/{document_id}/ingestion-jobs", response_model=list[IngestionJobResponse])
async def list_ingestion_jobs(
    document_id: int,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[IngestionJobResponse]:
    jobs = (
        db.execute(
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.id)
        )
        .scalars()
        .all()
    )
    return [
        IngestionJobResponse(
            id=job.id,
            stage=job.stage,
            status=job.status,
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            error_message=job.error_message,
            stage_metrics=job.stage_metrics,
        )
        for job in jobs
    ]
