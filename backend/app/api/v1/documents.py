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
from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.db.models.ingestion_jobs import IngestionJob
from app.ingestion.canonical_store import compute_document_hash
from app.ingestion.orchestrator import prepare_document_for_ingestion
from app.ingestion.paddleocr_vl_cascade import PAGE_RENDER_DPI_DEFAULT
from app.storage.factory import build_storage_client
from app.workers.tasks import run_ingestion_task

router = APIRouter(prefix="/documents", tags=["documents"])

WRITE = ["documents:write"]
READ = ["documents:read"]

_PT_TO_PX = PAGE_RENDER_DPI_DEFAULT / 72.0


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


class DocumentResponse(BaseModel):
    id: int
    title: str
    manufacturer: str | None
    model_family: str | None
    filename: str
    page_count: int | None
    status: str
    uploaded_at: str


class PageElementResponse(BaseModel):
    """One element on a page, for bbox highlight overlay (citation viewer,
    C2.8) — `bbox` format documented in
    `05-streaming-and-api-contract.md` §5.5 (PDF points, bottom-left
    origin)."""

    element_id: int
    element_type: str
    text: str | None
    bbox: dict | None
    # Presigned, directly usable as `<img src>` — `None` for non-visual
    # element types (paragraph/table/etc. never had an `image_uri` to begin
    # with).
    image_uri: str | None


class PageDetailResponse(BaseModel):
    document_id: int
    page_number: int
    # Presigned URL (`ObjectStorageClient.get_presigned_url`,
    # 04-provider-abstractions.md Bagian B) — never a raw canonical URI.
    page_image_uri: str | None
    # §5.5 gap fix — both units included so FE can pick either the
    # points-based or pixel-based bbox-scaling approach without hardcoding
    # an assumed render DPI.
    page_width_pt: float | None
    page_height_pt: float | None
    image_width_px: int | None
    image_height_px: int | None
    elements: list[PageElementResponse]


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


def _document_to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        manufacturer=document.manufacturer,
        model_family=document.model_family,
        filename=document.filename,
        page_count=document.page_count,
        status=document.status,
        uploaded_at=document.uploaded_at.isoformat(),
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[DocumentResponse]:
    documents = db.execute(select(Document).order_by(Document.id)).scalars().all()
    return [_document_to_response(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> DocumentResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _document_to_response(document)


@router.get("/{document_id}/pages/{page_number}", response_model=PageDetailResponse)
async def get_document_page(
    document_id: int,
    page_number: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> PageDetailResponse:
    page = db.execute(
        select(Page).where(Page.document_id == document_id, Page.page_number == page_number)
    ).scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found")

    storage = build_storage_client(settings)
    page_image_uri = (
        storage.get_presigned_url(
            page.page_image_uri, settings.OBJECT_STORAGE_PRESIGNED_URL_EXPIRY_SECONDS
        )
        if page.page_image_uri
        else None
    )

    elements = (
        db.execute(select(Element).where(Element.page_id == page.id).order_by(Element.id))
        .scalars()
        .all()
    )
    element_responses = [
        PageElementResponse(
            element_id=element.id,
            element_type=element.element_type,
            text=element.text,
            bbox=element.bbox,
            image_uri=(
                storage.get_presigned_url(
                    element.image_uri, settings.OBJECT_STORAGE_PRESIGNED_URL_EXPIRY_SECONDS
                )
                if element.image_uri
                else None
            ),
        )
        for element in elements
    ]

    image_width_px = (
        round(page.page_width_pt * _PT_TO_PX) if page.page_width_pt is not None else None
    )
    image_height_px = (
        round(page.page_height_pt * _PT_TO_PX) if page.page_height_pt is not None else None
    )

    return PageDetailResponse(
        document_id=document_id,
        page_number=page_number,
        page_image_uri=page_image_uri,
        page_width_pt=page.page_width_pt,
        page_height_pt=page.page_height_pt,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        elements=element_responses,
    )
