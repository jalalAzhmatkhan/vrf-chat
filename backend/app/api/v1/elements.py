"""`GET /api/v1/elements/{element_id}` — single element detail (figure/
table/icon/etc.), per `05-streaming-and-api-contract.md` §6: "Detail satu
elemen ... termasuk image_uri, visual_description, bbox — format bbox
didefinisikan di §5.5" plus (§5.2.1, F2-10, BARU 2026-08-10) "content_structured
PENUH (tidak terpotong) untuk element_type == table". Scope `documents:read`
(same scope family as `app/api/v1/documents.py` — elements are a
sub-resource of documents, not a separately-scoped resource per
`08-authentication-rbac.md`).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models.chunks import Chunk
from app.db.models.documents import Page
from app.db.models.elements import Element
from app.storage.factory import build_storage_client

router = APIRouter(prefix="/elements", tags=["elements"])

READ = ["documents:read"]


class ElementResponse(BaseModel):
    id: int
    document_id: int
    page_number: int
    element_type: str
    text: str | None
    bbox: dict | None
    section_path: list | None
    image_uri: str | None
    visual_description: dict | None
    # §5.2.1 (F2-10) — full, UN-truncated content_structured for
    # element_type == "table" (unlike Citation.content_structured, which may
    # be capped — see app/agent/context_builder.py
    # `cap_content_structured_for_citation`). `None` for every other
    # element_type, and also `None` for a table with no owning chunk found
    # (should not happen for real ingested data, but not an error either).
    content_structured: dict[str, Any] | None = None


def _find_table_content_structured(db: Session, element: Element) -> dict[str, Any] | None:
    """§5.2.1 full-data lookup — join `element_id` ∈ `chunks.element_ids`,
    same mapping verified 1:1 for 152/157 real table chunks (5/157 shared by
    2 chunks; first match wins, both are valid data for the same element).
    `element_ids` is a portable JSON column (not a native array in every
    supported DB engine, per `app/db/base.py` `PortableJSON`), so this
    filters in Python rather than relying on an engine-specific "array
    contains" SQL operator — acceptable here since it is scoped to one
    document's table chunks, not the whole corpus."""
    if element.element_type != "table":
        return None
    chunks = (
        db.execute(
            select(Chunk).where(
                Chunk.document_id == element.document_id, Chunk.chunk_type == "table"
            )
        )
        .scalars()
        .all()
    )
    for chunk in chunks:
        if element.id in (chunk.element_ids or []):
            return chunk.content_structured
    return None


@router.get("/{element_id}", response_model=ElementResponse)
async def get_element(
    element_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> ElementResponse:
    element = db.get(Element, element_id)
    if element is None:
        raise HTTPException(status_code=404, detail="Element not found")

    page = db.get(Page, element.page_id)
    page_number = page.page_number if page is not None else 0

    image_uri = None
    if element.image_uri:
        storage = build_storage_client(settings)
        image_uri = storage.get_presigned_url(
            element.image_uri, settings.OBJECT_STORAGE_PRESIGNED_URL_EXPIRY_SECONDS
        )

    return ElementResponse(
        id=element.id,
        document_id=element.document_id,
        page_number=page_number,
        element_type=element.element_type,
        text=element.text,
        bbox=element.bbox,
        section_path=element.section_path,
        image_uri=image_uri,
        visual_description=element.visual_description,
        content_structured=_find_table_content_structured(db, element),
    )
