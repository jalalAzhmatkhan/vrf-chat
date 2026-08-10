"""`GET /api/v1/elements/{element_id}` — single element detail (figure/
table/icon/etc.), per `05-streaming-and-api-contract.md` §6: "Detail satu
elemen ... termasuk image_uri, visual_description, bbox — format bbox
didefinisikan di §5.5". Scope `documents:read` (same scope family as
`app/api/v1/documents.py` — elements are a sub-resource of documents, not a
separately-scoped resource per `08-authentication-rbac.md`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Security
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
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
    )
