"""Internal route emulating a presigned URL for the `local` object storage
backend — see `app/storage/local_adapter.py` and
`Documentation/system-design/04-provider-abstractions.md` Bagian B.

Not mounted under the versioned `/api/v1` prefix on purpose: it's an
internal implementation detail of the `local` storage backend, not a public
API contract endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from app.core.config import Settings, get_settings
from app.storage.local_adapter import LocalFilesystemStorageClient, LocalStorageTokenError

router = APIRouter(tags=["internal-storage"])


@router.get("/internal/local-storage/{token}")
async def get_local_storage_object(
    token: str, settings: Settings = Depends(get_settings)  # noqa: B008 — idiomatic FastAPI DI
) -> Response:
    client = LocalFilesystemStorageClient(settings)
    try:
        data, content_type = client.resolve_token(token)
    except LocalStorageTokenError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(content=data, media_type=content_type)
