"""Object storage provider factory + fail-fast validation.

See `Documentation/system-design/04-provider-abstractions.md` Bagian B.
`OBJECT_STORAGE_BACKEND` picks the adapter purely via `.env` — application
code depends only on the `ObjectStorageClient` protocol.
"""

from __future__ import annotations

from app.core.config import Settings
from app.storage.base import ObjectStorageClient
from app.storage.local_adapter import LocalFilesystemStorageClient
from app.storage.s3_adapter import S3CompatibleStorageClient

SUPPORTED_BACKENDS = ("minio", "s3", "local")


class ObjectStorageConfigError(ValueError):
    """Raised (fail-fast) when an `OBJECT_STORAGE_*` env combination is invalid."""


def build_storage_client(settings: Settings) -> ObjectStorageClient:
    """Build the configured `ObjectStorageClient`.

    Raises `ObjectStorageConfigError` for invalid/incomplete configuration —
    call `validate_object_storage_or_raise` at app startup so this is caught
    immediately, not on first upload/download.
    """
    backend = settings.OBJECT_STORAGE_BACKEND

    if backend == "minio":
        if not settings.OBJECT_STORAGE_ENDPOINT_URL:
            raise ObjectStorageConfigError(
                "OBJECT_STORAGE_ENDPOINT_URL is required when "
                "OBJECT_STORAGE_BACKEND=minio."
            )
        return S3CompatibleStorageClient(settings)

    if backend == "s3":
        return S3CompatibleStorageClient(settings)

    if backend == "local":
        return LocalFilesystemStorageClient(settings)

    raise ObjectStorageConfigError(
        f"OBJECT_STORAGE_BACKEND: unknown backend '{backend}', "
        f"must be one of {SUPPORTED_BACKENDS}"
    )


def validate_object_storage_or_raise(settings: Settings) -> None:
    """Fail-fast validation — call once at app startup."""
    build_storage_client(settings)
