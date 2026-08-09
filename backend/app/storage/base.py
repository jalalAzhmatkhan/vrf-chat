"""`ObjectStorageClient` protocol — see
`Documentation/system-design/04-provider-abstractions.md` Bagian B §2.

All large binaries (source PDFs, page render images, figure/diagram crops)
go through this interface and are never stored in Postgres — only their
canonical URI is. Two adapters implement it: `S3CompatibleStorageClient`
(shared by `minio`/`s3` — both are S3 APIs via boto3, only `endpoint_url`/
`force_path_style` differ) and `LocalFilesystemStorageClient` (fallback for
dev without Docker storage).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStorageClient(Protocol):
    def put_object(self, key: str, data: bytes, content_type: str) -> str:
        """Store `data` under `key` and return its canonical URI
        (e.g. `s3://bucket/key` or `file://key`)."""
        ...

    def get_object(self, uri: str) -> bytes:
        """Fetch object bytes by canonical URI."""
        ...

    def get_presigned_url(self, uri: str, expires_in_seconds: int = 3600) -> str:
        """Return a URL usable directly as `<img src>`/download link by the
        frontend. For S3/MinIO this is a real presigned URL; for the local
        backend it is emulated via a signed token + internal FastAPI route
        (see `app/storage/local_adapter.py`) — callers must not assume the
        URL is a genuine presigned S3 URL in all cases."""
        ...

    # Protocol stubs — never executed (structural typing only).
    def delete_object(self, uri: str) -> None: ...  # pragma: no cover
    def exists(self, uri: str) -> bool: ...  # pragma: no cover
