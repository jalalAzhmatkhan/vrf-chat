"""Local filesystem object storage adapter.

Fallback for dev without Docker storage / fast local testing (see
`Documentation/system-design/04-provider-abstractions.md` Bagian B). Not a
real S3 API — implements the same `ObjectStorageClient` interface, but
`get_presigned_url` is emulated: a signed, time-limited token is generated
and embedded in a URL pointing at the internal route
`GET /internal/local-storage/{token}` (see `app/api/v1/internal_storage.py`),
so the contract's return type (a directly-usable URL string) stays
consistent across every backend for the frontend.

**F2-11** (`Documentation/qa-reports/phase-2-qa-report.md`): the returned URL
is made ABSOLUTE (prefixed with `Settings.APP_PUBLIC_BASE_URL`), not just a
path. A path-only URL is only ever "relative" in the sense that matters once
a *different* origin (the FE dev server, `http://localhost:5173`) resolves
it as `<img src>` — the browser resolves it against the page's own origin,
not the backend's, so every page image/icon crop 404'd in QA's real-browser
testing even though the backend route itself worked fine when hit directly.
MinIO/S3 (`s3_adapter.py`) does not have this problem — real presigned URLs
are absolute by construction.

Canonical URI format: `file://{key}`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

from app.core.config import Settings

_URI_PREFIX = "file://"


class LocalStorageTokenError(ValueError):
    """Raised when a local-storage presigned token is invalid/expired."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign(payload_b64: str, secret: str) -> str:
    signature = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    return _b64url_encode(signature)


class LocalFilesystemStorageClient:
    """`ObjectStorageClient` implementation backed by the local filesystem."""

    def __init__(self, settings: Settings) -> None:
        self._base_path = Path(settings.OBJECT_STORAGE_LOCAL_BASE_PATH)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._token_secret = settings.OBJECT_STORAGE_LOCAL_TOKEN_SECRET
        # F2-11 — see module docstring.
        self._public_base_url = settings.APP_PUBLIC_BASE_URL.rstrip("/")

    def _resolve(self, key: str) -> Path:
        return self._base_path / key

    def put_object(self, key: str, data: bytes, content_type: str) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._meta_path(path).write_text(json.dumps({"content_type": content_type}))
        return f"{_URI_PREFIX}{key}"

    def get_object(self, uri: str) -> bytes:
        key = self._key_from_uri(uri)
        return self._resolve(key).read_bytes()

    def get_presigned_url(self, uri: str, expires_in_seconds: int = 3600) -> str:
        key = self._key_from_uri(uri)
        token = self.issue_token(key, expires_in_seconds)
        return f"{self._public_base_url}/internal/local-storage/{token}"

    def delete_object(self, uri: str) -> None:
        key = self._key_from_uri(uri)
        path = self._resolve(key)
        path.unlink(missing_ok=True)
        self._meta_path(path).unlink(missing_ok=True)

    def exists(self, uri: str) -> bool:
        key = self._key_from_uri(uri)
        return self._resolve(key).exists()

    # ---- token issuance / verification (used by the internal route too) ----

    def issue_token(self, key: str, expires_in_seconds: int) -> str:
        payload = {"key": key, "exp": int(time.time()) + expires_in_seconds}
        payload_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))
        signature = _sign(payload_b64, self._token_secret)
        return f"{payload_b64}.{signature}"

    def resolve_token(self, token: str) -> tuple[bytes, str]:
        """Verify `token` and return `(data, content_type)`, or raise
        `LocalStorageTokenError`."""
        try:
            payload_b64, signature = token.split(".", 1)
        except ValueError as exc:
            raise LocalStorageTokenError("Malformed token") from exc

        expected_signature = _sign(payload_b64, self._token_secret)
        if not hmac.compare_digest(signature, expected_signature):
            raise LocalStorageTokenError("Invalid token signature")

        try:
            payload = json.loads(_b64url_decode(payload_b64))
        except (ValueError, UnicodeDecodeError) as exc:
            raise LocalStorageTokenError("Malformed token payload") from exc

        if payload.get("exp", 0) < time.time():
            raise LocalStorageTokenError("Token expired")

        key = payload["key"]
        path = self._resolve(key)
        if not path.exists():
            raise LocalStorageTokenError(f"Object not found for key: {key}")

        content_type = "application/octet-stream"
        meta_path = self._meta_path(path)
        if meta_path.exists():
            content_type = json.loads(meta_path.read_text()).get("content_type", content_type)

        return path.read_bytes(), content_type

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".meta.json")

    @staticmethod
    def _key_from_uri(uri: str) -> str:
        if not uri.startswith(_URI_PREFIX):
            raise ValueError(f"Not a valid file:// URI: {uri!r}")
        return uri[len(_URI_PREFIX) :]
