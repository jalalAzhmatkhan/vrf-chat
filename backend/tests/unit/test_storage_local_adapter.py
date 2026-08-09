import time

import pytest

from app.core.config import Settings
from app.storage.local_adapter import LocalFilesystemStorageClient, LocalStorageTokenError


def _settings(tmp_path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "OBJECT_STORAGE_BACKEND": "local",
        "OBJECT_STORAGE_LOCAL_BASE_PATH": str(tmp_path),
        "OBJECT_STORAGE_LOCAL_TOKEN_SECRET": "unit-test-secret",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_put_get_exists_delete_roundtrip(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))

    uri = client.put_object("docs/manual.pdf", b"%PDF-1.4 ...", "application/pdf")

    assert uri == "file://docs/manual.pdf"
    assert client.exists(uri) is True
    assert client.get_object(uri) == b"%PDF-1.4 ..."

    client.delete_object(uri)

    assert client.exists(uri) is False


def test_get_object_rejects_non_file_uri(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))

    with pytest.raises(ValueError, match="Not a valid file:// URI"):
        client.get_object("s3://bucket/key")


def test_presigned_url_token_roundtrip(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))
    uri = client.put_object("icons/reset.png", b"icon-bytes", "image/png")

    url = client.get_presigned_url(uri, expires_in_seconds=60)

    assert url.startswith("/internal/local-storage/")
    token = url.removeprefix("/internal/local-storage/")

    data, content_type = client.resolve_token(token)
    assert data == b"icon-bytes"
    assert content_type == "image/png"


def test_resolve_token_defaults_content_type_when_no_metadata(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))
    # Write directly, bypassing put_object, so no .meta.json sidecar exists.
    (tmp_path / "raw.bin").write_bytes(b"raw")

    token = client.issue_token("raw.bin", 60)

    data, content_type = client.resolve_token(token)
    assert data == b"raw"
    assert content_type == "application/octet-stream"


def test_resolve_token_rejects_malformed_token(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))

    with pytest.raises(LocalStorageTokenError, match="Malformed token"):
        client.resolve_token("not-a-valid-token")


def test_resolve_token_rejects_bad_signature(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))
    token = client.issue_token("some/key", 60)
    payload_b64, _signature = token.split(".", 1)
    tampered = f"{payload_b64}.tampered-signature"

    with pytest.raises(LocalStorageTokenError, match="Invalid token signature"):
        client.resolve_token(tampered)


def test_resolve_token_rejects_malformed_payload(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))
    # Craft a token whose payload isn't valid JSON but has a correct signature.
    from app.storage.local_adapter import _b64url_encode, _sign

    bad_payload_b64 = _b64url_encode(b"not-json")
    signature = _sign(bad_payload_b64, "unit-test-secret")
    token = f"{bad_payload_b64}.{signature}"

    with pytest.raises(LocalStorageTokenError, match="Malformed token payload"):
        client.resolve_token(token)


def test_resolve_token_rejects_expired_token(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))
    client.put_object("expired.txt", b"data", "text/plain")

    token = client.issue_token("expired.txt", -10)
    time.sleep(0)  # no-op, keeps intent explicit that exp is already in the past

    with pytest.raises(LocalStorageTokenError, match="Token expired"):
        client.resolve_token(token)


def test_resolve_token_rejects_missing_object(tmp_path) -> None:
    client = LocalFilesystemStorageClient(_settings(tmp_path))

    token = client.issue_token("never-created.txt", 60)

    with pytest.raises(LocalStorageTokenError, match="Object not found"):
        client.resolve_token(token)
