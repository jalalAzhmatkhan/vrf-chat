from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.storage.factory import (
    ObjectStorageConfigError,
    build_storage_client,
    validate_object_storage_or_raise,
)
from app.storage.local_adapter import LocalFilesystemStorageClient
from app.storage.s3_adapter import S3CompatibleStorageClient


def _settings(tmp_path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "OBJECT_STORAGE_BACKEND": "local",
        "OBJECT_STORAGE_LOCAL_BASE_PATH": str(tmp_path),
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_build_storage_client_local(tmp_path) -> None:
    client = build_storage_client(_settings(tmp_path))

    assert isinstance(client, LocalFilesystemStorageClient)


def test_build_storage_client_minio_requires_endpoint_url(tmp_path) -> None:
    settings = _settings(tmp_path, OBJECT_STORAGE_BACKEND="minio", OBJECT_STORAGE_ENDPOINT_URL=None)

    with pytest.raises(ObjectStorageConfigError, match="OBJECT_STORAGE_ENDPOINT_URL"):
        build_storage_client(settings)


def test_build_storage_client_minio_with_endpoint_url(tmp_path) -> None:
    settings = _settings(
        tmp_path, OBJECT_STORAGE_BACKEND="minio", OBJECT_STORAGE_ENDPOINT_URL="http://minio:9000"
    )

    client = build_storage_client(settings)

    assert isinstance(client, S3CompatibleStorageClient)


def test_build_storage_client_s3(tmp_path) -> None:
    settings = _settings(tmp_path, OBJECT_STORAGE_BACKEND="s3")

    client = build_storage_client(settings)

    assert isinstance(client, S3CompatibleStorageClient)


def test_build_storage_client_unknown_backend_raises() -> None:
    fake_settings = SimpleNamespace(OBJECT_STORAGE_BACKEND="dropbox")

    with pytest.raises(ObjectStorageConfigError, match="unknown backend"):
        build_storage_client(fake_settings)  # type: ignore[arg-type]


def test_validate_object_storage_or_raise_succeeds(tmp_path) -> None:
    validate_object_storage_or_raise(_settings(tmp_path))


def test_validate_object_storage_or_raise_propagates_error(tmp_path) -> None:
    settings = _settings(tmp_path, OBJECT_STORAGE_BACKEND="minio", OBJECT_STORAGE_ENDPOINT_URL=None)

    with pytest.raises(ObjectStorageConfigError):
        validate_object_storage_or_raise(settings)
