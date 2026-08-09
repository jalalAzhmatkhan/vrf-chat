from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.storage.local_adapter import LocalFilesystemStorageClient


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        OBJECT_STORAGE_BACKEND="local",
        OBJECT_STORAGE_LOCAL_BASE_PATH=str(tmp_path),
        OBJECT_STORAGE_LOCAL_TOKEN_SECRET="integration-test-secret",
    )  # type: ignore[arg-type]


def test_get_local_storage_object_serves_bytes(tmp_path) -> None:
    settings = _settings(tmp_path)
    client_lib = LocalFilesystemStorageClient(settings)
    uri = client_lib.put_object("icons/reset.png", b"icon-bytes", "image/png")
    token = client_lib.get_presigned_url(uri).removeprefix("/internal/local-storage/")

    app = create_app(settings)
    client = TestClient(app)

    response = client.get(f"/internal/local-storage/{token}")

    assert response.status_code == 200
    assert response.content == b"icon-bytes"
    assert response.headers["content-type"] == "image/png"


def test_get_local_storage_object_404_for_invalid_token(tmp_path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.get("/internal/local-storage/not-a-real-token")

    assert response.status_code == 404
