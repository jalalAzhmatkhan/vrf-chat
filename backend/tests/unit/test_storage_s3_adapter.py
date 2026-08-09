import boto3
import pytest
from moto import mock_aws

from app.core.config import Settings
from app.storage.s3_adapter import S3CompatibleStorageClient, _parse_s3_uri


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "OBJECT_STORAGE_BACKEND": "s3",
        "OBJECT_STORAGE_BUCKET": "vrf-manuals-test",
        "OBJECT_STORAGE_REGION": "us-east-1",
        "OBJECT_STORAGE_ACCESS_KEY": "test-access-key",
        "OBJECT_STORAGE_SECRET_KEY": "test-secret-key",
        "OBJECT_STORAGE_ENDPOINT_URL": None,
        "OBJECT_STORAGE_FORCE_PATH_STYLE": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def s3_bucket():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="vrf-manuals-test")
        yield


def test_parse_s3_uri_valid() -> None:
    assert _parse_s3_uri("s3://bucket/key/nested.pdf") == ("bucket", "key/nested.pdf")


@pytest.mark.parametrize("bad_uri", ["not-s3://bucket/key", "s3://bucket-only", "s3://"])
def test_parse_s3_uri_invalid(bad_uri: str) -> None:
    with pytest.raises(ValueError, match="Not a valid s3:// URI"):
        _parse_s3_uri(bad_uri)


def test_put_get_exists_delete_roundtrip(s3_bucket) -> None:
    client = S3CompatibleStorageClient(_settings())

    uri = client.put_object("manuals/puhy-p.pdf", b"pdf-bytes", "application/pdf")

    assert uri == "s3://vrf-manuals-test/manuals/puhy-p.pdf"
    assert client.exists(uri) is True
    assert client.get_object(uri) == b"pdf-bytes"

    client.delete_object(uri)

    assert client.exists(uri) is False


def test_exists_returns_false_for_missing_object(s3_bucket) -> None:
    client = S3CompatibleStorageClient(_settings())

    assert client.exists("s3://vrf-manuals-test/does/not/exist.pdf") is False


def test_get_presigned_url_returns_url(s3_bucket) -> None:
    client = S3CompatibleStorageClient(_settings())
    uri = client.put_object("icons/reset.png", b"icon", "image/png")

    url = client.get_presigned_url(uri, expires_in_seconds=120)

    assert "vrf-manuals-test" in url
    assert "Signature=" in url


def test_minio_style_configures_path_addressing() -> None:
    client = S3CompatibleStorageClient(
        _settings(
            OBJECT_STORAGE_ENDPOINT_URL="http://localhost:9000",
            OBJECT_STORAGE_FORCE_PATH_STYLE=True,
        )
    )

    assert client._client.meta.config.s3["addressing_style"] == "path"


def test_s3_style_configures_virtual_addressing() -> None:
    client = S3CompatibleStorageClient(_settings(OBJECT_STORAGE_FORCE_PATH_STYLE=False))

    assert client._client.meta.config.s3["addressing_style"] == "virtual"
