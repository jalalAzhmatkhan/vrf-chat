"""S3-compatible object storage adapter — shared by `minio` and `s3` backends.

Both are S3 APIs via `boto3`; only `endpoint_url`/`force_path_style` differ
(see `Documentation/system-design/04-provider-abstractions.md` Bagian B).
Canonical URI format: `s3://{bucket}/{key}`.
"""

from __future__ import annotations

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from app.core.config import Settings


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid s3:// URI: {uri!r}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Not a valid s3:// URI: {uri!r}")
    return bucket, key


class S3CompatibleStorageClient:
    """`ObjectStorageClient` implementation for MinIO and AWS S3."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.OBJECT_STORAGE_BUCKET
        addressing_style = "path" if settings.OBJECT_STORAGE_FORCE_PATH_STYLE else "virtual"
        self._client = boto3.client(
            "s3",
            region_name=settings.OBJECT_STORAGE_REGION,
            endpoint_url=settings.OBJECT_STORAGE_ENDPOINT_URL or None,
            use_ssl=settings.OBJECT_STORAGE_USE_SSL,
            aws_access_key_id=settings.OBJECT_STORAGE_ACCESS_KEY,
            aws_secret_access_key=settings.OBJECT_STORAGE_SECRET_KEY,
            config=BotoConfig(s3={"addressing_style": addressing_style}),
        )

    def put_object(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return f"s3://{self._bucket}/{key}"

    def get_object(self, uri: str) -> bytes:
        bucket, key = _parse_s3_uri(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body: bytes = response["Body"].read()
        return body

    def get_presigned_url(self, uri: str, expires_in_seconds: int = 3600) -> str:
        bucket, key = _parse_s3_uri(uri)
        url: str = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in_seconds,
        )
        return url

    def delete_object(self, uri: str) -> None:
        bucket, key = _parse_s3_uri(uri)
        self._client.delete_object(Bucket=bucket, Key=key)

    def exists(self, uri: str) -> bool:
        bucket, key = _parse_s3_uri(uri)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
        return True
