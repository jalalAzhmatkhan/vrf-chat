"""`VectorStoreClient` abstraction, see
`Documentation/system-design/04-provider-abstractions.md` Bagian D (I1.8
wires this — previously documented as existing from Fase 0, but verified
absent from the actual codebase/git history when I1.8 started; implemented
here instead of merely "wired").

Dense-only, portable-across-providers interface (per Bagian D §1/§3) — used
generically for basic dense similarity search. **Hybrid dense+sparse+RRF
stays a Qdrant-native operation** (`app/ingestion/embedder.py` calls
`qdrant_client.QdrantClient` directly for that, not through this Protocol),
per the explicit MVP decision in Bagian D §1: building a
backend-agnostic BM25/RRF layer now would be premature complexity for a
capability only Qdrant actually needs today.

Only `qdrant` is implemented — `chroma`/`milvus` are declared in `.env`
schema (Bagian D §4) for future use but `build_vector_store_client` raises
`NotImplementedError` for them (Bagian D §6: "bukan kebutuhan MVP").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import Settings

SUPPORTED_PROVIDERS = ("qdrant", "chroma", "milvus")


class VectorStoreConfigError(ValueError):
    """Raised (fail-fast) for an invalid `VECTOR_STORE_*` config combination,
    or an unimplemented provider selection."""


@dataclass(slots=True)
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CollectionStats:
    vector_count: int
    dimension: int
    status: str


class VectorStoreClient(Protocol):
    def upsert_vectors(
        self,
        collection: str,
        ids: list[str],
        dense_vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None: ...  # pragma: no cover

    def query(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]: ...  # pragma: no cover

    def delete(self, collection: str, ids: list[str]) -> None: ...  # pragma: no cover

    def get_collection_stats(self, collection: str) -> CollectionStats: ...  # pragma: no cover


def _build_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions: list[qmodels.Condition] = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return qmodels.Filter(must=conditions)


class QdrantVectorStoreClient:
    """`VectorStoreClient` implementation backed by Qdrant. Dense vectors
    are written/read under the named vector `"dense"` (see
    `app/ingestion/embedder.py` for the collection schema, which also has a
    `"sparse"` named vector this class does not touch — hybrid queries are
    Qdrant-native, called directly by `embedder.py`, not through this
    class)."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def upsert_vectors(
        self,
        collection: str,
        ids: list[str],
        dense_vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        points = [
            qmodels.PointStruct(id=point_id, vector={"dense": vector}, payload=payload)
            for point_id, vector, payload in zip(ids, dense_vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def query(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=_build_filter(filters),
            with_payload=True,
        )
        return [
            VectorSearchResult(id=str(point.id), score=point.score, payload=point.payload or {})
            for point in response.points
        ]

    def delete(self, collection: str, ids: list[str]) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=list(ids)),
        )

    def get_collection_stats(self, collection: str) -> CollectionStats:
        info = self._client.get_collection(collection_name=collection)
        vectors_config = info.config.params.vectors
        dimension = 0
        if isinstance(vectors_config, dict) and "dense" in vectors_config:
            dimension = vectors_config["dense"].size
        return CollectionStats(
            vector_count=info.points_count or 0,
            dimension=dimension,
            status=str(info.status),
        )


def build_qdrant_client(settings: Settings) -> QdrantClient:
    # `check_compatibility=False`: the default `True` makes `QdrantClient.__init__`
    # eagerly call the server to check its version — a real network round-trip
    # at construction time, which contradicts this module's "fail-fast
    # validation without a real network connection" contract
    # (`validate_vector_store_config_or_raise`) and, in practice, left a
    # dangling connection that crashed the Python interpreter at test-suite
    # shutdown (`_enter_buffered_busy` fatal error) when no server was
    # actually reachable at the configured URL.
    return QdrantClient(
        url=settings.VECTOR_STORE_QDRANT_URL,
        api_key=settings.VECTOR_STORE_QDRANT_API_KEY,
        check_compatibility=False,
    )


def build_vector_store_client(settings: Settings) -> VectorStoreClient:
    if settings.VECTOR_STORE_PROVIDER == "qdrant":
        return QdrantVectorStoreClient(build_qdrant_client(settings))
    if settings.VECTOR_STORE_PROVIDER in ("chroma", "milvus"):
        raise NotImplementedError(
            f"VECTOR_STORE_PROVIDER={settings.VECTOR_STORE_PROVIDER!r} is declared in the "
            "env schema (04-provider-abstractions.md Bagian D §4) but not implemented — "
            "only 'qdrant' is implemented for MVP (Bagian D §6)."
        )
    raise VectorStoreConfigError(
        f"VECTOR_STORE_PROVIDER: unknown provider '{settings.VECTOR_STORE_PROVIDER}', "
        f"must be one of {SUPPORTED_PROVIDERS}"
    )


def validate_vector_store_config_or_raise(settings: Settings) -> None:
    """Fail-fast validation without constructing a real client/network
    connection — call at ingestion-task startup, same rationale as
    `docling_parser`/`paddleocr_vl_cascade`'s validators (not wired into
    `create_app()` — `VECTOR_STORE_*` is only consumed by the ingestion
    pipeline in I1.8's current scope, not the chat API yet; retrieval wiring
    happens in a later phase)."""
    if settings.VECTOR_STORE_PROVIDER not in SUPPORTED_PROVIDERS:
        raise VectorStoreConfigError(
            f"VECTOR_STORE_PROVIDER: unknown provider '{settings.VECTOR_STORE_PROVIDER}', "
            f"must be one of {SUPPORTED_PROVIDERS}"
        )
    if settings.VECTOR_STORE_PROVIDER == "qdrant" and not settings.VECTOR_STORE_QDRANT_URL:
        raise VectorStoreConfigError(
            "VECTOR_STORE_QDRANT_URL is required when VECTOR_STORE_PROVIDER=qdrant."
        )
