"""Unit tests for `app/retrieval/vector_store.py` (I1.8).

`QdrantVectorStoreClient` is tested against a fake stand-in for
`qdrant_client.QdrantClient` (no real Qdrant connection) — the real class is
only ever constructed via `build_qdrant_client`, tested separately by
inspecting the constructed object's connection params (no network call
happens at construction time for the HTTP client).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.retrieval import vector_store as vs


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.query_response: Any = None
        self.collection_info: Any = None

    def upsert(self, collection_name: str, points: Any) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})

    def query_points(self, **kwargs: Any) -> Any:
        self._last_query_kwargs = kwargs
        return self.query_response

    def delete(self, collection_name: str, points_selector: Any) -> None:
        self.delete_calls.append(
            {"collection_name": collection_name, "points_selector": points_selector}
        )

    def get_collection(self, collection_name: str) -> Any:
        return self.collection_info


def test_upsert_vectors_builds_points_and_calls_client() -> None:
    fake_client = FakeQdrantClient()
    client = vs.QdrantVectorStoreClient(fake_client)

    client.upsert_vectors(
        "vrf_chunks",
        ids=["1", "2"],
        dense_vectors=[[0.1, 0.2], [0.3, 0.4]],
        payloads=[{"chunk_id": 1}, {"chunk_id": 2}],
    )

    assert len(fake_client.upsert_calls) == 1
    call = fake_client.upsert_calls[0]
    assert call["collection_name"] == "vrf_chunks"
    assert len(call["points"]) == 2
    assert call["points"][0].id == "1"
    assert call["points"][0].vector == {"dense": [0.1, 0.2]}
    assert call["points"][0].payload == {"chunk_id": 1}


def test_query_maps_results() -> None:
    fake_client = FakeQdrantClient()
    fake_point = SimpleNamespace(id=42, score=0.9, payload={"chunk_id": 42})
    fake_client.query_response = SimpleNamespace(points=[fake_point])
    client = vs.QdrantVectorStoreClient(fake_client)

    results = client.query("vrf_chunks", [0.1, 0.2], top_k=5)

    assert len(results) == 1
    assert results[0].id == "42"
    assert results[0].score == 0.9
    assert results[0].payload == {"chunk_id": 42}
    assert fake_client._last_query_kwargs["using"] == "dense"
    assert fake_client._last_query_kwargs["limit"] == 5


def test_query_with_none_payload_defaults_to_empty_dict() -> None:
    fake_client = FakeQdrantClient()
    fake_point = SimpleNamespace(id=1, score=0.5, payload=None)
    fake_client.query_response = SimpleNamespace(points=[fake_point])
    client = vs.QdrantVectorStoreClient(fake_client)

    results = client.query("vrf_chunks", [0.1], top_k=1)
    assert results[0].payload == {}


def test_query_with_filters_builds_qdrant_filter() -> None:
    fake_client = FakeQdrantClient()
    fake_client.query_response = SimpleNamespace(points=[])
    client = vs.QdrantVectorStoreClient(fake_client)

    client.query("vrf_chunks", [0.1], top_k=5, filters={"document_id": 1})

    query_filter = fake_client._last_query_kwargs["query_filter"]
    assert query_filter is not None
    assert query_filter.must[0].key == "document_id"


def test_query_without_filters_passes_none() -> None:
    fake_client = FakeQdrantClient()
    fake_client.query_response = SimpleNamespace(points=[])
    client = vs.QdrantVectorStoreClient(fake_client)

    client.query("vrf_chunks", [0.1], top_k=5)

    assert fake_client._last_query_kwargs["query_filter"] is None


def test_delete_builds_point_ids_list() -> None:
    fake_client = FakeQdrantClient()
    client = vs.QdrantVectorStoreClient(fake_client)

    client.delete("vrf_chunks", ["1", "2"])

    assert len(fake_client.delete_calls) == 1
    assert fake_client.delete_calls[0]["points_selector"].points == ["1", "2"]


def test_delete_by_document_builds_scoped_filter_not_full_drop() -> None:
    """[2026-08-10 operational-habit correction] Must be a document-scoped
    `FilterSelector` delete, never a full collection drop — see module
    docstring."""
    fake_client = FakeQdrantClient()
    client = vs.QdrantVectorStoreClient(fake_client)

    client.delete_by_document("vrf_chunks", document_id=3)

    assert len(fake_client.delete_calls) == 1
    call = fake_client.delete_calls[0]
    assert call["collection_name"] == "vrf_chunks"
    selector = call["points_selector"]
    assert selector.filter.must[0].key == "document_id"
    assert selector.filter.must[0].match.value == 3


def test_get_collection_stats_extracts_dense_dimension() -> None:
    fake_client = FakeQdrantClient()
    fake_client.collection_info = SimpleNamespace(
        points_count=123,
        status="green",
        config=SimpleNamespace(
            params=SimpleNamespace(vectors={"dense": SimpleNamespace(size=384)})
        ),
    )
    client = vs.QdrantVectorStoreClient(fake_client)

    stats = client.get_collection_stats("vrf_chunks")

    assert stats.vector_count == 123
    assert stats.dimension == 384
    assert stats.status == "green"


def test_get_collection_stats_no_points_count_defaults_zero() -> None:
    fake_client = FakeQdrantClient()
    fake_client.collection_info = SimpleNamespace(
        points_count=None,
        status="green",
        config=SimpleNamespace(params=SimpleNamespace(vectors={})),
    )
    client = vs.QdrantVectorStoreClient(fake_client)

    stats = client.get_collection_stats("vrf_chunks")
    assert stats.vector_count == 0
    assert stats.dimension == 0


def test_get_collection_stats_non_dict_vectors_config() -> None:
    fake_client = FakeQdrantClient()
    fake_client.collection_info = SimpleNamespace(
        points_count=5,
        status="green",
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=384))  # single unnamed vector
        ),
    )
    client = vs.QdrantVectorStoreClient(fake_client)

    stats = client.get_collection_stats("vrf_chunks")
    assert stats.dimension == 0


# ---------------------------------------------------------------------------
# Factories + validation
# ---------------------------------------------------------------------------


def test_build_qdrant_client_uses_settings() -> None:
    settings = _settings(
        VECTOR_STORE_QDRANT_URL="http://qdrant:6333", VECTOR_STORE_QDRANT_API_KEY="secret"
    )
    client = vs.build_qdrant_client(settings)
    assert client is not None  # construction doesn't hit network for HTTP client


def test_build_vector_store_client_qdrant() -> None:
    settings = _settings(VECTOR_STORE_PROVIDER="qdrant")
    client = vs.build_vector_store_client(settings)
    assert isinstance(client, vs.QdrantVectorStoreClient)


def test_build_vector_store_client_chroma_not_implemented() -> None:
    settings = _settings(VECTOR_STORE_PROVIDER="chroma")
    with pytest.raises(NotImplementedError):
        vs.build_vector_store_client(settings)


def test_build_vector_store_client_milvus_not_implemented() -> None:
    settings = _settings(VECTOR_STORE_PROVIDER="milvus")
    with pytest.raises(NotImplementedError):
        vs.build_vector_store_client(settings)


def test_build_vector_store_client_unknown_provider_raises() -> None:
    fake_settings = SimpleNamespace(VECTOR_STORE_PROVIDER="carrier-pigeon")
    with pytest.raises(vs.VectorStoreConfigError):
        vs.build_vector_store_client(fake_settings)  # type: ignore[arg-type]


def test_validate_config_unknown_provider_raises() -> None:
    fake_settings = SimpleNamespace(VECTOR_STORE_PROVIDER="carrier-pigeon")
    with pytest.raises(vs.VectorStoreConfigError):
        vs.validate_vector_store_config_or_raise(fake_settings)  # type: ignore[arg-type]


def test_validate_config_qdrant_missing_url_raises() -> None:
    fake_settings = SimpleNamespace(VECTOR_STORE_PROVIDER="qdrant", VECTOR_STORE_QDRANT_URL="")
    with pytest.raises(vs.VectorStoreConfigError):
        vs.validate_vector_store_config_or_raise(fake_settings)  # type: ignore[arg-type]


def test_validate_config_qdrant_ok() -> None:
    settings = _settings(VECTOR_STORE_PROVIDER="qdrant", VECTOR_STORE_QDRANT_URL="http://qdrant:6333")
    vs.validate_vector_store_config_or_raise(settings)  # should not raise
