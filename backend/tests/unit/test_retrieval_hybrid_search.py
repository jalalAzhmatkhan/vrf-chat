"""Unit tests for `app/retrieval/hybrid_search.py` (C2.1).

`search_documents` is tested end-to-end against an in-memory SQLite session
(real `Chunk` rows) + a fake Qdrant client (`FakeQdrantClient`, mirroring the
fakes already used in `tests/unit/test_ingestion_embedder.py`/
`tests/unit/test_retrieval_vector_store.py`) + fake dense/sparse embedding
models — no real Qdrant/fastembed dependency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.chunks import Chunk
from app.retrieval import hybrid_search as hs


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_chunk(db: Session, **overrides: Any) -> Chunk:
    base = dict(
        document_id=3,
        chunk_type="text",
        section_path=["Ch7", "7.3 Troubleshooting"],
        page_start=238,
        page_end=239,
        content_text="Check the compressor for error P8.",
        content_structured=None,
        element_ids=[101, 102],
        content_hash="hash-1",
        embedding_status="embedded",
        vector_id=None,
    )
    base.update(overrides)
    chunk = Chunk(**base)
    db.add(chunk)
    db.commit()
    return chunk


class FakeDenseModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeSparseModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[hs.SparseVectorData]:
        self.calls.append(texts)
        return [hs.SparseVectorData(indices=[1, 2], values=[0.5, 0.5]) for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.response: Any = SimpleNamespace(points=[])
        self.raise_error: Exception | None = None

    def query_points(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


def _fake_point(chunk_id: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id, score=score, payload={"chunk_id": chunk_id, "model_family": "REYQ"}
    )


# ---------------------------------------------------------------------------
# _build_effective_query / _build_filter (pure helpers)
# ---------------------------------------------------------------------------


def test_build_effective_query_no_extras() -> None:
    assert hs._build_effective_query("outdoor unit stops", None, None) == "outdoor unit stops"


def test_build_effective_query_with_error_code_and_component() -> None:
    result = hs._build_effective_query("check", "P8", "compressor")
    assert result == "check P8 compressor"


def test_build_effective_query_with_error_code_only() -> None:
    assert hs._build_effective_query("check", "P8", None) == "check P8"


def test_build_filter_none() -> None:
    assert hs._build_filter(None, None) is None


def test_build_filter_model_family_only() -> None:
    f = hs._build_filter("REYQ", None)
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "model_family"


def test_build_filter_both() -> None:
    f = hs._build_filter("REYQ", "table")
    assert f is not None
    assert {c.key for c in f.must} == {"model_family", "chunk_type"}


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


def test_search_documents_happy_path_enriches_from_postgres() -> None:
    db = _make_session()
    chunk = _make_chunk(db)
    client = FakeQdrantClient()
    client.response = SimpleNamespace(points=[_fake_point(chunk.id, 0.95)])

    result = hs.search_documents(
        db,
        client,
        FakeDenseModel(),
        FakeSparseModel(),
        query="compressor error",
        collection="vrf_chunks",
    )

    assert result.circuit_breaker_triggered is False
    assert len(result.chunks) == 1
    hit = result.chunks[0]
    assert hit.chunk_id == chunk.id
    assert hit.document_id == 3
    assert hit.content_text == "Check the compressor for error P8."
    assert hit.page_start == 238
    assert hit.element_ids == [101, 102]
    assert hit.score == 0.95
    assert hit.rank == 1
    assert result.elapsed_ms >= 0


def test_search_documents_folds_error_code_and_component_into_embed_calls() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    dense = FakeDenseModel()
    sparse = FakeSparseModel()

    hs.search_documents(
        db,
        client,
        dense,
        sparse,
        query="check",
        collection="vrf_chunks",
        error_code="P8",
        component="compressor",
    )

    assert dense.calls == [["check P8 compressor"]]
    assert sparse.calls == [["check P8 compressor"]]


def test_search_documents_builds_hybrid_prefetch_with_rrf_fusion() -> None:
    db = _make_session()
    client = FakeQdrantClient()

    hs.search_documents(
        db,
        client,
        FakeDenseModel(),
        FakeSparseModel(),
        query="thermistor",
        collection="vrf_chunks",
        top_k=5,
    )

    kwargs = client.last_kwargs
    assert kwargs is not None
    assert len(kwargs["prefetch"]) == 2
    used = {p.using for p in kwargs["prefetch"]}
    assert used == {"dense", "sparse"}
    assert kwargs["query"].fusion.value == "rrf"
    assert kwargs["limit"] == 5


def test_search_documents_applies_model_family_and_chunk_type_filters() -> None:
    db = _make_session()
    client = FakeQdrantClient()

    hs.search_documents(
        db,
        client,
        FakeDenseModel(),
        FakeSparseModel(),
        query="wiring",
        collection="vrf_chunks",
        model_family="REYQ",
        chunk_type="figure",
    )

    kwargs = client.last_kwargs
    assert kwargs is not None
    assert kwargs["query_filter"] is not None
    keys = {c.key for c in kwargs["query_filter"].must}
    assert keys == {"model_family", "chunk_type"}
    for prefetch in kwargs["prefetch"]:
        assert prefetch.filter is not None


def test_search_documents_circuit_breaker_on_timeout_returns_empty_best_effort() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.raise_error = TimeoutError("qdrant took too long")

    result = hs.search_documents(
        db,
        client,
        FakeDenseModel(),
        FakeSparseModel(),
        query="compressor",
        collection="vrf_chunks",
        circuit_breaker_seconds=8.0,
    )

    assert result.circuit_breaker_triggered is True
    assert result.chunks == []


def test_search_documents_skips_points_missing_chunk_id_payload() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.response = SimpleNamespace(
        points=[SimpleNamespace(id=1, score=0.5, payload={})]
    )

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert result.chunks == []


def test_search_documents_skips_stale_chunk_ids_not_in_postgres() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    # chunk_id 999 does not exist in Postgres (e.g. re-ingested since upsert).
    client.response = SimpleNamespace(points=[_fake_point(999, 0.8)])

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert result.chunks == []


def test_search_documents_ranks_preserve_qdrant_order() -> None:
    db = _make_session()
    chunk_a = _make_chunk(db, content_text="A", content_hash="a")
    chunk_b = _make_chunk(db, content_text="B", content_hash="b")
    client = FakeQdrantClient()
    # Qdrant returns b before a (fusion re-ranked) — result order must follow.
    client.response = SimpleNamespace(
        points=[_fake_point(chunk_b.id, 0.9), _fake_point(chunk_a.id, 0.7)]
    )

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert [c.chunk_id for c in result.chunks] == [chunk_b.id, chunk_a.id]
    assert [c.rank for c in result.chunks] == [1, 2]


def test_search_documents_no_chunks_no_postgres_query_needed() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.response = SimpleNamespace(points=[])

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert result.chunks == []
    assert result.circuit_breaker_triggered is False


def test_search_documents_default_effective_query_equals_query_when_no_extras() -> None:
    db = _make_session()
    client = FakeQdrantClient()

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="hello", collection="vrf_chunks"
    )

    assert result.query == "hello"
    assert result.effective_query == "hello"
