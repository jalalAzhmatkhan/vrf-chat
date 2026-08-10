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
    """§6.1: `search_documents` now issues TWO `query_points` calls per
    invocation — the main hybrid (RRF fusion, has a `prefetch` kwarg) query,
    then the dense-only relevance probe (no `prefetch` kwarg). `self.calls`
    records both, in order; `self.response`/`self.raise_error` are only
    consulted for the main query, `self.probe_response`/`self.probe_raise_error`
    only for the probe (defaulting to `self.response` if unset, so existing
    tests that never touch the probe-specific attributes keep working
    unchanged)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: Any = SimpleNamespace(points=[])
        self.raise_error: Exception | None = None
        self.probe_response: Any | None = None
        self.probe_raise_error: Exception | None = None

    @property
    def last_kwargs(self) -> dict[str, Any] | None:
        """Kwargs of the main hybrid query specifically (the first call) —
        existing tests inspect this to assert on `prefetch`/`query_filter`,
        which only the main query has."""
        return self.calls[0] if self.calls else None

    def query_points(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "prefetch" not in kwargs:
            if self.probe_raise_error is not None:
                raise self.probe_raise_error
            return self.probe_response if self.probe_response is not None else self.response
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


# ---------------------------------------------------------------------------
# search_documents — §6.1 dense-only relevance probe (top_dense_score)
# ---------------------------------------------------------------------------


def test_search_documents_returns_top_dense_score_from_probe() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.probe_response = SimpleNamespace(points=[SimpleNamespace(score=0.8251)])

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="P8 error", collection="vrf_chunks"
    )

    assert result.top_dense_score == 0.8251


def test_search_documents_probe_call_has_no_prefetch_no_fusion_limit_one() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.probe_response = SimpleNamespace(points=[SimpleNamespace(score=0.9)])

    hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="P8 error", collection="vrf_chunks"
    )

    assert len(client.calls) == 2
    probe_kwargs = client.calls[1]
    assert "prefetch" not in probe_kwargs
    assert probe_kwargs["using"] == hs.DENSE_VECTOR_NAME
    assert probe_kwargs["limit"] == 1
    assert probe_kwargs["with_payload"] is False


def test_search_documents_probe_reuses_already_computed_dense_vector() -> None:
    """No extra embedding computation for the probe — `dense_model.embed`
    is called exactly once per `search_documents` call, same as before §6.1."""
    db = _make_session()
    client = FakeQdrantClient()
    dense = FakeDenseModel()

    hs.search_documents(
        db, client, dense, FakeSparseModel(), query="P8 error", collection="vrf_chunks"
    )

    assert dense.calls == [["P8 error"]]


def test_search_documents_top_dense_score_none_when_probe_returns_no_points() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.probe_response = SimpleNamespace(points=[])

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert result.top_dense_score is None


def test_search_documents_top_dense_score_none_when_probe_raises() -> None:
    """The probe is advisory-only — a failure there must never fail/degrade
    the main hybrid search result."""
    db = _make_session()
    client = FakeQdrantClient()
    client.response = SimpleNamespace(points=[])
    client.probe_raise_error = TimeoutError("probe timed out")

    result = hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    assert result.top_dense_score is None
    assert result.circuit_breaker_triggered is False


def test_search_documents_probe_not_called_when_main_query_circuit_breaks() -> None:
    db = _make_session()
    client = FakeQdrantClient()
    client.raise_error = TimeoutError("qdrant took too long")

    hs.search_documents(
        db, client, FakeDenseModel(), FakeSparseModel(), query="x", collection="vrf_chunks"
    )

    # Only the (failing) main query is attempted — no point paying for a
    # second round-trip once the circuit breaker has already triggered.
    assert len(client.calls) == 1


def test_dense_relevance_probe_returns_none_on_exception() -> None:
    class RaisingClient:
        def query_points(self, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    score = hs._dense_relevance_probe(
        RaisingClient(), "vrf_chunks", [0.1, 0.2], timeout_seconds=8.0
    )
    assert score is None


def test_dense_relevance_probe_returns_score() -> None:
    class ScoringClient:
        def query_points(self, **kwargs: Any) -> Any:
            return SimpleNamespace(points=[SimpleNamespace(score=0.75)])

    score = hs._dense_relevance_probe(
        ScoringClient(), "vrf_chunks", [0.1, 0.2], timeout_seconds=8.0
    )
    assert score == 0.75
