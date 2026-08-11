"""Unit tests for `app/ingestion/embedder.py` (I1.8).

Real fastembed model construction (`_build_dense_model`/
`_build_sparse_model`) is `# pragma: no cover` (see module docstring
"Coverage/testing trade-off") — `FastEmbedDenseModel`/`FastEmbedSparseModel`
orchestration (build-once caching) IS tested here via monkeypatching those
two builder functions with fakes, same pattern as `DoclingParser`/
`LocalPaddleOCRVLClient`. `embed_and_upsert_chunks` is tested end-to-end
against an in-memory SQLite session + a fake Qdrant client + fake embedding
models implementing the plain `DenseEmbeddingModel`/`SparseEmbeddingModel`
protocols directly (no fastembed/Qdrant dependency at all).
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.chunks import Chunk
from app.ingestion import embedder as em


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_chunk(db: Session, **overrides: Any) -> Chunk:
    base = dict(
        document_id=1,
        chunk_type="text",
        section_path=["Ch1"],
        page_start=1,
        page_end=1,
        content_text="Check the compressor.",
        content_structured=None,
        element_ids=[1],
        content_hash="hash-1",
        embedding_status="pending",
    )
    base.update(overrides)
    chunk = Chunk(**base)
    db.add(chunk)
    db.commit()
    return chunk


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


def test_build_payload() -> None:
    db = _make_session()
    chunk = _make_chunk(db)

    payload = em._build_payload(chunk, model_family="PUHY-P_YKB-A1")

    assert payload == {
        "chunk_id": chunk.id,
        "document_id": 1,
        "chunk_type": "text",
        "section_path": ["Ch1"],
        "page_start": 1,
        "page_end": 1,
        "element_ids": [1],
        "model_family": "PUHY-P_YKB-A1",
        "content_hash": "hash-1",
    }


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


class FakeQdrantClient:
    def __init__(self, exists: bool = False) -> None:
        self._exists = exists
        self.create_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def collection_exists(self, collection: str) -> bool:
        return self._exists

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)

    def upsert(self, collection_name: str, points: Any) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})

    def delete(self, collection_name: str, points_selector: Any) -> None:
        self.delete_calls.append(
            {"collection_name": collection_name, "points_selector": points_selector}
        )


def test_delete_chunks_by_document_scopes_by_document_id_filter() -> None:
    """[2026-08-10 operational-habit correction] Must be a scoped
    `FilterSelector` delete (one document's points), never a full
    collection drop — see module docstring."""
    client = FakeQdrantClient()
    em.delete_chunks_by_document(client, "vrf_chunks", document_id=3)

    assert len(client.delete_calls) == 1
    call = client.delete_calls[0]
    assert call["collection_name"] == "vrf_chunks"
    selector = call["points_selector"]
    assert selector.filter.must[0].key == "document_id"
    assert selector.filter.must[0].match.value == 3


def test_ensure_collection_skips_if_exists() -> None:
    client = FakeQdrantClient(exists=True)
    em.ensure_collection(client, "vrf_chunks", dense_dim=384)
    assert client.create_calls == []


def test_ensure_collection_creates_if_missing() -> None:
    client = FakeQdrantClient(exists=False)
    em.ensure_collection(client, "vrf_chunks", dense_dim=384)
    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert call["collection_name"] == "vrf_chunks"
    assert em.DENSE_VECTOR_NAME in call["vectors_config"]
    assert em.SPARSE_VECTOR_NAME in call["sparse_vectors_config"]


# ---------------------------------------------------------------------------
# FastEmbedDenseModel / FastEmbedSparseModel — orchestration
# ---------------------------------------------------------------------------


class _FakeNumpyVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeSparseOutput:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = _FakeNumpyVector(indices)  # type: ignore[assignment]
        self.values = _FakeNumpyVector(values)  # type: ignore[assignment]


class _FakeDenseModel:
    def embed(self, texts: list[str]) -> list[_FakeNumpyVector]:
        return [_FakeNumpyVector([0.1, 0.2]) for _ in texts]


class _FakeSparseModel:
    def embed(self, texts: list[str]) -> list[_FakeSparseOutput]:
        return [_FakeSparseOutput([1, 2], [0.5, 0.5]) for _ in texts]


def test_fastembed_dense_model_caches_and_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = []

    def fake_build(model_name: str) -> _FakeDenseModel:
        build_calls.append(model_name)
        return _FakeDenseModel()

    monkeypatch.setattr(em, "_build_dense_model", fake_build)
    model = em.FastEmbedDenseModel("BAAI/bge-small-en-v1.5")
    assert model.is_loaded is False

    vectors = model.embed(["a", "b"])
    model.embed(["c"])  # second call must not rebuild

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert build_calls == ["BAAI/bge-small-en-v1.5"]
    assert model.is_loaded is True


def test_fastembed_sparse_model_caches_and_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = []

    def fake_build(model_name: str) -> _FakeSparseModel:
        build_calls.append(model_name)
        return _FakeSparseModel()

    monkeypatch.setattr(em, "_build_sparse_model", fake_build)
    model = em.FastEmbedSparseModel("Qdrant/bm25")
    assert model.is_loaded is False

    vectors = model.embed(["a"])
    model.embed(["b"])

    assert vectors == [em.SparseVectorData(indices=[1, 2], values=[0.5, 0.5])]
    assert build_calls == ["Qdrant/bm25"]
    assert model.is_loaded is True


# ---------------------------------------------------------------------------
# embed_and_upsert_chunks
# ---------------------------------------------------------------------------


class FakeDenseEmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeSparseEmbeddingModel:
    def embed(self, texts: list[str]) -> list[em.SparseVectorData]:
        return [em.SparseVectorData(indices=[1], values=[1.0]) for _ in texts]


def test_embed_and_upsert_chunks_embeds_pending_only() -> None:
    db = _make_session()
    pending = _make_chunk(db, content_text="Pending chunk.", embedding_status="pending")
    already_embedded = _make_chunk(
        db, content_text="Already embedded.", embedding_status="embedded", vector_id="99"
    )
    # [2026-08-11, round 3 memory fix] capture plain ints BEFORE the call —
    # `embed_and_upsert_chunks` now `db.expunge()`s the `Chunk` rows it
    # processes (see module docstring "round 3"), so `pending`/
    # `already_embedded` (the SAME identity-mapped objects, this test's own
    # session) become detached afterward; re-query fresh copies instead of
    # `db.refresh()`ing a detached instance (which raises).
    pending_id = pending.id
    already_embedded_id = already_embedded.id

    client = FakeQdrantClient()
    count = em.embed_and_upsert_chunks(
        db,
        client,
        document_id=1,
        collection="vrf_chunks",
        dense_model=FakeDenseEmbeddingModel(),
        sparse_model=FakeSparseEmbeddingModel(),
        model_family="PUHY-P_YKB-A1",
    )

    assert count == 1
    assert len(client.upsert_calls) == 1
    point = client.upsert_calls[0]["points"][0]
    assert point.id == pending_id
    assert point.vector[em.DENSE_VECTOR_NAME] == [0.1, 0.2]
    assert point.vector[em.SPARSE_VECTOR_NAME].indices == [1]

    refreshed_pending = db.get(Chunk, pending_id)
    refreshed_already_embedded = db.get(Chunk, already_embedded_id)
    assert refreshed_pending is not None
    assert refreshed_already_embedded is not None
    assert refreshed_pending.embedding_status == em.EMBEDDING_STATUS_EMBEDDED
    assert refreshed_pending.vector_id == str(pending_id)
    # Untouched — was already embedded before this call.
    assert refreshed_already_embedded.vector_id == "99"


def test_embed_and_upsert_chunks_no_pending_returns_zero() -> None:
    db = _make_session()
    _make_chunk(db, embedding_status="embedded", vector_id="1")
    client = FakeQdrantClient()

    count = em.embed_and_upsert_chunks(
        db,
        client,
        document_id=1,
        collection="vrf_chunks",
        dense_model=FakeDenseEmbeddingModel(),
        sparse_model=FakeSparseEmbeddingModel(),
    )

    assert count == 0
    assert client.upsert_calls == []


def test_embed_and_upsert_chunks_scoped_to_document_id() -> None:
    db = _make_session()
    _make_chunk(db, document_id=1, embedding_status="pending")
    _make_chunk(db, document_id=2, embedding_status="pending")
    client = FakeQdrantClient()

    count = em.embed_and_upsert_chunks(
        db,
        client,
        document_id=1,
        collection="vrf_chunks",
        dense_model=FakeDenseEmbeddingModel(),
        sparse_model=FakeSparseEmbeddingModel(),
    )

    assert count == 1
    doc2_chunk = db.execute(
        select(Chunk).where(Chunk.document_id == 2)
    ).scalar_one()
    assert doc2_chunk.embedding_status == "pending"


# ---------------------------------------------------------------------------
# batch_size (2026-08-11, round 3 memory fix)
# ---------------------------------------------------------------------------


def test_embed_and_upsert_chunks_rejects_non_positive_batch_size() -> None:
    db = _make_session()
    client = FakeQdrantClient()

    with pytest.raises(ValueError, match="batch_size"):
        em.embed_and_upsert_chunks(
            db,
            client,
            document_id=1,
            collection="vrf_chunks",
            dense_model=FakeDenseEmbeddingModel(),
            sparse_model=FakeSparseEmbeddingModel(),
            batch_size=0,
        )


def test_embed_and_upsert_chunks_processes_multiple_batches() -> None:
    """5 pending chunks, `batch_size=2` -> 3 inner iterations (2, 2, 1) —
    ALL 5 still get embedded/upserted/marked-embedded, in 3 separate
    `qdrant_client.upsert()` calls rather than 1 (proving the loop is
    actually batching, not just accepting the param and ignoring it)."""
    db = _make_session()
    chunk_ids = [
        _make_chunk(db, content_text=f"Chunk {i}", embedding_status="pending").id
        for i in range(5)
    ]
    client = FakeQdrantClient()
    embed_call_sizes: list[int] = []

    class SizeTrackingDenseModel:
        def embed(self, texts: list[str]) -> list[list[float]]:
            embed_call_sizes.append(len(texts))
            return [[0.1, 0.2] for _ in texts]

    count = em.embed_and_upsert_chunks(
        db,
        client,
        document_id=1,
        collection="vrf_chunks",
        dense_model=SizeTrackingDenseModel(),
        sparse_model=FakeSparseEmbeddingModel(),
        batch_size=2,
    )

    assert count == 5
    assert embed_call_sizes == [2, 2, 1]
    assert len(client.upsert_calls) == 3
    upserted_ids = sorted(
        point.id for call in client.upsert_calls for point in call["points"]
    )
    assert upserted_ids == sorted(chunk_ids)

    stored = db.execute(select(Chunk).where(Chunk.document_id == 1)).scalars().all()
    assert all(chunk.embedding_status == em.EMBEDDING_STATUS_EMBEDDED for chunk in stored)


def test_embed_and_upsert_chunks_expunges_processed_chunks_from_session() -> None:
    """[2026-08-11, round 3 memory fix] Regression guard, same pattern as
    `canonical_store.py`/`chunker.py`'s analogous tests — processed `Chunk`
    rows must not remain resident in the session's identity map after each
    inner batch (this project's `Session` has `expire_on_commit=False`,
    `app/db/engine.py`, so a plain `db.commit()` alone would not release
    them)."""
    db = _make_session()
    for i in range(3):
        _make_chunk(db, content_text=f"Chunk {i}", embedding_status="pending")
    client = FakeQdrantClient()

    em.embed_and_upsert_chunks(
        db,
        client,
        document_id=1,
        collection="vrf_chunks",
        dense_model=FakeDenseEmbeddingModel(),
        sparse_model=FakeSparseEmbeddingModel(),
    )

    identity_map_classes = {type(state.object) for state in db.identity_map.all_states()}
    assert identity_map_classes == set()
