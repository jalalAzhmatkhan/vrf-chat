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

    def collection_exists(self, collection: str) -> bool:
        return self._exists

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)

    def upsert(self, collection_name: str, points: Any) -> None:
        self.upsert_calls.append({"collection_name": collection_name, "points": points})


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
    assert point.id == pending.id
    assert point.vector[em.DENSE_VECTOR_NAME] == [0.1, 0.2]
    assert point.vector[em.SPARSE_VECTOR_NAME].indices == [1]

    db.refresh(pending)
    db.refresh(already_embedded)
    assert pending.embedding_status == em.EMBEDDING_STATUS_EMBEDDED
    assert pending.vector_id == str(pending.id)
    # Untouched — was already embedded before this call.
    assert already_embedded.vector_id == "99"


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
