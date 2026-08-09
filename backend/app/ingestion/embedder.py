"""Chunk embedding (dense + sparse) into Qdrant, see
`Documentation/system-design/03-retrieval-chunking.md` §6 and
`06-data-schema.md` §2 (I1.8).

Collection schema (`vrf_chunks`, per `06-data-schema.md` §2): named vectors
`dense` (fastembed `EMBEDDER_DENSE_MODEL`, default `BAAI/bge-small-en-v1.5`,
384-dim) + `sparse` (fastembed `EMBEDDER_SPARSE_MODEL`, default
`Qdrant/bm25`, Qdrant-native sparse vector with `Modifier.IDF` for BM25-style
scoring). Payload: `chunk_id`/`document_id`/`chunk_type`/`section_path`/
`page_start`/`page_end`/`element_ids`/`model_family`/`content_hash`.

Per `04-provider-abstractions.md` Bagian D §1, this module calls
`qdrant_client.QdrantClient` **directly** for the hybrid dense+sparse
upsert (not through `app/retrieval/vector_store.py`'s generic
`VectorStoreClient`, which is dense-only/portable) — hybrid stays a
Qdrant-native capability by design, this is not an oversight.

Coverage/testing trade-off: constructing real fastembed models
(`_build_dense_model`/`_build_sparse_model`) downloads/initializes real ONNX
models (~10-20MB, seconds not GPU-minutes — much lighter than Docling/
PaddleOCR-VL, but still a real model, not meaningfully unit-testable) —
`# pragma: no cover` with this rationale; the vector-building/payload/
idempotency orchestration around them is fully unit-tested with fakes. Real
usage verified via a live smoke test against a real Qdrant instance (see
STATUS REPORT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.observability import get_logger
from app.db.models.chunks import Chunk

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

EMBEDDING_STATUS_PENDING = "pending"
EMBEDDING_STATUS_EMBEDDED = "embedded"
EMBEDDING_STATUS_FAILED = "failed"


@dataclass(slots=True)
class SparseVectorData:
    indices: list[int]
    values: list[float]


class DenseEmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover


class SparseEmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVectorData]: ...  # pragma: no cover


def _build_dense_model(model_name: str) -> Any:  # pragma: no cover
    # See module docstring "Coverage/testing trade-off".
    from fastembed import TextEmbedding

    return TextEmbedding(model_name)


def _build_sparse_model(model_name: str) -> Any:  # pragma: no cover
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name)


class FastEmbedDenseModel:
    """Thin wrapper adapting fastembed's `TextEmbedding` (lazy numpy-array
    generator) to the plain-list `DenseEmbeddingModel` protocol above.

    `_ensure_loaded`/`embed` orchestration (build-once caching) IS unit
    tested, by monkeypatching `_build_dense_model` — only that builder
    function itself (real model construction) is excluded from coverage,
    same pattern as `DoclingParser`/`LocalPaddleOCRVLClient`."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("embedder.loading_dense_model", extra={"model": self._model_name})
            self._model = _build_dense_model(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        return [vector.tolist() for vector in model.embed(texts)]


class FastEmbedSparseModel:
    """See `FastEmbedDenseModel` docstring — same pattern."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("embedder.loading_sparse_model", extra={"model": self._model_name})
            self._model = _build_sparse_model(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[SparseVectorData]:
        model = self._ensure_loaded()
        return [
            SparseVectorData(indices=v.indices.tolist(), values=v.values.tolist())
            for v in model.embed(texts)
        ]


def ensure_collection(client: QdrantClient, collection: str, dense_dim: int) -> None:
    """Idempotent collection creation — safe to call on every ingestion run."""
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=dense_dim, distance=qmodels.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF,
            )
        },
    )


def _build_payload(chunk: Chunk, model_family: str | None) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_type": chunk.chunk_type,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "element_ids": chunk.element_ids,
        "model_family": model_family,
        "content_hash": chunk.content_hash,
    }


def embed_and_upsert_chunks(
    db: Session,
    qdrant_client: QdrantClient,
    *,
    document_id: int,
    collection: str,
    dense_model: DenseEmbeddingModel,
    sparse_model: SparseEmbeddingModel,
    model_family: str | None = None,
) -> int:
    """Embed every `pending` chunk for `document_id` and upsert to
    `collection`, updating `chunks.embedding_status`/`vector_id`.

    Returns the number of chunks embedded. Idempotent within a single call
    (only `pending` chunks are processed) — re-running after a full success
    is a no-op (nothing left `pending`); a chunk left `pending` after a
    partial failure is retried on the next call.
    """
    pending_chunks = (
        db.execute(
            select(Chunk).where(
                Chunk.document_id == document_id,
                Chunk.embedding_status == EMBEDDING_STATUS_PENDING,
            )
        )
        .scalars()
        .all()
    )
    if not pending_chunks:
        return 0

    texts = [chunk.content_text for chunk in pending_chunks]
    dense_vectors = dense_model.embed(texts)
    sparse_vectors = sparse_model.embed(texts)

    points = []
    for chunk, dense_vector, sparse_vector in zip(
        pending_chunks, dense_vectors, sparse_vectors, strict=True
    ):
        points.append(
            qmodels.PointStruct(
                id=chunk.id,
                vector={
                    DENSE_VECTOR_NAME: dense_vector,
                    SPARSE_VECTOR_NAME: qmodels.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                },
                payload=_build_payload(chunk, model_family),
            )
        )

    qdrant_client.upsert(collection_name=collection, points=points)

    for chunk in pending_chunks:
        chunk.embedding_status = EMBEDDING_STATUS_EMBEDDED
        chunk.vector_id = str(chunk.id)

    db.commit()
    return len(pending_chunks)
