"""Query-time embedding model factory (C2.1).

Deliberately reuses `app.ingestion.embedder`'s `FastEmbedDenseModel`/
`FastEmbedSparseModel` (the exact same fastembed model classes used to embed
chunks at ingestion time, I1.8) rather than a second implementation — using a
different model/dimension for query-time embedding than for stored chunk
embedding would silently break retrieval (dense vector dimension mismatch,
or a different sparse vocabulary). `EMBEDDER_DENSE_MODEL`/
`EMBEDDER_SPARSE_MODEL` (`app/core/config.py`) are shared config, single
source of truth for both ingestion and retrieval.
"""

from __future__ import annotations

from app.core.config import Settings
from app.ingestion.embedder import FastEmbedDenseModel, FastEmbedSparseModel
from app.retrieval.hybrid_search import DenseEmbeddingModel, SparseEmbeddingModel


def build_dense_embedding_model(settings: Settings) -> DenseEmbeddingModel:
    return FastEmbedDenseModel(settings.EMBEDDER_DENSE_MODEL)


def build_sparse_embedding_model(settings: Settings) -> SparseEmbeddingModel:
    return FastEmbedSparseModel(settings.EMBEDDER_SPARSE_MODEL)
