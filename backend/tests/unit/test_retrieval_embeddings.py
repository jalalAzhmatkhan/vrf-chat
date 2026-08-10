"""Unit tests for `app/retrieval/embeddings.py` (C2.1) — verifies the query-
time factory wires the configured model names into the same
`FastEmbedDenseModel`/`FastEmbedSparseModel` classes ingestion uses, without
constructing a real fastembed model (`is_loaded` stays False until `.embed`
is actually called, which none of these tests trigger)."""

from __future__ import annotations

from app.core.config import Settings
from app.ingestion.embedder import FastEmbedDenseModel, FastEmbedSparseModel
from app.retrieval import embeddings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_build_dense_embedding_model_uses_configured_model_name() -> None:
    settings = _settings(EMBEDDER_DENSE_MODEL="BAAI/bge-small-en-v1.5")
    model = embeddings.build_dense_embedding_model(settings)
    assert isinstance(model, FastEmbedDenseModel)
    assert model._model_name == "BAAI/bge-small-en-v1.5"
    assert model.is_loaded is False


def test_build_sparse_embedding_model_uses_configured_model_name() -> None:
    settings = _settings(EMBEDDER_SPARSE_MODEL="Qdrant/bm25")
    model = embeddings.build_sparse_embedding_model(settings)
    assert isinstance(model, FastEmbedSparseModel)
    assert model._model_name == "Qdrant/bm25"
    assert model.is_loaded is False
