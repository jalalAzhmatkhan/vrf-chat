"""Public interface of `app/retrieval/` — see
`Documentation/system-design/01-architecture-overview.md` §3 module boundary
principle ("setiap modul hanya boleh saling memanggil lewat interface publik
... tidak mengimpor internal modul lain secara langsung").

Other modules (`app/agent/tools.py`, C2.3) should import retrieval
primitives from here, not reach into `app.retrieval.hybrid_search`/
`app.retrieval.vector_store` submodules directly.
"""

from __future__ import annotations

from app.ingestion.embedder import SparseVectorData
from app.retrieval.hybrid_search import (
    DenseEmbeddingModel,
    RetrievalResult,
    RetrievedChunk,
    SparseEmbeddingModel,
    search_documents,
)
from app.retrieval.vector_store import (
    CollectionStats,
    VectorSearchResult,
    VectorStoreClient,
    VectorStoreConfigError,
    build_qdrant_client,
    build_vector_store_client,
    validate_vector_store_config_or_raise,
)

__all__ = [
    "CollectionStats",
    "DenseEmbeddingModel",
    "RetrievalResult",
    "RetrievedChunk",
    "SparseEmbeddingModel",
    "SparseVectorData",
    "VectorSearchResult",
    "VectorStoreClient",
    "VectorStoreConfigError",
    "build_qdrant_client",
    "build_vector_store_client",
    "search_documents",
    "validate_vector_store_config_or_raise",
]
