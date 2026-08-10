"""Deterministic hybrid retrieval (`search_documents`) over the Qdrant
`vrf_chunks` collection — dense + sparse (BM25) legs fused with Qdrant-native
RRF in a single query, per
`Documentation/system-design/03-retrieval-chunking.md` §6 (C2.1).

This module is the ONLY place that decides *how* a search is executed
(fusion strategy, filters, circuit breaker). Agent tools (`app/agent/tools.py`,
C2.3) call `search_documents` with explicit, structured arguments — the
agent itself never chooses retrieval strategy, per
`01-architecture-overview.md` §4 ("deterministic tools, not model judgment").

Qdrant payload (see `app/ingestion/embedder.py` `_build_payload`) does NOT
include `content_text` (kept out of the vector store payload by design —
`06-data-schema.md` §2 lists chunk_id/document_id/chunk_type/section_path/
page_start/page_end/element_ids/model_family/content_hash only). This
function therefore does two round-trips per call: a single Qdrant hybrid
query, followed by one Postgres `SELECT ... WHERE id IN (...)` to fetch the
full `content_text`/`content_structured` for the ranked chunk ids returned by
Qdrant — necessary because the LLM/agent context needs the actual chunk
text, not just IDs.

`error_code`/`component` parameters are deliberately NOT applied as Qdrant
payload filters — the payload has no free-text field to filter against.
Per `03-retrieval-chunking.md` §6 ("identifier teknis harus diperlakukan
lexical"), they are folded into the *query text* sent to both embedding
legs, relying on the sparse/BM25 leg (which IS lexical) to surface exact
identifier matches (`P8`, `CN105`, `TH3`, ...). This is a deliberate,
documented implementation decision, not an oversight — see also the
`error_codes` table note below.

**Known data gap (2026-08-10, verified against the real document_id=3
corpus)**: the denormalized `error_codes` lookup table
(`app/db/models/error_codes.py`, described in `06-data-schema.md` §1 as "for
exact match cepat") is never populated by any ingestion stage — `SELECT
count(*) FROM error_codes` is 0 even after document_id=3's full ingestion.
`search_documents`'s lexical (sparse/BM25) leg is therefore the only
functioning exact-identifier retrieval path today, not a fallback for an
empty table. This gap plus a proposed fix belongs to the ingestion
pipeline (`app/ingestion/`), out of scope for C2.1 — flagged in the
STATUS REPORT rather than fixed here to avoid touching ingestion modules a
parallel KG Foundation work stream owns.

Circuit breaker (`05-streaming-and-api-contract.md` §3 point 5): the Qdrant
call is bounded by `circuit_breaker_seconds` (default
`RETRIEVAL_CIRCUIT_BREAKER_SECONDS`, 8s) via Qdrant's own client-side
`timeout` kwarg. On timeout/any transport error, `search_documents` returns
a `RetrievalResult` with an empty chunk list and
`circuit_breaker_triggered=True` instead of raising — the rest of the chat
pipeline (agent/chat service, C2.3/C2.4) proceeds best-effort with whatever
TTFT budget is left, per the design's explicit "lanjutkan dengan
best-effort ... alih-alih blocking" instruction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.observability import get_logger
from app.db.models.chunks import Chunk
from app.ingestion.embedder import SparseVectorData

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_TOP_K = 20
DEFAULT_CIRCUIT_BREAKER_SECONDS = 8.0
# Qdrant hybrid RRF fusion looks at more candidates per leg than the final
# `top_k` returned — a wider prefetch improves fusion quality (per-leg
# recall) without changing the final result count.
PREFETCH_MULTIPLIER = 4


class DenseEmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover


class SparseEmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[SparseVectorData]: ...  # pragma: no cover


@dataclass(slots=True)
class RetrievedChunk:
    """One fused, Postgres-enriched retrieval hit — the unit agent tools
    (C2.3) build `Citation`/context out of."""

    chunk_id: int
    document_id: int
    chunk_type: str
    section_path: list[str] | None
    page_start: int | None
    page_end: int | None
    element_ids: list[int]
    content_text: str
    content_structured: dict[str, Any] | None
    model_family: str | None
    score: float
    rank: int


@dataclass(slots=True)
class RetrievalResult:
    query: str
    effective_query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    elapsed_ms: int = 0
    circuit_breaker_triggered: bool = False


def _build_effective_query(query: str, error_code: str | None, component: str | None) -> str:
    """Fold `error_code`/`component` into the query text sent to both
    embedding legs — see module docstring for why this replaces Qdrant
    payload filtering for these two parameters."""
    extra = [token for token in (error_code, component) if token]
    if not extra:
        return query
    return f"{query} {' '.join(extra)}".strip()


def _build_filter(model_family: str | None, chunk_type: str | None) -> qmodels.Filter | None:
    conditions: list[qmodels.Condition] = []
    if model_family:
        conditions.append(
            qmodels.FieldCondition(
                key="model_family", match=qmodels.MatchValue(value=model_family)
            )
        )
    if chunk_type:
        conditions.append(
            qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value=chunk_type))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def _fetch_chunks_by_ids(db: Session, chunk_ids: list[int]) -> dict[int, Chunk]:
    if not chunk_ids:
        return {}
    rows = db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids))).scalars().all()
    return {row.id: row for row in rows}


def search_documents(
    db: Session,
    qdrant_client: QdrantClient,
    dense_model: DenseEmbeddingModel,
    sparse_model: SparseEmbeddingModel,
    *,
    query: str,
    collection: str,
    model_family: str | None = None,
    error_code: str | None = None,
    component: str | None = None,
    chunk_type: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    circuit_breaker_seconds: float = DEFAULT_CIRCUIT_BREAKER_SECONDS,
) -> RetrievalResult:
    """Deterministic hybrid (dense + sparse, Qdrant-native RRF fusion) search.

    This is the single retrieval primitive every agent tool
    (`search_error_code`, `find_component`, `find_troubleshooting_procedure`,
    `find_wiring_diagram`, the plain `search_documents` tool — C2.3) is built
    on top of. Signature matches
    `01-architecture-overview.md` §4: `search_documents(query, model=None,
    error_code=None, component=None, top_k=20)` (`model_family` here is that
    doc's `model` param, renamed to match the Qdrant payload field it filters
    on; `chunk_type` is an additional, documented extension used by
    chunk-type-specific tools, e.g. `find_wiring_diagram` -> `chunk_type="figure"`).
    """
    start = time.monotonic()
    effective_query = _build_effective_query(query, error_code, component)

    dense_vector = dense_model.embed([effective_query])[0]
    sparse_vector = sparse_model.embed([effective_query])[0]
    query_filter = _build_filter(model_family, chunk_type)
    prefetch_limit = max(top_k * PREFETCH_MULTIPLIER, top_k)

    try:
        response = qdrant_client.query_points(
            collection_name=collection,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            timeout=int(circuit_breaker_seconds),
        )
    except Exception:
        # Circuit breaker (see module docstring): ANY transport/timeout
        # error from Qdrant is treated as best-effort-empty, never fatal —
        # the caller (agent tool -> chat service) keeps going with whatever
        # TTFT budget remains rather than blocking/propagating.
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "retrieval.circuit_breaker_triggered",
            extra={"query": query, "elapsed_ms": elapsed_ms, "collection": collection},
        )
        return RetrievalResult(
            query=query,
            effective_query=effective_query,
            chunks=[],
            elapsed_ms=elapsed_ms,
            circuit_breaker_triggered=True,
        )

    ordered_ids: list[int] = []
    payload_by_chunk_id: dict[int, dict[str, Any]] = {}
    scores: dict[int, float] = {}
    for point in response.points:
        payload = point.payload or {}
        raw_chunk_id = payload.get("chunk_id")
        if raw_chunk_id is None:
            continue
        chunk_id = int(raw_chunk_id)
        ordered_ids.append(chunk_id)
        payload_by_chunk_id[chunk_id] = payload
        scores[chunk_id] = point.score

    chunk_rows = _fetch_chunks_by_ids(db, ordered_ids)

    chunks: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        row = chunk_rows.get(chunk_id)
        if row is None:
            # Payload references a chunk no longer in Postgres (deleted /
            # re-ingested since the point was upserted) — skip rather than
            # fabricate content for a citation that can't be traced back.
            continue
        payload = payload_by_chunk_id[chunk_id]
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                section_path=row.section_path,
                page_start=row.page_start,
                page_end=row.page_end,
                element_ids=row.element_ids or [],
                content_text=row.content_text,
                content_structured=row.content_structured,
                model_family=payload.get("model_family"),
                score=scores[chunk_id],
                rank=rank,
            )
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return RetrievalResult(
        query=query,
        effective_query=effective_query,
        chunks=chunks,
        elapsed_ms=elapsed_ms,
        circuit_breaker_triggered=False,
    )
