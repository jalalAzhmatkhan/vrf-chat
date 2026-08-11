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

**§6.1 relevance-floor probe (2026-08-11, F2-03 "never invent" gap fix)**:
after the main hybrid query succeeds, one additional dense-only Qdrant
query (`_dense_relevance_probe`, `limit=1`, reuses the already-computed
dense vector — no extra embedding cost) measures `top_dense_score`, a
signal consumed ONLY by `enforce_never_invent_safety_net`
(`app/agent/answer_postprocess.py`) to decide whether to force a refusal —
it never changes which chunks are selected as context. See
`MIN_RELEVANCE_SCORE`'s docstring and
`Documentation/system-design/03-retrieval-chunking.md` §6.1 for the full
empirical calibration and its explicitly-acknowledged limits.
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

# §6.1.4 (`Documentation/system-design/03-retrieval-chunking.md`) — relevance
# floor for the "never invent" safety net (`app/agent/answer_postprocess.py`
# `enforce_never_invent_safety_net`), NOT used to filter/rank the chunks
# returned to the LLM (RRF fusion above remains the sole source of truth for
# *context*).
#
# [REVISED 2026-08-11, cycle 2] 0.68 -> 0.72. QA recalibrated
# (`Documentation/qa-reports/phase-2-qa-report-cycle2.md` §5,
# `vrf-qa/scripts/phase2_cycle2_recalibrate_relevance.py`, same methodology
# as the original §6.1.3 calibration) against the live `vrf_chunks`
# collection now that the full 6-document corpus is ingested (11,993
# points, up from 3,607/1 document), n=62 (up from 19), same production
# embedding model (`BAAI/bge-small-en-v1.5`):
#   - Clearly out-of-scope queries: top-1 dense cosine 0.5232-0.7101
#     (up from 0.4932-0.6609 at 1 document — more chunks means a higher
#     chance *something* looks "close enough" to any given query)
#   - Genuine in-scope queries (all sub-classes: verbatim/terse/per-vendor/
#     obscure-tail): top-1 dense cosine 0.7356-0.9197
#   - 0.68 had drifted INSIDE the out-of-scope distribution by this point —
#     5/16 clearly-out-of-scope queries scored above it (one confirmed live:
#     "What is the square root of 144?", 0.6819, still answered). 0.72 is
#     QA's threshold-sweep recommendation: 16/16 out-of-scope rejected,
#     0/36 in-scope wrongly rejected at n=62. Margin to the lowest genuine
#     in-scope score (0.7356, "Mitsubishi check code 1102 PUHY-P") is thin
#     (0.0156) — do NOT raise this above 0.72 without a much larger
#     in-scope sample (0.74 already misclassifies 1/36 in-scope on QA's
#     sweep).
#
# Explicitly does NOT separate domain-adjacent near-miss queries (e.g. "how
# do I fix a window AC that isn't cooling") from genuine in-scope ones —
# confirmed again at this corpus size (domain-adjacent 0.6646-0.8007, full
# overlap with in-scope); this is an accepted residual risk under active
# escalation (§6.1.8: cross-encoder reranker promoted to "Fase 2, active
# next priority"), not something this constant is meant to solve.
#
# RECALIBRATION POLICY (§6.1.4, tightened this cycle): NOT "recalibrate once
# after a big corpus change" — recalibrate EVERY time a new source document
# finishes ingesting (`status=ready`), no "significant" threshold judgment
# call. The 1->6 document shift alone was enough to invalidate 0.68; there
# is no evidence a smaller shift is safe to ignore. Run
# `vrf-qa/scripts/phase2_cycle2_recalibrate_relevance.py` (read-only, no
# LLM/GPU, well under 1 minute at 11,993 points) before treating a newly
# ingested document as production-ready, and update this docstring's
# snapshot (n, corpus size, date, source) every time the constant changes —
# never leave a stale snapshot next to a live value.
MIN_RELEVANCE_SCORE = 0.72


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
    # §6.1 relevance-floor signal — top-1 dense-only cosine similarity
    # (`MIN_RELEVANCE_SCORE` docstring above), `None` if the dense-only
    # probe itself failed/timed out (treated as "no relevance signal
    # available", never as "definitely irrelevant" — see
    # `_dense_relevance_probe`) or if the circuit breaker already
    # short-circuited the whole call before reaching it.
    top_dense_score: float | None = None


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


def _dense_relevance_probe(
    qdrant_client: QdrantClient,
    collection: str,
    dense_vector: list[float],
    *,
    timeout_seconds: float,
) -> float | None:
    """§6.1 relevance-floor signal — a second, dense-only query (no fusion,
    `limit=1`, `with_payload=False`) reusing the already-computed
    `dense_vector`, so this costs one extra local Qdrant round-trip and zero
    extra embedding computation. Purely advisory: any failure here (timeout,
    transport error, empty collection) returns `None` rather than raising —
    this signal must never be able to break/degrade the main hybrid search
    it rides alongside, and `None` is treated by the safety net
    (`app/agent/answer_postprocess.py`) as "no signal available", never as
    "definitely irrelevant"."""
    try:
        response = qdrant_client.query_points(
            collection_name=collection,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=1,
            with_payload=False,
            timeout=int(timeout_seconds),
        )
    except Exception:  # noqa: BLE001 — advisory-only signal, never fatal
        logger.warning("retrieval.dense_relevance_probe_failed", extra={"collection": collection})
        return None
    if not response.points:
        return None
    return response.points[0].score


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

    # §6.1 relevance-floor probe — deliberately AFTER the main hybrid query
    # has already succeeded (never adds latency to a call that was already
    # going to be circuit-broken/fail) and BEFORE we compute the final
    # `elapsed_ms`, so its cost is fully visible in that number (measured,
    # not estimated — see `03-retrieval-chunking.md` §6.1.5).
    dense_probe_start = time.monotonic()
    top_dense_score = _dense_relevance_probe(
        qdrant_client, collection, dense_vector, timeout_seconds=circuit_breaker_seconds
    )
    dense_probe_elapsed_ms = int((time.monotonic() - dense_probe_start) * 1000)
    logger.info(
        "retrieval.dense_relevance_probe",
        extra={
            "collection": collection,
            "top_dense_score": top_dense_score,
            "elapsed_ms": dense_probe_elapsed_ms,
        },
    )

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
        top_dense_score=top_dense_score,
    )
