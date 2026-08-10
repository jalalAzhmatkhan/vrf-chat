"""Deterministic agent tools — per
`Documentation/system-design/01-architecture-overview.md` §4: "Agent tidak
menentukan strategi retrieval sendiri — retrieval (hybrid dense+BM25+RRF)
adalah logika deterministic di `retrieval/`, dipanggil oleh tools." Every
function in this module has an explicit, structured signature; the LLM
picks *which* tool and *what arguments*, never *how* a search executes
internally (fusion strategy, filters, circuit breaker — all fixed logic in
`app/retrieval/hybrid_search.py`/`app/agent/context_builder.py`).

Each tool:
1. (optionally) runs heuristic query expansion (`app/domain/query_expansion.py`
   — no LLM call, §6 03-retrieval-chunking.md).
2. Retrieves via `app.retrieval.search_documents` and/or direct Postgres
   element lookups.
3. Runs the retrieved chunks/elements through
   `app.agent.context_builder.build_context`/`fetch_context_elements` to get
   marker-annotated text (§5.1 layer 1) — the string returned to the LLM as
   the tool's result **already contains `{{el:ID}}` markers** where
   applicable.
4. Merges every element it touched into `AgentDeps.context_elements` — the
   running whitelist `app/agent/answer_postprocess.py` validates the final
   answer's markers against (every tool call in a turn contributes, not
   just the last one).

These are plain functions (not decorated with `@agent.tool`) so they are
independently unit-testable without constructing a Pydantic AI `Agent` at
all — `app/agent/vrf_agent.py` registers thin wrappers around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.context_builder import (
    MARKER_ELEMENT_TYPES,
    BuiltContext,
    ContextElement,
    build_context,
    build_marker,
    fetch_context_elements,
)
from app.db.models.documents import Page
from app.db.models.elements import Element
from app.db.models.error_codes import ErrorCode
from app.domain.query_expansion import KnownEntities, expand_query
from app.retrieval.hybrid_search import (
    DenseEmbeddingModel,
    SparseEmbeddingModel,
    search_documents,
)

NO_RESULTS_MESSAGE = "No relevant content found in the indexed manuals for this query."
CIRCUIT_BREAKER_MESSAGE = (
    "Retrieval timed out (circuit breaker triggered) — no results available for this "
    "query right now. Do not invent an answer based on this; say so explicitly or refuse "
    "if this was the only avenue for the user's question."
)
NOT_FOUND_PAGE_MESSAGE = "The requested document/page was not found."
NOT_FOUND_FIGURE_MESSAGE = "The requested figure/diagram element was not found."
KG_STUB_MESSAGE = (
    "Knowledge graph search is not available yet (planned for a later phase). "
    "Use search_documents/find_component instead."
)


@dataclass(slots=True)
class AgentDeps:
    """Per-turn dependencies injected into every tool call — constructed
    once per chat request (`app/api/v1/chat.py`, C2.4), never chosen/altered
    by the LLM itself."""

    db: Session
    qdrant_client: QdrantClient
    dense_model: DenseEmbeddingModel
    sparse_model: SparseEmbeddingModel
    collection: str
    model_family: str | None = None
    default_top_k: int = 20
    max_context_chunks: int = 8
    circuit_breaker_seconds: float = 8.0
    known_entities: KnownEntities | None = None
    # Running whitelist merged across every tool call this turn — see
    # module docstring point 4. `app/agent/answer_postprocess.py` validates
    # the final answer's markers against this.
    context_elements: dict[int, ContextElement] = field(default_factory=dict)
    tool_call_count: int = 0
    any_chunks_retrieved: bool = False


def _merge_context(deps: AgentDeps, context: BuiltContext) -> None:
    deps.context_elements.update(context.elements_by_id)
    if context.chunks:
        deps.any_chunks_retrieved = True


def _render_context_or_message(context: BuiltContext, *, circuit_breaker_triggered: bool) -> str:
    if context.chunks:
        return context.context_text
    if circuit_breaker_triggered:
        return CIRCUIT_BREAKER_MESSAGE
    return NO_RESULTS_MESSAGE


def tool_search_documents(
    deps: AgentDeps,
    query: str,
    *,
    model_family: str | None = None,
    error_code: str | None = None,
    component: str | None = None,
    top_k: int | None = None,
) -> str:
    """General-purpose hybrid search over the indexed manuals."""
    deps.tool_call_count += 1
    expanded = expand_query(query, deps.known_entities)
    result = search_documents(
        deps.db,
        deps.qdrant_client,
        deps.dense_model,
        deps.sparse_model,
        query=expanded.expanded_query_text,
        collection=deps.collection,
        model_family=model_family or deps.model_family,
        error_code=error_code,
        component=component,
        top_k=top_k or deps.default_top_k,
        circuit_breaker_seconds=deps.circuit_breaker_seconds,
    )
    context = build_context(deps.db, result.chunks, max_chunks=deps.max_context_chunks)
    _merge_context(deps, context)
    return _render_context_or_message(
        context, circuit_breaker_triggered=result.circuit_breaker_triggered
    )


def _exact_error_code_lookup(deps: AgentDeps, error_code: str, model_family: str | None) -> str:
    """Exact-match lookup against the denormalized `error_codes` table
    (`06-data-schema.md` §1 "for exact match cepat"). Currently a no-op in
    practice for document_id=3 (table verified empty, see
    `app/retrieval/hybrid_search.py` module docstring "Known data gap") —
    kept as the documented fast path for whenever that gap is fixed by the
    ingestion pipeline, rather than assuming it will always be empty."""
    stmt = select(ErrorCode).where(ErrorCode.code == error_code)
    if model_family:
        stmt = stmt.where(ErrorCode.model_family == model_family)
    rows = deps.db.execute(stmt).scalars().all()
    element_ids = [row.element_id for row in rows if row.element_id is not None]
    if not element_ids:
        return ""

    elements_by_id = fetch_context_elements(deps.db, element_ids)
    deps.context_elements.update(elements_by_id)
    if elements_by_id:
        deps.any_chunks_retrieved = True

    parts = []
    for row in rows:
        row_element_id = row.element_id
        element = elements_by_id.get(row_element_id) if row_element_id is not None else None
        marker = ""
        if (
            element is not None
            and row_element_id is not None
            and element.element_type in MARKER_ELEMENT_TYPES
        ):
            marker = build_marker(row_element_id)
        description = row.description or (element.text if element is not None else "")
        parts.append(f"{marker} Error code {row.code}: {description}".strip())
    return "\n".join(parts)


def tool_search_error_code(
    deps: AgentDeps, error_code: str, *, model_family: str | None = None
) -> str:
    """Look up a specific error/malfunction code — combines the exact
    `error_codes` table lookup (fast path, currently empty in practice) with
    a hybrid search biased toward this identifier (the sparse/BM25 leg is
    what actually surfaces exact codes today, see
    `app/retrieval/hybrid_search.py`)."""
    exact = _exact_error_code_lookup(deps, error_code, model_family)
    searched = tool_search_documents(
        deps,
        f"error code {error_code}",
        model_family=model_family,
        error_code=error_code,
    )
    if exact and searched not in (NO_RESULTS_MESSAGE, CIRCUIT_BREAKER_MESSAGE):
        return f"{exact}\n\n---\n\n{searched}"
    return exact or searched


def tool_find_component(
    deps: AgentDeps, component_name: str, *, model_family: str | None = None
) -> str:
    """Look up a component/part (e.g. "compressor", "TH3", "CN105")."""
    return tool_search_documents(
        deps, component_name, model_family=model_family, component=component_name
    )


def tool_find_troubleshooting_procedure(
    deps: AgentDeps, symptom: str, *, model_family: str | None = None
) -> str:
    """Find a step-by-step troubleshooting procedure for `symptom`. Prefers
    `procedure`-type chunks (ordered steps preserved,
    `03-retrieval-chunking.md` §2) — falls back to an unrestricted search if
    no procedure chunk matches, since the relevant guidance may live in a
    narrative `text` chunk instead."""
    deps.tool_call_count += 1
    expanded = expand_query(symptom, deps.known_entities)
    procedure_result = search_documents(
        deps.db,
        deps.qdrant_client,
        deps.dense_model,
        deps.sparse_model,
        query=expanded.expanded_query_text,
        collection=deps.collection,
        model_family=model_family or deps.model_family,
        chunk_type="procedure",
        top_k=deps.default_top_k,
        circuit_breaker_seconds=deps.circuit_breaker_seconds,
    )
    context = build_context(deps.db, procedure_result.chunks, max_chunks=deps.max_context_chunks)
    _merge_context(deps, context)
    if context.chunks:
        return context.context_text
    if procedure_result.circuit_breaker_triggered:
        return CIRCUIT_BREAKER_MESSAGE
    return tool_search_documents(deps, symptom, model_family=model_family)


def tool_find_wiring_diagram(
    deps: AgentDeps, component: str, *, model_family: str | None = None
) -> str:
    """Find an electrical wiring/circuit diagram (`figure`-type chunks)
    related to `component`."""
    deps.tool_call_count += 1
    expanded = expand_query(component, deps.known_entities)
    result = search_documents(
        deps.db,
        deps.qdrant_client,
        deps.dense_model,
        deps.sparse_model,
        query=expanded.expanded_query_text,
        collection=deps.collection,
        model_family=model_family or deps.model_family,
        component=component,
        chunk_type="figure",
        top_k=deps.default_top_k,
        circuit_breaker_seconds=deps.circuit_breaker_seconds,
    )
    context = build_context(deps.db, result.chunks, max_chunks=deps.max_context_chunks)
    _merge_context(deps, context)
    return _render_context_or_message(
        context, circuit_breaker_triggered=result.circuit_breaker_triggered
    )


def tool_get_document_page(deps: AgentDeps, document_id: int, page_number: int) -> str:
    """Return every element on a specific page of a specific document, in
    reading order — for when the user (or a citation) explicitly references
    a page number directly, bypassing semantic/lexical search."""
    deps.tool_call_count += 1
    page = deps.db.execute(
        select(Page).where(Page.document_id == document_id, Page.page_number == page_number)
    ).scalar_one_or_none()
    if page is None:
        return NOT_FOUND_PAGE_MESSAGE

    element_ids = [
        row[0]
        for row in deps.db.execute(
            select(Element.id).where(Element.page_id == page.id).order_by(Element.id)
        ).all()
    ]
    elements_by_id = fetch_context_elements(deps.db, element_ids)
    deps.context_elements.update(elements_by_id)
    if not elements_by_id:
        return NOT_FOUND_PAGE_MESSAGE
    deps.any_chunks_retrieved = True

    parts: list[str] = []
    for element_id in element_ids:
        element = elements_by_id.get(element_id)
        if element is None:
            continue
        if element.element_type in {"icon", "figure", "diagram", "figure_caption"}:
            parts.append(build_marker(element_id))
        if element.text:
            parts.append(element.text)
    return "\n".join(parts)


def tool_get_figure(deps: AgentDeps, figure_id: int) -> str:
    """Return a single figure/diagram/icon element by its internal element
    id (e.g. one already seen in a previous tool call's `{{el:ID}}` marker,
    when the LLM wants more detail on it)."""
    deps.tool_call_count += 1
    elements_by_id = fetch_context_elements(deps.db, [figure_id])
    element = elements_by_id.get(figure_id)
    if element is None:
        return NOT_FOUND_FIGURE_MESSAGE

    deps.context_elements.update(elements_by_id)
    deps.any_chunks_retrieved = True
    marker = build_marker(figure_id)
    visual_description = element.visual_description or {}
    visual_description_text = visual_description.get("description")
    description = element.text or visual_description_text
    return f"{marker} {description or ''}".strip()


def tool_search_knowledge_graph(
    deps: AgentDeps, entity: str, *, relation: str | None = None
) -> str:
    """Stub — knowledge graph search is Fase 3 scope
    (`01-architecture-overview.md` §4: "search_knowledge_graph(entity,
    relation=None)  # Fase 3 saja, stub di MVP"). Always returns a fixed
    message; deliberately touches neither `db` nor `qdrant_client` so it
    can never silently do a *different* kind of search under the hood."""
    deps.tool_call_count += 1
    return KG_STUB_MESSAGE


__all__ = [
    "AgentDeps",
    "tool_find_component",
    "tool_find_troubleshooting_procedure",
    "tool_find_wiring_diagram",
    "tool_get_document_page",
    "tool_get_figure",
    "tool_search_documents",
    "tool_search_error_code",
    "tool_search_knowledge_graph",
]
