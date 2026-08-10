"""Deterministic Context Builder — layer 1 of the 3-layer inline-marker
design in `Documentation/system-design/05-streaming-and-api-contract.md`
§5.1 ("Context Builder (deterministik) sudah menyisipkan marker ke teks
chunk SEBELUM dikirim sebagai context ke LLM").

Every chunk's `element_ids` (`03-retrieval-chunking.md` §1/§4) already
records reading order, computed once during ingestion
(`app/ingestion/chunker.py` — icon/figure_caption elements are always
appended to their parent's chunk in reading order, the "most critical
requirement in the whole project" per that module's docstring). This module
does **no model judgment** — it re-walks that already-known order and
formats it into text with `{{el:<element_id>}}` markers spliced in at
visual-element positions, so the LLM receives context where marker
placement is already correct and is instructed (`app/agent/vrf_agent.py`
system prompt) to preserve, not invent, markers.

**Granularity note (documented limitation)**: `chunks.content_text` (as
persisted by the chunker) is a `"\\n"`-joined concatenation of each
text-bearing element in the chunk — it does not preserve character-level
mid-sentence position for an icon that Docling positioned inline within a
single source sentence. This module's reconstruction is therefore
line/element-level, not character-level: a marker is inserted immediately
before/after the icon/figure element's own text (if any) in element order,
which is the finest granularity actually available from the canonical
element representation today. This is still a strict improvement over no
positional information at all, and is explicitly what
"05-streaming-and-api-contract.md" §5.1 describes ("Context Builder cukup
memformat ulang urutan yang **sudah diketahui**").

**Chunk-type scoping**: marker reconstruction only applies to `text`/
`procedure`/`figure` chunks — narrative prose where "point of reference"
markers make sense. `table`/`entity` chunks are passed through with their
original `content_text` unchanged: `entity` chunk text is a synthesized
summary sentence (`app/ingestion/chunker.py` `build_entity_chunks`), not a
reconstruction of its constituent elements' text, so per-element
reconstruction would produce garbled, unrelated text; `table` chunks are
covered by the FE's separate Visual Reference Card (§5.2 scoping), not
inline markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.documents import Page
from app.db.models.elements import Element
from app.retrieval.hybrid_search import RetrievedChunk

# Element types that get a `{{el:ID}}` marker inserted at their position —
# per 05-streaming-and-api-contract.md §5.2 scoping (icon/figure/diagram).
# `figure_caption` is included too: it is always joined into its parent
# figure/text chunk (chunker.py `_ALWAYS_JOIN_PARENT_ELEMENT_TYPES`) and
# refers to the same visual the marker anchors to.
MARKER_ELEMENT_TYPES = frozenset({"icon", "figure", "diagram", "figure_caption"})

# Chunk types eligible for per-element marker reconstruction — see module
# docstring "Chunk-type scoping".
ANNOTATABLE_CHUNK_TYPES = frozenset({"text", "procedure", "figure"})

DEFAULT_MAX_CONTEXT_CHUNKS = 8

# F2-07 (`Documentation/qa-reports/phase-2-qa-report.md`) — a single
# `search_documents` tool result was observed at 17,000-50,558 characters
# (~4,500-13,000 tokens) with `DEFAULT_MAX_CONTEXT_CHUNKS` chunks sent at
# full `content_text` length, which alone exceeds `gpt-3.5-turbo`'s 16,385
# token window (`context_length_exceeded` from the provider). §3 point 1 of
# `05-streaming-and-api-contract.md` only bounds chunk *count*, not size per
# chunk — this bounds size too. 1,500 chars/chunk (QA's recommended figure)
# keeps a worst-case `DEFAULT_MAX_CONTEXT_CHUNKS`-chunk result to roughly
# 12,000 chars (~3,000 tokens), comfortably inside even the smallest
# supported chat model's context window with headroom for the system
# prompt/tool schemas/conversation history.
DEFAULT_MAX_CHUNK_CHARS = 1500

# Matches a `{{el:<digits>}}` marker that got cut short by truncation (e.g.
# "...{{el:61" or "...{{el:6127}" with the final brace missing) — stripped
# from a truncated chunk's tail so a partial marker is never handed to the
# LLM as context (it would not match `MARKER_PATTERN` if copied verbatim,
# but there is no reason to send a broken one at all).
_TRAILING_PARTIAL_MARKER_PATTERN = re.compile(r"\{\{el:\d*\}?$")


def build_marker(element_id: int) -> str:
    return f"{{{{el:{element_id}}}}}"


def _truncate_chunk_text(text: str, max_chars: int) -> str:
    """F2-07 per-chunk size cap — see `DEFAULT_MAX_CHUNK_CHARS` docstring.
    `max_chars <= 0` disables truncation (treated as "no limit"), matching
    how `max_chunks<=0`-style "unlimited" sentinels are usually spelled
    elsewhere in this codebase."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    truncated = _TRAILING_PARTIAL_MARKER_PATTERN.sub("", truncated)
    return f"{truncated} …[truncated]"


@dataclass(slots=True)
class ContextElement:
    """Element metadata needed both for marker placement and for building
    a `Citation` (`app/agent/schemas.py`) if the LLM ends up referencing it."""

    element_id: int
    element_type: str
    text: str | None
    image_uri: str | None
    visual_description: dict[str, Any] | None
    document_id: int
    page: int | None


@dataclass(slots=True)
class ContextChunk:
    chunk_id: int
    document_id: int
    chunk_type: str
    section_path: list[str] | None
    page_start: int | None
    page_end: int | None
    annotated_text: str
    score: float


@dataclass(slots=True)
class BuiltContext:
    chunks: list[ContextChunk] = field(default_factory=list)
    # Union of every element referenced by any included chunk — the
    # whitelist layer 3 validation (`app/agent/answer_postprocess.py`) uses
    # to decide which `{{el:ID}}` markers in the LLM's answer are legitimate
    # for *this turn*.
    elements_by_id: dict[int, ContextElement] = field(default_factory=dict)
    context_text: str = ""


def fetch_context_elements(db: Session, element_ids: list[int]) -> dict[int, ContextElement]:
    """Fetch `Element` rows (+ their page number) for `element_ids` and
    convert to `ContextElement`. Public — also used directly by
    `app/agent/tools.py` for tools that return single elements
    (`get_figure`) or a page's worth of elements (`get_document_page`)
    without going through hybrid retrieval."""
    if not element_ids:
        return {}
    rows = db.execute(
        select(Element, Page.page_number)
        .join(Page, Element.page_id == Page.id)
        .where(Element.id.in_(element_ids))
    ).all()
    return {
        element.id: ContextElement(
            element_id=element.id,
            element_type=element.element_type,
            text=element.text,
            image_uri=element.image_uri,
            visual_description=element.visual_description,
            document_id=element.document_id,
            page=page_number,
        )
        for element, page_number in rows
    }


def _annotate_chunk_text(
    chunk: RetrievedChunk,
    elements_by_id: dict[int, ContextElement],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    if chunk.chunk_type not in ANNOTATABLE_CHUNK_TYPES:
        return _truncate_chunk_text(chunk.content_text, max_chunk_chars)

    parts: list[str] = []
    for element_id in chunk.element_ids:
        element = elements_by_id.get(element_id)
        if element is None:
            continue
        if element.element_type in MARKER_ELEMENT_TYPES:
            parts.append(build_marker(element_id))
        if element.text:
            parts.append(element.text)

    annotated = "\n".join(parts) if parts else chunk.content_text
    return _truncate_chunk_text(annotated, max_chunk_chars)


def _format_chunk_header(chunk: RetrievedChunk) -> str:
    section = " > ".join(chunk.section_path) if chunk.section_path else ""
    pages = (
        f"{chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"{chunk.page_start}-{chunk.page_end}"
    )
    return f"[document_id={chunk.document_id} page={pages} section=\"{section}\"]"


def build_context(
    db: Session,
    retrieved_chunks: list[RetrievedChunk],
    *,
    max_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> BuiltContext:
    """Build the marker-annotated context text sent to the LLM for one tool
    call's retrieval results — per `05-streaming-and-api-contract.md` §3
    point 1 ("Batasi context yang dikirim ke LLM ... 6-10 chunk paling
    relevan"), only the top `max_chunks` (already ranked by
    `search_documents`'s RRF fusion) are included. `max_chunk_chars` (F2-07)
    additionally bounds each individual chunk's *size*, not just the count —
    see `DEFAULT_MAX_CHUNK_CHARS` docstring.
    """
    selected = retrieved_chunks[:max_chunks]
    all_element_ids = sorted({element_id for c in selected for element_id in c.element_ids})
    elements_by_id = fetch_context_elements(db, all_element_ids)

    context_chunks: list[ContextChunk] = []
    formatted_sections: list[str] = []
    for chunk in selected:
        annotated_text = _annotate_chunk_text(
            chunk, elements_by_id, max_chunk_chars=max_chunk_chars
        )
        context_chunks.append(
            ContextChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                annotated_text=annotated_text,
                score=chunk.score,
            )
        )
        formatted_sections.append(f"{_format_chunk_header(chunk)}\n{annotated_text}")

    return BuiltContext(
        chunks=context_chunks,
        elements_by_id=elements_by_id,
        context_text="\n\n---\n\n".join(formatted_sections),
    )
