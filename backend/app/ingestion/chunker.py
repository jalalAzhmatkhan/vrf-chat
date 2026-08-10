"""Hierarchical chunker, see
`Documentation/system-design/03-retrieval-chunking.md` §1-3 (I1.7).

Groups canonically-stored `elements` (I1.5) into `chunks` rows (5 types:
text/table/figure/procedure/entity), following document structure
(Chapter/Section/Sub-section, via each element's own `section_path` —
already computed by Docling during Stage 2, see `docling_parser.py`, and
reused here rather than recomputed).

**Most critical requirement in the whole project** (`CLAUDE.md` §4,
`02-ingestion-pipeline.md` §6): an inline icon must never end up in a
different chunk than the sentence it's part of. This module enforces it
structurally, not incidentally: any `icon`/`figure_caption` element with a
resolvable `parent_id` is **always** placed into the same chunk as its
parent element (via an explicit `element_id -> chunk index` map maintained
during the walk), regardless of any other chunk-boundary rule (section
change, size cap, etc.) that would otherwise apply.

Decoupled from a live DB session on purpose (same pattern as every other
ingestion stage module): `ChunkableElement` is a minimal, plain-dataclass
view of a stored `elements` row (+ its page number), so this module is
fully unit-testable without a DB. The caller (I1.10 orchestrator) queries
`Element` joined with `Page` and maps rows to `ChunkableElement`, then
`store_chunks` persists the result.

Known simplification (documented, not silently skipped): this MVP does not
build separate "container" chunks for each section level (e.g. a chunk
representing "Chapter 7.3" itself, with `parent_chunk_id` pointing child
chunks at it) — `parent_chunk_id` is left `None` for every chunk produced
here. `section_path` metadata alone (present on every chunk) already gives
the hierarchical breadcrumb needed for citation/retrieval filtering;
building actual parent/child chunk rows for context-expansion is real,
separable follow-up work, not required for retrieval to function, per the
project's stated preference for adding complexity only when proven
necessary (see e.g. `04-provider-abstractions.md` Bagian D §1 rationale
applying the same principle elsewhere).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db.models.chunks import Chunk

CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_TABLE = "table"
CHUNK_TYPE_FIGURE = "figure"
CHUNK_TYPE_PROCEDURE = "procedure"
CHUNK_TYPE_ENTITY = "entity"

# Element types swept into a running "text" chunk (headings included — a
# heading's own `section_path` differs from what preceded it, which the
# grouping-boundary rule below already treats as a natural chunk break, so
# no special-casing is needed for headings to start fresh chunks at section
# transitions).
_TEXT_GROUP_ELEMENT_TYPES = frozenset({"paragraph", "heading", "warning", "note"})
_ALWAYS_JOIN_PARENT_ELEMENT_TYPES = frozenset({"icon", "figure_caption"})

DEFAULT_MAX_CHUNK_CHARS = 2000


@dataclass(slots=True)
class ChunkableElement:
    """Minimal view of a stored `elements` row (+ its page number) needed
    for chunking. See module docstring for why this is decoupled from the
    ORM `Element` model directly."""

    element_id: int
    element_type: str
    text: str | None
    page_number: int
    parent_id: int | None
    section_path: list[str]
    image_uri: str | None
    visual_description: dict[str, Any] | None
    kg_candidate_entities: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ChunkDraft:
    chunk_type: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    content_text: str
    content_structured: dict[str, Any] | None
    element_ids: list[int]


def _parse_markdown_table(markdown: str | None) -> list[dict[str, str]]:
    """Best-effort re-parse of a Docling-exported markdown table
    (`| Header1 | Header2 |` / separator row / data rows) back into
    `rows: list[dict]` — the only structured table data available to this
    module (only the markdown rendering was persisted to `elements.text` in
    I1.2/I1.5, not the original `TableItem`/`DataFrame`). Returns `[]` for
    anything that doesn't look like a markdown table (defensive, not an
    error — `content_text` still preserves the raw markdown regardless)."""
    if not markdown or "|" not in markdown:
        return []

    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    header = _cells(lines[0])
    body_lines = lines[1:]
    # Skip a markdown separator row (e.g. `| --- | --- |`) if present.
    if body_lines and all(set(c) <= {"-", ":"} for c in _cells(body_lines[0]) if c):
        body_lines = body_lines[1:]

    rows: list[dict[str, str]] = []
    for line in body_lines:
        cells = _cells(line)
        # `_cells` always returns >= 1 element for any non-empty line (even
        # `"".split("|")` yields `['']`), so `row` below is never an empty
        # dict — no `if row:` guard needed.
        row = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        rows.append(row)
    return rows


class _HTMLTableCellParser(HTMLParser):
    """Minimal `<table>`/`<tr>`/`<td>`/`<th>` extractor for real PaddleOCR-VL
    table-reparse output (`extraction_method='docling+paddle_table'`, QA
    phase-1-qa-report.md §1.3 F1) — e.g.
    `<table border=1 ...><tr><td colspan="2">Warning</td></tr>...`, NOT
    markdown. Uses stdlib `html.parser` (no new dependency) rather than a
    full HTML tree — this module only ever needs flat row/cell text, never
    nested-table or block-level structure. `colspan` is expanded (the
    cell's text repeated across every column it spans) so downstream
    consumers never need to special-case merged cells. `<img>` tags
    naturally disappear (they produce no `handle_data` text) — dropping
    embedded images from `rows` is intentional; the image itself is not
    lost, `content_text` still preserves the full raw HTML.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell_parts: list[str] | None = None
        self._current_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell_parts = []
            self._current_colspan = 1
            for name, value in attrs:
                if name == "colspan" and value:
                    try:
                        self._current_colspan = max(1, int(value))
                    except ValueError:
                        self._current_colspan = 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._current_row is not None:
            text = "".join(self._current_cell_parts or []).strip()
            self._current_row.extend([text] * self._current_colspan)
            self._current_cell_parts = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell_parts is not None:
            self._current_cell_parts.append(data)


def _parse_html_table(html_text: str) -> list[dict[str, str]]:
    """Real PaddleOCR-VL table-reparse HTML -> `rows: list[dict]` (see
    `_HTMLTableCellParser`). Unlike `_parse_markdown_table`, keys are
    positional (`col_0`, `col_1`, ...) rather than inferred from the first
    row: real `docling+paddle_table` HTML output commonly has no true
    header row at all (e.g. a `colspan=2` "Warning" title row is not a set
    of column names) — inferring semantic headers from it would be
    misleading, unlike markdown pipe-tables where the first row genuinely
    is the header by syntax convention."""
    parser = _HTMLTableCellParser()
    try:
        parser.feed(html_text)
    except Exception:  # pragma: no cover - defensive against malformed HTML
        return []

    raw_rows = [row for row in parser.rows if row]
    if not raw_rows:
        return []

    max_cols = max(len(row) for row in raw_rows)
    return [
        {f"col_{i}": (row[i] if i < len(row) else "") for i in range(max_cols)}
        for row in raw_rows
    ]


def _parse_table_rows(content_text: str | None) -> list[dict[str, str]]:
    """Format-detecting dispatcher — see `03-retrieval-chunking.md` §3 and
    QA phase-1-qa-report.md §1.3 (F1). `elements.text` for a table can be
    EITHER Docling's native markdown export OR, when Stage 4 table-reparse
    (`extraction_method='docling+paddle_table'`) overwrote it, real
    PaddleOCR-VL HTML output (`app/ingestion/canonical_store.py`
    `store_pages_and_elements`: `text = cascade_result.table_reparse.markdown
    or text` — despite the `TableReparseResult.markdown` field name, the
    real value is HTML, not markdown, confirmed against real data).

    Detection is "`<table` appears anywhere" (case-insensitive), NOT merely
    a prefix check — real PaddleOCR-VL output sometimes prepends a caption
    OUTSIDE the table itself (e.g. `<div style="text-align: center;">
    Parameter [A] Refrigerant amount...</div>\\n\\n<table ...>`, a real
    example found during F1 fix verification against document_id=3, chunk
    id 3734) which a strict `startswith("<table")` check would miss and
    silently fall through to the markdown parser (returning `[]` again —
    exactly the bug this fix addresses). `_parse_html_table`'s
    `HTMLParser`-based extraction only ever looks at `tr`/`td`/`th` tags,
    so feeding it the caption `<div>` alongside the table is harmless (the
    `<div>` text is simply never emitted as row data)."""
    if not content_text:
        return []
    if "<table" in content_text.lower():
        return _parse_html_table(content_text)
    return _parse_markdown_table(content_text)


def _new_chunk(element: ChunkableElement, chunk_type: str) -> ChunkDraft:
    return ChunkDraft(
        chunk_type=chunk_type,
        section_path=list(element.section_path),
        page_start=element.page_number,
        page_end=element.page_number,
        content_text="",
        content_structured=None,
        element_ids=[],
    )


def _append_text(chunk: ChunkDraft, element: ChunkableElement) -> None:
    if element.text:
        chunk.content_text = f"{chunk.content_text}\n{element.text}".strip()
    chunk.element_ids.append(element.element_id)
    chunk.page_end = element.page_number


def build_chunks(elements: list[ChunkableElement]) -> list[ChunkDraft]:
    """Build text/table/figure/procedure chunks from a document's elements,
    in the order given (caller must supply reading order — page_number then
    original insertion/id order, matching `canonical_store.py`'s insert
    order)."""
    chunks: list[ChunkDraft] = []
    element_id_to_chunk_index: dict[int, int] = {}
    current_text_chunk_index: int | None = None
    current_procedure_chunk_index: int | None = None

    for element in elements:
        # WAJIB: icon/figure_caption always join their parent's chunk,
        # overriding every other grouping rule below. See module docstring.
        if (
            element.element_type in _ALWAYS_JOIN_PARENT_ELEMENT_TYPES
            and element.parent_id is not None
            and element.parent_id in element_id_to_chunk_index
        ):
            chunk_index = element_id_to_chunk_index[element.parent_id]
            _append_text(chunks[chunk_index], element)
            element_id_to_chunk_index[element.element_id] = chunk_index
            continue

        if element.element_type == "table":
            chunk = _new_chunk(element, CHUNK_TYPE_TABLE)
            chunk.content_text = element.text or ""
            chunk.content_structured = {
                "markdown": element.text,
                "rows": _parse_table_rows(element.text),
            }
            chunk.element_ids = [element.element_id]
            chunks.append(chunk)
            element_id_to_chunk_index[element.element_id] = len(chunks) - 1
            current_text_chunk_index = None
            current_procedure_chunk_index = None
            continue

        if element.element_type == "figure":
            chunk = _new_chunk(element, CHUNK_TYPE_FIGURE)
            description = (element.visual_description or {}).get("description")
            chunk.content_text = element.text or description or ""
            chunk.content_structured = {
                "image_uri": element.image_uri,
                "visual_description": element.visual_description,
            }
            chunk.element_ids = [element.element_id]
            chunks.append(chunk)
            element_id_to_chunk_index[element.element_id] = len(chunks) - 1
            current_text_chunk_index = None
            current_procedure_chunk_index = None
            continue

        if element.element_type == "list":
            reuse_procedure = (
                current_procedure_chunk_index is not None
                and chunks[current_procedure_chunk_index].section_path == element.section_path
            )
            if reuse_procedure:
                assert current_procedure_chunk_index is not None  # narrows for mypy
                procedure_index = current_procedure_chunk_index
            else:
                chunks.append(_new_chunk(element, CHUNK_TYPE_PROCEDURE))
                procedure_index = len(chunks) - 1
                steps: list[dict[str, Any]] = []
                chunks[procedure_index].content_structured = {"steps": steps}
            current_procedure_chunk_index = procedure_index

            chunk = chunks[procedure_index]
            steps = chunk.content_structured["steps"]  # type: ignore[index]
            steps.append(
                {
                    "step_number": len(steps) + 1,
                    "step_text": element.text or "",
                    "element_id": element.element_id,
                }
            )
            _append_text(chunk, element)
            element_id_to_chunk_index[element.element_id] = procedure_index
            current_text_chunk_index = None
            continue

        if element.element_type in _TEXT_GROUP_ELEMENT_TYPES:
            reuse_text = (
                current_text_chunk_index is not None
                and chunks[current_text_chunk_index].section_path == element.section_path
                and len(chunks[current_text_chunk_index].content_text) < DEFAULT_MAX_CHUNK_CHARS
            )
            if reuse_text:
                assert current_text_chunk_index is not None  # narrows for mypy
                text_index = current_text_chunk_index
            else:
                chunks.append(_new_chunk(element, CHUNK_TYPE_TEXT))
                text_index = len(chunks) - 1
            current_text_chunk_index = text_index

            chunk = chunks[text_index]
            _append_text(chunk, element)
            element_id_to_chunk_index[element.element_id] = text_index
            current_procedure_chunk_index = None
            continue

        # Unrecognized/other element types (defensive fallback): standalone
        # text chunk, doesn't disturb the running text/procedure grouping.
        chunk = _new_chunk(element, CHUNK_TYPE_TEXT)
        _append_text(chunk, element)
        chunks.append(chunk)
        element_id_to_chunk_index[element.element_id] = len(chunks) - 1

    return chunks


def build_entity_chunks(elements: list[ChunkableElement]) -> list[ChunkDraft]:
    """Aggregate `kg_candidate_entities` (I1.9) across a document's elements
    into one `entity` chunk per unique `(name, entity_type)` pair — per
    `03-retrieval-chunking.md` §2 ("membantu query 'apa itu TH3?' tanpa
    nunggu Neo4j"). Occurrences are merged (all pages/element_ids kept),
    keyed on `KGCandidateEntity` fields as persisted in
    `elements.kg_candidate_entities` (I1.9/I1.5)."""
    aggregated: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for element in elements:
        for candidate in element.kg_candidate_entities:
            key = (candidate["name"], candidate["entity_type"])
            aggregated.setdefault(key, []).append(candidate)

    chunks: list[ChunkDraft] = []
    for (name, entity_type), occurrences in sorted(aggregated.items()):
        pages = sorted({o["page"] for o in occurrences})
        element_ids = sorted({o["element_id"] for o in occurrences})
        best = max(occurrences, key=lambda o: o["confidence"])
        content_text = (
            f"{name} ({entity_type}) — referenced on page(s) "
            f"{', '.join(str(p) for p in pages)}."
        )
        chunks.append(
            ChunkDraft(
                chunk_type=CHUNK_TYPE_ENTITY,
                section_path=[],
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                content_text=content_text,
                content_structured={
                    "name": name,
                    "entity_type": entity_type,
                    "occurrence_count": len(occurrences),
                    "pages": pages,
                    "best_confidence": best["confidence"],
                },
                element_ids=element_ids,
            )
        )
    return chunks


def compute_content_hash(draft: ChunkDraft) -> str:
    payload = "|".join(
        [
            draft.chunk_type,
            ">".join(draft.section_path),
            draft.content_text,
            ",".join(str(i) for i in draft.element_ids),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def store_chunks(db: Session, document_id: int, drafts: list[ChunkDraft]) -> list[Chunk]:
    """Persist `drafts` as `chunks` rows for `document_id`.

    Idempotency strategy (simpler than I1.5's page-level hash diffing,
    deliberately — see module docstring rationale): chunking is a cheap,
    purely-derived recomputation from already-stored `elements` (unlike
    Docling/PaddleOCR-VL, which are the actually expensive stages), so
    re-running the chunker for a document **replaces** all of its existing
    chunks wholesale rather than diffing chunk-by-chunk. `content_hash` is
    still computed and stored per chunk for I1.8's embedding-idempotency use
    (`06-data-schema.md` §1 "content_hash # idempotency re-embedding").
    """
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    db.flush()

    rows: list[Chunk] = []
    for draft in drafts:
        row = Chunk(
            document_id=document_id,
            chunk_type=draft.chunk_type,
            parent_chunk_id=None,
            section_path=draft.section_path,
            page_start=draft.page_start,
            page_end=draft.page_end,
            content_text=draft.content_text,
            content_structured=draft.content_structured,
            element_ids=draft.element_ids,
            content_hash=compute_content_hash(draft),
            embedding_status="pending",
        )
        db.add(row)
        rows.append(row)

    db.flush()
    db.commit()
    return rows
