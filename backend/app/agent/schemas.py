"""`TechnicalAnswer`/`Citation`/`Warning` — the chatbot's structured output
contract, per `Documentation/system-design/05-streaming-and-api-contract.md`
§5 (revised 2026-08-10 responding to UI/UX Designer wireframe gaps —
see §5.1-§5.4 for the full rationale behind each field below). This is the
**official FE-BE contract** — do not add/rename/remove fields here without a
corresponding update to that design doc first (per its BROADCAST block).

C2.3 scope note: `Citation.element_type`/`image_uri`/`visual_description`
(§5.2) and `Warning.severity` (§5.3) are implemented here even though the
design doc's priority note focuses C2.3 on §5.1 (inline marker)/§5.4
(markdown subset) — `app/agent/context_builder.py`'s marker-validation
citation backfill (§5.1 layer 3) already has the element metadata needed to
populate them at essentially no extra cost, so there is no value in
deferring a *partial* schema to C2.5.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# §5.2 scoping — image_uri/visual_description are only ever populated for
# these element types (icon/figure/diagram); other element types (table,
# paragraph, procedure, entity, ...) keep both fields `None`.
VISUAL_ELEMENT_TYPES = frozenset({"icon", "figure", "diagram"})


class Citation(BaseModel):
    document_id: str
    page: int
    element_id: str
    # "icon"|"figure"|"diagram"|"table"|"paragraph"|"procedure"|"entity"|...
    # — same value space as elements.element_type/chunks.chunk_type
    # (02-ingestion-pipeline.md §5).
    element_type: str
    quote: str | None = None
    # Populated only when element_type in VISUAL_ELEMENT_TYPES (§5.2).
    image_uri: str | None = None
    visual_description: dict[str, Any] | None = None
    # BARU §5.2.1 (F2-10) — HANYA diisi jika element_type == "table"; shape
    # matches chunks.content_structured (`{"rows": [...]}`), capped per the
    # size limits documented in `app/agent/context_builder.py`
    # `cap_content_structured_for_citation`.
    content_structured: dict[str, Any] | None = None


class Warning(BaseModel):
    message: str
    severity: Literal["safety", "note"] = "note"


class TechnicalAnswer(BaseModel):
    # Markdown-lite (constrained subset, §5.4): paragraphs (`\n\n`),
    # `1. `/`- `/`* ` lists, `**bold**`, and `{{el:<element_id>}}` inline
    # reference markers (§5.1). No headings, markdown images/tables, raw
    # HTML, or links — enforced primarily via system prompt
    # (`app/agent/vrf_agent.py`), see that module's docstring.
    answer: str
    # [C2.4] Default `0.0` (not just a bound constraint) is a deliberate
    # streaming-quality fix, not an arbitrary choice: pydantic_ai's
    # `stream_output()` only surfaces a partial object once every field
    # *without* a default has appeared in the streamed JSON so far.
    # `confidence` immediately follows `answer` in this model/the LLM's
    # tool-call JSON — if it stayed a bare required field (no default), no
    # partial `answer` text could stream at all until the model finished
    # emitting `confidence` too, which (since `answer` is usually the
    # longest field, emitted first) would arrive only once `answer` is
    # ALREADY fully generated, defeating incremental `token` SSE events
    # entirely. Verified empirically via a scripted `FunctionModel`
    # (`tests/unit/test_agent_streaming.py`). The FINAL `confidence` in the
    # `done` payload is always the model's real value, never this default —
    # full (non-partial) validation runs once the stream completes.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    related_components: list[str] = Field(default_factory=list)
    related_error_codes: list[str] = Field(default_factory=list)
    # Explicit "insufficient retrieved evidence" flag — the "never invent"
    # safety rule (05-streaming-and-api-contract.md §5, brainstorming §31).
    refused: bool = False
