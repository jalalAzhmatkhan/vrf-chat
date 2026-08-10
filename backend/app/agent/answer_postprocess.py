"""Deterministic post-generation validation — layer 3 of the 3-layer
inline-marker design, `Documentation/system-design/05-streaming-and-api-contract.md`
§5.1: "Backend menjalankan validasi deterministik pasca-generasi (gerbang
wajib, bukan opsional) ... strip marker mana pun yang ID-nya tidak ada di
element_ids konteks yang benar-benar dipakai turn ini". Marked
**non-negotiable** by System Analyst — this is the anti-hallucination gate
for the project's single most emphasized requirement
(`CLAUDE.md` §4 inline icon association): without it, an LLM-invented
`element_id` would silently produce a citation pointing at the wrong
page/element.

Three responsibilities, always run together on every turn's final answer
before it is sent in the `done` SSE event (`app/api/v1/chat.py`, C2.4):

1. **Strip invalid markers.** Any `{{el:ID}}` in `answer.answer` whose `ID`
   is not in this turn's context element whitelim
   (`BuiltContext.elements_by_id`, i.e. an element actually retrieved and
   shown to the LLM this turn) is removed — the surrounding sentence is
   left untouched, only the marker substring itself is deleted. Every
   removal is logged as an anomaly (observability,
   `01-architecture-overview.md` §5), never silently dropped.
2. **Validate `citations[]` against the same whitelist (F2-02).** The
   marker gate above only ever covered `{{el:ID}}` references inside
   `answer.answer` — `answer.citations` (the model's own structured
   output field) was never checked at all, so a model could emit e.g.
   ``Citation(element_id="Revision History", page=282, element_type="section")``
   for an element it never actually retrieved, and it would sail straight
   through to the user (`Documentation/qa-reports/phase-2-qa-report.md`
   F2-02/F2-03 — this exact scenario was observed producing 2 fabricated
   citations for a question entirely outside the VRF/VRV domain). Same
   philosophy as layer 3 above ("model mengusulkan id, backend memutuskan
   sisanya"): any citation whose `element_id` does not resolve to an
   integer present in `context.elements_by_id` is dropped outright: every
   citation that survives has its `document_id`/`page`/`element_type`/
   `image_uri`/`visual_description`/`content_structured` (§5.2.1, F2-10)
   **overwritten** from the authoritative `ContextElement` — the model's
   own values for those fields are never
   trusted, only `element_id` (the proposal) and `quote` (a free-text
   excerpt, not positional/type metadata) are kept as given. This directly
   fixes F2-03 too, as a pure consequence: `enforce_never_invent_safety_net`
   below (unchanged) inspects `answer.citations` *after* this validation
   runs (see `app/agent/vrf_agent.py::run_agent_turn`'s call order), so a
   turn that only ever had fabricated citations now correctly ends up with
   an empty `citations` list by the time the safety net looks at it.
3. **Backfill citations.** Every marker that *does* survive validation is
   guaranteed a matching `Citation` entry (`element_id` matches) — created
   automatically from the context element's metadata if the LLM didn't
   already include one (post-validation) in its structured `citations`
   output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.context_builder import BuiltContext, ContextElement
from app.agent.schemas import VISUAL_ELEMENT_TYPES, Citation, TechnicalAnswer, Warning
from app.core.observability import get_logger
from app.retrieval.hybrid_search import MIN_RELEVANCE_SCORE

logger = get_logger(__name__)

MARKER_PATTERN = re.compile(r"\{\{el:(\d+)\}\}")
_COLLAPSE_WHITESPACE_PATTERN = re.compile(r"[ \t]{2,}")


@dataclass(slots=True)
class PostProcessResult:
    answer: TechnicalAnswer
    stripped_element_ids: tuple[int, ...]
    backfilled_element_ids: tuple[int, ...]
    # F2-02 — `citations[]` entries dropped because their `element_id` did
    # not resolve to a whitelisted context element (raw string as the model
    # sent it, since a dropped id may not even be numeric, e.g.
    # "Revision History").
    dropped_citation_ids: tuple[str, ...]


def _strip_marker(text: str, element_id: int) -> str:
    return text.replace(f"{{{{el:{element_id}}}}}", "")


def _citation_from_context_element(element: ContextElement) -> Citation:
    is_visual = element.element_type in VISUAL_ELEMENT_TYPES
    is_table = element.element_type == "table"
    return Citation(
        document_id=str(element.document_id),
        page=element.page or 0,
        element_id=str(element.element_id),
        element_type=element.element_type,
        quote=element.text,
        image_uri=element.image_uri if is_visual else None,
        visual_description=element.visual_description if is_visual else None,
        content_structured=element.content_structured if is_table else None,
    )


def _resolve_context_element(element_id: str, context: BuiltContext) -> ContextElement | None:
    """Resolve a model-proposed `Citation.element_id` (always a `str` in the
    schema) to the authoritative `ContextElement` for this turn, or `None` if
    it is not a legitimate reference — either because it is not even an
    integer (e.g. `"Revision History"`, `"3.51"`, both observed from a real
    model in QA) or because it is an integer that was never actually
    retrieved this turn (whitelist miss)."""
    try:
        parsed = int(element_id)
    except (TypeError, ValueError):
        return None
    return context.elements_by_id.get(parsed)


def _validate_and_normalize_citations(
    citations: list[Citation], context: BuiltContext
) -> tuple[list[Citation], tuple[str, ...]]:
    """F2-02 gate — see module docstring point 2. Never trust `citations[]`
    as sent by the model: drop anything outside this turn's whitelist, and
    overwrite every positional/type field on what survives from the
    authoritative `ContextElement` (`quote` is the one field left as the
    model provided it — it is free text, not something the backend has an
    authoritative value for)."""
    validated: list[Citation] = []
    dropped_ids: list[str] = []
    for citation in citations:
        element = _resolve_context_element(citation.element_id, context)
        if element is None:
            dropped_ids.append(citation.element_id)
            continue
        is_visual = element.element_type in VISUAL_ELEMENT_TYPES
        is_table = element.element_type == "table"
        validated.append(
            citation.model_copy(
                update={
                    "document_id": str(element.document_id),
                    "page": element.page or 0,
                    "element_type": element.element_type,
                    "image_uri": element.image_uri if is_visual else None,
                    "visual_description": element.visual_description if is_visual else None,
                    "content_structured": element.content_structured if is_table else None,
                }
            )
        )
    return validated, tuple(dropped_ids)


def postprocess_answer(answer: TechnicalAnswer, context: BuiltContext) -> PostProcessResult:
    """Apply the two non-negotiable deterministic gates to `answer` (produced
    by the agent this turn) against `context` (everything retrieved/shown to
    the LLM this turn, from every tool call): §5.1 layer-3 marker validation,
    and F2-02 citation validation/normalization.
    """
    valid_element_ids = set(context.elements_by_id.keys())
    found_element_ids = {int(match) for match in MARKER_PATTERN.findall(answer.answer)}
    invalid_element_ids = found_element_ids - valid_element_ids
    surviving_element_ids = found_element_ids & valid_element_ids

    fixed_text = answer.answer
    for element_id in invalid_element_ids:
        fixed_text = _strip_marker(fixed_text, element_id)
    fixed_text = _COLLAPSE_WHITESPACE_PATTERN.sub(" ", fixed_text)

    if invalid_element_ids:
        logger.warning(
            "agent.answer.invalid_marker_stripped",
            extra={"element_ids": sorted(invalid_element_ids)},
        )

    validated_citations, dropped_citation_ids = _validate_and_normalize_citations(
        answer.citations, context
    )
    if dropped_citation_ids:
        logger.warning(
            "agent.answer.invalid_citation_dropped",
            extra={"element_ids": dropped_citation_ids},
        )

    existing_by_element_id = {citation.element_id: citation for citation in validated_citations}
    citations = list(validated_citations)
    backfilled_element_ids: list[int] = []
    for element_id in sorted(surviving_element_ids):
        key = str(element_id)
        if key in existing_by_element_id:
            continue
        citations.append(_citation_from_context_element(context.elements_by_id[element_id]))
        backfilled_element_ids.append(element_id)

    updated_answer = answer.model_copy(update={"answer": fixed_text, "citations": citations})
    return PostProcessResult(
        answer=updated_answer,
        stripped_element_ids=tuple(sorted(invalid_element_ids)),
        backfilled_element_ids=tuple(backfilled_element_ids),
        dropped_citation_ids=dropped_citation_ids,
    )


NO_EVIDENCE_WARNING = Warning(
    message=(
        "No relevant information was found in the indexed manuals for this question — "
        "refusing rather than guessing."
    ),
    severity="note",
)


def enforce_never_invent_safety_net(
    answer: TechnicalAnswer,
    *,
    tool_call_count: int,
    any_chunks_retrieved: bool,
    max_dense_relevance_score: float | None = None,
    any_exact_evidence_found: bool = False,
) -> TechnicalAnswer:
    """A second, deterministic "never invent" gate independent of the system
    prompt (`app/agent/vrf_agent.py` STATIC_SYSTEM_PROMPT already instructs
    this, but that alone is a request to the model, not an enforceable
    guarantee — see task instructions: "Implementasikan sebagai perilaku
    yang bisa dites, bukan sekadar kalimat di system prompt").

    Never touches an already-`refused=True` answer, and never fires for a
    no-tool-call conversational turn (`tool_call_count == 0` — small talk,
    clarifying question) — both preserved unchanged from the original
    (pre-§6.1) version of this gate.

    Two independent ways to force `refused=True` (§6.1,
    `Documentation/system-design/03-retrieval-chunking.md`):

    1. **Original branch, unchanged**: `not any_chunks_retrieved and not
       answer.citations` — the agent called a tool but genuinely got
       nothing back at all (circuit breaker triggered, or a
       `chunk_type`/`model_family` filter matched zero points). The
       `not answer.citations` half is logically redundant in the real
       pipeline (F2-02 always empties `citations` when
       `context.elements_by_id` is empty, which is exactly what
       `any_chunks_retrieved is False` implies) — kept anyway (System
       Analyst §6.1.4 explicitly permits dropping it as "purely cosmetic",
       but this module's existing direct-unit-test coverage constructs
       `citations` independently of a real retrieval pipeline, so keeping
       it preserves that test suite's existing, intentional behavior
       unchanged rather than requiring it to be rewritten for a
       code-generalization that doesn't need to touch this branch at all).
    2. **§6.1, new**: `not any_exact_evidence_found AND
       max_dense_relevance_score is not None AND
       max_dense_relevance_score < MIN_RELEVANCE_SCORE` — the agent *did*
       retrieve chunks (branch 1 doesn't apply), but every semantic search
       this turn scored below the calibrated relevance floor (empirical
       basis + residual limitations: `03-retrieval-chunking.md` §6.1.3).
       **Deliberately NOT additionally gated on `not answer.citations`** —
       unlike branch 1, `answer.citations` can be genuinely non-empty here
       even when nothing is actually relevant: F2-02's whitelist only
       proves an `element_id` was *retrieved* this turn, not that it is
       *relevant* to the answer, so a citation built from a low-relevance
       candidate pool can pass F2-02 validation while still being
       fabricated in every way that matters to the user. This is the
       precise mechanism that let the real F2-03 QA repro ("What is the
       capital city of France...") slip through with 2 real-looking
       citations even after F2-02 was fixed — the citations pointed at
       elements that were genuinely retrieved (so F2-02 correctly kept
       them) but were not remotely relevant to the question.
       `any_exact_evidence_found` unconditionally exempts a turn from this
       branch — an exact-match lookup (error code / page number / element
       id the user or a prior turn already named explicitly) is either
       correct or absent, never "topically similar but irrelevant", so it
       has no need for a similarity threshold at all; without this
       exemption a turn that correctly resolved an exact lookup but also
       happened to make one low-scoring exploratory semantic call would be
       wrongly refused.

    `max_dense_relevance_score is None` (no semantic search tool was called
    this turn at all — e.g. a turn that only used `tool_get_document_page`/
    `tool_get_figure`) makes branch 2 inapplicable entirely, by design:
    there is no relevance signal to threshold against, and branch 1 remains
    the only applicable check in that case, exactly as before §6.1.

    **Explicitly out of scope, by calibrated design** (`03-retrieval-chunking.md`
    §6.1.3 round 2 / §6.1.5): domain-adjacent near-miss questions (e.g. "how
    do I fix a window AC that isn't cooling" against a VRF/VRV commercial
    manual corpus) score in the same range as genuine in-scope questions —
    this gate does not and cannot separate those two classes; that is an
    explicitly accepted residual risk pending a Fase 2 cross-encoder
    reranker, not a bug in this implementation.
    """
    if answer.refused:
        return answer
    if tool_call_count == 0:
        return answer

    force_refuse = (not any_chunks_retrieved and not answer.citations) or (
        not any_exact_evidence_found
        and max_dense_relevance_score is not None
        and max_dense_relevance_score < MIN_RELEVANCE_SCORE
    )
    if force_refuse:
        return answer.model_copy(
            update={"refused": True, "warnings": [*answer.warnings, NO_EVIDENCE_WARNING]}
        )
    return answer


def log_confidence_anomaly_if_needed(answer: TechnicalAnswer) -> None:
    """F2-09 (§5.0, `05-streaming-and-api-contract.md`) — **wajib**: sebelum
    `done` dikirim (or the equivalent non-streaming return), if
    `confidence == 0.0` **and** `refused == False`, log it as an
    observability anomaly (warning level, never blocks/mutates the answer).
    Not a perfect disambiguator (a model can legitimately score confidence
    0.0 for an answer it still chooses to show), but §5.0's own reasoning
    stands: that specific combination is inherently suspicious in a "never
    invent" domain, and is cheap to flag. Call this AFTER both deterministic
    gates above have run, on the final answer that is actually about to be
    sent to the user — never on an intermediate/pre-gate value.
    """
    if answer.confidence == 0.0 and not answer.refused:
        logger.warning(
            "agent.answer.confidence_zero_not_refused_anomaly",
            extra={"answer_length": len(answer.answer)},
        )
