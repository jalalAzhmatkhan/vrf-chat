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

Two responsibilities, always run together on every turn's final answer
before it is sent in the `done` SSE event (`app/api/v1/chat.py`, C2.4):

1. **Strip invalid markers.** Any `{{el:ID}}` in `answer.answer` whose `ID`
   is not in this turn's context element whitelim
   (`BuiltContext.elements_by_id`, i.e. an element actually retrieved and
   shown to the LLM this turn) is removed — the surrounding sentence is
   left untouched, only the marker substring itself is deleted. Every
   removal is logged as an anomaly (observability,
   `01-architecture-overview.md` §5), never silently dropped.
2. **Backfill citations.** Every marker that *does* survive validation is
   guaranteed a matching `Citation` entry (`element_id` matches) — created
   automatically from the context element's metadata if the LLM didn't
   already include one in its structured `citations` output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.context_builder import BuiltContext, ContextElement
from app.agent.schemas import VISUAL_ELEMENT_TYPES, Citation, TechnicalAnswer, Warning
from app.core.observability import get_logger

logger = get_logger(__name__)

MARKER_PATTERN = re.compile(r"\{\{el:(\d+)\}\}")
_COLLAPSE_WHITESPACE_PATTERN = re.compile(r"[ \t]{2,}")


@dataclass(slots=True)
class PostProcessResult:
    answer: TechnicalAnswer
    stripped_element_ids: tuple[int, ...]
    backfilled_element_ids: tuple[int, ...]


def _strip_marker(text: str, element_id: int) -> str:
    return text.replace(f"{{{{el:{element_id}}}}}", "")


def _citation_from_context_element(element: ContextElement) -> Citation:
    is_visual = element.element_type in VISUAL_ELEMENT_TYPES
    return Citation(
        document_id=str(element.document_id),
        page=element.page or 0,
        element_id=str(element.element_id),
        element_type=element.element_type,
        quote=element.text,
        image_uri=element.image_uri if is_visual else None,
        visual_description=element.visual_description if is_visual else None,
    )


def postprocess_answer(answer: TechnicalAnswer, context: BuiltContext) -> PostProcessResult:
    """Apply the non-negotiable marker-validation gate to `answer` (produced
    by the agent this turn) against `context` (everything retrieved/shown to
    the LLM this turn, from every tool call).
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

    existing_by_element_id = {citation.element_id: citation for citation in answer.citations}
    citations = list(answer.citations)
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
    )


NO_EVIDENCE_WARNING = Warning(
    message=(
        "No relevant information was found in the indexed manuals for this question — "
        "refusing rather than guessing."
    ),
    severity="note",
)


def enforce_never_invent_safety_net(
    answer: TechnicalAnswer, *, tool_call_count: int, any_chunks_retrieved: bool
) -> TechnicalAnswer:
    """A second, deterministic "never invent" gate independent of the system
    prompt (`app/agent/vrf_agent.py` STATIC_SYSTEM_PROMPT already instructs
    this, but that alone is a request to the model, not an enforceable
    guarantee — see task instructions: "Implementasikan sebagai perilaku
    yang bisa dites, bukan sekadar kalimat di system prompt").

    Deliberately narrow (to avoid false-positiving on legitimate
    tool-free turns, e.g. a greeting): only forces `refused=True` when the
    agent *did* call at least one retrieval tool this turn
    (`tool_call_count > 0`, so it recognized the question needed evidence)
    but genuinely found nothing (`any_chunks_retrieved is False`) and ended
    up with zero citations, yet did not already mark itself refused. A
    no-tool-call conversational turn (small talk, clarifying question) is
    left untouched.
    """
    if answer.refused:
        return answer
    if tool_call_count > 0 and not any_chunks_retrieved and not answer.citations:
        return answer.model_copy(
            update={"refused": True, "warnings": [*answer.warnings, NO_EVIDENCE_WARNING]}
        )
    return answer
