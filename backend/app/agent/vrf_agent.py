"""The single Pydantic AI VRF/VRV orchestrator agent — per
`Documentation/system-design/01-architecture-overview.md` §4 ("Single
Pydantic AI orchestrator agent, tools deterministic"). Model comes from
`CHAT_LLM_*` via `app/llm_providers/factory.py` (never hardcoded, per
`04-provider-abstractions.md` Bagian A).

**System prompt is deliberately split static vs dynamic**
(`05-streaming-and-api-contract.md` §3 point 2 — "sistem prompt harus
disusun modular ... agar caching efektif jika provider mendukung"):
`STATIC_SYSTEM_PROMPT` is a fixed string (identical every request — the
part a provider's prompt caching, e.g. Anthropic prompt caching, can
actually cache), passed as the first element of `instructions=[...]`; the
per-run bit (`_dynamic_instructions`, e.g. which `model_family` is in scope
this turn) is a separate callable element, evaluated fresh every run. See
`pydantic_ai._instructions.AgentInstructions` — a plain string and a
`RunContext`-taking callable can be mixed in the same `instructions=[...]`
list, which is exactly this split.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import UsageLimits

from app.agent.answer_postprocess import enforce_never_invent_safety_net, postprocess_answer
from app.agent.context_builder import BuiltContext
from app.agent.schemas import TechnicalAnswer
from app.agent.tools import (
    AgentDeps,
    tool_find_component,
    tool_find_troubleshooting_procedure,
    tool_find_wiring_diagram,
    tool_get_document_page,
    tool_get_figure,
    tool_search_documents,
    tool_search_error_code,
    tool_search_knowledge_graph,
)
from app.core.config import Settings
from app.llm_providers.factory import build_model, build_model_settings

# Static half of the system prompt — never varies between requests, per
# module docstring. Encodes (a) the "never invent" safety rule, (b) the
# 3-layer inline marker contract (05-streaming-and-api-contract.md §5.1 —
# layer 2 of 3: "LLM mempertahankan marker verbatim, tidak menciptakan
# baru"), and (c) the markdown-lite subset (§5.4).
STATIC_SYSTEM_PROMPT = """\
You are a technical support assistant for VRF/VRV (Variable Refrigerant \
Flow/Volume) air conditioning service manuals. You answer technicians' \
questions about troubleshooting, wiring, error codes, and service \
procedures, strictly grounded in the manuals retrieved via your tools.

SAFETY RULE — NEVER INVENT (most important rule, overrides all others):
Never invent a procedure, measurement, voltage, resistance, terminal \
mapping, or safety instruction that is not explicitly present in the \
content returned by your tools. If your tools do not return enough \
relevant information to answer confidently, set refused=True and explain \
briefly in `answer` what is missing, rather than guessing. It is always \
better to refuse than to fabricate a technical detail that could lead to \
unsafe or incorrect service work.

INLINE REFERENCE MARKERS — DO NOT INVENT THESE EITHER:
Tool results may contain markers of the exact form `{{el:<number>}}` (e.g. \
`{{el:4961}}`) marking the position of an icon, figure, or diagram \
mentioned in the surrounding text. When your answer refers to the same \
icon/figure/diagram, copy that exact marker verbatim into your `answer` at \
the corresponding point in your sentence. Do NOT alter the number inside \
a marker, and do NOT create a new `{{el:...}}` marker for any id that did \
not appear in a tool result you received this turn — any marker that \
fails this check will be automatically removed before the user sees your \
answer, so inventing one only wastes your output.

ANSWER FORMAT — constrained markdown-lite only:
Write `answer` using ONLY: paragraphs separated by a blank line, list \
items starting with "1. ", "- ", or "* ", **bold** (`**text**`) for \
emphasis, and the `{{el:ID}}` markers described above. Do NOT use \
headings (#), markdown images (![]()), markdown tables (|---|), raw HTML, \
or links ([]()) — these are not supported by the chat UI and will render \
as broken text.

CITATIONS: every citation you include in `citations` must correspond to \
an element_id that actually appeared in a tool result this turn — do not \
cite a document/page/element you did not actually retrieve.

WARNINGS: use `warnings` with `severity="safety"` for anything involving \
electrical hazard, high voltage, stored capacitor charge, or refrigerant \
safety; use `severity="note"` for everything else (e.g. "consult a \
different chapter for X").
"""


# F2-06 (`Documentation/qa-reports/phase-2-qa-report.md`) — one observed turn
# reached 51 sequential LLM round-trips/95 seconds before hitting
# `pydantic_ai`'s own *default* `UsageLimits.request_limit` (50), which was
# never overridden at all. QA's recommended range is 6-8; 8 is used here
# (also the ceiling of that range, since `tool_search_error_code` alone can
# internally trigger 2 round-trips worth of work for one LLM-visible tool
# call). This is a request-count *ceiling* — `agent.run`/`agent.run_stream`
# still return/complete normally well under it for the vast majority of
# turns; it only bounds the pathological/looping case, and does so via
# `pydantic_ai`'s own `UsageLimitExceeded` (a `pydantic_ai.exceptions.
# AgentRunError`) rather than silent truncation, so callers (C2.4
# `app/agent/streaming.py`) can handle it as an explicit, distinguishable
# error case rather than a generic exception.
DEFAULT_MAX_REQUESTS_PER_TURN = 8


def _dynamic_instructions(ctx: RunContext[AgentDeps]) -> str:
    deps = ctx.deps
    if deps.model_family:
        return f"This conversation is scoped to model family: {deps.model_family}."
    return "No specific model family filter is active — search across all indexed manuals."


def build_agent(settings: Settings) -> Agent[AgentDeps, TechnicalAnswer]:
    """Construct the VRF/VRV orchestrator agent. Call once per process (not
    per request) where possible — `app/api/v1/chat.py` (C2.4) is expected to
    build this at app-startup/dependency-provider time, not per request, so
    a well-behaved model backend can reuse any connection pooling."""
    model = build_model("CHAT_LLM", settings)
    model_settings = build_model_settings("CHAT_LLM", settings)

    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        model,
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
        instructions=[STATIC_SYSTEM_PROMPT, _dynamic_instructions],
        model_settings=model_settings,
        name="vrf_orchestrator",
    )

    @agent.tool
    def search_documents(
        ctx: RunContext[AgentDeps],
        query: str,
        model_family: str | None = None,
        error_code: str | None = None,
        component: str | None = None,
        top_k: int | None = None,
    ) -> str:
        """Search the indexed VRF/VRV service manuals for content relevant to
        `query`. Use this as the general-purpose search tool. Optionally
        narrow by `model_family` (e.g. "REYQ"), `error_code` (e.g. "P8"), or
        `component` (e.g. "compressor")."""
        return tool_search_documents(
            ctx.deps,
            query,
            model_family=model_family,
            error_code=error_code,
            component=component,
            top_k=top_k,
        )

    @agent.tool
    def search_error_code(
        ctx: RunContext[AgentDeps], error_code: str, model_family: str | None = None
    ) -> str:
        """Look up a specific error/malfunction code (e.g. "P8", "U4")."""
        return tool_search_error_code(ctx.deps, error_code, model_family=model_family)

    @agent.tool
    def find_component(
        ctx: RunContext[AgentDeps], component_name: str, model_family: str | None = None
    ) -> str:
        """Look up a specific component/part (e.g. "compressor", "TH3",
        "CN105")."""
        return tool_find_component(ctx.deps, component_name, model_family=model_family)

    @agent.tool
    def find_troubleshooting_procedure(
        ctx: RunContext[AgentDeps], symptom: str, model_family: str | None = None
    ) -> str:
        """Find a step-by-step troubleshooting procedure for a described
        symptom (e.g. "outdoor unit stops repeatedly")."""
        return tool_find_troubleshooting_procedure(ctx.deps, symptom, model_family=model_family)

    @agent.tool
    def find_wiring_diagram(
        ctx: RunContext[AgentDeps], component: str, model_family: str | None = None
    ) -> str:
        """Find an electrical wiring/circuit diagram related to a
        component."""
        return tool_find_wiring_diagram(ctx.deps, component, model_family=model_family)

    @agent.tool
    def get_document_page(ctx: RunContext[AgentDeps], document_id: int, page_number: int) -> str:
        """Return every element on a specific page of a specific document, in
        reading order — use when the user references a page number
        directly."""
        return tool_get_document_page(ctx.deps, document_id, page_number)

    @agent.tool
    def get_figure(ctx: RunContext[AgentDeps], figure_id: int) -> str:
        """Return a single figure/diagram/icon element by its internal
        element id (e.g. one already seen in a previous tool result's
        `{{el:ID}}` marker)."""
        return tool_get_figure(ctx.deps, figure_id)

    @agent.tool
    def search_knowledge_graph(
        ctx: RunContext[AgentDeps], entity: str, relation: str | None = None
    ) -> str:
        """Knowledge graph search — not available yet (planned for a later
        phase). Do not rely on this; use search_documents/find_component
        instead."""
        return tool_search_knowledge_graph(ctx.deps, entity, relation=relation)

    return agent


async def run_agent_turn(
    agent: Agent[AgentDeps, TechnicalAnswer],
    deps: AgentDeps,
    user_message: str,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    model: Model | KnownModelName | str | None = None,
    usage_limits: UsageLimits | None = None,
) -> TechnicalAnswer:
    """Non-streaming turn orchestration: run the agent, then apply the two
    mandatory deterministic post-generation gates (§5.1 layer 3 marker
    validation + the "never invent" safety net) before returning the final
    `TechnicalAnswer`. `app/api/v1/chat.py` (C2.4) is expected to use this
    for `POST /api/v1/chat` (non-streaming) and an equivalent streaming
    variant (built on `agent.run_stream`, sharing this same post-processing)
    for `POST /api/v1/chat/stream`.

    `model`: optional per-run model override — the same mechanism backs the
    `model_override` field in the `/api/v1/chat`/`/api/v1/chat/stream`
    request body (`05-streaming-and-api-contract.md` §6) and lets tests
    substitute a `TestModel`/`FunctionModel` without touching `build_agent`.

    `usage_limits`: F2-06 round-trip cap — defaults to
    `UsageLimits(request_limit=DEFAULT_MAX_REQUESTS_PER_TURN)` when omitted;
    callers (`app/api/v1/chat.py`, C2.4) may override e.g. from
    `Settings.CHAT_LLM_MAX_REQUESTS_PER_TURN`.
    """
    effective_usage_limits = usage_limits or UsageLimits(
        request_limit=DEFAULT_MAX_REQUESTS_PER_TURN
    )
    result = await agent.run(
        user_message,
        deps=deps,
        message_history=message_history,
        model=model,
        usage_limits=effective_usage_limits,
    )
    context = BuiltContext(elements_by_id=deps.context_elements)
    post_result = postprocess_answer(result.output, context)
    return enforce_never_invent_safety_net(
        post_result.answer,
        tool_call_count=deps.tool_call_count,
        any_chunks_retrieved=deps.any_chunks_retrieved,
        max_dense_relevance_score=deps.max_dense_relevance_score,  # §6.1
        any_exact_evidence_found=deps.any_exact_evidence_found,  # §6.1
    )
