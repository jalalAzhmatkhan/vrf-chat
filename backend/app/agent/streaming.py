"""SSE streaming + non-streaming turn execution — per
`Documentation/system-design/05-streaming-and-api-contract.md` §4 (event
format) and §2-3 (TTFT budget/strategy). `app/api/v1/chat.py` (C2.4) is the
only intended caller.

**Stage sequencing (§4.1 `status` enum) — documented trade-off**: this
module's architecture has every tool (`app/agent/tools.py`) perform
retrieval AND context-building together, inside a single call triggered
*by the LLM* mid-generation — there is no separate "backend is retrieving"
phase that happens strictly *before* the LLM is invoked at all (unlike a
classic non-agentic RAG pipeline). The three fixed `stage` values are
therefore emitted in quick, deterministic succession right at turn start
(`searching_manual` -> `building_context` -> `generating_answer`), not
spaced out to match literal wall-clock sub-phases: `searching_manual` is
shown the instant the request is accepted (before any per-turn setup),
`building_context` right after per-turn setup (query-expansion inputs,
`AgentDeps`) is ready, `generating_answer` right as control is handed to
`agent.run_stream` (which internally interleaves tool calls — i.e.
retrieval AND context-building for real — with token generation). All the
*actual* waiting happens during `generating_answer`, which is exactly where
`ttft_ms` is measured from. This is a deliberate simplification, not an
oversight — flagged here for System Analyst/QA visibility.

**Marker validation timing (§5.1)**: raw `token` SSE deltas stream the
model's *unvalidated* text as it arrives — per §5.1 point 3's own wording
("... sebelum `answer` final dikirim di event `done`"), the non-negotiable
marker-strip/citation-backfill gate only guarantees the **`done` event's**
`answer`/`citations` are clean. A `{{el:ID}}` marker for a
since-invalidated/hallucinated id could theoretically flash through raw
`token` deltas before `done` corrects it — an accepted trade-off (the
alternative, buffering the entire answer server-side before streaming
anything, would defeat streaming's TTFT purpose entirely).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model

from app.agent.answer_postprocess import enforce_never_invent_safety_net, postprocess_answer
from app.agent.context_builder import BuiltContext
from app.agent.schemas import TechnicalAnswer
from app.agent.tools import AgentDeps
from app.agent.vrf_agent import run_agent_turn
from app.core.observability import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 25.0

TIMEOUT_ANSWER = TechnicalAnswer(
    answer=(
        "Sorry, the request took too long to process and was stopped. Please try again — "
        "if this keeps happening, try a shorter or more specific question."
    ),
    confidence=0.0,
    refused=True,
)


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@dataclass(slots=True)
class TurnResult:
    answer: TechnicalAnswer
    ttft_ms: int
    total_latency_ms: int
    timed_out: bool = False


async def run_turn_with_metrics(
    agent: Agent[AgentDeps, TechnicalAnswer],
    deps: AgentDeps,
    user_message: str,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    model: Model | KnownModelName | str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> TurnResult:
    """Non-streaming turn execution for `POST /api/v1/chat` — no
    intermediate token visibility, so `ttft_ms` is reported equal to
    `total_latency_ms` (honest, not an approximation left unset)."""
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            answer = await run_agent_turn(
                agent, deps, user_message, message_history=message_history, model=model
            )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning("chat.turn.timeout", extra={"timeout_seconds": timeout_seconds})
        return TurnResult(
            answer=TIMEOUT_ANSWER, ttft_ms=elapsed_ms, total_latency_ms=elapsed_ms, timed_out=True
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return TurnResult(answer=answer, ttft_ms=elapsed_ms, total_latency_ms=elapsed_ms)


async def stream_turn(
    agent: Agent[AgentDeps, TechnicalAnswer],
    deps: AgentDeps,
    user_message: str,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    model: Model | KnownModelName | str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> AsyncIterator[str]:
    """SSE event generator for `POST /api/v1/chat/stream` — yields fully
    formatted `event: ...\\ndata: ...\\n\\n` strings, per
    `05-streaming-and-api-contract.md` §4. See module docstring for the
    `status` stage-sequencing and marker-validation-timing trade-offs.
    """
    start = time.monotonic()
    ttft_ms: int | None = None

    yield format_sse_event("status", {"stage": "searching_manual"})

    try:
        async with asyncio.timeout(timeout_seconds):
            yield format_sse_event("status", {"stage": "building_context"})
            async with agent.run_stream(
                user_message, deps=deps, message_history=message_history, model=model
            ) as result:
                yield format_sse_event("status", {"stage": "generating_answer"})
                previous_text = ""
                async for partial in result.stream_output():
                    current_text = getattr(partial, "answer", None) or ""
                    if len(current_text) > len(previous_text):
                        if ttft_ms is None:
                            ttft_ms = int((time.monotonic() - start) * 1000)
                        yield format_sse_event(
                            "token", {"delta": current_text[len(previous_text) :]}
                        )
                        previous_text = current_text
                raw_output = await result.get_output()
    except TimeoutError:
        logger.warning("chat.stream.timeout", extra={"timeout_seconds": timeout_seconds})
        yield format_sse_event(
            "error",
            {
                "code": "timeout",
                "message": "The assistant took too long to respond. Please try again.",
            },
        )
        return
    except Exception as exc:  # noqa: BLE001 — any provider/transport failure becomes an `error` event
        logger.exception("chat.stream.provider_error")
        yield format_sse_event(
            "error", {"code": "provider_error", "message": f"{type(exc).__name__}: {exc}"}
        )
        return

    context = BuiltContext(elements_by_id=deps.context_elements)
    post_result = postprocess_answer(raw_output, context)
    final_answer = enforce_never_invent_safety_net(
        post_result.answer,
        tool_call_count=deps.tool_call_count,
        any_chunks_retrieved=deps.any_chunks_retrieved,
    )

    for citation in final_answer.citations:
        yield format_sse_event("citation", citation.model_dump())

    total_latency_ms = int((time.monotonic() - start) * 1000)
    effective_ttft_ms = ttft_ms if ttft_ms is not None else total_latency_ms

    done_payload = final_answer.model_dump()
    done_payload["ttft_ms"] = effective_ttft_ms
    done_payload["total_latency_ms"] = total_latency_ms
    yield format_sse_event("done", done_payload)
