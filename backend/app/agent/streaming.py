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

**F2-01 (`Documentation/qa-reports/phase-2-qa-report.md`) — graceful
degradation on output-validation failure**: `pydantic_ai` validates every
partial object during `result.stream_output()`; if the model's structured
output ever omits a required field on a nested model (e.g. `Citation.
element_type` — observed with `gpt-3.5-turbo` in real QA testing) it raises
`UnexpectedModelBehavior`, and `run_stream()` (unlike `agent.run()`) does
**not** support retries. Discarding the whole turn at that point would
throw away `answer` text that had already streamed successfully and was
already correct/useful. This module now catches that specific exception
(only — any other exception still hits the general `error` handler below)
and degrades: the already-streamed text becomes the final `answer`
(`confidence=0.0`, no citations from the invalid structured output — still
run through the same two mandatory gates below, so any `{{el:ID}}` markers
already in that partial text are still validated/backfilled normally), and
the turn still ends in a `done` event rather than being thrown away.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import UsageLimits

from app.agent.answer_postprocess import (
    enforce_never_invent_safety_net,
    log_confidence_anomaly_if_needed,
    postprocess_answer,
)
from app.agent.context_builder import BuiltContext
from app.agent.schemas import TechnicalAnswer
from app.agent.tools import AgentDeps
from app.agent.vrf_agent import DEFAULT_MAX_REQUESTS_PER_TURN, run_agent_turn
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

# F2-05 — a safe, user-facing message for any provider/transport failure not
# handled more specifically above (timeout has its own message; F2-01's
# UnexpectedModelBehavior degrades instead of erroring). Previously this
# event carried f"{type(exc).__name__}: {exc}" verbatim to the user — an
# internal exception message/traceback fragment is not something a
# technician should see, and it is not actionable for them either way; full
# detail still goes to the server log via `logger.exception` below.
SAFE_PROVIDER_ERROR_MESSAGE = (
    "Sorry, something went wrong while processing your request. Please try again — "
    "if this keeps happening, please contact support."
)


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _with_conversation_id(payload: dict[str, Any], conversation_id: int | None) -> dict[str, Any]:
    """F2-04 (§5.6) — include `conversation_id` (as a string, per the ratified
    contract type) on every event where it is documented to appear
    (`status`/`error`/`done`), whenever it has actually been resolved.
    `conversation_id` is `None` here unless the caller (`app/api/v1/chat.py`,
    C2.6) already resolved/created the conversation row before starting this
    generator — never resolved by this module itself (kept DB-agnostic, per
    `01-architecture-overview.md` §3 module boundary principle)."""
    if conversation_id is not None:
        payload["conversation_id"] = str(conversation_id)
    return payload


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
    usage_limits: UsageLimits | None = None,
) -> TurnResult:
    """Non-streaming turn execution for `POST /api/v1/chat` — no
    intermediate token visibility, so `ttft_ms` is reported equal to
    `total_latency_ms` (honest, not an approximation left unset).

    `usage_limits`: F2-06 round-trip cap, passed straight through to
    `run_agent_turn` (which supplies its own default when omitted here too).
    """
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout_seconds):
            answer = await run_agent_turn(
                agent,
                deps,
                user_message,
                message_history=message_history,
                model=model,
                usage_limits=usage_limits,
            )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning("chat.turn.timeout", extra={"timeout_seconds": timeout_seconds})
        return TurnResult(
            answer=TIMEOUT_ANSWER, ttft_ms=elapsed_ms, total_latency_ms=elapsed_ms, timed_out=True
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    log_confidence_anomaly_if_needed(answer)  # F2-09, §5.0
    return TurnResult(answer=answer, ttft_ms=elapsed_ms, total_latency_ms=elapsed_ms)


async def stream_turn(
    agent: Agent[AgentDeps, TechnicalAnswer],
    deps: AgentDeps,
    user_message: str,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    model: Model | KnownModelName | str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    usage_limits: UsageLimits | None = None,
    on_done: Callable[[TechnicalAnswer, int, int], None] | None = None,
    conversation_id: int | None = None,
) -> AsyncIterator[str]:
    """SSE event generator for `POST /api/v1/chat/stream` — yields fully
    formatted `event: ...\\ndata: ...\\n\\n` strings, per
    `05-streaming-and-api-contract.md` §4. See module docstring for the
    `status` stage-sequencing, marker-validation-timing, and F2-01
    graceful-degradation trade-offs/behavior.

    `usage_limits`: F2-06 round-trip cap, same default/rationale as
    `app/agent/vrf_agent.py::run_agent_turn`.

    `on_done` (optional): called synchronously with
    `(final_answer, ttft_ms, total_latency_ms)` right before the `done`
    event is yielded — this is the hook `app/api/v1/chat.py` (C2.6) uses to
    persist the turn to `messages`/`citations` without this module (which
    stays DB-agnostic on purpose, see `01-architecture-overview.md` §3
    module boundary principle) needing any DB/persistence knowledge itself.
    Never called on the timeout/error paths (nothing to persist for a turn
    that produced no answer) — see F2-08 for why that gap is otherwise
    handled defensively at the persistence layer instead.

    `conversation_id` (optional): F2-04/§5.6 — see `_with_conversation_id`.
    Pure passthrough; this module never resolves/creates a conversation
    itself (kept DB-agnostic, same module-boundary principle as `on_done`).
    """
    start = time.monotonic()
    ttft_ms: int | None = None
    effective_usage_limits = usage_limits or UsageLimits(
        request_limit=DEFAULT_MAX_REQUESTS_PER_TURN
    )

    yield format_sse_event(
        "status", _with_conversation_id({"stage": "searching_manual"}, conversation_id)
    )

    raw_output: TechnicalAnswer | None = None
    previous_text = ""

    try:
        async with asyncio.timeout(timeout_seconds):
            yield format_sse_event(
                "status", _with_conversation_id({"stage": "building_context"}, conversation_id)
            )
            async with agent.run_stream(
                user_message,
                deps=deps,
                message_history=message_history,
                model=model,
                usage_limits=effective_usage_limits,
            ) as result:
                yield format_sse_event(
                    "status",
                    _with_conversation_id({"stage": "generating_answer"}, conversation_id),
                )
                try:
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
                except UnexpectedModelBehavior:
                    # F2-01 — see module docstring. Degrade instead of
                    # discarding the turn: keep whatever `answer` text
                    # already streamed successfully. `raw_output` stays
                    # `None` here; built below once we're out of the
                    # `agent.run_stream()` context manager.
                    logger.warning(
                        "chat.stream.output_validation_failed_degraded",
                        extra={"partial_answer_length": len(previous_text)},
                    )
    except TimeoutError:
        logger.warning("chat.stream.timeout", extra={"timeout_seconds": timeout_seconds})
        yield format_sse_event(
            "error",
            _with_conversation_id(
                {
                    "code": "timeout",
                    "message": "The assistant took too long to respond. Please try again.",
                },
                conversation_id,
            ),
        )
        return
    except Exception:  # noqa: BLE001 — any other provider/transport failure becomes an `error` event
        logger.exception("chat.stream.provider_error")
        yield format_sse_event(
            "error",
            _with_conversation_id(
                {"code": "provider_error", "message": SAFE_PROVIDER_ERROR_MESSAGE},  # F2-05
                conversation_id,
            ),
        )
        return

    if raw_output is None:
        # F2-01 degraded path — build the best available TechnicalAnswer
        # from whatever streamed successfully, then run it through the
        # exact same two mandatory gates as the happy path below (so any
        # `{{el:ID}}` markers already in `previous_text` are still
        # validated/backfilled, not just passed through raw).
        raw_output = TechnicalAnswer(answer=previous_text, confidence=0.0)

    context = BuiltContext(elements_by_id=deps.context_elements)
    post_result = postprocess_answer(raw_output, context)
    final_answer = enforce_never_invent_safety_net(
        post_result.answer,
        tool_call_count=deps.tool_call_count,
        any_chunks_retrieved=deps.any_chunks_retrieved,
    )
    log_confidence_anomaly_if_needed(final_answer)  # F2-09, §5.0

    for citation in final_answer.citations:
        yield format_sse_event("citation", citation.model_dump())

    total_latency_ms = int((time.monotonic() - start) * 1000)
    effective_ttft_ms = ttft_ms if ttft_ms is not None else total_latency_ms

    if on_done is not None:
        on_done(final_answer, effective_ttft_ms, total_latency_ms)

    done_payload = final_answer.model_dump()
    done_payload["ttft_ms"] = effective_ttft_ms
    done_payload["total_latency_ms"] = total_latency_ms
    yield format_sse_event("done", _with_conversation_id(done_payload, conversation_id))
