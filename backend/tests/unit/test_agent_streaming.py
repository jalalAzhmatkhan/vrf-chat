"""Unit tests for `app/agent/streaming.py` (C2.4) — SSE event framing,
TTFT/total-latency instrumentation, timeout enforcement, and the
non-streaming `run_turn_with_metrics` equivalent.

Uses `pydantic_ai`'s `FunctionModel` with a real `stream_function` (verified
against the installed `pydantic-ai` version's actual partial-streaming
behavior — see STATUS REPORT) to produce genuine incremental `answer` text
deltas, not a single-shot mock — this is what caught the need for
`TechnicalAnswer.confidence`'s default value (see that field's docstring).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent import streaming as st
from app.agent.context_builder import ContextElement
from app.agent.schemas import TechnicalAnswer
from app.agent.tools import AgentDeps
from app.db.base import Base


def _make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return Session(engine)


def _deps(db: Session) -> AgentDeps:
    return AgentDeps(
        db=db,
        qdrant_client=object(),
        dense_model=object(),
        sparse_model=object(),
        collection="vrf_chunks",
    )


def _parse_sse_events(raw_events: list[str]) -> list[tuple[str, dict[str, Any]]]:
    parsed = []
    for raw in raw_events:
        lines = raw.strip("\n").split("\n")
        event_name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event_name, data))
    return parsed


# ---------------------------------------------------------------------------
# format_sse_event
# ---------------------------------------------------------------------------


def test_format_sse_event() -> None:
    formatted = st.format_sse_event("status", {"stage": "searching_manual"})
    assert formatted == 'event: status\ndata: {"stage": "searching_manual"}\n\n'


# ---------------------------------------------------------------------------
# run_turn_with_metrics
# ---------------------------------------------------------------------------


async def test_run_turn_with_metrics_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()
    expected = TechnicalAnswer(answer="Hi there.", confidence=0.9)

    async def _fake_run_agent_turn(*args: Any, **kwargs: Any) -> TechnicalAnswer:
        return expected

    monkeypatch.setattr(st, "run_agent_turn", _fake_run_agent_turn)

    result = await st.run_turn_with_metrics(object(), _deps(db), "hello")

    assert result.answer == expected
    assert result.timed_out is False
    assert result.ttft_ms == result.total_latency_ms
    assert result.ttft_ms >= 0


async def test_run_turn_with_metrics_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_session()

    async def _slow_run_agent_turn(*args: Any, **kwargs: Any) -> TechnicalAnswer:
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled by the timeout")  # pragma: no cover

    monkeypatch.setattr(st, "run_agent_turn", _slow_run_agent_turn)

    result = await st.run_turn_with_metrics(
        object(), _deps(db), "hello", timeout_seconds=0.05
    )

    assert result.timed_out is True
    assert result.answer == st.TIMEOUT_ANSWER
    assert result.answer.refused is True


# ---------------------------------------------------------------------------
# stream_turn — happy path (real incremental streaming via stream_function)
# ---------------------------------------------------------------------------


def _make_final_result_stream_function(
    payload: dict[str, Any], *, chunk_size: int = 8, delay_seconds: float = 0.0
):
    async def _stream_fn(messages: list[ModelMessage], info: AgentInfo):
        tool_name = info.output_tools[0].name
        full_args = json.dumps(payload)
        for i in range(0, len(full_args), chunk_size):
            piece = full_args[i : i + chunk_size]
            yield {0: DeltaToolCall(name=tool_name if i == 0 else None, json_args=piece)}
            # `result.stream_output()` (app/agent/streaming.py) uses its
            # default `debounce_by=0.1` (100ms) — real LLM token pacing
            # naturally exceeds that, but a synthetic stream_function needs
            # an explicit delay to produce genuinely separate `token` SSE
            # events instead of one coalesced update.
            await asyncio.sleep(delay_seconds)

    return _stream_fn


async def test_stream_turn_happy_path_event_sequence_and_marker_validation() -> None:
    db = _make_session()
    deps = _deps(db)
    icon_element = ContextElement(
        element_id=42,
        element_type="icon",
        text=None,
        image_uri="s3://bucket/icon.png",
        visual_description=None,
        document_id=3,
        page=238,
    )
    deps.context_elements = {42: icon_element}
    deps.tool_call_count = 1
    deps.any_chunks_retrieved = True

    payload = {
        "confidence": 0.9,
        "citations": [],
        "warnings": [],
        "related_components": [],
        "related_error_codes": [],
        "refused": False,
        "answer": "Press {{el:42}} and also {{el:999}} to reset. " * 3,
    }
    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(
            function=lambda m, i: None,
            stream_function=_make_final_result_stream_function(payload, delay_seconds=0.12),
        ),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [event async for event in st.stream_turn(agent, deps, "how do I reset it?")]
    events = _parse_sse_events(raw_events)
    event_names = [name for name, _ in events]

    assert event_names[0] == "status"
    assert events[0][1] == {"stage": "searching_manual"}
    assert event_names[1] == "status"
    assert events[1][1] == {"stage": "building_context"}
    assert event_names[2] == "status"
    assert events[2][1] == {"stage": "generating_answer"}

    token_events = [data for name, data in events if name == "token"]
    assert len(token_events) > 1  # genuinely incremental, not one big chunk
    reconstructed = "".join(t["delta"] for t in token_events)
    assert reconstructed == payload["answer"]

    citation_events = [data for name, data in events if name == "citation"]
    assert len(citation_events) == 1
    assert citation_events[0]["element_id"] == "42"

    assert event_names[-1] == "done"
    done_payload = events[-1][1]
    assert "{{el:42}}" in done_payload["answer"]
    assert "{{el:999}}" not in done_payload["answer"]  # invalid marker stripped
    assert done_payload["refused"] is False
    assert isinstance(done_payload["ttft_ms"], int)
    assert isinstance(done_payload["total_latency_ms"], int)
    assert done_payload["ttft_ms"] <= done_payload["total_latency_ms"]


async def test_stream_turn_no_token_deltas_ttft_falls_back_to_total_latency() -> None:
    """When the model's answer is empty (an edge case, e.g. a refusal with
    empty text), no `token` events fire — `ttft_ms` must still be reported
    (equal to total latency), never left null."""
    db = _make_session()
    deps = _deps(db)
    payload = {
        "answer": "",
        "confidence": 0.0,
        "citations": [],
        "warnings": [],
        "related_components": [],
        "related_error_codes": [],
        "refused": True,
    }
    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(
            function=lambda m, i: None,
            stream_function=_make_final_result_stream_function(payload),
        ),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [event async for event in st.stream_turn(agent, deps, "hello")]
    events = _parse_sse_events(raw_events)

    assert all(name != "token" for name, _ in events)
    done_payload = events[-1][1]
    assert done_payload["ttft_ms"] == done_payload["total_latency_ms"]


# ---------------------------------------------------------------------------
# stream_turn — timeout / provider error
# ---------------------------------------------------------------------------


async def test_stream_turn_timeout_emits_error_event() -> None:
    db = _make_session()
    deps = _deps(db)

    async def _slow_stream_fn(messages: list[ModelMessage], info: AgentInfo):
        await asyncio.sleep(10)
        yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args="{}")}  # pragma: no cover

    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(function=lambda m, i: None, stream_function=_slow_stream_fn),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [
        event
        async for event in st.stream_turn(agent, deps, "hello", timeout_seconds=0.05)
    ]
    events = _parse_sse_events(raw_events)

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "timeout"


async def test_stream_turn_provider_error_emits_error_event() -> None:
    db = _make_session()
    deps = _deps(db)

    async def _raising_stream_fn(messages: list[ModelMessage], info: AgentInfo):
        raise RuntimeError("boom")
        yield {}  # pragma: no cover — makes this an async generator function

    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(function=lambda m, i: None, stream_function=_raising_stream_fn),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [event async for event in st.stream_turn(agent, deps, "hello")]
    events = _parse_sse_events(raw_events)

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "provider_error"
    assert "RuntimeError" in events[-1][1]["message"]
