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
# stream_turn — F2-01 graceful degradation on output validation failure
# ---------------------------------------------------------------------------


async def test_stream_turn_output_validation_failure_degrades_instead_of_discarding() -> None:
    """Deterministic reproduction of the real QA repro's shape (a citation
    field the model got wrong — here an unparseable `page`, since
    `element_type` now has a default and can no longer trigger this by
    itself): `UnexpectedModelBehavior` raised mid-stream must still end in
    `done` with whatever `answer` text had already streamed, not be thrown
    away as an `error`."""
    db = _make_session()
    deps = _deps(db)
    payload = {
        "answer": "See the reset button on the panel.",
        "confidence": 0.9,
        "citations": [
            {"document_id": "3", "page": "not-a-number", "element_id": "42", "element_type": "icon"}
        ],
        "warnings": [],
        "related_components": [],
        "related_error_codes": [],
        "refused": False,
    }
    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(
            function=lambda m, i: None,
            # delay_seconds > the 0.1s stream_output() debounce window, same
            # as the happy-path test — needed for genuinely separate partial
            # updates to be observed before the invalid citation arrives.
            stream_function=_make_final_result_stream_function(payload, delay_seconds=0.12),
        ),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [event async for event in st.stream_turn(agent, deps, "how do I reset it?")]
    events = _parse_sse_events(raw_events)

    assert all(name != "error" for name, _ in events)
    assert events[-1][0] == "done"
    done_payload = events[-1][1]
    # Whatever text streamed before the invalid Citation arrived — "answer"
    # is emitted before "citations" in the schema/JSON, so at least some
    # prefix of it must have streamed successfully.
    assert done_payload["answer"] != ""
    assert payload["answer"].startswith(done_payload["answer"])
    assert done_payload["confidence"] == 0.0
    assert done_payload["citations"] == []  # invalid structured output -> none survive
    assert "ttft_ms" in done_payload
    assert "total_latency_ms" in done_payload


async def test_stream_turn_output_validation_failure_still_validates_markers() -> None:
    """The degraded `answer` text still goes through the same §5.1 layer-3
    marker gate as the happy path — a valid `{{el:ID}}` marker already in
    the streamed prefix is still backfilled with a citation."""
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
    payload = {
        "answer": "Press {{el:42}} to reset.",
        "confidence": 0.9,
        "citations": [{"document_id": "3", "page": "bad", "element_id": "1", "element_type": "x"}],
        "warnings": [],
        "related_components": [],
        "related_error_codes": [],
        "refused": False,
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

    # "answer" is fully emitted before "citations" begins in the JSON, so by
    # the time the invalid citation aborts the stream, the full answer text
    # (including the marker) is already captured in `previous_text`.
    done_payload = events[-1][1]
    assert "{{el:42}}" in done_payload["answer"]
    citation_events = [data for name, data in events if name == "citation"]
    assert any(c["element_id"] == "42" for c in citation_events)


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
    # F2-05: no internal exception details leak to the user-facing message —
    # only the safe, generic message is sent (full detail still goes to the
    # server log via `logger.exception`, not asserted here).
    assert events[-1][1]["message"] == st.SAFE_PROVIDER_ERROR_MESSAGE
    assert "RuntimeError" not in events[-1][1]["message"]
    assert "boom" not in events[-1][1]["message"]


# ---------------------------------------------------------------------------
# stream_turn — F2-04 conversation_id threading (§5.6)
# ---------------------------------------------------------------------------


def _happy_agent(payload: dict[str, Any]) -> Agent[AgentDeps, TechnicalAnswer]:
    return Agent(
        FunctionModel(
            function=lambda m, i: None,
            stream_function=_make_final_result_stream_function(payload),
        ),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )


_HAPPY_PAYLOAD: dict[str, Any] = {
    "answer": "Hi.",
    "confidence": 0.9,
    "citations": [],
    "warnings": [],
    "related_components": [],
    "related_error_codes": [],
    "refused": False,
}


async def test_stream_turn_conversation_id_included_on_status_events() -> None:
    db = _make_session()
    deps = _deps(db)

    raw_events = [
        event
        async for event in st.stream_turn(
            _happy_agent(_HAPPY_PAYLOAD), deps, "hi", conversation_id=42
        )
    ]
    events = _parse_sse_events(raw_events)

    status_events = [data for name, data in events if name == "status"]
    assert len(status_events) == 3
    for status in status_events:
        # §5.6: sent as a string on the wire, even though the internal
        # representation (a DB primary key) is an int.
        assert status["conversation_id"] == "42"


async def test_stream_turn_conversation_id_included_on_done() -> None:
    db = _make_session()
    deps = _deps(db)

    raw_events = [
        event
        async for event in st.stream_turn(
            _happy_agent(_HAPPY_PAYLOAD), deps, "hi", conversation_id=42
        )
    ]
    events = _parse_sse_events(raw_events)

    assert events[-1][1]["conversation_id"] == "42"


async def test_stream_turn_conversation_id_included_on_error_event() -> None:
    db = _make_session()
    deps = _deps(db)

    async def _raising_stream_fn(messages: list[ModelMessage], info: AgentInfo):
        raise RuntimeError("boom")
        yield {}  # pragma: no cover

    agent: Agent[AgentDeps, TechnicalAnswer] = Agent(
        FunctionModel(function=lambda m, i: None, stream_function=_raising_stream_fn),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )

    raw_events = [
        event async for event in st.stream_turn(agent, deps, "hello", conversation_id=7)
    ]
    events = _parse_sse_events(raw_events)

    assert events[-1][0] == "error"
    assert events[-1][1]["conversation_id"] == "7"


async def test_stream_turn_conversation_id_included_on_timeout_error_event() -> None:
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
        async for event in st.stream_turn(
            agent, deps, "hello", timeout_seconds=0.05, conversation_id=99
        )
    ]
    events = _parse_sse_events(raw_events)

    assert events[-1][0] == "error"
    assert events[-1][1]["conversation_id"] == "99"


async def test_stream_turn_conversation_id_omitted_when_not_given() -> None:
    db = _make_session()
    deps = _deps(db)

    raw_events = [event async for event in st.stream_turn(_happy_agent(_HAPPY_PAYLOAD), deps, "hi")]
    events = _parse_sse_events(raw_events)

    for _name, data in events:
        assert "conversation_id" not in data


# ---------------------------------------------------------------------------
# stream_turn / run_turn_with_metrics — F2-06 usage_limits passthrough
# ---------------------------------------------------------------------------


async def test_stream_turn_passes_through_custom_usage_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.usage import UsageLimits

    db = _make_session()
    deps = _deps(db)
    seen: dict[str, Any] = {}
    original_run_stream = Agent.run_stream

    def _spy_run_stream(self, *args: Any, **kwargs: Any):
        seen["usage_limits"] = kwargs.get("usage_limits")
        return original_run_stream(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "run_stream", _spy_run_stream)

    custom_limits = UsageLimits(request_limit=3)
    _ = [
        event
        async for event in st.stream_turn(
            _happy_agent(_HAPPY_PAYLOAD), deps, "hi", usage_limits=custom_limits
        )
    ]

    assert seen["usage_limits"] is custom_limits


async def test_stream_turn_defaults_usage_limits_when_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    deps = _deps(db)
    seen: dict[str, Any] = {}
    original_run_stream = Agent.run_stream

    def _spy_run_stream(self, *args: Any, **kwargs: Any):
        seen["usage_limits"] = kwargs.get("usage_limits")
        return original_run_stream(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "run_stream", _spy_run_stream)

    _ = [event async for event in st.stream_turn(_happy_agent(_HAPPY_PAYLOAD), deps, "hi")]

    assert seen["usage_limits"] is not None
    assert seen["usage_limits"].request_limit == st.DEFAULT_MAX_REQUESTS_PER_TURN


async def test_run_turn_with_metrics_passes_through_usage_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.usage import UsageLimits

    db = _make_session()
    seen: dict[str, Any] = {}

    async def _fake_run_agent_turn(*args: Any, **kwargs: Any) -> TechnicalAnswer:
        seen["usage_limits"] = kwargs.get("usage_limits")
        return TechnicalAnswer(answer="Hi.", confidence=0.9)

    monkeypatch.setattr(st, "run_agent_turn", _fake_run_agent_turn)

    custom_limits = UsageLimits(request_limit=3)
    await st.run_turn_with_metrics(object(), _deps(db), "hello", usage_limits=custom_limits)

    assert seen["usage_limits"] is custom_limits


# ---------------------------------------------------------------------------
# run_turn_with_metrics / stream_turn — F2-09 confidence anomaly (§5.0)
# ---------------------------------------------------------------------------


async def test_run_turn_with_metrics_logs_confidence_anomaly(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    db = _make_session()
    zero_confidence_not_refused = TechnicalAnswer(answer="Looks fine.", confidence=0.0)

    async def _fake_run_agent_turn(*args: Any, **kwargs: Any) -> TechnicalAnswer:
        return zero_confidence_not_refused

    monkeypatch.setattr(st, "run_agent_turn", _fake_run_agent_turn)

    with caplog.at_level("WARNING"):
        await st.run_turn_with_metrics(object(), _deps(db), "hello")

    assert any(
        r.message == "agent.answer.confidence_zero_not_refused_anomaly" for r in caplog.records
    )


async def test_stream_turn_logs_confidence_anomaly(caplog) -> None:
    db = _make_session()
    deps = _deps(db)
    payload = {**_HAPPY_PAYLOAD, "confidence": 0.0, "refused": False}

    with caplog.at_level("WARNING"):
        _ = [event async for event in st.stream_turn(_happy_agent(payload), deps, "hi")]

    assert any(
        r.message == "agent.answer.confidence_zero_not_refused_anomaly" for r in caplog.records
    )
