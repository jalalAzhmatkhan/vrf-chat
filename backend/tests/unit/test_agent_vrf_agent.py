"""Unit tests for `app/agent/vrf_agent.py` (C2.3) — agent/tool wiring and
`run_agent_turn` end-to-end orchestration (marker validation + safety net),
using `pydantic_ai`'s `TestModel`/`FunctionModel` (no real LLM provider
network call — `CHAT_LLM_*` settings only need to be *syntactically* valid
for `build_model`'s fail-fast validation, per `tests/conftest.py`'s
dummy-env-var convention).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent import tools as tools_module
from app.agent import vrf_agent
from app.agent.schemas import TechnicalAnswer
from app.agent.tools import AgentDeps
from app.core.config import Settings
from app.db.base import Base
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.retrieval.hybrid_search import RetrievalResult, RetrievedChunk


def _settings(**overrides: Any) -> Settings:
    base = dict(
        CHAT_LLM_PROVIDER="anthropic",
        CHAT_LLM_MODEL="claude-sonnet-4-5-20250929",
        CHAT_LLM_API_KEY="sk-ant-test-dummy",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _make_session() -> Session:
    # `check_same_thread=False` + `StaticPool`: pydantic_ai runs sync tool
    # functions in a worker thread (not the event loop thread) — an
    # in-memory SQLite connection must be explicitly shared across threads
    # for that to work, same pattern as
    # `tests/integration/conftest.py`'s `db_engine` fixture.
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


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------


def test_build_agent_registers_all_required_tools() -> None:
    agent = vrf_agent.build_agent(_settings())
    tool_names = set(agent._function_toolset.tools.keys())
    assert tool_names == {
        "search_documents",
        "search_error_code",
        "find_component",
        "find_troubleshooting_procedure",
        "find_wiring_diagram",
        "get_document_page",
        "get_figure",
        "search_knowledge_graph",
    }


async def test_build_agent_runs_with_test_model_produces_technical_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke test: `TestModel` calls every registered tool with
    schema-appropriate dummy args and synthesizes a `TechnicalAnswer` —
    verifies the whole agent/tool wiring doesn't raise, without a real LLM
    or Qdrant."""
    db = _make_session()

    def _fake_search_documents(*args: Any, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(query="x", effective_query="x", chunks=[])

    monkeypatch.setattr(tools_module, "search_documents", _fake_search_documents)

    agent = vrf_agent.build_agent(_settings())
    deps = _deps(db)

    result = await agent.run("What is error P8?", deps=deps, model=TestModel())
    assert isinstance(result.output, TechnicalAnswer)


# ---------------------------------------------------------------------------
# run_agent_turn — marker validation + citation backfill end-to-end
# ---------------------------------------------------------------------------


def _seed_document_and_page(db: Session, page_number: int = 238) -> tuple[Document, Page]:
    document = Document(title="Manual", filename="m.pdf", source_hash="hash-1")
    db.add(document)
    db.commit()
    page = Page(document_id=document.id, page_number=page_number)
    db.add(page)
    db.commit()
    return document, page


async def test_run_agent_turn_preserves_valid_marker_and_backfills_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()
    document, page = _seed_document_and_page(db)
    icon = Element(
        document_id=document.id,
        page_id=page.id,
        element_type="icon",
        text=None,
        source_hash="h",
    )
    para = Element(
        document_id=document.id,
        page_id=page.id,
        element_type="paragraph",
        text="Press the button to reset.",
        source_hash="h",
    )
    db.add_all([icon, para])
    db.commit()

    chunk = RetrievedChunk(
        chunk_id=1,
        document_id=document.id,
        chunk_type="text",
        section_path=["Ch1"],
        page_start=page.page_number,
        page_end=page.page_number,
        element_ids=[para.id, icon.id],
        content_text="fallback",
        content_structured=None,
        model_family="REYQ",
        score=0.9,
        rank=1,
    )

    def _fake_search_documents(*args: Any, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(query="x", effective_query="x", chunks=[chunk])

    monkeypatch.setattr(tools_module, "search_documents", _fake_search_documents)

    call_count = {"n": 0}

    def _script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_documents",
                        args={"query": "reset button"},
                        tool_call_id="call-1",
                    )
                ]
            )
        out_tool_name = info.output_tools[0].name
        marker = f"{{{{el:{icon.id}}}}}"
        invented_marker = "{{el:999999}}"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=out_tool_name,
                    args={
                        "answer": f"Press {marker} {invented_marker} to reset.",
                        "confidence": 0.9,
                        "citations": [],
                        "warnings": [],
                        "related_components": [],
                        "related_error_codes": [],
                        "refused": False,
                    },
                    tool_call_id="call-2",
                )
            ]
        )

    agent = vrf_agent.build_agent(_settings())
    deps = _deps(db)

    final = await vrf_agent.run_agent_turn(
        agent, deps, "How do I reset it?", model=FunctionModel(_script)
    )

    assert f"{{{{el:{icon.id}}}}}" in final.answer
    assert "999999" not in final.answer
    assert len(final.citations) == 1
    assert final.citations[0].element_id == str(icon.id)
    assert final.refused is False


async def test_run_agent_turn_forces_refusal_when_nothing_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_session()

    def _fake_search_documents(*args: Any, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(query="x", effective_query="x", chunks=[])

    monkeypatch.setattr(tools_module, "search_documents", _fake_search_documents)

    call_count = {"n": 0}

    def _script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_documents",
                        args={"query": "nonexistent thing"},
                        tool_call_id="call-1",
                    )
                ]
            )
        out_tool_name = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=out_tool_name,
                    args={
                        "answer": "It works like this...",
                        "confidence": 0.9,
                        "citations": [],
                        "warnings": [],
                        "related_components": [],
                        "related_error_codes": [],
                        "refused": False,
                    },
                    tool_call_id="call-2",
                )
            ]
        )

    agent = vrf_agent.build_agent(_settings())
    deps = _deps(db)

    final = await vrf_agent.run_agent_turn(
        agent, deps, "What about nonexistent thing?", model=FunctionModel(_script)
    )

    assert final.refused is True


async def test_run_agent_turn_greeting_no_tool_calls_not_forced_refused() -> None:
    db = _make_session()

    def _script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        out_tool_name = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=out_tool_name,
                    args={
                        "answer": "Hello! How can I help you today?",
                        "confidence": 0.95,
                        "citations": [],
                        "warnings": [],
                        "related_components": [],
                        "related_error_codes": [],
                        "refused": False,
                    },
                    tool_call_id="call-1",
                )
            ]
        )

    agent = vrf_agent.build_agent(_settings())
    deps = _deps(db)

    final = await vrf_agent.run_agent_turn(agent, deps, "hello", model=FunctionModel(_script))

    assert final.refused is False
    assert final.answer == "Hello! How can I help you today?"


def test_dynamic_instructions_with_model_family() -> None:
    db = _make_session()
    deps = _deps(db)
    deps.model_family = "REYQ"

    class _FakeRunContext:
        def __init__(self, deps: AgentDeps) -> None:
            self.deps = deps

    text = vrf_agent._dynamic_instructions(_FakeRunContext(deps))  # type: ignore[arg-type]
    assert "REYQ" in text


def test_dynamic_instructions_without_model_family() -> None:
    db = _make_session()
    deps = _deps(db)

    class _FakeRunContext:
        def __init__(self, deps: AgentDeps) -> None:
            self.deps = deps

    text = vrf_agent._dynamic_instructions(_FakeRunContext(deps))  # type: ignore[arg-type]
    assert "No specific model family filter" in text


def test_static_system_prompt_mentions_never_invent_and_marker_rule() -> None:
    assert "NEVER INVENT" in vrf_agent.STATIC_SYSTEM_PROMPT
    assert "{{el:" in vrf_agent.STATIC_SYSTEM_PROMPT
