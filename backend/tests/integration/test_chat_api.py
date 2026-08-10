"""Integration tests for `POST /api/v1/chat` / `POST /api/v1/chat/stream`
(C2.4) — HTTP-layer wiring (scope enforcement, request/response shape, SSE
framing over a real `TestClient`), not agent/retrieval internals (already
covered by `tests/unit/test_agent_*`).

`app.state.chat_agent` is swapped for a `FunctionModel`-driven agent after
app construction so no real LLM/Qdrant network call happens — `qdrant_client`/
`dense_embedding_model`/`sparse_embedding_model` stay as plain placeholders
since these scripted agents never call a retrieval tool.
"""

from __future__ import annotations

import json
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from app.agent.schemas import TechnicalAnswer
from app.agent.tools import AgentDeps
from app.core.config import Settings
from app.db.engine import get_db
from app.db.models.conversations import Conversation, Message
from app.main import create_app
from tests.integration.conftest import USER_EMAIL, USER_PASSWORD


def _final_result_response(payload: dict[str, Any]) -> Any:
    def _func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args=payload, tool_call_id="call-1"
                )
            ]
        )

    return _func


def _final_result_stream(payload: dict[str, Any]) -> Any:
    async def _stream_func(messages: list[ModelMessage], info: AgentInfo):
        yield {
            0: DeltaToolCall(
                name=info.output_tools[0].name, json_args=json.dumps(payload), tool_call_id="call-1"
            )
        }

    return _stream_func


@pytest.fixture
def chat_client(
    db_session_factory, seed_data, fake_redis: fakeredis.FakeRedis, test_settings: Settings
):
    app = create_app(test_settings)

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    payload = {
        "answer": "Hello! How can I help you with your VRF/VRV system today?",
        "confidence": 0.9,
        "citations": [],
        "warnings": [],
        "related_components": [],
        "related_error_codes": [],
        "refused": False,
    }
    app.state.chat_agent = Agent(
        FunctionModel(
            function=_final_result_response(payload),
            stream_function=_final_result_stream(payload),
        ),
        deps_type=AgentDeps,
        output_type=TechnicalAnswer,
    )
    app.state.qdrant_client = object()
    app.state.dense_embedding_model = object()
    app.state.sparse_embedding_model = object()

    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client, app, db_session_factory


def _login(client: TestClient) -> str:
    response = client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------


def test_chat_requires_scope(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_happy_path(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    token = _login(client)

    response = client.post("/api/v1/chat", json={"message": "hello"}, headers=_auth_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello! How can I help you with your VRF/VRV system today?"
    assert body["refused"] is False
    assert body["confidence"] == 0.9


def test_chat_accepts_conversation_id_and_model_override(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    token = _login(client)

    response = client.post(
        "/api/v1/chat",
        json={"message": "hello", "conversation_id": None, "model_override": None},
        headers=_auth_headers(token),
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/chat/stream
# ---------------------------------------------------------------------------


def _parse_sse_stream(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        event_name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event_name, data))
    return events


def test_chat_stream_requires_scope(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    response = client.post("/api/v1/chat/stream", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_stream_happy_path_event_sequence(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    token = _login(client)

    response = client.post(
        "/api/v1/chat/stream", json={"message": "hello"}, headers=_auth_headers(token)
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_stream(response.text)
    event_names = [name for name, _ in events]
    assert event_names[:3] == ["status", "status", "status"]
    assert event_names[-1] == "done"
    done_payload = events[-1][1]
    assert done_payload["answer"] == "Hello! How can I help you with your VRF/VRV system today?"
    assert "ttft_ms" in done_payload
    assert "total_latency_ms" in done_payload


# ---------------------------------------------------------------------------
# C2.6 — conversation/message/citation persistence
# ---------------------------------------------------------------------------


def test_chat_response_includes_conversation_id_and_persists_messages(chat_client) -> None:
    client, _app, db_session_factory = chat_client
    token = _login(client)

    response = client.post("/api/v1/chat", json={"message": "hello"}, headers=_auth_headers(token))

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    # F2-04/§5.6: serialized as a string on the wire, even though the
    # internal representation (a DB primary key) is an int.
    assert isinstance(conversation_id, str)

    session = db_session_factory()
    try:
        conversation = session.get(Conversation, int(conversation_id))
        assert conversation is not None
        assert conversation.title == "hello"
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == int(conversation_id))
            .order_by(Message.id)
            .all()
        )
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[0].content == "hello"
        assert messages[1].content == "Hello! How can I help you with your VRF/VRV system today?"
        assert messages[1].ttft_ms is not None
        assert messages[1].total_latency_ms is not None
        assert messages[1].model_provider is not None
    finally:
        session.close()


def test_chat_reuses_existing_conversation_id(chat_client) -> None:
    client, _app, db_session_factory = chat_client
    token = _login(client)

    first = client.post("/api/v1/chat", json={"message": "hello"}, headers=_auth_headers(token))
    conversation_id = first.json()["conversation_id"]

    second = client.post(
        "/api/v1/chat",
        # `conversation_id` on the wire is a string (§5.6); ChatRequest's
        # `conversation_id: int | None` field coerces a numeric string
        # (standard pydantic lenient-mode behavior) — this also exercises
        # that coercion path, not just a same-type roundtrip.
        json={"message": "follow-up", "conversation_id": conversation_id},
        headers=_auth_headers(token),
    )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id

    session = db_session_factory()
    try:
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == int(conversation_id))
            .order_by(Message.id)
            .all()
        )
        assert len(messages) == 4  # user+assistant x2 turns
    finally:
        session.close()


def test_chat_unknown_conversation_id_returns_404(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    token = _login(client)

    response = client.post(
        "/api/v1/chat",
        json={"message": "hello", "conversation_id": 999999},
        headers=_auth_headers(token),
    )

    assert response.status_code == 404


def test_chat_stream_done_event_includes_conversation_id_and_persists(chat_client) -> None:
    client, _app, db_session_factory = chat_client
    token = _login(client)

    response = client.post(
        "/api/v1/chat/stream", json={"message": "hello"}, headers=_auth_headers(token)
    )
    events = _parse_sse_stream(response.text)
    done_payload = events[-1][1]
    conversation_id = done_payload["conversation_id"]
    assert isinstance(conversation_id, str)

    session = db_session_factory()
    try:
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == int(conversation_id))
            .order_by(Message.id)
            .all()
        )
        assert [m.role for m in messages] == ["user", "assistant"]
    finally:
        session.close()


def test_chat_stream_unknown_conversation_id_returns_404(chat_client) -> None:
    client, _app, _db_session_factory = chat_client
    token = _login(client)

    response = client.post(
        "/api/v1/chat/stream",
        json={"message": "hello", "conversation_id": 999999},
        headers=_auth_headers(token),
    )

    assert response.status_code == 404
