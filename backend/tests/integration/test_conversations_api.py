"""Integration tests for `GET /api/v1/conversations` (list) /
`GET /api/v1/conversations/{id}` (detail) — C2.6.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db.engine import get_db
from app.db.models.conversations import Citation, Conversation, Message
from app.main import create_app
from tests.integration.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, USER_EMAIL, USER_PASSWORD


@pytest.fixture
def conversations_client(db_session_factory, seed_data, fake_redis, test_settings):
    app = create_app(test_settings)

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client, db_session_factory


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_conversation(conversations_client):
    _client, db_session_factory = conversations_client
    session = db_session_factory()
    try:
        conversation = Conversation(title="How do I fix error P8?")
        session.add(conversation)
        session.commit()

        user_message = Message(conversation_id=conversation.id, role="user", content="P8 error?")
        session.add(user_message)
        session.commit()

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Check the fan motor.",
            structured_answer={"answer": "Check the fan motor.", "confidence": 0.9},
            model_provider="anthropic",
            model_name="claude",
            ttft_ms=1000,
            total_latency_ms=2000,
        )
        session.add(assistant_message)
        session.commit()

        session.add(
            Citation(
                message_id=assistant_message.id,
                document_id=1,
                page=10,
                element_id=None,
                quote="Check the fan motor.",
                rank=1,
            )
        )
        session.commit()

        return conversation.id
    finally:
        session.close()


def test_list_conversations_requires_scope(conversations_client) -> None:
    client, _factory = conversations_client
    response = client.get("/api/v1/conversations")
    assert response.status_code == 401


def test_list_conversations_returns_seeded(conversations_client, seeded_conversation) -> None:
    client, _factory = conversations_client
    token = _login(client, USER_EMAIL, USER_PASSWORD)

    response = client.get("/api/v1/conversations", headers=_auth_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == seeded_conversation
    assert body[0]["title"] == "How do I fix error P8?"


def test_list_conversations_empty(conversations_client) -> None:
    client, _factory = conversations_client
    token = _login(client, USER_EMAIL, USER_PASSWORD)

    response = client.get("/api/v1/conversations", headers=_auth_headers(token))

    assert response.status_code == 200
    assert response.json() == []


def test_get_conversation_detail_includes_messages_and_citations(
    conversations_client, seeded_conversation
) -> None:
    client, _factory = conversations_client
    token = _login(client, USER_EMAIL, USER_PASSWORD)

    response = client.get(
        f"/api/v1/conversations/{seeded_conversation}", headers=_auth_headers(token)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == seeded_conversation
    assert len(body["messages"]) == 2
    user_msg, assistant_msg = body["messages"]
    assert user_msg["role"] == "user"
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["ttft_ms"] == 1000
    assert assistant_msg["total_latency_ms"] == 2000
    assert len(assistant_msg["citations"]) == 1
    assert assistant_msg["citations"][0]["document_id"] == 1
    assert assistant_msg["citations"][0]["page"] == 10
    assert user_msg["citations"] == []


def test_get_conversation_not_found(conversations_client) -> None:
    client, _factory = conversations_client
    token = _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = client.get("/api/v1/conversations/999999", headers=_auth_headers(token))

    assert response.status_code == 404


def test_get_conversation_requires_scope(conversations_client, seeded_conversation) -> None:
    client, _factory = conversations_client
    response = client.get(f"/api/v1/conversations/{seeded_conversation}")
    assert response.status_code == 401
