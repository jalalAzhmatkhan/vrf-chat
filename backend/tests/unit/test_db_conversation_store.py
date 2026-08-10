"""Unit tests for `app/db/conversation_store.py` (C2.6)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.schemas import Citation, TechnicalAnswer, Warning
from app.db import conversation_store as cstore
from app.db.base import Base
from app.db.models.conversations import Conversation, Message
from app.db.models.documents import Document, Page
from app.db.models.elements import Element


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_document(db: Session, document_id: int) -> Document:
    document = Document(
        id=document_id, title="Manual", filename="m.pdf", source_hash=f"hash-{document_id}"
    )
    db.add(document)
    db.commit()
    return document


def _seed_element(db: Session, element_id: int, document_id: int) -> Element:
    page = Page(document_id=document_id, page_number=1)
    db.add(page)
    db.commit()
    element = Element(
        id=element_id,
        document_id=document_id,
        page_id=page.id,
        element_type="icon",
        source_hash=f"hash-el-{element_id}",
    )
    db.add(element)
    db.commit()
    return element


# ---------------------------------------------------------------------------
# get_or_create_conversation
# ---------------------------------------------------------------------------


def test_get_or_create_conversation_creates_new_with_title_from_message() -> None:
    db = _make_session()
    conversation = cstore.get_or_create_conversation(
        db, None, owner_id=7, title_hint="How do I fix error P8?"
    )
    assert conversation.id is not None
    assert conversation.owner_id == 7
    assert conversation.title == "How do I fix error P8?"


def test_get_or_create_conversation_truncates_long_title() -> None:
    db = _make_session()
    long_message = "a" * 500
    conversation = cstore.get_or_create_conversation(
        db, None, owner_id=None, title_hint=long_message
    )
    assert len(conversation.title) == cstore.TITLE_MAX_LENGTH
    assert conversation.title.endswith("…")


def test_get_or_create_conversation_reuses_existing() -> None:
    db = _make_session()
    existing = Conversation(owner_id=1, title="Existing")
    db.add(existing)
    db.commit()

    conversation = cstore.get_or_create_conversation(
        db, existing.id, owner_id=1, title_hint="ignored"
    )
    assert conversation.id == existing.id
    assert conversation.title == "Existing"


def test_get_or_create_conversation_raises_when_id_not_found() -> None:
    db = _make_session()
    with pytest.raises(cstore.ConversationNotFoundError):
        cstore.get_or_create_conversation(db, 999, owner_id=1, title_hint="x")


# ---------------------------------------------------------------------------
# persist_user_message
# ---------------------------------------------------------------------------


def test_persist_user_message() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()

    message = cstore.persist_user_message(db, conversation.id, "hello")

    assert message.id is not None
    assert message.role == "user"
    assert message.content == "hello"
    stored = db.get(Message, message.id)
    assert stored.conversation_id == conversation.id


# ---------------------------------------------------------------------------
# persist_assistant_message
# ---------------------------------------------------------------------------


def test_persist_assistant_message_with_citations() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()
    _seed_document(db, 3)
    _seed_element(db, 42, document_id=3)
    _seed_element(db, 99, document_id=3)

    answer = TechnicalAnswer(
        answer="Press {{el:42}} to reset.",
        confidence=0.9,
        citations=[
            Citation(document_id="3", page=238, element_id="42", element_type="icon"),
            Citation(document_id="3", page=157, element_id="99", element_type="table"),
        ],
        warnings=[Warning(message="high voltage", severity="safety")],
        refused=False,
    )

    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude-sonnet-4-5",
        ttft_ms=1200,
        total_latency_ms=3400,
    )

    assert message.role == "assistant"
    assert message.content == "Press {{el:42}} to reset."
    assert message.model_provider == "anthropic"
    assert message.model_name == "claude-sonnet-4-5"
    assert message.ttft_ms == 1200
    assert message.total_latency_ms == 3400
    assert message.structured_answer["confidence"] == 0.9
    assert message.structured_answer["warnings"][0]["severity"] == "safety"

    db.refresh(message)
    assert len(message.citations) == 2
    ranks = sorted(c.rank for c in message.citations)
    assert ranks == [1, 2]
    first = next(c for c in message.citations if c.rank == 1)
    assert first.document_id == 3
    assert first.element_id == 42
    assert first.page == 238


def test_persist_assistant_message_no_citations() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()

    answer = TechnicalAnswer(answer="Hi.", confidence=0.9)
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=100,
        total_latency_ms=200,
    )
    db.refresh(message)
    assert message.citations == []


def test_persist_assistant_message_skips_citation_with_invalid_document_id() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()

    answer = TechnicalAnswer(
        answer="See it.",
        confidence=0.9,
        citations=[
            Citation(document_id="not-a-number", page=1, element_id="1", element_type="table")
        ],
    )
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=1,
        total_latency_ms=2,
    )
    db.refresh(message)
    assert message.citations == []


def test_persist_assistant_message_citation_with_missing_element_id() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()
    _seed_document(db, 3)

    answer = TechnicalAnswer(
        answer="See it.",
        confidence=0.9,
        citations=[
            Citation(document_id="3", page=1, element_id="not-a-number", element_type="table")
        ],
    )
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=1,
        total_latency_ms=2,
    )
    db.refresh(message)
    assert len(message.citations) == 1
    assert message.citations[0].element_id is None


# ---------------------------------------------------------------------------
# persist_assistant_message — F2-08 FK existence guard
# ---------------------------------------------------------------------------


def test_persist_assistant_message_document_id_valid_int_but_no_such_document_skipped() -> None:
    """`document_id="999999"` parses fine as an int, but no such row exists
    (e.g. a hallucinated/since-deleted document) — must not attempt an
    insert that would violate the NOT NULL FK, and must not crash."""
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()

    answer = TechnicalAnswer(
        answer="See it.",
        confidence=0.9,
        citations=[
            Citation(document_id="999999", page=1, element_id="1", element_type="table")
        ],
    )
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=1,
        total_latency_ms=2,
    )
    db.refresh(message)
    assert message.citations == []


def test_persist_assistant_message_element_id_valid_int_but_no_such_element_nulled() -> None:
    """`element_id="999999"` parses fine as an int, but no such row exists.
    Deterministic reproduction of the F2-08 QA repro
    (`psycopg.errors.ForeignKeyViolation ... citations_element_id_fkey`,
    `element_id=13` not present in `elements`) — the whole citation must
    still be persisted (document_id/page/quote remain meaningful), just
    with `element_id=None`, and must not crash `persist_assistant_message`
    or the caller (`POST /api/v1/chat` 500 / `POST /api/v1/chat/stream`
    generator breaking mid-stream, per the QA report)."""
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()
    _seed_document(db, 3)

    answer = TechnicalAnswer(
        answer="See it.",
        confidence=0.9,
        citations=[
            Citation(document_id="3", page=60, element_id="999999", element_type="text")
        ],
    )
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=1,
        total_latency_ms=2,
    )
    db.refresh(message)
    assert len(message.citations) == 1
    assert message.citations[0].element_id is None
    assert message.citations[0].document_id == 3
    assert message.citations[0].page == 60


def test_persist_assistant_message_mixed_valid_and_fk_violating_citations() -> None:
    db = _make_session()
    conversation = Conversation(title="C")
    db.add(conversation)
    db.commit()
    _seed_document(db, 3)
    _seed_element(db, 42, document_id=3)

    answer = TechnicalAnswer(
        answer="See it.",
        confidence=0.9,
        citations=[
            Citation(document_id="3", page=1, element_id="42", element_type="icon"),
            Citation(document_id="999999", page=1, element_id="1", element_type="table"),
            Citation(document_id="3", page=2, element_id="888888", element_type="table"),
        ],
    )
    message = cstore.persist_assistant_message(
        db,
        conversation.id,
        answer,
        model_provider="anthropic",
        model_name="claude",
        ttft_ms=1,
        total_latency_ms=2,
    )
    db.refresh(message)
    assert len(message.citations) == 2  # the document_id=999999 one is dropped
    element_ids = sorted((c.element_id for c in message.citations), key=lambda v: (v is None, v))
    assert element_ids == [42, None]
