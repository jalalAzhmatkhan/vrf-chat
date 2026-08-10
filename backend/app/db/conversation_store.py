"""Conversation/message/citation persistence — per `06-data-schema.md` §1
and milestone C2.6 ("simpan messages/citations per turn chat"). Used by
`app/api/v1/chat.py` after a turn's final `TechnicalAnswer` is ready (both
the non-streaming and SSE-streaming paths persist identically, once, after
post-processing/the safety net — never the raw pre-validation answer).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session

from app.agent.schemas import TechnicalAnswer
from app.core.observability import get_logger
from app.db.models.conversations import Citation, Conversation, Message
from app.db.models.documents import Document
from app.db.models.elements import Element

logger = get_logger(__name__)

TITLE_MAX_LENGTH = 200


class ConversationNotFoundError(ValueError):
    """Raised when a caller supplies a `conversation_id` that doesn't
    exist — distinct from "no conversation_id given" (which creates a new
    conversation)."""


def _title_from_message(message: str) -> str:
    stripped = message.strip()
    if len(stripped) <= TITLE_MAX_LENGTH:
        return stripped
    return stripped[: TITLE_MAX_LENGTH - 1].rstrip() + "…"


def get_or_create_conversation(
    db: Session,
    conversation_id: int | None,
    *,
    owner_id: int | None,
    title_hint: str,
) -> Conversation:
    """If `conversation_id` is given, it MUST already exist (raises
    `ConversationNotFoundError` otherwise — silently creating a new
    conversation under a caller-supplied id the caller didn't actually own
    would be confusing, not helpful). If `conversation_id` is `None`, a new
    conversation is created, titled from the first ~200 chars of the user's
    message."""
    if conversation_id is not None:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        return conversation

    conversation = Conversation(owner_id=owner_id, title=_title_from_message(title_hint))
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def persist_user_message(db: Session, conversation_id: int, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_ids(db: Session, id_column: Mapped[int], ids: set[int]) -> set[int]:
    if not ids:
        return set()
    return set(db.execute(select(id_column).where(id_column.in_(ids))).scalars().all())


def persist_assistant_message(
    db: Session,
    conversation_id: int,
    answer: TechnicalAnswer,
    *,
    model_provider: str | None,
    model_name: str | None,
    ttft_ms: int,
    total_latency_ms: int,
) -> Message:
    """Persists the assistant's turn (`messages` row, `structured_answer`
    = the full `TechnicalAnswer`, per `06-data-schema.md` §1) plus one
    `citations` row per `answer.citations` entry, `rank` = 1-indexed
    position in that list.

    **F2-08** (`Documentation/qa-reports/phase-2-qa-report.md`): both
    `citations.document_id` (`NOT NULL` FK to `documents.id`) and
    `citations.element_id` (nullable FK to `elements.id`) are verified to
    reference a row that actually exists **before** any `Citation` row is
    added to the session — QA observed an unguarded `IntegrityError` here
    (`element_id` pointing at a since-deleted/nonexistent element) crash
    `POST /api/v1/chat` with a bare HTTP 500, and silently break the SSE
    generator mid-stream on `POST /api/v1/chat/stream` (this function is
    called from `stream_turn`'s `on_done` hook, *inside* the generator,
    before `done` is yielded). F2-02 (citation validation against this
    turn's retrieval whitelist) already eliminates most real-world cases —
    every citation that reaches here should already reference a row F2-02
    itself just read from these same tables — but this is a second,
    independent layer of defense at the persistence boundary, per QA's
    explicit request, not a replacement for F2-02. A citation with a
    missing/nonexistent `document_id` is skipped entirely (that FK is
    required, not nullable); one with a missing/nonexistent `element_id`
    is still persisted, just with `element_id=None` (matches the column's
    own `ondelete="SET NULL"` semantics — the citation's document/page/quote
    are still meaningful without a specific element link).
    """
    message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer.answer,
        structured_answer=answer.model_dump(mode="json"),
        model_provider=model_provider,
        model_name=model_name,
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    candidate_document_ids = {
        parsed
        for citation in answer.citations
        if (parsed := _safe_int(citation.document_id)) is not None
    }
    candidate_element_ids = {
        parsed
        for citation in answer.citations
        if (parsed := _safe_int(citation.element_id)) is not None
    }
    existing_document_ids = _existing_ids(db, Document.id, candidate_document_ids)
    existing_element_ids = _existing_ids(db, Element.id, candidate_element_ids)

    for rank, citation in enumerate(answer.citations, start=1):
        document_id = _safe_int(citation.document_id)
        if document_id is None or document_id not in existing_document_ids:
            logger.warning(
                "conversation_store.citation_skipped_invalid_document_id",
                extra={"document_id": citation.document_id, "message_id": message.id},
            )
            continue

        element_id = _safe_int(citation.element_id)
        if element_id is not None and element_id not in existing_element_ids:
            logger.warning(
                "conversation_store.citation_element_id_not_found",
                extra={"element_id": citation.element_id, "message_id": message.id},
            )
            element_id = None

        db.add(
            Citation(
                message_id=message.id,
                document_id=document_id,
                page=citation.page,
                element_id=element_id,
                quote=citation.quote,
                rank=rank,
            )
        )
    db.commit()

    return message
