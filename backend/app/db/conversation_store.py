"""Conversation/message/citation persistence — per `06-data-schema.md` §1
and milestone C2.6 ("simpan messages/citations per turn chat"). Used by
`app/api/v1/chat.py` after a turn's final `TechnicalAnswer` is ready (both
the non-streaming and SSE-streaming paths persist identically, once, after
post-processing/the safety net — never the raw pre-validation answer).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.agent.schemas import TechnicalAnswer
from app.core.observability import get_logger
from app.db.models.conversations import Citation, Conversation, Message

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
    position in that list. A citation whose `document_id` doesn't parse as
    an int (should never happen — this backend always generates it from a
    real integer `documents.id`, defensive only) is skipped with a warning
    rather than aborting the whole message persist.
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

    for rank, citation in enumerate(answer.citations, start=1):
        document_id = _safe_int(citation.document_id)
        if document_id is None:
            logger.warning(
                "conversation_store.citation_skipped_invalid_document_id",
                extra={"document_id": citation.document_id, "message_id": message.id},
            )
            continue
        db.add(
            Citation(
                message_id=message.id,
                document_id=document_id,
                page=citation.page,
                element_id=_safe_int(citation.element_id),
                quote=citation.quote,
                rank=rank,
            )
        )
    db.commit()

    return message
