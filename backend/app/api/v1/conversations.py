"""`GET /api/v1/conversations` (list) / `GET /api/v1/conversations/{id}`
(detail) — per `05-streaming-and-api-contract.md` §6 (C2.6). Scope
`chat:read`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Security
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.db.engine import get_db
from app.db.models.conversations import Conversation, Message

router = APIRouter(prefix="/conversations", tags=["conversations"])

READ = ["chat:read"]


class ConversationSummaryResponse(BaseModel):
    id: int
    title: str | None
    created_at: str


class CitationResponse(BaseModel):
    id: int
    document_id: int
    page: int | None
    element_id: int | None
    quote: str | None
    rank: int | None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str | None
    structured_answer: dict | None
    model_provider: str | None
    model_name: str | None
    ttft_ms: int | None
    total_latency_ms: int | None
    created_at: str
    citations: list[CitationResponse]


class ConversationDetailResponse(ConversationSummaryResponse):
    messages: list[MessageResponse]


def _conversation_summary(conversation: Conversation) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation.id, title=conversation.title, created_at=conversation.created_at.isoformat()
    )


@router.get("", response_model=list[ConversationSummaryResponse])
async def list_conversations(
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[ConversationSummaryResponse]:
    conversations = (
        db.execute(select(Conversation).order_by(Conversation.created_at.desc())).scalars().all()
    )
    return [_conversation_summary(conversation) for conversation in conversations]


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> ConversationDetailResponse:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        )
        .scalars()
        .all()
    )

    message_responses = [
        MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            structured_answer=message.structured_answer,
            model_provider=message.model_provider,
            model_name=message.model_name,
            ttft_ms=message.ttft_ms,
            total_latency_ms=message.total_latency_ms,
            created_at=message.created_at.isoformat(),
            citations=[
                CitationResponse(
                    id=citation.id,
                    document_id=citation.document_id,
                    page=citation.page,
                    element_id=citation.element_id,
                    quote=citation.quote,
                    rank=citation.rank,
                )
                for citation in sorted(message.citations, key=lambda c: c.rank or 0)
            ],
        )
        for message in messages
    ]

    summary = _conversation_summary(conversation)
    return ConversationDetailResponse(
        id=summary.id,
        title=summary.title,
        created_at=summary.created_at,
        messages=message_responses,
    )
