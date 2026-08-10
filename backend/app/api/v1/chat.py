"""`POST /api/v1/chat` (non-streaming) / `POST /api/v1/chat/stream` (SSE) —
per `Documentation/system-design/05-streaming-and-api-contract.md` §4/§6.
Scope `chat:write` (both endpoints send a message and get a response, per
the `chat:write` = "Mengirim pesan chat" scope description,
`08-authentication-rbac.md`).

**C2.6**: every turn (both endpoints) is persisted —
`app.db.conversation_store.get_or_create_conversation`/
`persist_user_message`/`persist_assistant_message`, per `06-data-schema.md`
§1. `conversation_id` is resolved (created if absent, looked up if given)
*before* the turn runs, so the user's own message is recorded even if the
LLM call itself times out/errors afterward.

**`ChatResponse.conversation_id`** — ratified in
`05-streaming-and-api-contract.md` §5.6 as `str` (this fix, F2-04
`Documentation/qa-reports/phase-2-qa-report.md`; originally added here as
`int` and flagged for confirmation, since corrected to match the ratified
contract type — see `app/agent/streaming.py` `stream_turn`'s
`conversation_id`/`_with_conversation_id` for the identical treatment on
the SSE `status`/`error`/`done` events).

**Why the agent/qdrant client/embedding models are built once at app
startup, not per-request** (`app/main.py`): re-constructing
`FastEmbedDenseModel`/`FastEmbedSparseModel` per request would reload the
underlying ONNX model from disk on every single chat message — a real,
avoidable latency/resource cost directly inside the TTFT budget. A fresh
`Session` (`app.db.engine.get_db`) is still created per-request as usual
(SQLAlchemy sessions are not safe to share across concurrent requests).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai.usage import UsageLimits
from sqlalchemy.orm import Session

from app.agent.schemas import TechnicalAnswer
from app.agent.streaming import run_turn_with_metrics, stream_turn
from app.agent.tools import AgentDeps
from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.conversation_store import (
    ConversationNotFoundError,
    get_or_create_conversation,
    persist_assistant_message,
    persist_user_message,
)
from app.db.engine import get_db
from app.db.models.conversations import Conversation
from app.domain.query_expansion import load_known_entities

router = APIRouter(prefix="/chat", tags=["chat"])

WRITE = ["chat:write"]


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str
    model_override: str | None = None


class ChatResponse(TechnicalAnswer):
    # F2-04/§5.6 — serialized as a string on the wire, per the ratified
    # contract type (`app/agent/streaming.py` does the same for the SSE
    # `status`/`error`/`done` events' `conversation_id`). Internal
    # representation stays an int DB primary key everywhere else.
    conversation_id: str


def _build_agent_deps(
    request: Request, db: Session, settings: Settings, chat_request: ChatRequest
) -> AgentDeps:
    known_entities = load_known_entities(db)
    return AgentDeps(
        db=db,
        qdrant_client=request.app.state.qdrant_client,
        dense_model=request.app.state.dense_embedding_model,
        sparse_model=request.app.state.sparse_embedding_model,
        collection=settings.VECTOR_STORE_COLLECTION,
        default_top_k=settings.RETRIEVAL_DEFAULT_TOP_K,
        circuit_breaker_seconds=settings.RETRIEVAL_CIRCUIT_BREAKER_SECONDS,
        known_entities=known_entities,
        max_chunk_chars=settings.CONTEXT_BUILDER_MAX_CHUNK_CHARS,  # F2-07
    )


def _build_usage_limits(settings: Settings) -> UsageLimits:
    """F2-06 — wires `Settings.CHAT_LLM_MAX_REQUESTS_PER_TURN` into the
    per-turn round-trip cap (`app/agent/vrf_agent.py`/`app/agent/streaming.py`
    otherwise fall back to a hardcoded default of their own if this is never
    called)."""
    return UsageLimits(request_limit=settings.CHAT_LLM_MAX_REQUESTS_PER_TURN)


def _resolve_conversation(
    db: Session, chat_request: ChatRequest, owner_id: int | None
) -> Conversation:
    try:
        return get_or_create_conversation(
            db,
            chat_request.conversation_id,
            owner_id=owner_id,
            title_hint=chat_request.message,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> ChatResponse:
    """Non-streaming — used by the `vrf-qa/` evaluation harness so it
    doesn't need to parse SSE for batch runs (§6)."""
    conversation = _resolve_conversation(db, chat_request, user.id)
    persist_user_message(db, conversation.id, chat_request.message)

    deps = _build_agent_deps(request, db, settings, chat_request)
    result = await run_turn_with_metrics(
        request.app.state.chat_agent,
        deps,
        chat_request.message,
        model=chat_request.model_override,
        timeout_seconds=settings.CHAT_LLM_TIMEOUT_SECONDS,
        usage_limits=_build_usage_limits(settings),
    )

    persist_assistant_message(
        db,
        conversation.id,
        result.answer,
        model_provider=settings.CHAT_LLM_PROVIDER,
        model_name=chat_request.model_override or settings.CHAT_LLM_MODEL,
        ttft_ms=result.ttft_ms,
        total_latency_ms=result.total_latency_ms,
    )

    return ChatResponse(conversation_id=str(conversation.id), **result.answer.model_dump())


@router.post("/stream")
async def chat_stream(
    chat_request: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> StreamingResponse:
    conversation = _resolve_conversation(db, chat_request, user.id)
    persist_user_message(db, conversation.id, chat_request.message)

    deps = _build_agent_deps(request, db, settings, chat_request)

    def _on_done(answer: TechnicalAnswer, ttft_ms: int, total_latency_ms: int) -> None:
        persist_assistant_message(
            db,
            conversation.id,
            answer,
            model_provider=settings.CHAT_LLM_PROVIDER,
            model_name=chat_request.model_override or settings.CHAT_LLM_MODEL,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
        )

    generator = stream_turn(
        request.app.state.chat_agent,
        deps,
        chat_request.message,
        model=chat_request.model_override,
        timeout_seconds=settings.CHAT_LLM_TIMEOUT_SECONDS,
        usage_limits=_build_usage_limits(settings),
        on_done=_on_done,
        conversation_id=conversation.id,
    )
    return StreamingResponse(generator, media_type="text/event-stream")
