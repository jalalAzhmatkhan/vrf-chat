"""`POST /api/v1/chat` (non-streaming) / `POST /api/v1/chat/stream` (SSE) —
per `Documentation/system-design/05-streaming-and-api-contract.md` §4/§6.
Scope `chat:write` (both endpoints send a message and get a response, per
the `chat:write` = "Mengirim pesan chat" scope description,
`08-authentication-rbac.md`).

**C2.4 scope note**: `conversation_id` is accepted in the request body (per
the documented contract) but not yet persisted/resolved against
`conversations`/`messages` — that's `app/api/v1/chat.py`'s C2.6 addition
(`06-data-schema.md` §1). This branch focuses on the SSE/non-streaming
mechanics + TTFT instrumentation + circuit breaker + timeout enforcement.

**Why the agent/qdrant client/embedding models are built once at app
startup, not per-request** (`app/main.py`): re-constructing
`FastEmbedDenseModel`/`FastEmbedSparseModel` per request would reload the
underlying ONNX model from disk on every single chat message — a real,
avoidable latency/resource cost directly inside the TTFT budget. A fresh
`Session` (`app.db.engine.get_db`) is still created per-request as usual
(SQLAlchemy sessions are not safe to share across concurrent requests).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Security
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
from app.db.engine import get_db
from app.domain.query_expansion import load_known_entities

router = APIRouter(prefix="/chat", tags=["chat"])

WRITE = ["chat:write"]


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str
    model_override: str | None = None


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


@router.post("", response_model=TechnicalAnswer)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> TechnicalAnswer:
    """Non-streaming — used by the `vrf-qa/` evaluation harness so it
    doesn't need to parse SSE for batch runs (§6)."""
    deps = _build_agent_deps(request, db, settings, chat_request)
    result = await run_turn_with_metrics(
        request.app.state.chat_agent,
        deps,
        chat_request.message,
        model=chat_request.model_override,
        timeout_seconds=settings.CHAT_LLM_TIMEOUT_SECONDS,
        usage_limits=_build_usage_limits(settings),
    )
    return result.answer


@router.post("/stream")
async def chat_stream(
    chat_request: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> StreamingResponse:
    deps = _build_agent_deps(request, db, settings, chat_request)
    generator = stream_turn(
        request.app.state.chat_agent,
        deps,
        chat_request.message,
        model=chat_request.model_override,
        timeout_seconds=settings.CHAT_LLM_TIMEOUT_SECONDS,
        usage_limits=_build_usage_limits(settings),
    )
    return StreamingResponse(generator, media_type="text/event-stream")
