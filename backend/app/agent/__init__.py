"""Public interface of `app/agent/` — see
`Documentation/system-design/01-architecture-overview.md` §3 module boundary
principle. `app/api/v1/chat.py` (C2.4) should import from here."""

from __future__ import annotations

from app.agent.answer_postprocess import (
    PostProcessResult,
    enforce_never_invent_safety_net,
    postprocess_answer,
)
from app.agent.context_builder import BuiltContext, ContextChunk, ContextElement, build_context
from app.agent.schemas import Citation, TechnicalAnswer, Warning
from app.agent.tools import AgentDeps
from app.agent.vrf_agent import build_agent, run_agent_turn

__all__ = [
    "AgentDeps",
    "BuiltContext",
    "Citation",
    "ContextChunk",
    "ContextElement",
    "PostProcessResult",
    "TechnicalAnswer",
    "Warning",
    "build_agent",
    "build_context",
    "enforce_never_invent_safety_net",
    "postprocess_answer",
    "run_agent_turn",
]
