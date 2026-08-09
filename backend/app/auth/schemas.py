"""Pydantic request/response models for `/auth/*` — contract per
`Documentation/system-design/08-authentication-rbac.md` §8.5.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MeResponse(BaseModel):
    id: int
    username: str
    role: str
    scopes: list[str]


class LogoutResponse(BaseModel):
    detail: str = "Logged out"


class RateLimitedResponse(BaseModel):
    detail: str = Field(default="Too many login attempts, try again later")
    retry_after_seconds: int


class AuthenticatedUser(BaseModel):
    """Decoded-JWT view of the caller — constructed purely from token
    claims (no DB round-trip), see 08-authentication-rbac.md §3.1."""

    id: int
    role: str
    scopes: list[str]
