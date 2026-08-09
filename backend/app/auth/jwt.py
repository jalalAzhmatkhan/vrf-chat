"""JWT access token issuance/verification — see
`Documentation/system-design/08-authentication-rbac.md` §1/§3.

Access tokens are short-lived (30 min default), carry `scopes` so
`SecurityScopes` can validate without a DB round-trip, and always include a
`jti` (unique token id) — unused for revocation checks in MVP (logout = Opsi
1, refresh-token-only revocation, §8.4), but present from day one as a cheap
hook for an access-token blocklist (Opsi 2) if that's ever needed later
without a token schema change.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import Settings


class InvalidTokenError(ValueError):
    """Raised when a JWT access token fails signature/expiry verification."""


def create_access_token(
    *,
    user_id: int,
    role: str,
    scopes: list[str],
    settings: Settings,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "scopes": scopes,
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_and_verify_jwt(token: str, settings: Settings) -> dict[str, Any]:
    """Verify signature + expiry and return the decoded claims.

    Raises `InvalidTokenError` for any failure (expired, bad signature,
    malformed) — callers should treat this uniformly as "not authenticated",
    not branch on the specific PyJWT exception type.
    """
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
