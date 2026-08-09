"""`SecurityScopes`/`get_current_user` dependency — see
`Documentation/system-design/08-authentication-rbac.md` §3.1.

Stateless by design: scopes are validated purely against the JWT's `scopes`
claim, no DB round-trip per request (the trade-off — role/scope changes
apply only after the access token is refreshed/expires — is accepted
explicitly in the design doc §1).

Two-stage error semantics (revised 2026-08-09 responding to QA finding F-8,
`Documentation/qa-reports/phase-0-qa-report.md`):
  - Stage 1 (AUTHENTICATION — missing/malformed/expired/bad-signature token)
    -> 401 Unauthorized, unchanged.
  - Stage 2 (AUTHORIZATION — token is valid, but lacks a required scope)
    -> 403 Forbidden (was 401 previously), with
    ``WWW-Authenticate: Bearer scope="...", error="insufficient_scope"``
    per RFC 6750 §3.1.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

from app.auth.jwt import InvalidTokenError, decode_and_verify_jwt
from app.auth.schemas import AuthenticatedUser
from app.core.config import Settings, get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": f'Bearer scope="{security_scopes.scope_str}"'},
    )

    if token is None:
        raise auth_error

    try:
        payload = decode_and_verify_jwt(token, settings)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": f'Bearer scope="{security_scopes.scope_str}"'},
        ) from exc

    token_scopes: list[str] = payload.get("scopes", [])

    # Stage 2 — AUTHORIZATION: identity is already known/valid at this
    # point, it just lacks a required scope -> 403, not 401 (F-8).
    missing = [s for s in security_scopes.scopes if s not in token_scopes]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required scope(s): {missing}",
            headers={
                "WWW-Authenticate": (
                    f'Bearer scope="{security_scopes.scope_str}", '
                    f'error="insufficient_scope"'
                )
            },
        )

    return AuthenticatedUser(id=int(payload["sub"]), role=payload["role"], scopes=token_scopes)
