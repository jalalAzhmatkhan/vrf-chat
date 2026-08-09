"""`SecurityScopes`/`get_current_user` dependency — see
`Documentation/system-design/08-authentication-rbac.md` §3.1.

Stateless by design: scopes are validated purely against the JWT's `scopes`
claim, no DB round-trip per request (the trade-off — role/scope changes
apply only after the access token is refreshed/expires — is accepted
explicitly in the design doc §1).
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

    missing = [s for s in security_scopes.scopes if s not in token_scopes]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required scope(s): {missing}",
            headers={"WWW-Authenticate": f'Bearer scope="{security_scopes.scope_str}"'},
        )

    return AuthenticatedUser(id=int(payload["sub"]), role=payload["role"], scopes=token_scopes)
