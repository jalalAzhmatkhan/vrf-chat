"""`POST /auth/login`, `/auth/refresh`, `/auth/logout`, `GET /auth/me` — see
`Documentation/system-design/08-authentication-rbac.md` §6/§8.5.

Refresh token is transported *only* via an `httpOnly; Secure; SameSite=Strict`
cookie set by this router — never in a JSON response body, and `/auth/refresh`
+ `/auth/logout` never accept it from the request body/header either.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.jwt import create_access_token
from app.auth.password import get_dummy_hash, verify_password
from app.auth.rate_limiter import LoginRateLimiter, get_redis_client
from app.auth.refresh_tokens import (
    InvalidRefreshTokenError,
    hash_token,
    issue_refresh_token,
    revoke_token_family,
    rotate_refresh_token,
)
from app.auth.schemas import (
    AuthenticatedUser,
    LoginRequest,
    LogoutResponse,
    MeResponse,
    TokenResponse,
)
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models.auth import RefreshToken, Role, User

router = APIRouter(prefix="/auth", tags=["auth"])


def get_rate_limiter(settings: Settings = Depends(get_settings)) -> LoginRateLimiter:
    return LoginRateLimiter(get_redis_client(settings), settings)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_refresh_cookie(response: Response, raw_token: str, settings: Settings) -> None:
    response.set_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=True,
        samesite="strict",
        path=settings.REFRESH_TOKEN_COOKIE_PATH,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )


def _clear_refresh_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        path=settings.REFRESH_TOKEN_COOKIE_PATH,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    rate_limiter: LoginRateLimiter = Depends(get_rate_limiter),
) -> TokenResponse | Response:
    ip = _client_ip(request)

    retry_after = rate_limiter.check_blocked(body.username, ip)
    if retry_after is not None:
        # Built as a raw JSONResponse (not HTTPException) so the body is a
        # FLAT {"detail": ..., "retry_after_seconds": ...} object per
        # 08-authentication-rbac.md §5.1 — FastAPI's default HTTPException
        # handler would otherwise nest a dict `detail=` under another
        # "detail" key.
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Too many login attempts, try again later",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    user = db.execute(select(User).where(User.username == body.username)).scalar_one_or_none()

    if user is not None and user.is_active:
        password_ok = await verify_password(body.password, user.password_hash)
    else:
        # Run a decoy bcrypt verify even when the user doesn't exist/is
        # inactive, so response timing doesn't leak "does this username
        # exist" (08-authentication-rbac.md §2).
        await verify_password(body.password, get_dummy_hash(settings.BCRYPT_COST_FACTOR))
        password_ok = False

    if not password_ok:
        rate_limiter.record_failed_attempt(body.username, ip)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    assert user is not None  # narrows type for mypy — guaranteed by password_ok branch above
    rate_limiter.reset(body.username, ip)

    role = db.get(Role, user.role_id)
    scopes = [rs.scope.code for rs in role.role_scopes] if role else []

    access_token = create_access_token(
        user_id=user.id, role=role.name if role else "", scopes=scopes, settings=settings
    )
    _, raw_refresh_token = issue_refresh_token(
        db, user, settings, user_agent=request.headers.get("user-agent"), ip_address=ip
    )
    _set_refresh_cookie(response, raw_refresh_token, settings)

    user.last_login_at = datetime.now(UTC)
    db.add(user)
    db.commit()

    return TokenResponse(
        access_token=access_token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    raw_token = request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if not raw_token:
        raise HTTPException(status_code=401, detail="Refresh token invalid, expired, or revoked")

    try:
        new_row, new_raw_token = rotate_refresh_token(
            db,
            raw_token,
            settings,
            user_agent=request.headers.get("user-agent"),
            ip_address=_client_ip(request),
        )
    except InvalidRefreshTokenError as exc:
        _clear_refresh_cookie(response, settings)
        raise HTTPException(
            status_code=401, detail="Refresh token invalid, expired, or revoked"
        ) from exc

    user = db.get(User, new_row.user_id)
    if user is None or not user.is_active:  # pragma: no cover
        # Defensive/unreachable in practice: rotate_refresh_token() above
        # already looks up the user and raises InvalidRefreshTokenError
        # (caught above) if they're missing/inactive, so `new_row` is only
        # ever returned for an active user. Kept as defense-in-depth in
        # case that invariant ever changes.
        raise HTTPException(status_code=401, detail="Refresh token invalid, expired, or revoked")

    role = db.get(Role, user.role_id)
    scopes = [rs.scope.code for rs in role.role_scopes] if role else []

    access_token = create_access_token(
        user_id=user.id, role=role.name if role else "", scopes=scopes, settings=settings
    )
    _set_refresh_cookie(response, new_raw_token, settings)

    return TokenResponse(
        access_token=access_token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> LogoutResponse:
    raw_token = request.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    if raw_token:
        row = db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw_token))
        ).scalar_one_or_none()
        if row is not None:
            revoke_token_family(db, row.token_family_id)

    _clear_refresh_cookie(response, settings)
    return LogoutResponse()


@router.get("/me", response_model=MeResponse)
async def me(
    user: AuthenticatedUser = Security(get_current_user, scopes=["chat:read"]),
    db: Session = Depends(get_db),
) -> MeResponse:
    db_user = db.get(User, user.id)
    if db_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return MeResponse(id=db_user.id, username=db_user.username, role=user.role, scopes=user.scopes)
