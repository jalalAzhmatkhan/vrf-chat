import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes

from app.auth.jwt import create_access_token
from app.auth.security import get_current_user
from app.core.config import Settings


def _settings() -> Settings:
    return Settings(_env_file=None, JWT_SECRET_KEY="unit-test-secret")


async def test_get_current_user_valid_token_no_required_scopes() -> None:
    settings = _settings()
    token = create_access_token(user_id=7, role="user", scopes=["chat:read"], settings=settings)

    user = await get_current_user(SecurityScopes([]), token, settings)

    assert user.id == 7
    assert user.role == "user"
    assert user.scopes == ["chat:read"]


async def test_get_current_user_valid_token_with_satisfied_scope() -> None:
    settings = _settings()
    token = create_access_token(
        user_id=1, role="admin", scopes=["admin:rbac:read", "chat:read"], settings=settings
    )

    user = await get_current_user(SecurityScopes(["admin:rbac:read"]), token, settings)

    assert user.id == 1


async def test_get_current_user_missing_token_raises_401() -> None:
    settings = _settings()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(SecurityScopes([]), None, settings)

    assert exc_info.value.status_code == 401


async def test_get_current_user_invalid_token_raises_401() -> None:
    settings = _settings()

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(SecurityScopes([]), "not-a-real-jwt", settings)

    assert exc_info.value.status_code == 401


async def test_get_current_user_missing_scope_raises_403() -> None:
    """F-8 (Documentation/qa-reports/phase-0-qa-report.md): a valid token
    that merely lacks a required scope is a stage-2 AUTHORIZATION failure
    (403), not a stage-1 AUTHENTICATION failure (401)."""
    settings = _settings()
    token = create_access_token(user_id=1, role="user", scopes=["chat:read"], settings=settings)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(SecurityScopes(["admin:rbac:write"]), token, settings)

    assert exc_info.value.status_code == 403
    assert "admin:rbac:write" in exc_info.value.detail
    assert exc_info.value.headers is not None
    assert 'error="insufficient_scope"' in exc_info.value.headers["WWW-Authenticate"]
