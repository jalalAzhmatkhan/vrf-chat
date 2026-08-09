from datetime import UTC, datetime, timedelta

import jwt as pyjwt
import pytest

from app.auth.jwt import InvalidTokenError, create_access_token, decode_and_verify_jwt
from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "JWT_SECRET_KEY": "unit-test-secret",
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_create_and_decode_access_token_roundtrip() -> None:
    settings = _settings()

    token = create_access_token(
        user_id=42, role="admin", scopes=["chat:read", "chat:write"], settings=settings
    )
    payload = decode_and_verify_jwt(token, settings)

    assert payload["sub"] == "42"
    assert payload["role"] == "admin"
    assert payload["scopes"] == ["chat:read", "chat:write"]
    assert "jti" in payload
    assert "exp" in payload


def test_each_token_has_a_unique_jti() -> None:
    settings = _settings()

    token_a = create_access_token(user_id=1, role="user", scopes=[], settings=settings)
    token_b = create_access_token(user_id=1, role="user", scopes=[], settings=settings)

    payload_a = decode_and_verify_jwt(token_a, settings)
    payload_b = decode_and_verify_jwt(token_b, settings)

    assert payload_a["jti"] != payload_b["jti"]


def test_decode_rejects_bad_signature() -> None:
    settings = _settings()
    other_settings = _settings(JWT_SECRET_KEY="a-different-secret")
    token = create_access_token(user_id=1, role="user", scopes=[], settings=settings)

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt(token, other_settings)


def test_decode_rejects_expired_token() -> None:
    settings = _settings()
    now = datetime.now(UTC)
    expired_payload = {
        "sub": "1",
        "role": "user",
        "scopes": [],
        "jti": "test-jti",
        "iat": now - timedelta(minutes=60),
        "exp": now - timedelta(minutes=30),
    }
    token = pyjwt.encode(expired_payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt(token, settings)


def test_decode_rejects_malformed_token() -> None:
    settings = _settings()

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt("not-a-jwt-at-all", settings)


def test_decode_rejects_wrong_algorithm() -> None:
    settings = _settings()
    # Craft a token signed with a completely different algorithm/secret to
    # make sure the PyJWTError branch (not just signature mismatch) is hit.
    token = pyjwt.encode({"sub": "1"}, "some-secret", algorithm="HS512")

    with pytest.raises(InvalidTokenError):
        decode_and_verify_jwt(token, settings)
