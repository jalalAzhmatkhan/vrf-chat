import pytest

from app.auth.password import (
    MAX_PASSWORD_BYTES,
    PasswordTooLongError,
    hash_password,
    hash_password_sync,
    verify_password,
    verify_password_sync,
)

COST = 4  # bcrypt minimum, fast for tests


def test_hash_and_verify_sync_roundtrip() -> None:
    hashed = hash_password_sync("correct horse battery staple", COST)

    assert verify_password_sync("correct horse battery staple", hashed) is True
    assert verify_password_sync("wrong password", hashed) is False


async def test_hash_and_verify_async_roundtrip() -> None:
    hashed = await hash_password("correct horse battery staple", COST)

    assert await verify_password("correct horse battery staple", hashed) is True
    assert await verify_password("wrong password", hashed) is False


def test_hash_password_rejects_over_72_bytes() -> None:
    too_long = "x" * (MAX_PASSWORD_BYTES + 1)

    with pytest.raises(PasswordTooLongError):
        hash_password_sync(too_long, COST)


def test_verify_password_rejects_over_72_bytes() -> None:
    too_long = "x" * (MAX_PASSWORD_BYTES + 1)

    with pytest.raises(PasswordTooLongError):
        verify_password_sync(too_long, "irrelevant-hash")


def test_hash_password_accepts_exactly_72_bytes() -> None:
    exactly_max = "x" * MAX_PASSWORD_BYTES

    hashed = hash_password_sync(exactly_max, COST)

    assert verify_password_sync(exactly_max, hashed) is True
