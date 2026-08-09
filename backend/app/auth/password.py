"""bcrypt password hashing — see
`Documentation/system-design/08-authentication-rbac.md` §2.

Two hard requirements from the design doc, both implemented here:

1. bcrypt verify/hash MUST run off the asyncio event loop (it's a ~1s
   CPU-bound synchronous call at cost factor 14; running it inline would
   block the whole server, including concurrent chat/streaming requests).
2. bcrypt silently truncates input beyond 72 bytes — enforced explicitly
   here (`PasswordTooLongError`) rather than left as a silent footgun.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import bcrypt

MAX_PASSWORD_BYTES = 72


class PasswordTooLongError(ValueError):
    """Raised when a password exceeds bcrypt's 72-byte input limit."""


def _check_length(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise PasswordTooLongError(
            f"Password exceeds bcrypt's {MAX_PASSWORD_BYTES}-byte limit "
            f"({len(encoded)} bytes given)."
        )
    return encoded


def hash_password_sync(password: str, cost_factor: int) -> str:
    """Synchronous bcrypt hash — call via a thread pool, never directly from
    an async request handler. Exposed separately so it's trivially unit
    testable without asyncio machinery."""
    encoded = _check_length(password)
    salt = bcrypt.gensalt(rounds=cost_factor)
    return bcrypt.hashpw(encoded, salt).decode("ascii")


def verify_password_sync(password: str, password_hash: str) -> bool:
    """Synchronous bcrypt verify — call via a thread pool."""
    encoded = _check_length(password)
    return bcrypt.checkpw(encoded, password_hash.encode("ascii"))


async def hash_password(password: str, cost_factor: int) -> str:
    """Thread-pool-offloaded bcrypt hash — use this from request handlers."""
    return await asyncio.to_thread(hash_password_sync, password, cost_factor)


async def verify_password(password: str, password_hash: str) -> bool:
    """Thread-pool-offloaded bcrypt verify — use this from request handlers."""
    return await asyncio.to_thread(verify_password_sync, password, password_hash)


@lru_cache
def get_dummy_hash(cost_factor: int) -> str:
    """A fixed bcrypt hash used to run a decoy `verify_password` call when a
    login's `username` doesn't match any user — without this, a login
    attempt for a nonexistent username would skip bcrypt entirely and
    return faster than one for a real username + wrong password, leaking
    "does this username exist" via response timing (08-authentication-rbac.md
    §2 note on avoiding exactly this)."""
    return hash_password_sync("dummy-password-for-timing-normalization", cost_factor)
