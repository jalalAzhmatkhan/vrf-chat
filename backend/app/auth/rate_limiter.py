"""Redis-backed `/auth/login` rate limiter — see
`Documentation/system-design/08-authentication-rbac.md` §5/§5.1.

Two-phase design so a blocked caller's request never reaches bcrypt verify
(the whole point is to prevent CPU-exhaustion DoS against bcrypt cost 14,
§2):

1. `check_blocked` — read-only, called *before* attempting authentication.
2. `record_failed_attempt` — called only after a failed authentication,
   increments the counter (`EXPIRE` set only on the first increment in the
   window, per §5.1's sliding-window-lite description).

`retry_after_seconds` is read directly from Redis `TTL` (§5.1) — if the key
has no TTL (edge case: `-2` "key expired mid-check"), fall back to the full
window instead of a negative/zero value.
"""

from __future__ import annotations

import redis

from app.core.config import Settings


def get_redis_client(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


class LoginRateLimiter:
    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._max_attempts = settings.LOGIN_RATE_LIMIT_MAX_ATTEMPTS
        self._window_seconds = settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS

    @staticmethod
    def _key(username: str, ip: str) -> str:
        return f"login_attempts:{username}:{ip}"

    def check_blocked(self, username: str, ip: str) -> int | None:
        """Return `retry_after_seconds` if blocked, else `None`."""
        raw = self._redis.get(self._key(username, ip))
        if raw is None or int(raw) < self._max_attempts:
            return None

        ttl = self._redis.ttl(self._key(username, ip))
        if ttl is None or ttl < 0:
            ttl = self._window_seconds
        return int(ttl)

    def record_failed_attempt(self, username: str, ip: str) -> int:
        """Increment the failed-attempt counter, return the new count."""
        key = self._key(username, ip)
        attempts = self._redis.incr(key)
        if attempts == 1:
            self._redis.expire(key, self._window_seconds)
        return int(attempts)

    def reset(self, username: str, ip: str) -> None:
        """Clear the counter — called after a successful login."""
        self._redis.delete(self._key(username, ip))
