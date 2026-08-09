import fakeredis
import pytest

from app.auth.rate_limiter import LoginRateLimiter, get_redis_client
from app.core.config import Settings


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, LOGIN_RATE_LIMIT_MAX_ATTEMPTS=3, LOGIN_RATE_LIMIT_WINDOW_SECONDS=60
    )


@pytest.fixture
def limiter(redis_client, settings) -> LoginRateLimiter:
    return LoginRateLimiter(redis_client, settings)


def test_not_blocked_when_no_attempts_recorded(limiter: LoginRateLimiter) -> None:
    assert limiter.check_blocked("alice", "1.2.3.4") is None


def test_not_blocked_below_max_attempts(limiter: LoginRateLimiter) -> None:
    limiter.record_failed_attempt("alice", "1.2.3.4")
    limiter.record_failed_attempt("alice", "1.2.3.4")

    assert limiter.check_blocked("alice", "1.2.3.4") is None


def test_blocked_at_max_attempts(limiter: LoginRateLimiter) -> None:
    for _ in range(3):
        limiter.record_failed_attempt("alice", "1.2.3.4")

    retry_after = limiter.check_blocked("alice", "1.2.3.4")

    assert retry_after is not None
    assert 0 < retry_after <= 60


def test_reset_clears_block(limiter: LoginRateLimiter) -> None:
    for _ in range(3):
        limiter.record_failed_attempt("alice", "1.2.3.4")
    assert limiter.check_blocked("alice", "1.2.3.4") is not None

    limiter.reset("alice", "1.2.3.4")

    assert limiter.check_blocked("alice", "1.2.3.4") is None


def test_attempts_scoped_per_username_and_ip(limiter: LoginRateLimiter) -> None:
    for _ in range(3):
        limiter.record_failed_attempt("alice", "1.2.3.4")

    assert limiter.check_blocked("alice", "5.6.7.8") is None
    assert limiter.check_blocked("bob", "1.2.3.4") is None


def test_check_blocked_falls_back_to_window_when_ttl_missing(
    limiter: LoginRateLimiter, redis_client
) -> None:
    """Simulates the documented edge case (08-authentication-rbac.md §5.1):
    TTL races to -2 (key just expired) — fall back to the full window
    instead of a negative/zero value."""
    key = "login_attempts:alice:1.2.3.4"
    redis_client.set(key, 5)  # no EXPIRE set -> TTL is -1 ("no expiry")

    retry_after = limiter.check_blocked("alice", "1.2.3.4")

    assert retry_after == 60


def test_get_redis_client_builds_from_settings(settings: Settings) -> None:
    client = get_redis_client(settings)

    assert client is not None
