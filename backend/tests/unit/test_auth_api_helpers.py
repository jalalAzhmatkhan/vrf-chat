from unittest.mock import MagicMock

from app.api.v1.auth import _client_ip, get_rate_limiter
from app.auth.rate_limiter import LoginRateLimiter
from app.core.config import Settings


def test_get_rate_limiter_builds_a_login_rate_limiter() -> None:
    settings = Settings(_env_file=None)

    limiter = get_rate_limiter(settings)

    assert isinstance(limiter, LoginRateLimiter)


def test_client_ip_returns_unknown_when_no_client_info() -> None:
    request = MagicMock()
    request.client = None

    assert _client_ip(request) == "unknown"


def test_client_ip_returns_host_when_client_present() -> None:
    request = MagicMock()
    request.client.host = "203.0.113.5"

    assert _client_ip(request) == "203.0.113.5"
