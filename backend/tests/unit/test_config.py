from pathlib import Path

from app.core.config import Settings, get_settings

_ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_default_settings_have_expected_app_name() -> None:
    settings = Settings(_env_file=None)

    assert settings.APP_NAME == "vrf-chat-backend"
    assert settings.APP_ENV == "development"
    assert settings.API_V1_PREFIX == "/api/v1"


def test_allowed_origins_accepts_native_list() -> None:
    settings = Settings(_env_file=None, ALLOWED_ORIGINS=["http://a", "http://b"])

    assert settings.ALLOWED_ORIGINS == ["http://a", "http://b"]


def test_allowed_origins_splits_comma_separated_string() -> None:
    settings = Settings(_env_file=None, ALLOWED_ORIGINS="http://a, http://b")

    assert settings.ALLOWED_ORIGINS == ["http://a", "http://b"]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()


def test_allowed_origins_from_real_os_environ_variable(monkeypatch) -> None:
    """Regression test: pydantic-settings attempts JSON-decoding env values
    for list-typed fields by default, which fails for a plain
    comma-separated string coming from a real OS env var / docker-compose
    `environment:` block (as opposed to a Python kwarg in tests, which
    bypasses that source entirely) unless the field is marked `NoDecode`.
    See Documentation/qa-reports/phase-0-qa-report.md F-1."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173,http://example.com")

    settings = Settings(_env_file=None)

    assert settings.ALLOWED_ORIGINS == ["http://localhost:5173", "http://example.com"]


def test_otel_endpoint_blank_string_from_real_env_var_becomes_none(monkeypatch) -> None:
    """Regression test: `AnyHttpUrl | None` does not treat `""` as `None` on
    its own — `.env.example`/docker-compose commonly ship an empty value for
    optional URL settings. See Documentation/qa-reports/phase-0-qa-report.md
    F-1."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    settings = Settings(_env_file=None)

    assert settings.OTEL_EXPORTER_OTLP_ENDPOINT is None


def test_settings_constructs_from_shipped_env_example() -> None:
    """Regression test for the exact QA finding (F-1): every backend branch
    tested crashed on `Settings()` construction after following the README's
    own `cp .env.example .env` step, because unit tests only ever built
    `Settings` via explicit kwargs/monkeypatched env vars — never by
    actually loading the shipped `.env.example`. This loads the real file
    pydantic-settings-side (`_env_file=...`), the same mechanism
    `docker-compose.yml`'s `env_file:` directive relies on."""
    assert _ENV_EXAMPLE.exists(), f"{_ENV_EXAMPLE} not found"

    settings = Settings(_env_file=_ENV_EXAMPLE)

    assert settings.ALLOWED_ORIGINS == ["http://localhost:5173"]
    assert settings.OTEL_EXPORTER_OTLP_ENDPOINT is None
