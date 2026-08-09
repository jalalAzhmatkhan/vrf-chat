"""Centralized application settings.

All environment variables consumed by the backend MUST be declared here as
fields of :class:`Settings`, and read through the ``get_settings()`` accessor
elsewhere in the codebase. Do not call ``os.getenv()``/``os.environ`` directly
in application code outside this module (and Alembic env scripts, which run
outside the FastAPI app context) — this is what makes provider abstractions
(LLM, object storage, database engine, vector store) genuinely swappable via
``.env`` only, per ``Documentation/system-design/04-provider-abstractions.md``.

Field groups are added incrementally as their respective modules are
implemented (LLM provider abstraction, object storage abstraction, database
engine, auth/RBAC, etc.) — this module (B0.1) only defines the generic
application-level settings needed for the FastAPI scaffold + health check.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---- Generic application settings ----
    APP_NAME: str = "vrf-chat-backend"
    APP_ENV: Literal["development", "test", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    API_V1_PREFIX: str = "/api/v1"

    # CORS — single-tenant app, explicit allow-list required (never wildcard).
    # `NoDecode`: disables pydantic-settings' default JSON-decode-then-validate
    # for complex (list) env values, so a plain comma-separated string from a
    # real `.env`/env var (as opposed to a Python kwarg in tests, which
    # bypasses that source entirely) is handled by `_split_allowed_origins`
    # below instead of failing JSON parsing first — see
    # Documentation/qa-reports/phase-0-qa-report.md F-1.
    ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    # OpenTelemetry-compatible tracing (see app/core/observability.py). Left
    # unset by default -> tracing spans are created but exported nowhere
    # (no-op exporter) until an OTLP collector endpoint is configured.
    OTEL_EXPORTER_OTLP_ENDPOINT: AnyHttpUrl | None = None
    OTEL_SERVICE_NAME: str = "vrf-chat-backend"

    # First superuser bootstrap (consumed by the Alembic data migration that
    # seeds the initial admin account — see app/db/migrations, B0.6). Kept
    # here so the value is validated/typed centrally, even though the
    # migration environment reads it independently (Alembic env.py runs
    # outside the FastAPI settings lifecycle).
    FIRST_SUPERUSER_EMAIL: str | None = None
    FIRST_SUPERUSER_PASSWORD: str | None = None

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _split_allowed_origins(cls, value: object) -> object:
        """Allow ``ALLOWED_ORIGINS`` to be provided as a comma-separated string
        in ``.env`` (e.g. ``ALLOWED_ORIGINS=http://a,http://b``) in addition to
        a native list, since dotenv values are always strings."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("OTEL_EXPORTER_OTLP_ENDPOINT", mode="before")
    @classmethod
    def _blank_otel_endpoint_to_none(cls, value: object) -> object:
        """Treat an empty string (what `.env.example` ships,
        ``OTEL_EXPORTER_OTLP_ENDPOINT=``) as unset, rather than an invalid
        URL — Pydantic's ``AnyHttpUrl`` does not treat ``""`` as ``None`` on
        its own. See Documentation/qa-reports/phase-0-qa-report.md F-1."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` instance.

    Cached via ``lru_cache`` so settings are parsed once per process; tests
    that need to override environment variables should call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
