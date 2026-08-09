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

    # ---- Database engine abstraction (Postgres <-> MySQL), see
    # Documentation/system-design/04-provider-abstractions.md Bagian C ----
    DB_ENGINE: Literal["postgresql", "mysql"] = "postgresql"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "vrf_chatbot"
    DB_USER: str = "vrf_app"
    DB_PASSWORD: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 5
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_PRE_PING: bool = True
    DB_SSL_MODE: str = "prefer"

    # ---- LLM provider abstraction, see
    # Documentation/system-design/04-provider-abstractions.md Bagian A.
    # Two structurally-identical slots (chat model vs eval judge model) so
    # LLM-as-judge can use a different provider than the chat model. ----
    CHAT_LLM_PROVIDER: Literal["anthropic", "google", "openai", "local"] = "anthropic"
    CHAT_LLM_MODEL: str = "claude-sonnet-4-5-20250929"
    CHAT_LLM_API_KEY: str | None = None
    CHAT_LLM_BASE_URL: str | None = None
    CHAT_LLM_TEMPERATURE: float = 0.1
    CHAT_LLM_MAX_TOKENS: int = 4096
    CHAT_LLM_TIMEOUT_SECONDS: int = 25

    EVAL_JUDGE_LLM_PROVIDER: Literal["anthropic", "google", "openai", "local"] = "google"
    EVAL_JUDGE_LLM_MODEL: str = "gemini-2.5-pro"
    EVAL_JUDGE_LLM_API_KEY: str | None = None
    EVAL_JUDGE_LLM_BASE_URL: str | None = None
    EVAL_JUDGE_LLM_TEMPERATURE: float = 0.0
    EVAL_JUDGE_LLM_MAX_TOKENS: int = 4096
    EVAL_JUDGE_LLM_TIMEOUT_SECONDS: int = 60

    # ---- Object storage abstraction, see
    # Documentation/system-design/04-provider-abstractions.md Bagian B. ----
    OBJECT_STORAGE_BACKEND: Literal["minio", "s3", "local"] = "minio"
    OBJECT_STORAGE_BUCKET: str = "vrf-manuals"
    OBJECT_STORAGE_REGION: str = "us-east-1"
    OBJECT_STORAGE_ACCESS_KEY: str | None = None
    OBJECT_STORAGE_SECRET_KEY: str | None = None
    OBJECT_STORAGE_ENDPOINT_URL: str | None = None
    OBJECT_STORAGE_USE_SSL: bool = False
    OBJECT_STORAGE_FORCE_PATH_STYLE: bool = True
    OBJECT_STORAGE_LOCAL_BASE_PATH: str = "./data/object-storage"
    OBJECT_STORAGE_PRESIGNED_URL_EXPIRY_SECONDS: int = 3600
    # HMAC secret used to sign the emulated "presigned" tokens issued by the
    # local filesystem adapter (see app/storage/local_adapter.py). Only
    # relevant when OBJECT_STORAGE_BACKEND=local.
    OBJECT_STORAGE_LOCAL_TOKEN_SECRET: str = "dev-only-insecure-local-storage-token-secret"

    # ---- Redis (Celery broker/cache + /auth/login rate limiter), see
    # Documentation/system-design/01-architecture-overview.md §2/§6 and
    # Documentation/system-design/08-authentication-rbac.md §5 ----
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    # ---- Celery ----
    CELERY_TASK_DEFAULT_QUEUE: str = "default"
    CELERY_GPU_QUEUE: str = "gpu"

    # ---- Authentication & RBAC, see
    # Documentation/system-design/08-authentication-rbac.md ----
    JWT_SECRET_KEY: str = "dev-only-insecure-jwt-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    REFRESH_TOKEN_COOKIE_NAME: str = "refresh_token"
    REFRESH_TOKEN_COOKIE_PATH: str = "/api/v1/auth"
    # bcrypt cost factor — see 08-authentication-rbac.md §2 (14 evaluated as
    # appropriate for this project's scale, MUST be offloaded to a thread
    # pool since it's a ~1s CPU-bound synchronous call, see app/auth/password.py)
    BCRYPT_COST_FACTOR: int = 14
    # /auth/login rate limiting — see 08-authentication-rbac.md §5.1
    LOGIN_RATE_LIMIT_MAX_ATTEMPTS: int = 5
    LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = 900

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

    @property
    def redis_url(self) -> str:
        """Build the Redis connection URL used by both Celery and the
        `/auth/login` rate limiter (see 08-authentication-rbac.md §5)."""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` instance.

    Cached via ``lru_cache`` so settings are parsed once per process; tests
    that need to override environment variables should call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
