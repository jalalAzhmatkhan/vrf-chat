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

    # ---- Ingestion Stage 1 — native PDF probe (PyMuPDF), see
    # Documentation/system-design/02-ingestion-pipeline.md §3 and
    # app/ingestion/native_probe.py. Starting-point heuristics, calibrated by
    # I1.6 benchmark (backend/docs/i1.6-vram-benchmark-report.md). ----
    NATIVE_PROBE_MIN_TEXT_CHARS: int = 40
    NATIVE_PROBE_VECTOR_DRAWING_THRESHOLD: int = 50
    NATIVE_PROBE_TEXT_DENSITY_MIN: float = 0.0005

    # ---- Ingestion Stage 2 — Docling structural parser, see
    # Documentation/system-design/02-ingestion-pipeline.md §3-4 and
    # app/ingestion/docling_parser.py. Only consumed by the GPU ingestion
    # Celery task (backend-worker-gpu), never validated at API startup — see
    # app/ingestion/docling_parser.py module docstring for rationale. ----
    DOCLING_DEVICE: Literal["cuda", "cpu"] = "cuda"
    DOCLING_ICON_MAX_AREA_RATIO: float = 0.02
    DOCLING_ICON_MAX_DIM_PT: float = 80.0
    # WAJIB true in dev (RTX 3060 6GB) — Docling and PaddleOCR-VL must never
    # be GPU-resident simultaneously. See 02-ingestion-pipeline.md §4.
    DOCLING_UNLOAD_BEFORE_PADDLE_STAGE: bool = True

    # ---- Ingestion Stage 3 — deterministic cascade trigger rules, see
    # Documentation/system-design/02-ingestion-pipeline.md §3 and
    # app/ingestion/cascade_trigger.py. Starting-point values per design doc
    # §3 — WAJIB dikalibrasi ulang via I1.6 benchmark before full-volume
    # ingestion. ----
    THRESHOLD_TABLE: float = 0.75
    THRESHOLD_TEXT: float = 0.6

    # ---- Ingestion Stage 4 — PaddleOCR-VL cascade, see
    # Documentation/system-design/02-ingestion-pipeline.md §4 and
    # app/ingestion/paddleocr_vl_cascade.py. Only consumed by the GPU
    # ingestion Celery task, not validated at API startup (same rationale as
    # DOCLING_* — see docling_parser.py module docstring). ----
    PADDLE_OCR_VL_BACKEND: Literal["local", "remote_api"] = "local"
    PADDLE_OCR_VL_DEVICE: Literal["cuda", "cpu"] = "cuda"
    # WAJIB fp16 in dev (RTX 3060 6GB) — bf16/fp32 only legitimate on a
    # larger cloud GPU. See 02-ingestion-pipeline.md §4.
    PADDLE_OCR_VL_DTYPE: Literal["fp16", "bf16", "fp32"] = "fp16"
    # WAJIB 1 in dev, not raised without explicit benchmark data (I1.6).
    PADDLE_OCR_VL_BATCH_SIZE: int = 1
    PADDLE_OCR_VL_MAX_CONCURRENT_WORKERS: int = 1
    PADDLE_OCR_VL_API_URL: str | None = None
    PADDLE_OCR_VL_API_KEY: str | None = None
    # [I1.10 live finding, confirmed with real 286-page corpus] 60s then
    # 180s (earlier revisions of this default) both proved too short — a
    # subset of real cascade tasks (electrical diagrams/complex tables,
    # under the ~5.96-5.98GB VRAM pressure documented in
    # paddleocr-vl-service/README.md) genuinely take longer than 180s to
    # generate, CONFIRMED via GPU utilization at 100% throughout (real
    # computation, not a hang) and via a live run at 420s/0-retries where
    # the full paddle_cascade stage for all 286 pages completed
    # successfully. 420s is therefore the evidence-based default, not a
    # guess — see docs/i1.10-e2e-findings.md for the full investigation
    # (including the ruled-out hypothesis that this was purely an
    # event-loop-blocking artifact — that WAS a real, separate bug, fixed
    # in paddleocr-vl-service, but did not by itself explain requests this
    # slow).
    PADDLE_OCR_VL_TIMEOUT_SECONDS: int = 420
    # [I1.10 live finding, revised] a bounded retry on timeout/connection
    # error is a better mitigation than an ever-larger timeout alone for
    # genuine transient failures (e.g. a "cold start after idle-unload" —
    # see RemoteAPIPaddleOCRVLClient._post docstring). Lowered from 2 to 1:
    # now that PADDLE_OCR_VL_TIMEOUT_SECONDS=420 already accounts for
    # genuinely-slow-but-working requests (confirmed via 100% GPU
    # utilization throughout, not a hang), a *second* full-length retry on
    # top of that mostly just multiplies worst-case per-element wait time
    # (up to 3x420s=21min with the old default) without addressing a
    # different failure mode than the first retry already covers.
    PADDLE_OCR_VL_MAX_RETRIES: int = 1

    # ---- Vector Store abstraction, see
    # Documentation/system-design/04-provider-abstractions.md Bagian D and
    # app/retrieval/vector_store.py. Default MVP: Qdrant — the only provider
    # actually implemented; chroma/milvus fields are declared (per the
    # design doc's full .env schema) but `build_vector_store_client` raises
    # NotImplementedError for them (documented future work, Bagian D §6). ----
    VECTOR_STORE_PROVIDER: Literal["qdrant", "chroma", "milvus"] = "qdrant"
    VECTOR_STORE_COLLECTION: str = "vrf_chunks"
    # 384 = BAAI/bge-small-en-v1.5 (EMBEDDER_DENSE_MODEL default) output dim.
    VECTOR_STORE_DENSE_DIM: int = 384
    VECTOR_STORE_QDRANT_URL: str = "http://localhost:6333"
    VECTOR_STORE_QDRANT_API_KEY: str | None = None
    VECTOR_STORE_CHROMA_MODE: Literal["embedded", "http"] = "embedded"
    VECTOR_STORE_CHROMA_HOST: str | None = None
    VECTOR_STORE_CHROMA_PORT: int = 8000
    VECTOR_STORE_CHROMA_PERSIST_DIR: str = "./data/chroma"
    VECTOR_STORE_MILVUS_URI: str = "http://localhost:19530"
    VECTOR_STORE_MILVUS_TOKEN: str | None = None

    # ---- Ingestion Stage: chunk embedding (I1.8), see
    # Documentation/system-design/03-retrieval-chunking.md §6. fastembed
    # (ONNX runtime, CPU-friendly — no GPU/VRAM contention with Docling/
    # PaddleOCR-VL) for both dense and sparse (Qdrant-native BM25). ----
    EMBEDDER_DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDER_SPARSE_MODEL: str = "Qdrant/bm25"

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
