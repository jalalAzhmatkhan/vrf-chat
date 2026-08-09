"""Database engine / session factory.

Builds a SQLAlchemy `Engine` + session factory from `Settings.DB_*` fields —
see `Documentation/system-design/04-provider-abstractions.md` Bagian C. This
is the *only* place a database connection string is assembled; application
code must never construct dialect-specific connection strings itself, which
is what keeps `DB_ENGINE=postgresql|mysql` genuinely swappable via `.env`.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings

_DRIVER_BY_ENGINE = {
    "postgresql": "postgresql+psycopg",
    "mysql": "mysql+pymysql",
}


def build_database_url(settings: Settings) -> str:
    """Assemble the SQLAlchemy connection URL for the configured DB engine."""
    driver = _DRIVER_BY_ENGINE[settings.DB_ENGINE]
    return (
        f"{driver}://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )


def build_engine(settings: Settings) -> Engine:
    """Build a pooled SQLAlchemy `Engine` from settings.

    Pool parameters are entirely configurable via env (`DB_POOL_*`) — see
    04-provider-abstractions.md Bagian C §3 for rationale/defaults.
    """
    return create_engine(
        build_database_url(settings),
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
    )


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide cached `Engine`."""
    return build_engine(get_settings())


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide cached session factory."""
    return sessionmaker(
        bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped `Session`."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
