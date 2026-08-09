from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.engine import (
    build_database_url,
    build_engine,
    get_db,
    get_engine,
    get_session_factory,
)


def test_build_database_url_postgresql() -> None:
    settings = Settings(
        _env_file=None,
        DB_ENGINE="postgresql",
        DB_HOST="dbhost",
        DB_PORT=5432,
        DB_NAME="vrf_chatbot",
        DB_USER="vrf_app",
        DB_PASSWORD="secret",
    )

    url = build_database_url(settings)

    assert url == "postgresql+psycopg://vrf_app:secret@dbhost:5432/vrf_chatbot"


def test_build_database_url_mysql() -> None:
    settings = Settings(
        _env_file=None,
        DB_ENGINE="mysql",
        DB_HOST="dbhost",
        DB_PORT=3306,
        DB_NAME="vrf_chatbot",
        DB_USER="vrf_app",
        DB_PASSWORD="secret",
    )

    url = build_database_url(settings)

    assert url == "mysql+pymysql://vrf_app:secret@dbhost:3306/vrf_chatbot"


def test_build_engine_returns_engine_with_configured_pool() -> None:
    settings = Settings(
        _env_file=None,
        DB_ENGINE="postgresql",
        DB_HOST="dbhost",
        DB_PORT=5432,
        DB_NAME="vrf_chatbot",
        DB_USER="vrf_app",
        DB_PASSWORD="secret",
        DB_POOL_SIZE=7,
        DB_MAX_OVERFLOW=3,
    )

    engine = build_engine(settings)

    assert isinstance(engine, Engine)
    assert engine.pool.size() == 7
    engine.dispose()


def test_get_engine_and_session_factory_are_cached() -> None:
    get_engine.cache_clear()
    get_session_factory.cache_clear()

    first_engine = get_engine()
    second_engine = get_engine()
    first_factory = get_session_factory()
    second_factory = get_session_factory()

    assert first_engine is second_engine
    assert isinstance(first_factory, sessionmaker)
    assert first_factory is second_factory

    get_engine.cache_clear()
    get_session_factory.cache_clear()


def test_get_db_yields_and_closes_session(monkeypatch) -> None:
    get_session_factory.cache_clear()

    closed = {"value": False}

    class DummySession:
        def close(self) -> None:
            closed["value"] = True

    class DummyFactory:
        def __call__(self) -> Session:
            return DummySession()  # type: ignore[return-value]

    monkeypatch.setattr("app.db.engine.get_session_factory", lambda: DummyFactory())

    generator = get_db()
    session = next(generator)
    assert isinstance(session, DummySession)

    generator_exhausted = False
    try:
        next(generator)
    except StopIteration:
        generator_exhausted = True

    assert generator_exhausted
    assert closed["value"] is True
