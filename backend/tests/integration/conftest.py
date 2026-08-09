"""Shared fixtures for auth integration tests: in-memory SQLite DB (seeded
with roles/scopes/users) + fakeredis, wired into the FastAPI app via
`dependency_overrides` so no real Postgres/Redis is needed.
"""

from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.auth import get_rate_limiter
from app.auth.password import hash_password_sync
from app.auth.rate_limiter import LoginRateLimiter
from app.core.config import Settings
from app.db.base import Base
from app.db.engine import get_db
from app.db.models.auth import Role, RoleScope, Scope, User
from app.main import create_app

TEST_BCRYPT_COST = 4  # bcrypt minimum — fast tests, never use in production

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "AdminPass123!"
USER_EMAIL = "user@example.com"
USER_PASSWORD = "UserPass123!"
INACTIVE_EMAIL = "inactive@example.com"
INACTIVE_PASSWORD = "InactivePass123!"

ALL_SCOPES = [
    "chat:read",
    "chat:write",
    "documents:read",
    "documents:write",
    "admin:rbac:read",
    "admin:rbac:write",
]
USER_SCOPES = ["chat:read", "chat:write", "documents:read"]


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session_factory(db_engine):
    return sessionmaker(bind=db_engine, autoflush=False, autocommit=False)


@pytest.fixture
def seed_data(db_session_factory):
    session = db_session_factory()
    try:
        admin_role = Role(name="admin", description="Full access")
        user_role = Role(name="user", description="Basic")
        session.add_all([admin_role, user_role])
        session.flush()

        scopes = {code: Scope(code=code, description=code, category="test") for code in ALL_SCOPES}
        session.add_all(scopes.values())
        session.flush()

        for scope in scopes.values():
            session.add(RoleScope(role_id=admin_role.id, scope_id=scope.id))
        for code in USER_SCOPES:
            session.add(RoleScope(role_id=user_role.id, scope_id=scopes[code].id))

        session.add_all(
            [
                User(
                    username=ADMIN_EMAIL,
                    password_hash=hash_password_sync(ADMIN_PASSWORD, TEST_BCRYPT_COST),
                    role_id=admin_role.id,
                    is_active=True,
                ),
                User(
                    username=USER_EMAIL,
                    password_hash=hash_password_sync(USER_PASSWORD, TEST_BCRYPT_COST),
                    role_id=user_role.id,
                    is_active=True,
                ),
                User(
                    username=INACTIVE_EMAIL,
                    password_hash=hash_password_sync(INACTIVE_PASSWORD, TEST_BCRYPT_COST),
                    role_id=user_role.id,
                    is_active=False,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()


@pytest.fixture
def fake_redis():
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def test_settings():
    return Settings(
        _env_file=None,
        LOGIN_RATE_LIMIT_MAX_ATTEMPTS=3,
        LOGIN_RATE_LIMIT_WINDOW_SECONDS=60,
        BCRYPT_COST_FACTOR=TEST_BCRYPT_COST,
    )


@pytest.fixture
def auth_client(db_session_factory, seed_data, fake_redis, test_settings):
    app = create_app(test_settings)

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    def override_get_rate_limiter() -> LoginRateLimiter:
        return LoginRateLimiter(fake_redis, test_settings)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rate_limiter] = override_get_rate_limiter

    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client
