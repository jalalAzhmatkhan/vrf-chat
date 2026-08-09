from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.auth.refresh_tokens import (
    InvalidRefreshTokenError,
    RefreshTokenReuseDetectedError,
    _ensure_aware_utc,
    hash_token,
    issue_refresh_token,
    revoke_all_user_tokens,
    revoke_token_family,
    rotate_refresh_token,
)
from app.core.config import Settings
from app.db.base import Base
from app.db.models.auth import RefreshToken, Role, User


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, REFRESH_TOKEN_EXPIRE_DAYS=7)


@pytest.fixture
def user(db: Session) -> User:
    role = Role(name="user", description="basic")
    db.add(role)
    db.flush()
    u = User(username="a@example.com", password_hash="hash", role_id=role.id, is_active=True)
    db.add(u)
    db.commit()
    return u


def test_issue_refresh_token_creates_row_with_new_family(
    db: Session, settings: Settings, user: User
) -> None:
    row, raw_token = issue_refresh_token(
        db, user, settings, user_agent="pytest", ip_address="127.0.0.1"
    )

    assert row.user_id == user.id
    assert row.token_hash == hash_token(raw_token)
    assert row.token_family_id
    assert row.revoked_at is None


def test_issue_refresh_token_reuses_given_family(
    db: Session, settings: Settings, user: User
) -> None:
    row, _ = issue_refresh_token(db, user, settings, token_family_id="family-123")

    assert row.token_family_id == "family-123"


def test_rotate_refresh_token_revokes_old_and_issues_new_in_same_family(
    db: Session, settings: Settings, user: User
) -> None:
    old_row, raw_token = issue_refresh_token(db, user, settings)

    new_row, new_raw_token = rotate_refresh_token(db, raw_token, settings)

    db.refresh(old_row)
    assert old_row.revoked_at is not None
    assert old_row.replaced_by_token_id == new_row.id
    assert new_row.token_family_id == old_row.token_family_id
    assert new_raw_token != raw_token


def test_rotate_refresh_token_unknown_token_raises(db: Session, settings: Settings) -> None:
    with pytest.raises(InvalidRefreshTokenError):
        rotate_refresh_token(db, "does-not-exist", settings)


def test_rotate_refresh_token_reuse_detection_revokes_whole_family(
    db: Session, settings: Settings, user: User
) -> None:
    old_row, raw_token = issue_refresh_token(db, user, settings)
    # First rotation succeeds — old_row is now revoked.
    new_row, new_raw_token = rotate_refresh_token(db, raw_token, settings)

    # Reusing the already-revoked old token should be detected as theft.
    with pytest.raises(RefreshTokenReuseDetectedError):
        rotate_refresh_token(db, raw_token, settings)

    db.refresh(new_row)
    assert new_row.revoked_at is not None  # entire family revoked, including the newer token


def test_rotate_refresh_token_expired_raises(db: Session, settings: Settings, user: User) -> None:
    row, raw_token = issue_refresh_token(db, user, settings)
    row.expires_at = datetime.now(UTC) - timedelta(days=1)
    db.add(row)
    db.commit()

    with pytest.raises(InvalidRefreshTokenError, match="expired"):
        rotate_refresh_token(db, raw_token, settings)


def test_rotate_refresh_token_naive_expires_at_is_handled(
    db: Session, settings: Settings, user: User
) -> None:
    """SQLite doesn't preserve tzinfo — simulate a naive `expires_at` read
    back from the DB and ensure comparison against timezone-aware `now()`
    doesn't raise `TypeError`."""
    row, raw_token = issue_refresh_token(db, user, settings)
    row.expires_at = (datetime.now(UTC) + timedelta(days=7)).replace(tzinfo=None)
    db.add(row)
    db.commit()

    new_row, _ = rotate_refresh_token(db, raw_token, settings)
    assert new_row is not None


def test_rotate_refresh_token_inactive_user_raises(
    db: Session, settings: Settings, user: User
) -> None:
    _, raw_token = issue_refresh_token(db, user, settings)
    user.is_active = False
    db.add(user)
    db.commit()

    with pytest.raises(InvalidRefreshTokenError, match="inactive"):
        rotate_refresh_token(db, raw_token, settings)


def test_revoke_token_family(db: Session, settings: Settings, user: User) -> None:
    row, _ = issue_refresh_token(db, user, settings)

    revoke_token_family(db, row.token_family_id)

    db.refresh(row)
    assert row.revoked_at is not None


def test_revoke_all_user_tokens(db: Session, settings: Settings, user: User) -> None:
    row_a, _ = issue_refresh_token(db, user, settings, token_family_id="fam-a")
    row_b, _ = issue_refresh_token(db, user, settings, token_family_id="fam-b")

    revoke_all_user_tokens(db, user.id)

    db.refresh(row_a)
    db.refresh(row_b)
    assert row_a.revoked_at is not None
    assert row_b.revoked_at is not None


def test_refresh_token_model_repr_smoke(db: Session, user: User) -> None:
    # Just exercises the ORM mapping end-to-end (already covered above),
    # kept as an explicit smoke test for the RefreshToken model import.
    assert RefreshToken.__tablename__ == "refresh_tokens"


def test_ensure_aware_utc_converts_naive_datetime() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)

    result = _ensure_aware_utc(naive)

    assert result.tzinfo == UTC


def test_ensure_aware_utc_leaves_aware_datetime_unchanged() -> None:
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    result = _ensure_aware_utc(aware)

    assert result is aware
