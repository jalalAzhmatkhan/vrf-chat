"""Refresh token issuance, rotation (single-use) and reuse detection — see
`Documentation/system-design/08-authentication-rbac.md` §8.2/§8.3.

Refresh tokens are opaque (not JWTs): a random value is returned to the
caller once, only its SHA-256 hash is ever persisted (`token_hash`) — this
mirrors the design doc's explicit "hash from a token, not password-derived"
requirement (no need for bcrypt's slow KDF here, this is a high-entropy
random token, not a user-chosen password).
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models.auth import RefreshToken, User

_TOKEN_BYTES = 48


class InvalidRefreshTokenError(ValueError):
    """Raised when a refresh token is missing, expired, or otherwise unusable."""


class RefreshTokenReuseDetectedError(InvalidRefreshTokenError):
    """Raised when an already-rotated (revoked) refresh token is presented
    again — signals likely token theft; caller must ensure the whole
    `token_family_id` has been revoked (this module does that before
    raising, see `rotate_refresh_token`)."""


def _generate_raw_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _ensure_aware_utc(value: datetime) -> datetime:
    """Normalize a possibly-naive `datetime` (e.g. read back from SQLite,
    which drops tzinfo even for `DateTime(timezone=True)` columns) to
    timezone-aware UTC, so it can be safely compared against
    `datetime.now(UTC)`. PostgreSQL/MySQL both return aware datetimes for
    `DateTime(timezone=True)` columns, so this is a no-op there."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def issue_refresh_token(
    db: Session,
    user: User,
    settings: Settings,
    *,
    token_family_id: str | None = None,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> tuple[RefreshToken, str]:
    """Issue a brand-new refresh token. `token_family_id` is left `None` for
    a fresh login (a new family is started); pass the existing family id
    when issuing as part of a rotation (see `rotate_refresh_token`)."""
    raw_token = _generate_raw_token()
    now = datetime.now(UTC)
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_token),
        token_family_id=token_family_id or str(uuid.uuid4()),
        issued_at=now,
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        user_agent=user_agent,
        ip_address=ip_address,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, raw_token


def revoke_token_family(db: Session, token_family_id: str) -> None:
    """Revoke every non-revoked token in a family — used on reuse detection
    and on logout."""
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.token_family_id == token_family_id)
        .where(RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    db.commit()


def revoke_all_user_tokens(db: Session, user_id: int) -> None:
    """Revoke every non-revoked refresh token for a user — used when an
    admin deactivates a user (see 08-authentication-rbac.md §6, B0.7)."""
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id)
        .where(RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    db.commit()


def rotate_refresh_token(
    db: Session,
    raw_token: str,
    settings: Settings,
    *,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> tuple[RefreshToken, str]:
    """Validate `raw_token`, revoke it, and issue+return a new one in the
    same family (rotation). Raises `RefreshTokenReuseDetectedError` (a
    subclass of `InvalidRefreshTokenError`) if `raw_token` was already
    revoked — the entire family is revoked before raising."""
    row = (
        db.query(RefreshToken)
        .filter(RefreshToken.token_hash == hash_token(raw_token))
        .one_or_none()
    )

    if row is None:
        raise InvalidRefreshTokenError("Refresh token not found.")

    if row.revoked_at is not None:
        revoke_token_family(db, row.token_family_id)
        raise RefreshTokenReuseDetectedError(
            "Refresh token reuse detected — entire token family revoked."
        )

    if _ensure_aware_utc(row.expires_at) < datetime.now(UTC):
        raise InvalidRefreshTokenError("Refresh token expired.")

    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        raise InvalidRefreshTokenError("User not found or inactive.")

    new_row, new_raw_token = issue_refresh_token(
        db,
        user,
        settings,
        token_family_id=row.token_family_id,
        user_agent=user_agent,
        ip_address=ip_address,
    )

    row.revoked_at = datetime.now(UTC)
    row.replaced_by_token_id = new_row.id
    db.add(row)
    db.commit()

    return new_row, new_raw_token
