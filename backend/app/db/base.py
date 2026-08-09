"""Declarative base + shared column helpers for all SQLAlchemy models.

`PortableJSON` is the WAJIB pattern for every column documented as `jsonb` in
`Documentation/system-design/06-data-schema.md` /
`07-evaluation-system.md` / `08-authentication-rbac.md`: it becomes a real
`JSONB` column on PostgreSQL (binary, GIN-indexable) and a native `JSON`
column on MySQL, keeping the MySQL migration path open — see
`Documentation/system-design/04-provider-abstractions.md` Bagian C §2.1.
Never import `sqlalchemy.dialects.postgresql.JSONB` directly in a model.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PortableJSON = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class TimestampMixin:
    """Adds a server-side-defaulted `created_at` column."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
