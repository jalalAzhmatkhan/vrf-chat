"""`roles`, `scopes`, `role_scopes`, `users`, `refresh_tokens` — see
`Documentation/system-design/08-authentication-rbac.md` §4.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    role_scopes: Mapped[list[RoleScope]] = relationship(
        back_populates="role", cascade="all, delete-orphan"
    )
    users: Mapped[list[User]] = relationship(back_populates="role")


class Scope(Base):
    __tablename__ = "scopes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    role_scopes: Mapped[list[RoleScope]] = relationship(
        back_populates="scope", cascade="all, delete-orphan"
    )


class RoleScope(Base):
    __tablename__ = "role_scopes"

    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    scope_id: Mapped[int] = mapped_column(
        ForeignKey("scopes.id", ondelete="CASCADE"), primary_key=True
    )
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    granted_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    role: Mapped[Role] = relationship(back_populates="role_scopes")
    scope: Mapped[Scope] = relationship(back_populates="role_scopes")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Login identifier. Named `username` per the schema in
    # 08-authentication-rbac.md §4, but this project logs in via email
    # (see FIRST_SUPERUSER_EMAIL) — this column stores the email address.
    username: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="RESTRICT"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    org_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    role: Mapped[Role] = relationship(back_populates="users", foreign_keys=[role_id])
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="RefreshToken.user_id",
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    token_family_id: Mapped[str] = mapped_column(String(64), index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by_token_id: Mapped[int | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL"), nullable=True
    )
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="refresh_tokens", foreign_keys=[user_id])
