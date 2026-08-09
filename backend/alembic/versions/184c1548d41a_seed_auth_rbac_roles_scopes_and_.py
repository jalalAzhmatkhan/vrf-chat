"""seed auth rbac roles scopes and bootstrap first superuser

Revision ID: 184c1548d41a
Revises: 9c71b9879a35
Create Date: 2026-08-09 08:46:23.409392

Data migration — see Documentation/system-design/08-authentication-rbac.md
§4 (seed data) and §3.4 (admin gets every scope via explicit `role_scopes`
rows, never a hardcoded `if role == "admin"` bypass). This migration
replaces the standalone bootstrap CLI script originally sketched as B0.8 —
per Jalal's explicit instruction, admin account bootstrap is an idempotent
Alembic data migration instead, reading FIRST_SUPERUSER_EMAIL/
FIRST_SUPERUSER_PASSWORD from the environment.

Idempotency: every INSERT here first checks for existing rows (by unique
key) and skips if already present, so re-running `alembic upgrade head`
against an already-seeded database (or a database where this migration
already ran) never creates duplicates.
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '184c1548d41a'
down_revision: Union[str, Sequence[str], None] = '9c71b9879a35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.seed_auth_rbac")

# Lightweight table definitions for this migration — deliberately NOT
# imported from app.db.models.auth, so this migration keeps working
# unchanged even if the ORM models evolve later (standard Alembic practice).
roles_table = sa.table(
    "roles",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("description", sa.String),
)
scopes_table = sa.table(
    "scopes",
    sa.column("id", sa.Integer),
    sa.column("code", sa.String),
    sa.column("description", sa.String),
    sa.column("category", sa.String),
)
role_scopes_table = sa.table(
    "role_scopes",
    sa.column("role_id", sa.Integer),
    sa.column("scope_id", sa.Integer),
)
users_table = sa.table(
    "users",
    sa.column("id", sa.Integer),
    sa.column("username", sa.String),
    sa.column("password_hash", sa.String),
    sa.column("role_id", sa.Integer),
    sa.column("is_active", sa.Boolean),
)

ROLES = [
    {"name": "admin", "description": "Full access — semua scope"},
    {"name": "user", "description": "Akses chat & lihat dokumen sumber saja"},
]

# (code, description, category) — 10 scopes per 08-authentication-rbac.md §3.3
SCOPES = [
    ("chat:read", "Melihat riwayat percakapan sendiri", "chat"),
    ("chat:write", "Mengirim pesan chat", "chat"),
    ("documents:read", "Melihat dokumen/halaman sumber (citation viewer)", "documents"),
    ("documents:write", "Memicu ingestion/upload dokumen baru", "documents"),
    ("admin:eval:read", "Melihat dashboard metrik (tab Retrieval & Generation)", "admin_eval"),
    ("admin:eval:write", "Memicu eval run baru", "admin_eval"),
    ("admin:review:read", "Melihat manual review queue", "admin_review"),
    ("admin:review:write", "Submit label review, promote ke dataset", "admin_review"),
    (
        "admin:rbac:read",
        "Melihat halaman Permission Management & User Management (read-only)",
        "admin_rbac",
    ),
    ("admin:rbac:write", "Mengedit mapping role->scope dan role user (write)", "admin_rbac"),
]

USER_ROLE_SCOPES = ["chat:read", "chat:write", "documents:read"]


def _get_or_create_role(conn: sa.Connection, name: str, description: str) -> int:
    existing = conn.execute(
        sa.select(roles_table.c.id).where(roles_table.c.name == name)
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    conn.execute(sa.insert(roles_table).values(name=name, description=description))
    return int(
        conn.execute(sa.select(roles_table.c.id).where(roles_table.c.name == name)).scalar_one()
    )


def _get_or_create_scope(conn: sa.Connection, code: str, description: str, category: str) -> int:
    existing = conn.execute(
        sa.select(scopes_table.c.id).where(scopes_table.c.code == code)
    ).scalar_one_or_none()
    if existing is not None:
        return int(existing)

    conn.execute(
        sa.insert(scopes_table).values(code=code, description=description, category=category)
    )
    return int(
        conn.execute(sa.select(scopes_table.c.id).where(scopes_table.c.code == code)).scalar_one()
    )


def _ensure_role_scope(conn: sa.Connection, role_id: int, scope_id: int) -> None:
    existing = conn.execute(
        sa.select(role_scopes_table.c.role_id).where(
            role_scopes_table.c.role_id == role_id, role_scopes_table.c.scope_id == scope_id
        )
    ).scalar_one_or_none()
    if existing is None:
        conn.execute(sa.insert(role_scopes_table).values(role_id=role_id, scope_id=scope_id))


def _bootstrap_first_superuser(conn: sa.Connection, admin_role_id: int) -> None:
    # Imported lazily (not at module top) so `alembic downgrade`/tooling
    # that doesn't need app settings never pays the import cost, and so a
    # missing/misconfigured Settings field can't break `alembic history`
    # or other read-only Alembic commands.
    from app.auth.password import hash_password_sync
    from app.core.config import get_settings

    settings = get_settings()
    email = settings.FIRST_SUPERUSER_EMAIL
    password = settings.FIRST_SUPERUSER_PASSWORD

    if not email or not password:
        logger.info(
            "FIRST_SUPERUSER_EMAIL/FIRST_SUPERUSER_PASSWORD not set — "
            "skipping first superuser bootstrap (fine for CI/test databases)."
        )
        return

    existing = conn.execute(
        sa.select(users_table.c.id).where(users_table.c.username == email)
    ).scalar_one_or_none()
    if existing is not None:
        logger.info("First superuser %s already exists — skipping bootstrap.", email)
        return

    password_hash = hash_password_sync(password, settings.BCRYPT_COST_FACTOR)
    conn.execute(
        sa.insert(users_table).values(
            username=email,
            password_hash=password_hash,
            role_id=admin_role_id,
            is_active=True,
        )
    )
    logger.info("Bootstrapped first superuser: %s", email)


def upgrade() -> None:
    conn = op.get_bind()

    role_ids = {r["name"]: _get_or_create_role(conn, r["name"], r["description"]) for r in ROLES}
    scope_ids = {
        code: _get_or_create_scope(conn, code, description, category)
        for code, description, category in SCOPES
    }

    # admin: every scope, via explicit role_scopes rows (08-authentication-rbac.md §3.4)
    for scope_id in scope_ids.values():
        _ensure_role_scope(conn, role_ids["admin"], scope_id)

    # user: the 3-scope starting set
    for code in USER_ROLE_SCOPES:
        _ensure_role_scope(conn, role_ids["user"], scope_ids[code])

    _bootstrap_first_superuser(conn, role_ids["admin"])


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.delete(role_scopes_table))
    conn.execute(sa.delete(scopes_table))
    # Deliberately NOT deleting `users` here — downgrading this migration
    # must not silently delete real admin accounts created after bootstrap.
    conn.execute(sa.delete(roles_table).where(roles_table.c.name.in_(["admin", "user"])))
