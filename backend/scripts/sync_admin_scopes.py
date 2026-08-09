"""Idempotent safety net: assign any `scopes` row not yet present in the
`admin` role's `role_scopes` mapping.

Per `Documentation/system-design/08-authentication-rbac.md` §3.4: admin
access is granted via **explicit, auditable `role_scopes` rows** (never a
hardcoded `if role == "admin"` bypass). The primary mechanism for this is a
*migration convention* — every Alembic migration that adds a new row to
`scopes` MUST also insert the corresponding `role_scopes` row for `admin` in
the same migration (already followed by
`alembic/versions/184c1548d41a_seed_auth_rbac_roles_scopes_and_.py`).

This script is the secondary safety net the design doc also calls for: if a
future migration ever forgets that convention, running this script
(manually, or wired into CI/deploy) closes the gap — the *result* is still
an explicit, auditable `role_scopes` row (not a runtime bypass), only the
*process* of inserting it is automated here.

Usage:
    uv run python scripts/sync_admin_scopes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `uv run python scripts/sync_admin_scopes.py` (script,
# not module) without needing `app` to already be on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.engine import get_session_factory  # noqa: E402
from app.db.models.auth import Role, RoleScope, Scope  # noqa: E402


class AdminRoleNotFoundError(RuntimeError):
    """Raised when no role named `admin` exists — run migrations first."""


def sync_admin_scopes(db: Session) -> list[str]:
    """Assign every scope not yet in the `admin` role's `role_scopes` to it.

    Returns the list of newly-added scope codes (empty if already in sync).
    Safe to call repeatedly — never creates duplicate `role_scopes` rows.
    """
    admin_role = db.execute(select(Role).where(Role.name == "admin")).scalar_one_or_none()
    if admin_role is None:
        raise AdminRoleNotFoundError(
            "No role named 'admin' found — run `alembic upgrade head` first."
        )

    all_scopes = db.execute(select(Scope)).scalars().all()
    existing_scope_ids = {
        rs.scope_id
        for rs in db.execute(
            select(RoleScope).where(RoleScope.role_id == admin_role.id)
        ).scalars()
    }

    newly_added: list[str] = []
    for scope in all_scopes:
        if scope.id in existing_scope_ids:
            continue
        db.add(RoleScope(role_id=admin_role.id, scope_id=scope.id))
        newly_added.append(scope.code)

    if newly_added:
        db.commit()

    return newly_added


def main() -> None:  # pragma: no cover — thin CLI wrapper, exercised manually
    session_factory = get_session_factory()
    db = session_factory()
    try:
        newly_added = sync_admin_scopes(db)
    finally:
        db.close()

    if newly_added:
        print(f"Synced {len(newly_added)} missing scope(s) to admin: {newly_added}")
    else:
        print("admin role already has every seeded scope — nothing to do.")


if __name__ == "__main__":  # pragma: no cover
    main()
