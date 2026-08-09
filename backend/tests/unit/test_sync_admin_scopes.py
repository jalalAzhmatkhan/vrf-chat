import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.auth import Role, RoleScope, Scope
from scripts.sync_admin_scopes import AdminRoleNotFoundError, sync_admin_scopes


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


def test_raises_when_no_admin_role_exists(db: Session) -> None:
    with pytest.raises(AdminRoleNotFoundError):
        sync_admin_scopes(db)


def test_assigns_all_scopes_when_none_assigned_yet(db: Session) -> None:
    admin_role = Role(name="admin", description="Full access")
    db.add(admin_role)
    db.add_all(
        [
            Scope(code="chat:read", description="d", category="chat"),
            Scope(code="chat:write", description="d", category="chat"),
        ]
    )
    db.commit()

    newly_added = sync_admin_scopes(db)

    assert set(newly_added) == {"chat:read", "chat:write"}
    role_scope_count = db.query(RoleScope).filter(RoleScope.role_id == admin_role.id).count()
    assert role_scope_count == 2


def test_is_idempotent_no_duplicates_on_rerun(db: Session) -> None:
    admin_role = Role(name="admin", description="Full access")
    db.add(admin_role)
    db.add(Scope(code="chat:read", description="d", category="chat"))
    db.commit()

    sync_admin_scopes(db)
    second_run_result = sync_admin_scopes(db)

    assert second_run_result == []
    role_scope_count = db.query(RoleScope).filter(RoleScope.role_id == admin_role.id).count()
    assert role_scope_count == 1


def test_only_assigns_the_missing_scope(db: Session) -> None:
    admin_role = Role(name="admin", description="Full access")
    db.add(admin_role)
    existing_scope = Scope(code="chat:read", description="d", category="chat")
    new_scope = Scope(code="admin:rbac:write", description="d", category="admin_rbac")
    db.add_all([existing_scope, new_scope])
    db.flush()
    db.add(RoleScope(role_id=admin_role.id, scope_id=existing_scope.id))
    db.commit()

    newly_added = sync_admin_scopes(db)

    assert newly_added == ["admin:rbac:write"]
