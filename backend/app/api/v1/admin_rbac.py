"""`/admin/rbac/*` — Permission Management & User Management, see
`Documentation/system-design/08-authentication-rbac.md` §6.

Scope `admin:rbac:read` gates every GET; `admin:rbac:write` gates every
mutating endpoint. Scopes themselves are never created/deleted here (§0/§3.3
— only via migration) — this API only manages the `role_scopes` mapping and
`users`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Security
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.auth.password import hash_password
from app.auth.rbac_schemas import (
    CreateUserRequest,
    RoleResponse,
    RoleScopesResponse,
    ScopeResponse,
    UpdateRoleScopesRequest,
    UpdateUserRequest,
    UserResponse,
)
from app.auth.refresh_tokens import revoke_all_user_tokens
from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user
from app.core.config import Settings, get_settings
from app.db.engine import get_db
from app.db.models.auth import Role, RoleScope, Scope, User

router = APIRouter(prefix="/admin/rbac", tags=["admin-rbac"])

READ = ["admin:rbac:read"]
WRITE = ["admin:rbac:write"]


def _role_to_response(role: Role) -> RoleResponse:
    return RoleResponse(id=role.id, name=role.name, description=role.description)


def _user_to_response(user: User, role_name: str) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        role_id=user.role_id,
        role_name=role_name,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.get("/roles", response_model=list[RoleResponse])
async def list_roles(
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[RoleResponse]:
    roles = db.execute(select(Role).order_by(Role.id)).scalars().all()
    return [_role_to_response(r) for r in roles]


@router.get("/scopes", response_model=list[ScopeResponse])
async def list_scopes(
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[ScopeResponse]:
    scopes = db.execute(select(Scope).order_by(Scope.id)).scalars().all()
    return [
        ScopeResponse(id=s.id, code=s.code, description=s.description, category=s.category)
        for s in scopes
    ]


@router.get("/roles/{role_id}/scopes", response_model=RoleScopesResponse)
async def get_role_scopes(
    role_id: int,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> RoleScopesResponse:
    role = db.get(Role, role_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    return RoleScopesResponse(
        role_id=role.id, scope_codes=[rs.scope.code for rs in role.role_scopes]
    )


@router.put("/roles/{role_id}/scopes", response_model=RoleScopesResponse)
async def update_role_scopes(
    role_id: int,
    body: UpdateRoleScopesRequest,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> RoleScopesResponse:
    role = db.get(Role, role_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")

    requested_codes = set(body.scope_codes)
    scopes = db.execute(select(Scope).where(Scope.code.in_(requested_codes))).scalars().all()
    found_codes = {s.code for s in scopes}
    unknown = requested_codes - found_codes
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown scope code(s): {sorted(unknown)}")

    db.execute(delete(RoleScope).where(RoleScope.role_id == role_id))
    for scope in scopes:
        db.add(RoleScope(role_id=role_id, scope_id=scope.id, granted_by=_user.id))
    db.commit()

    db.refresh(role)
    return RoleScopesResponse(
        role_id=role.id, scope_codes=[rs.scope.code for rs in role.role_scopes]
    )


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    db: Session = Depends(get_db),
    role_id: int | None = None,
    search: str | None = None,
    is_active: bool | None = None,
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> list[UserResponse]:
    stmt = select(User)
    if role_id is not None:
        stmt = stmt.where(User.role_id == role_id)
    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)
    if search:
        stmt = stmt.where(User.username.ilike(f"%{search}%"))

    users = db.execute(stmt.order_by(User.id)).scalars().all()
    role_names = {r.id: r.name for r in db.execute(select(Role)).scalars().all()}
    return [_user_to_response(u, role_names.get(u.role_id, "")) for u in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=READ),
) -> UserResponse:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    role = db.get(Role, user.role_id)
    return _user_to_response(user, role.name if role else "")


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    body: CreateUserRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> UserResponse:
    role = db.get(Role, body.role_id)
    if role is None:
        raise HTTPException(status_code=400, detail="Unknown role_id")

    existing = db.execute(select(User).where(User.username == body.username)).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="A user with this username already exists")

    password_hash = await hash_password(body.initial_password, settings.BCRYPT_COST_FACTOR)
    new_user = User(
        username=body.username, password_hash=password_hash, role_id=role.id, is_active=True
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return _user_to_response(new_user, role.name)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    body: UpdateUserRequest,
    db: Session = Depends(get_db),
    _user: AuthenticatedUser = Security(get_current_user, scopes=WRITE),
) -> UserResponse:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.role_id is not None:
        role = db.get(Role, body.role_id)
        if role is None:
            raise HTTPException(status_code=400, detail="Unknown role_id")
        user.role_id = role.id

    if body.is_active is not None:
        user.is_active = body.is_active

    db.add(user)
    db.commit()
    db.refresh(user)

    if body.is_active is False:
        # Deactivating a user immediately revokes their ability to obtain
        # new access tokens via refresh — see 08-authentication-rbac.md §1
        # trade-off / §6.
        revoke_all_user_tokens(db, user.id)

    role = db.get(Role, user.role_id)
    return _user_to_response(user, role.name if role else "")
