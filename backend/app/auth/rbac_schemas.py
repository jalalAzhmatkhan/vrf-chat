"""Pydantic request/response models for `/admin/rbac/*` — contract per
`Documentation/system-design/08-authentication-rbac.md` §6.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RoleResponse(BaseModel):
    id: int
    name: str
    description: str | None = None


class ScopeResponse(BaseModel):
    id: int
    code: str
    description: str | None = None
    category: str | None = None


class RoleScopesResponse(BaseModel):
    role_id: int
    scope_codes: list[str]


class UpdateRoleScopesRequest(BaseModel):
    scope_codes: list[str]


class UserResponse(BaseModel):
    id: int
    username: str
    role_id: int
    role_name: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None


class CreateUserRequest(BaseModel):
    username: str
    initial_password: str = Field(min_length=1, max_length=72)
    role_id: int


class UpdateUserRequest(BaseModel):
    role_id: int | None = None
    is_active: bool | None = None
