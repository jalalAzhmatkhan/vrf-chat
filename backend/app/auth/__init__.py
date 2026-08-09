"""Authentication & RBAC — password hashing, JWT issuance/verification,
refresh token rotation, `SecurityScopes` dependency, login rate limiting.
See `Documentation/system-design/08-authentication-rbac.md`.
"""

from app.auth.schemas import AuthenticatedUser
from app.auth.security import get_current_user

__all__ = ["AuthenticatedUser", "get_current_user"]
