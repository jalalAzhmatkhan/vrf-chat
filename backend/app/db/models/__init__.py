"""SQLAlchemy ORM models — re-exported here so Alembic autogenerate (and
anything importing `app.db.models`) sees the full metadata without needing
to import each submodule individually.
"""

from app.db.models.auth import RefreshToken, Role, RoleScope, Scope, User
from app.db.models.chunks import Chunk
from app.db.models.conversations import Citation, Conversation, Message
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from app.db.models.error_codes import ErrorCode
from app.db.models.ingestion_jobs import IngestionJob

__all__ = [
    "Chunk",
    "Citation",
    "Conversation",
    "Document",
    "Element",
    "ErrorCode",
    "IngestionJob",
    "Message",
    "Page",
    "RefreshToken",
    "Role",
    "RoleScope",
    "Scope",
    "User",
]
