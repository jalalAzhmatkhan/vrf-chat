"""`error_codes` — denormalized lookup for the `search_error_code` tool."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ErrorCode(Base):
    __tablename__ = "error_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(32), index=True)
    model_family: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    element_id: Mapped[int | None] = mapped_column(
        ForeignKey("elements.id", ondelete="SET NULL"), nullable=True
    )
