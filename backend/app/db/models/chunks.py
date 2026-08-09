"""`chunks` — see 06-data-schema.md §1 and 03-retrieval-chunking.md."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, PortableJSON

CHUNK_TYPES = ("text", "table", "figure", "procedure", "entity")
EMBEDDING_STATUSES = ("pending", "embedded", "failed")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_type: Mapped[str] = mapped_column(String(32))
    parent_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    section_path: Mapped[list | None] = mapped_column(PortableJSON, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_text: Mapped[str] = mapped_column(Text)
    content_structured: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    element_ids: Mapped[list] = mapped_column(PortableJSON, default=list)
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    vector_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    children: Mapped[list[Chunk]] = relationship(back_populates="parent")
    parent: Mapped[Chunk | None] = relationship(back_populates="children", remote_side=[id])
