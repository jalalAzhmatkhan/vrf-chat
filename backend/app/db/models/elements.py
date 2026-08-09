"""`elements` — see 06-data-schema.md §1 and 02-ingestion-pipeline.md §5.

Canonical extracted-element representation (heading/paragraph/table/figure/
icon/etc.), the unit that preserves inline icon <-> surrounding-text
association required by `CLAUDE.md` §4.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, PortableJSON

ELEMENT_TYPES = (
    "heading",
    "paragraph",
    "list",
    "table",
    "figure",
    "figure_caption",
    "diagram",
    "warning",
    "note",
    "error_code",
    "icon",
)


class Element(Base):
    __tablename__ = "elements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    element_type: Mapped[str] = mapped_column(String(32))
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("elements.id", ondelete="SET NULL"), nullable=True
    )
    section_path: Mapped[list | None] = mapped_column(PortableJSON, nullable=True)
    image_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_hash: Mapped[str] = mapped_column(String(128))
    extraction_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_description: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
    kg_candidate_entities: Mapped[list] = mapped_column(PortableJSON, default=list)
    kg_candidate_relations: Mapped[list] = mapped_column(PortableJSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    children: Mapped[list[Element]] = relationship(back_populates="parent")
    parent: Mapped[Element | None] = relationship(back_populates="children", remote_side=[id])
