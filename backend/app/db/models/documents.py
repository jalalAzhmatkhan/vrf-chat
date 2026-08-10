"""`documents` and `pages` — see 06-data-schema.md §1."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

DOCUMENT_STATUSES = ("queued", "processing", "ready", "failed")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    manufacturer: Mapped[str | None] = mapped_column(String(256), nullable=True)
    model_family: Mapped[str | None] = mapped_column(String(256), nullable=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    org_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    pages: Mapped[list[Page]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    page_image_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    text_layer_present: Mapped[bool] = mapped_column(Boolean, default=False)
    extraction_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Idempotency (I1.5, 02-ingestion-pipeline.md §5): hash of this page's
    # source-derived content (extracted text + vector drawing/image counts —
    # see app/ingestion/canonical_store.py `compute_page_hash`), NOT of any
    # ML model output, so re-ingesting an unchanged PDF always yields the
    # same page_hash regardless of Docling/PaddleOCR-VL run-to-run variance.
    page_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    document: Mapped[Document] = relationship(back_populates="pages")
