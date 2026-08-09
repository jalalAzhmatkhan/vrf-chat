"""`ingestion_jobs` — per-stage structured tracking, see
01-architecture-overview.md §5 (observability) and 06-data-schema.md §1.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, PortableJSON

INGESTION_STAGES = (
    "native_probe",
    "docling",
    "paddle_cascade",
    "chunking",
    "embedding",
    "kg_candidate",
)
INGESTION_JOB_STATUSES = ("queued", "running", "done", "failed")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_metrics: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)
