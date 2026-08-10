"""Unit tests for `scripts/backfill_page_dimensions.py` (C2.5).

`backfill_page_dimensions` (the testable function, not the CLI `main()`) is
exercised against an in-memory SQLite session + a small synthetic PDF built
via PyMuPDF's authoring API — no real Postgres/GPU dependency.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document, Page
from scripts.backfill_page_dimensions import backfill_page_dimensions


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _write_pdf(path: Path, page_count: int, *, width: float = 600, height: float = 800) -> None:
    doc = pymupdf.open()
    for _ in range(page_count):
        doc.new_page(width=width, height=height)
    doc.save(str(path))


def test_backfill_updates_null_dimensions(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, 2, width=609, height=793)
    db = _make_session()
    document = Document(title="Manual", filename="manual.pdf", source_hash="h")
    db.add(document)
    db.commit()
    db.add_all(
        [
            Page(document_id=document.id, page_number=1),
            Page(document_id=document.id, page_number=2),
        ]
    )
    db.commit()

    updated, skipped = backfill_page_dimensions(db, document.id, pdf_path)

    assert updated == 2
    assert skipped == 0
    pages = db.query(Page).order_by(Page.page_number).all()
    assert pages[0].page_width_pt == 609
    assert pages[0].page_height_pt == 793


def test_backfill_skips_pages_that_already_have_dimensions(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, 1)
    db = _make_session()
    document = Document(title="Manual", filename="manual.pdf", source_hash="h")
    db.add(document)
    db.commit()
    db.add(
        Page(
            document_id=document.id, page_number=1, page_width_pt=100.0, page_height_pt=200.0
        )
    )
    db.commit()

    updated, skipped = backfill_page_dimensions(db, document.id, pdf_path)

    assert updated == 0
    assert skipped == 1
    page = db.query(Page).one()
    assert page.page_width_pt == 100.0


def test_backfill_force_recomputes_even_if_present(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _write_pdf(pdf_path, 1, width=609, height=793)
    db = _make_session()
    document = Document(title="Manual", filename="manual.pdf", source_hash="h")
    db.add(document)
    db.commit()
    db.add(
        Page(
            document_id=document.id, page_number=1, page_width_pt=100.0, page_height_pt=200.0
        )
    )
    db.commit()

    updated, skipped = backfill_page_dimensions(db, document.id, pdf_path, force=True)

    assert updated == 1
    assert skipped == 0
    page = db.query(Page).one()
    assert page.page_width_pt == 609


def test_backfill_no_pages_returns_zero() -> None:
    db = _make_session()
    document = Document(title="Manual", filename="manual.pdf", source_hash="h")
    db.add(document)
    db.commit()

    updated, skipped = backfill_page_dimensions(db, document.id, Path("unused.pdf"))

    assert updated == 0
    assert skipped == 0
