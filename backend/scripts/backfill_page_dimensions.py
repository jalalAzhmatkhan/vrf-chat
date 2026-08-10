"""C2.5 — one-off backfill of `pages.page_width_pt`/`page_height_pt` for
pages ingested before the C2.5 migration (`4da71ff1bd4f_add_page_dimensions_
to_pages_for_.py`) added those columns, per
`Documentation/system-design/05-streaming-and-api-contract.md` §5.5.

Deliberately does **NOT** re-run ingestion (no Docling/PaddleOCR-VL, no
GPU) — page dimensions are read directly from the original source PDF via
`pymupdf.Page.rect`, which is all `store_pages_and_elements` itself does
for pages ingested from now on. Idempotent: only updates rows where
`page_width_pt IS NULL` (or `--force`), matched by `page_number` (1-indexed,
same convention as `app/ingestion/canonical_store.py`).

Usage (from the repo, Postgres reachable — see `scripts/i1_10_run_e2e.py`
for the equivalent env var pattern):
    DB_HOST=localhost DB_PORT=15432 DB_USER=vrf_app \
    DB_PASSWORD=vrf_app_dev_password DB_NAME=vrf_chatbot \
    uv run python scripts/backfill_page_dimensions.py --document-id 3 \
      --pdf-path "D:\\Training\\vrf-chatbot\\source-documents\\Zeggo VRV IV Service Manual REYQ.pdf"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db.engine import get_session_factory  # noqa: E402
from app.db.models.documents import Page  # noqa: E402


def backfill_page_dimensions(
    db, document_id: int, pdf_path: Path, *, force: bool = False
) -> tuple[int, int]:
    """Returns `(updated_count, skipped_count)`. Exposed as a plain function
    (not just CLI-only logic) so it's unit-testable without a real
    Postgres/PDF."""
    pages = (
        db.execute(select(Page).where(Page.document_id == document_id).order_by(Page.page_number))
        .scalars()
        .all()
    )
    if not pages:
        return 0, 0

    updated = 0
    skipped = 0
    pdf_doc = pymupdf.open(pdf_path)
    try:
        for page in pages:
            if not force and page.page_width_pt is not None and page.page_height_pt is not None:
                skipped += 1
                continue
            pymupdf_page = pdf_doc[page.page_number - 1]
            page.page_width_pt = pymupdf_page.rect.width
            page.page_height_pt = pymupdf_page.rect.height
            updated += 1
        db.commit()
    finally:
        pdf_doc.close()

    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--pdf-path", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="Recompute even for pages that already have values."
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise SystemExit(f"PDF not found at {args.pdf_path}")

    session_factory = get_session_factory()
    db = session_factory()
    try:
        updated, skipped = backfill_page_dimensions(
            db, args.document_id, args.pdf_path, force=args.force
        )
    finally:
        db.close()

    print(f"document_id={args.document_id} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
