"""add page dimensions to pages for citation viewer overlay

See `Documentation/system-design/05-streaming-and-api-contract.md` §5.5 —
`elements.bbox` (PDF points, bottom-left origin) needs the source page's
own dimensions for the FE citation viewer to scale a bbox overlay onto the
rendered `page_image_uri`; that reference was missing from `pages` until
now (C2.5). Populated going forward by
`app/ingestion/canonical_store.py` `store_pages_and_elements`; existing
rows (document_id=3) are backfilled separately by
`scripts/backfill_page_dimensions.py` (reads the original PDF via
`pymupdf`, does NOT re-run ingestion).

Revision ID: 4da71ff1bd4f
Revises: fbf392394c7d
Create Date: 2026-08-10 17:24:49.703984

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4da71ff1bd4f"
down_revision: str | Sequence[str] | None = "fbf392394c7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("page_width_pt", sa.Float(), nullable=True))
    op.add_column("pages", sa.Column("page_height_pt", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("pages", "page_height_pt")
    op.drop_column("pages", "page_width_pt")
