"""add page_hash to pages for idempotency

See `Documentation/system-design/02-ingestion-pipeline.md` §5 — idempotency
is `document_hash` (`documents.source_hash`, already present) / `page_hash`
/ `element_hash` (`elements.source_hash`, already present). Only `pages`
was missing its hash column (I1.5).

Revision ID: a0b89b55ab4c
Revises: 184c1548d41a
Create Date: 2026-08-09 14:02:44.484964

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a0b89b55ab4c"
down_revision: str | Sequence[str] | None = "184c1548d41a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("pages", sa.Column("page_hash", sa.String(length=128), nullable=True))
    op.create_index(op.f("ix_pages_page_hash"), "pages", ["page_hash"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_pages_page_hash"), table_name="pages")
    op.drop_column("pages", "page_hash")
