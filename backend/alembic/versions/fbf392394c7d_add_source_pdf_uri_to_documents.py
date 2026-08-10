"""add source_pdf_uri to documents

See `app/db/models/documents.py` `Document.source_pdf_uri` docstring
(I1.10) — canonical object storage URI of the original uploaded PDF, needed
by a Celery worker to re-materialize the file locally for PyMuPDF/Docling.

Revision ID: fbf392394c7d
Revises: a0b89b55ab4c
Create Date: 2026-08-09 15:23:02.579781

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fbf392394c7d"
down_revision: str | Sequence[str] | None = "a0b89b55ab4c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_pdf_uri", sa.String(length=1024), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "source_pdf_uri")
