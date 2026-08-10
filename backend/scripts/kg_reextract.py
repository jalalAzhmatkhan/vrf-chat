"""KG Foundation Wave 1 — CLI wrapper around
`app.ingestion.kg_candidate_reextractor.reextract_kg_candidates_for_document`.

Thin wrapper only (the real logic is the tested, importable function in
`app/ingestion/kg_candidate_reextractor.py` — same convention as
`scripts/sync_admin_scopes.py`): builds a real DB session from settings,
loops over the given `document_id`(s), prints a summary per document.

Per Wave 1 DoD (`Documentation/project-milestones/06-kg-candidate-extraction-foundation.md`):
this is the "≈seconds/document, not ~82 minutes/document" re-extraction
path — NOT a re-ingest, no Docling/PaddleOCR-VL/GPU/object-storage
involved, only reads/writes Postgres `elements` rows for
already-ingested documents.

Usage (from WSL or Windows, wherever Postgres is reachable — no GPU
needed):
    DB_HOST=localhost DB_PORT=15432 DB_USER=vrf_app \
    DB_PASSWORD=vrf_app_dev_password DB_NAME=vrf_chatbot \
    uv run python scripts/kg_reextract.py 3 4 5

Re-extracts every `document_id` given on the command line, in order. With no
arguments, re-extracts every document currently in `documents` (used for the
Wave 1 DoD's "jalankan sekali terhadap ketujuh dokumen" step, once all 7 are
ingested — see `09-kg-extraction-strategy.md` §4).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.engine import get_session_factory  # noqa: E402
from app.db.models.documents import Document  # noqa: E402
from app.ingestion.kg_candidate_reextractor import (  # noqa: E402
    DocumentNotFoundError,
    reextract_kg_candidates_for_document,
)


def main() -> None:  # pragma: no cover — thin CLI wrapper, exercised manually
    session_factory = get_session_factory()
    db = session_factory()
    try:
        document_ids = [int(arg) for arg in sys.argv[1:]]
        if not document_ids:
            document_ids = list(db.execute(select(Document.id).order_by(Document.id)).scalars())

        if not document_ids:
            print("No documents found — nothing to re-extract.")
            return

        for document_id in document_ids:
            try:
                summary = reextract_kg_candidates_for_document(db, document_id)
            except DocumentNotFoundError as exc:
                print(f"document_id={document_id}: SKIPPED ({exc})")
                continue
            print(
                f"document_id={summary.document_id}: "
                f"elements_scanned={summary.elements_scanned} "
                f"elements_with_candidates={summary.elements_with_candidates} "
                f"entities_written={summary.entities_written} "
                f"relations_written={summary.relations_written}"
            )
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    main()
