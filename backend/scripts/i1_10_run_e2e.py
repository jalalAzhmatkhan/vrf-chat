"""I1.10 — full end-to-end ingestion proof, Zeggo VRV IV REYQ (286 pages),
see `Documentation/project-milestones/02-phase-1-ingestion.md` I1.10 DoD
("Minimal 1 dokumen penuh... berhasil diingest end-to-end... tanpa error
fatal").

Runs the real orchestrator (`app.ingestion.orchestrator.run_ingestion_pipeline`)
directly (not via Celery — no worker/broker dependency needed for this
one-shot proof run) against real infrastructure: Postgres, Qdrant,
paddleocr-vl-service (all via WSL, see STATUS REPORT for exact
endpoints/ports), local object storage, and real Docling (GPU).

Usage (from WSL, with Postgres/Qdrant/paddleocr-vl-service all reachable):
    DB_HOST=localhost DB_PORT=15432 DB_USER=vrf_app \
    DB_PASSWORD=vrf_app_dev_password DB_NAME=vrf_chatbot \
    VECTOR_STORE_QDRANT_URL=http://localhost:6333 \
    PADDLE_OCR_VL_BACKEND=remote_api \
    PADDLE_OCR_VL_API_URL=http://localhost:8100 \
    OBJECT_STORAGE_BACKEND=local \
    uv run python scripts/i1_10_run_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db.engine import get_session_factory  # noqa: E402
from app.ingestion.orchestrator import (  # noqa: E402
    prepare_document_for_ingestion,
    run_ingestion_pipeline,
)
from app.storage.factory import build_storage_client  # noqa: E402

SOURCE_PDF = Path(r"D:\Training\vrf-chatbot\source-documents\Zeggo VRV IV Service Manual REYQ.pdf")
# WSL path equivalent (this script may run with a WSL-native venv against a
# Windows-path source file, mounted at /mnt/d).
SOURCE_PDF_WSL = Path(
    "/mnt/d/Training/vrf-chatbot/source-documents/Zeggo VRV IV Service Manual REYQ.pdf"
)
RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "i1.10-e2e-results.json"


def main() -> None:
    pdf_path = SOURCE_PDF_WSL if SOURCE_PDF_WSL.exists() else SOURCE_PDF
    if not pdf_path.exists():
        raise SystemExit(f"Source PDF not found at {pdf_path}")

    settings = get_settings()
    session_factory = get_session_factory()
    storage = build_storage_client(settings)
    db = session_factory()

    print(f"Reading {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)...")
    pdf_bytes = pdf_path.read_bytes()

    import hashlib

    document_hash = hashlib.sha256(pdf_bytes).hexdigest()

    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    doc.close()
    print(f"page_count={page_count}")

    document, created = prepare_document_for_ingestion(
        db,
        storage,
        pdf_bytes=pdf_bytes,
        filename="Zeggo VRV IV Service Manual REYQ.pdf",
        title="Zeggo VRV IV Service Manual REYQ",
        manufacturer="Zeggo",
        model_family="REYQ",
        document_hash=document_hash,
        page_count=page_count,
    )
    print(f"document_id={document.id} created={created}")

    start = time.perf_counter()
    result = run_ingestion_pipeline(
        db,
        storage,
        settings,
        document_id=document.id,
        pdf_path=pdf_path,
    )
    elapsed = time.perf_counter() - start

    print("=== RESULT ===")
    print(result)
    print(f"elapsed_seconds={elapsed:.1f}")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "document_id": result.document_id,
                "pages_stored": result.pages_stored,
                "elements_stored": result.elements_stored,
                "chunks_stored": result.chunks_stored,
                "chunks_embedded": result.chunks_embedded,
                "cascade_task_count": result.cascade_task_count,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
