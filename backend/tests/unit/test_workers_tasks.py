"""Unit tests for `app/workers/tasks.py` (I1.10).

`run_ingestion_task` is invoked directly as a plain function call (Celery
tasks are callable without a broker/worker — this exercises the task body
synchronously) with every collaborator (`get_session_factory`,
`build_storage_client`, `materialize_source_pdf`, `run_ingestion_pipeline`)
monkeypatched — no DB/storage/Celery broker/GPU dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.ingestion.orchestrator import OrchestratorResult
from app.workers import tasks


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_run_ingestion_task_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()
    monkeypatch.setattr(tasks, "get_session_factory", lambda: (lambda: fake_session))
    monkeypatch.setattr(tasks, "build_storage_client", lambda settings: object())

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    materialize_calls: list[Any] = []

    def fake_materialize(db: Any, storage: Any, document_id: int) -> Path:
        materialize_calls.append(document_id)
        return pdf_path

    monkeypatch.setattr(tasks, "materialize_source_pdf", fake_materialize)

    pipeline_calls: list[dict[str, Any]] = []

    def fake_run_pipeline(
        db: Any, storage: Any, settings: Any, **kwargs: Any
    ) -> OrchestratorResult:
        pipeline_calls.append(kwargs)
        return OrchestratorResult(
            document_id=kwargs["document_id"],
            pages_stored=1,
            elements_stored=2,
            chunks_stored=1,
            chunks_embedded=1,
            cascade_task_count=0,
        )

    monkeypatch.setattr(tasks, "run_ingestion_pipeline", fake_run_pipeline)

    result = tasks.run_ingestion_task(document_id=42)

    assert result == {
        "document_id": 42,
        "pages_stored": 1,
        "elements_stored": 2,
        "chunks_stored": 1,
        "chunks_embedded": 1,
        "cascade_task_count": 0,
    }
    assert materialize_calls == [42]
    assert pipeline_calls[0]["document_id"] == 42
    assert fake_session.closed is True
    assert not pdf_path.exists()  # temp PDF cleaned up


def test_run_ingestion_task_cleans_up_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_session = _FakeSession()
    monkeypatch.setattr(tasks, "get_session_factory", lambda: (lambda: fake_session))
    monkeypatch.setattr(tasks, "build_storage_client", lambda settings: object())

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf-bytes")
    monkeypatch.setattr(tasks, "materialize_source_pdf", lambda db, storage, document_id: pdf_path)

    def fake_run_pipeline(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(tasks, "run_ingestion_pipeline", fake_run_pipeline)

    with pytest.raises(RuntimeError, match="pipeline exploded"):
        tasks.run_ingestion_task(document_id=7)

    assert fake_session.closed is True
    assert not pdf_path.exists()
