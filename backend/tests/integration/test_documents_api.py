"""Integration tests for `POST /api/v1/documents` and
`GET /api/v1/documents/{id}/ingestion-jobs` (I1.10).

`run_ingestion_task.delay()` is monkeypatched to avoid touching a real
Celery broker (Redis) — this test verifies the endpoint enqueues correctly,
not that a real worker picks the job up (that's what the I1.10 live E2E run
verifies separately).
"""

from __future__ import annotations

from types import SimpleNamespace

import pymupdf
import pytest

from app.api.v1 import documents as documents_module
from tests.integration.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, USER_EMAIL, USER_PASSWORD


def _login(auth_client, username: str, password: str) -> str:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _sample_pdf_bytes(pages: int = 2) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=600, height=800)
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "Sample manual content.")
    data = doc.tobytes()
    doc.close()
    return bytes(data)


@pytest.fixture
def fake_delay(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _fake_delay(document_id: int):
        calls.append(document_id)
        return SimpleNamespace(id=f"task-for-{document_id}")

    monkeypatch.setattr(documents_module.run_ingestion_task, "delay", _fake_delay)
    return calls


def test_trigger_ingestion_requires_documents_write_scope(auth_client) -> None:
    user_token = _login(auth_client, USER_EMAIL, USER_PASSWORD)

    response = auth_client.post(
        "/api/v1/documents",
        headers=_auth_headers(user_token),
        params={"title": "Manual"},
        files={"file": ("manual.pdf", _sample_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 403


def test_trigger_ingestion_no_token_returns_401(auth_client) -> None:
    response = auth_client.post(
        "/api/v1/documents",
        params={"title": "Manual"},
        files={"file": ("manual.pdf", _sample_pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 401


def test_trigger_ingestion_success(auth_client, fake_delay) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.post(
        "/api/v1/documents",
        headers=_auth_headers(admin_token),
        params={"title": "Service Manual", "manufacturer": "Zeggo", "model_family": "REYQ"},
        files={"file": ("manual.pdf", _sample_pdf_bytes(3), "application/pdf")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["document_id"] is not None
    assert body["celery_task_id"] == f"task-for-{body['document_id']}"
    assert fake_delay == [body["document_id"]]


def test_trigger_ingestion_idempotent_on_identical_pdf(auth_client, fake_delay) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    pdf_bytes = _sample_pdf_bytes(1)

    first = auth_client.post(
        "/api/v1/documents",
        headers=_auth_headers(admin_token),
        params={"title": "Manual"},
        files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
    )
    second = auth_client.post(
        "/api/v1/documents",
        headers=_auth_headers(admin_token),
        params={"title": "Manual"},
        files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["document_id"] == second.json()["document_id"]
    assert second.json()["status"] == "already_ingested"
    assert second.json()["celery_task_id"] is None
    # Only enqueued once — the second (identical) upload is a no-op.
    assert fake_delay == [first.json()["document_id"]]


def test_list_ingestion_jobs_requires_documents_read_scope(auth_client) -> None:
    unauthorized = auth_client.get("/api/v1/documents/1/ingestion-jobs")
    assert unauthorized.status_code == 401


def test_list_ingestion_jobs_empty_for_unknown_document(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get(
        "/api/v1/documents/999/ingestion-jobs", headers=_auth_headers(admin_token)
    )

    assert response.status_code == 200
    assert response.json() == []
