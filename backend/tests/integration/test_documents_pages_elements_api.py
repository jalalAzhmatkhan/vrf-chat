"""Integration tests for `GET /api/v1/documents`, `GET /api/v1/documents/{id}`,
`GET /api/v1/documents/{id}/pages/{page_number}`, and
`GET /api/v1/elements/{element_id}` (C2.5).
"""

from __future__ import annotations

import pytest

from app.db.models.chunks import Chunk
from app.db.models.documents import Document, Page
from app.db.models.elements import Element
from tests.integration.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, USER_EMAIL, USER_PASSWORD


def _login(auth_client, username: str, password: str) -> str:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_document(db_session_factory):
    session = db_session_factory()
    try:
        document = Document(
            title="Zeggo VRV IV Service Manual REYQ",
            manufacturer="Zeggo",
            model_family="REYQ",
            filename="manual.pdf",
            source_hash="hash-1",
            page_count=286,
            status="ready",
        )
        session.add(document)
        session.commit()

        page = Page(
            document_id=document.id,
            page_number=238,
            page_image_uri="file://documents/1/pages/238/page.png",
            page_width_pt=609.0,
            page_height_pt=793.0,
        )
        session.add(page)
        session.commit()

        paragraph = Element(
            document_id=document.id,
            page_id=page.id,
            element_type="paragraph",
            text="Check the compressor.",
            bbox={"l": 1, "t": 2, "r": 3, "b": 4},
            source_hash="h1",
        )
        icon = Element(
            document_id=document.id,
            page_id=page.id,
            element_type="icon",
            text=None,
            image_uri="file://documents/1/pages/238/elements/1.png",
            visual_description={"description": "reset button icon"},
            source_hash="h2",
        )
        session.add_all([paragraph, icon])
        session.commit()

        return {
            "document_id": document.id,
            "page_number": 238,
            "paragraph_id": paragraph.id,
            "icon_id": icon.id,
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------


def test_list_documents_requires_scope(auth_client) -> None:
    response = auth_client.get("/api/v1/documents")
    assert response.status_code == 401


def test_list_documents_returns_seeded_document(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get("/api/v1/documents", headers=_auth_headers(token))
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["title"] == "Zeggo VRV IV Service Manual REYQ"
    assert body[0]["model_family"] == "REYQ"


def test_list_documents_empty(auth_client) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get("/api/v1/documents", headers=_auth_headers(token))
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{id}
# ---------------------------------------------------------------------------


def test_get_document_found(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/documents/{seeded_document['document_id']}", headers=_auth_headers(token)
    )
    assert response.status_code == 200
    assert response.json()["id"] == seeded_document["document_id"]


def test_get_document_not_found(auth_client) -> None:
    token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = auth_client.get("/api/v1/documents/999999", headers=_auth_headers(token))
    assert response.status_code == 404


def test_get_document_requires_scope(auth_client, seeded_document) -> None:
    response = auth_client.get(f"/api/v1/documents/{seeded_document['document_id']}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{id}/pages/{page_number}
# ---------------------------------------------------------------------------


def test_get_document_page_found(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/documents/{seeded_document['document_id']}/pages/238",
        headers=_auth_headers(token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page_number"] == 238
    assert body["page_width_pt"] == 609.0
    assert body["page_height_pt"] == 793.0
    assert body["image_width_px"] == round(609.0 * 200 / 72)
    assert body["image_height_px"] == round(793.0 * 200 / 72)
    assert body["page_image_uri"].startswith("http://localhost:8000/internal/local-storage/")
    assert len(body["elements"]) == 2
    icon_element = next(e for e in body["elements"] if e["element_type"] == "icon")
    assert icon_element["image_uri"].startswith("http://localhost:8000/internal/local-storage/")
    paragraph_element = next(e for e in body["elements"] if e["element_type"] == "paragraph")
    assert paragraph_element["image_uri"] is None
    assert paragraph_element["text"] == "Check the compressor."


def test_get_document_page_not_found(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/documents/{seeded_document['document_id']}/pages/9999",
        headers=_auth_headers(token),
    )
    assert response.status_code == 404


def test_get_document_page_requires_scope(auth_client, seeded_document) -> None:
    response = auth_client.get(
        f"/api/v1/documents/{seeded_document['document_id']}/pages/238"
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/elements/{element_id}
# ---------------------------------------------------------------------------


def test_get_element_found_visual_type(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/elements/{seeded_document['icon_id']}", headers=_auth_headers(token)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["element_type"] == "icon"
    assert body["page_number"] == 238
    assert body["image_uri"].startswith("http://localhost:8000/internal/local-storage/")
    assert body["visual_description"] == {"description": "reset button icon"}


def test_get_element_found_non_visual_type_no_image_uri(auth_client, seeded_document) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/elements/{seeded_document['paragraph_id']}", headers=_auth_headers(token)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_uri"] is None
    assert body["text"] == "Check the compressor."


def test_get_element_not_found(auth_client) -> None:
    token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    response = auth_client.get("/api/v1/elements/999999", headers=_auth_headers(token))
    assert response.status_code == 404


def test_get_element_requires_scope(auth_client, seeded_document) -> None:
    response = auth_client.get(f"/api/v1/elements/{seeded_document['paragraph_id']}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/elements/{element_id} — content_structured (F2-10, §5.2.1)
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_table_document(db_session_factory):
    """A document with two table elements: one with an owning `chunks` row
    (content_structured present) and one without (simulates a table element
    ingested before/without its chunk, or an edge case — must degrade to
    `None`, never fabricate data)."""
    session = db_session_factory()
    try:
        document = Document(
            title="Table Manual", filename="t.pdf", source_hash="hash-table", status="ready"
        )
        session.add(document)
        session.commit()

        page = Page(document_id=document.id, page_number=99)
        session.add(page)
        session.commit()

        table_with_chunk = Element(
            document_id=document.id,
            page_id=page.id,
            element_type="table",
            text="<table>...</table>",
            source_hash="h-table-1",
        )
        table_without_chunk = Element(
            document_id=document.id,
            page_id=page.id,
            element_type="table",
            text="<table>...</table>",
            source_hash="h-table-2",
        )
        session.add_all([table_with_chunk, table_without_chunk])
        session.commit()

        content_structured = {"rows": [["Warning", "High voltage"], ["Note", "See manual"]]}
        chunk = Chunk(
            document_id=document.id,
            chunk_type="table",
            content_text="Warning | High voltage",
            content_structured=content_structured,
            element_ids=[table_with_chunk.id],
            content_hash="chash-1",
        )
        session.add(chunk)
        session.commit()

        return {
            "with_chunk_id": table_with_chunk.id,
            "without_chunk_id": table_without_chunk.id,
            "content_structured": content_structured,
        }
    finally:
        session.close()


def test_get_element_table_with_chunk_returns_full_content_structured(
    auth_client, seeded_table_document
) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/elements/{seeded_table_document['with_chunk_id']}", headers=_auth_headers(token)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["element_type"] == "table"
    assert body["content_structured"] == seeded_table_document["content_structured"]


def test_get_element_table_without_owning_chunk_content_structured_none(
    auth_client, seeded_table_document
) -> None:
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/elements/{seeded_table_document['without_chunk_id']}",
        headers=_auth_headers(token),
    )
    assert response.status_code == 200
    assert response.json()["content_structured"] is None


def test_get_element_non_table_type_content_structured_none(auth_client, seeded_document) -> None:
    # Non-table short-circuit — must not even attempt a chunk query.
    token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.get(
        f"/api/v1/elements/{seeded_document['paragraph_id']}", headers=_auth_headers(token)
    )
    assert response.status_code == 200
    assert response.json()["content_structured"] is None
