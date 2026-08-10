"""Integration tests for the FastAPI app (`app/main.py`) — HTTP contract
matching `backend/app/ingestion/paddleocr_vl_cascade.py`
`RemoteAPIPaddleOCRVLClient`'s expectations. Uses a fake `PaddleOCRVLPipeline`
(no real model/GPU)."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline import OCRPageResponse, TableReparseResponse, VisualDescriptionResponse


class FakePipeline:
    def __init__(self) -> None:
        self.is_loaded = False
        self.calls: list[tuple[str, str]] = []
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def describe_figure(self, image_base64: str) -> VisualDescriptionResponse:
        self.calls.append(("describe_figure", image_base64))
        return VisualDescriptionResponse(
            description="Fan mode icon", figure_type="icon", components=["fan"], connections=[]
        )

    def reparse_table(self, image_base64: str) -> TableReparseResponse:
        self.calls.append(("reparse_table", image_base64))
        return TableReparseResponse(markdown="| A |", rows=[{"A": "1"}])

    def ocr_page(self, image_base64: str) -> OCRPageResponse:
        self.calls.append(("ocr_page", image_base64))
        return OCRPageResponse(text="scanned text")


def _client() -> TestClient:
    fake = FakePipeline()
    app = create_app(pipeline=fake)  # type: ignore[arg-type]
    client = TestClient(app)
    client.fake_pipeline = fake  # type: ignore[attr-defined]
    return client


def _b64(data: bytes = b"png-bytes") -> str:
    return base64.b64encode(data).decode("ascii")


def test_health() -> None:
    with _client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": False}


def test_describe_figure() -> None:
    with _client() as client:
        response = client.post("/describe_figure", json={"image_base64": _b64()})
        assert response.status_code == 200
        assert response.json() == {
            "description": "Fan mode icon",
            "figure_type": "icon",
            "components": ["fan"],
            "connections": [],
        }


def test_reparse_table() -> None:
    with _client() as client:
        response = client.post("/reparse_table", json={"image_base64": _b64()})
        assert response.status_code == 200
        assert response.json() == {"markdown": "| A |", "rows": [{"A": "1"}]}


def test_ocr_page() -> None:
    with _client() as client:
        response = client.post("/ocr_page", json={"image_base64": _b64()})
        assert response.status_code == 200
        assert response.json() == {"text": "scanned text"}


def test_describe_figure_missing_body_returns_422() -> None:
    with _client() as client:
        response = client.post("/describe_figure", json={})
        assert response.status_code == 422
