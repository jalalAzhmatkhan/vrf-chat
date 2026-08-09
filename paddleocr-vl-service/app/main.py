"""`paddleocr-vl-service` — standalone PaddleOCR-VL inference service, see
`Documentation/system-design/02-ingestion-pipeline.md` §4.0.

Implements the HTTP JSON contract `backend/app/ingestion/paddleocr_vl_cascade.py`
`RemoteAPIPaddleOCRVLClient` expects: `POST /describe_figure`,
`POST /reparse_table`, `POST /ocr_page`, each accepting
`{"image_base64": "<base64 PNG>"}` and returning the corresponding response
shape. `GET /health` for Docker Compose healthcheck/orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.pipeline import PaddleOCRVLPipeline


class ImageRequest(BaseModel):
    image_base64: str


class VisualDescriptionResponseModel(BaseModel):
    description: str | None
    figure_type: str | None = None
    components: list[str] = []
    connections: list[str] = []


class TableReparseResponseModel(BaseModel):
    markdown: str | None
    rows: list[dict] = []


class OCRPageResponseModel(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def create_app(pipeline: PaddleOCRVLPipeline | None = None) -> FastAPI:
    """`pipeline` is injectable so tests can pass a fake/mock instead of a
    real `PaddleOCRVLPipeline` (which would start a real idle-unload thread
    and, on first inference request, try to load a real GPU model)."""
    if pipeline is None:
        pipeline = PaddleOCRVLPipeline(get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            pipeline.shutdown()

    app = FastAPI(title="paddleocr-vl-service", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=pipeline.is_loaded)

    @app.post("/describe_figure", response_model=VisualDescriptionResponseModel)
    async def describe_figure(request: ImageRequest) -> VisualDescriptionResponseModel:
        result = pipeline.describe_figure(request.image_base64)
        return VisualDescriptionResponseModel(
            description=result.description,
            figure_type=result.figure_type,
            components=result.components,
            connections=result.connections,
        )

    @app.post("/reparse_table", response_model=TableReparseResponseModel)
    async def reparse_table(request: ImageRequest) -> TableReparseResponseModel:
        result = pipeline.reparse_table(request.image_base64)
        return TableReparseResponseModel(markdown=result.markdown, rows=result.rows)

    @app.post("/ocr_page", response_model=OCRPageResponseModel)
    async def ocr_page(request: ImageRequest) -> OCRPageResponseModel:
        result = pipeline.ocr_page(request.image_base64)
        return OCRPageResponseModel(text=result.text)

    return app


app = create_app()
