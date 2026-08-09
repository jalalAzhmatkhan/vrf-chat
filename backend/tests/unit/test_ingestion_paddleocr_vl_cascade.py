"""Unit tests for `app/ingestion/paddleocr_vl_cascade.py` (I1.4).

Real local PaddleOCR-VL inference (`_build_local_pipeline`, `LocalPaddleOCRVLClient
._run_predict`) is `# pragma: no cover` (see module docstring "Coverage/testing
trade-off") — everything else, including the `remote_api` HTTP contract (via
`httpx.MockTransport`, no real network), is exercised here with no GPU/model
download required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pymupdf
import pytest

from app.core.config import Settings
from app.ingestion import paddleocr_vl_cascade as pv
from app.ingestion.cascade_trigger import (
    TASK_FULL_PAGE_OCR,
    TASK_TABLE_REPARSE,
    TASK_VISUAL_DESCRIPTION,
    CascadePlan,
    CascadeTask,
)
from app.ingestion.docling_parser import DoclingParser, ElementDraft


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render_bbox_crop
# ---------------------------------------------------------------------------


def test_render_bbox_crop_full_page(tmp_path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "Hello world")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()

    image = pv.render_bbox_crop(pdf_path, page_number=1)

    assert image[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_render_bbox_crop_with_bbox(tmp_path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "Hello world")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()

    bbox = {"l": 50.0, "t": 700.0, "r": 200.0, "b": 650.0}
    image = pv.render_bbox_crop(pdf_path, page_number=1, bbox=bbox, dpi=100)

    assert image[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Pure mapping helpers
# ---------------------------------------------------------------------------


def test_first_prediction_to_dict_none() -> None:
    assert pv._first_prediction_to_dict(None) == {}


def test_first_prediction_to_dict_json_property() -> None:
    prediction = SimpleNamespace(json={"markdown": "# Title"})
    assert pv._first_prediction_to_dict(prediction) == {"markdown": "# Title"}


def test_first_prediction_to_dict_markdown_attribute() -> None:
    prediction = SimpleNamespace(markdown="# Title", json=None)
    assert pv._first_prediction_to_dict(prediction) == {"markdown": "# Title"}


def test_first_prediction_to_dict_plain_dict() -> None:
    assert pv._first_prediction_to_dict({"text": "hi"}) == {"text": "hi"}


def test_first_prediction_to_dict_unrecognized_object() -> None:
    assert pv._first_prediction_to_dict(object()) == {}


def test_extract_markdown_text_string() -> None:
    assert pv._extract_markdown_text({"markdown": "# Title"}) == "# Title"


def test_extract_markdown_text_dict_markdown_texts() -> None:
    assert pv._extract_markdown_text({"markdown": {"markdown_texts": "hi"}}) == "hi"


def test_extract_markdown_text_dict_text_key() -> None:
    assert pv._extract_markdown_text({"markdown": {"text": "hi"}}) == "hi"


def test_extract_markdown_text_dict_no_known_key() -> None:
    assert pv._extract_markdown_text({"markdown": {"other": "x"}}) is None


def test_extract_markdown_text_fallback_top_level_text() -> None:
    assert pv._extract_markdown_text({"text": "top level"}) == "top level"


def test_extract_markdown_text_nothing_found() -> None:
    assert pv._extract_markdown_text({}) is None


# ---------------------------------------------------------------------------
# LocalPaddleOCRVLClient
# ---------------------------------------------------------------------------


def test_local_client_rejects_invalid_device() -> None:
    # Real `Settings` already restricts this via `Literal["cuda", "cpu"]` at
    # the pydantic layer — this guard exercises the defensive belt-and-braces
    # check in `LocalPaddleOCRVLClient.__init__` itself (e.g. if constructed
    # directly with a non-Settings config object).
    fake_settings = SimpleNamespace(
        PADDLE_OCR_VL_DEVICE="tpu",
        PADDLE_OCR_VL_BATCH_SIZE=1,
        PADDLE_OCR_VL_DTYPE="fp16",
    )
    with pytest.raises(pv.PaddleOCRVLConfigError):
        pv.LocalPaddleOCRVLClient(fake_settings)  # type: ignore[arg-type]


def test_local_client_warns_on_batch_size_above_one(caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings(PADDLE_OCR_VL_BATCH_SIZE=4)
    pv.LocalPaddleOCRVLClient(settings)  # should not raise, only warn


def test_local_client_warns_on_non_fp16_cuda_dtype() -> None:
    settings = _settings(PADDLE_OCR_VL_DEVICE="cuda", PADDLE_OCR_VL_DTYPE="fp32")
    pv.LocalPaddleOCRVLClient(settings)  # should not raise, only warn


def test_local_client_describe_reparse_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    client = pv.LocalPaddleOCRVLClient(settings)
    assert client.is_loaded is False

    monkeypatch.setattr(client, "_run_predict", lambda image: {"markdown": "# Fan icon"})

    visual = client.describe_figure(b"png-bytes")
    assert visual.description == "# Fan icon"
    assert visual.raw == {"markdown": "# Fan icon"}

    table = client.reparse_table(b"png-bytes")
    assert table.markdown == "# Fan icon"

    ocr = client.ocr_page(b"png-bytes")
    assert ocr.text == "# Fan icon"


def test_local_client_ocr_page_empty_text_defaults_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    client = pv.LocalPaddleOCRVLClient(settings)
    monkeypatch.setattr(client, "_run_predict", lambda image: {})

    ocr = client.ocr_page(b"png-bytes")
    assert ocr.text == ""


def _track_empty_cache(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    called = {"empty_cache": False}

    def _mark() -> None:
        called["empty_cache"] = True

    monkeypatch.setattr(pv.torch.cuda, "empty_cache", _mark)
    return called


def test_local_client_unload_cpu_skips_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _track_empty_cache(monkeypatch)
    settings = _settings(PADDLE_OCR_VL_DEVICE="cpu")
    client = pv.LocalPaddleOCRVLClient(settings)
    client._pipeline = object()

    client.unload()

    assert client.is_loaded is False
    assert called["empty_cache"] is False


def test_local_client_unload_cuda_empties_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _track_empty_cache(monkeypatch)
    monkeypatch.setattr(pv.torch.cuda, "is_available", lambda: True)
    settings = _settings(PADDLE_OCR_VL_DEVICE="cuda")
    client = pv.LocalPaddleOCRVLClient(settings)
    client._pipeline = object()

    client.unload()

    assert called["empty_cache"] is True


def test_local_client_ensure_loaded_builds_once(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = []

    def fake_build(settings: Settings) -> object:
        build_calls.append(settings.PADDLE_OCR_VL_DEVICE)
        return object()

    monkeypatch.setattr(pv, "_build_local_pipeline", fake_build)
    settings = _settings()
    client = pv.LocalPaddleOCRVLClient(settings)

    client._ensure_loaded()
    client._ensure_loaded()

    assert len(build_calls) == 1
    assert client.is_loaded is True


# ---------------------------------------------------------------------------
# RemoteAPIPaddleOCRVLClient
# ---------------------------------------------------------------------------


def test_remote_client_requires_api_url() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="remote_api", PADDLE_OCR_VL_API_URL=None)
    with pytest.raises(pv.PaddleOCRVLConfigError):
        pv.RemoteAPIPaddleOCRVLClient(settings)


def _mock_client(handler) -> pv.RemoteAPIPaddleOCRVLClient:
    settings = _settings(
        PADDLE_OCR_VL_BACKEND="remote_api",
        PADDLE_OCR_VL_API_URL="https://paddle-vl.example.com",
        PADDLE_OCR_VL_API_KEY="secret-key",
    )
    transport = httpx.MockTransport(handler)
    return pv.RemoteAPIPaddleOCRVLClient(settings, transport=transport)


def test_remote_client_describe_figure_success() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "description": "Fan mode icon",
                "figure_type": "icon",
                "components": ["fan"],
                "connections": [],
            },
        )

    client = _mock_client(handler)
    result = client.describe_figure(b"png-bytes")

    assert result.description == "Fan mode icon"
    assert result.figure_type == "icon"
    assert result.components == ["fan"]
    assert captured["path"] == "/describe_figure"
    assert captured["auth"] == "Bearer secret-key"


def test_remote_client_reparse_table_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markdown": "| A |", "rows": [{"A": "1"}]})

    client = _mock_client(handler)
    result = client.reparse_table(b"png-bytes")

    assert result.markdown == "| A |"
    assert result.rows == [{"A": "1"}]


def test_remote_client_ocr_page_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "scanned text"})

    client = _mock_client(handler)
    result = client.ocr_page(b"png-bytes")

    assert result.text == "scanned text"


def test_remote_client_ocr_page_missing_text_defaults_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _mock_client(handler)
    result = client.ocr_page(b"png-bytes")

    assert result.text == ""


def test_remote_client_http_error_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = _mock_client(handler)
    with pytest.raises(pv.PaddleOCRVLRemoteError, match="HTTP 500"):
        client.describe_figure(b"png-bytes")


def test_remote_client_transport_error_raises_after_exhausting_retries() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    client = _mock_client(handler)
    with pytest.raises(
        pv.PaddleOCRVLRemoteError, match="failed after 3 attempt\\(s\\)"
    ):
        client.describe_figure(b"png-bytes")

    # 1 initial attempt + PADDLE_OCR_VL_MAX_RETRIES (default 2) retries.
    assert calls["count"] == 3


def test_remote_client_recovers_after_transient_timeout() -> None:
    """[I1.10 live finding] a request that times out once (e.g. the service
    was mid idle-unload-then-reload) should succeed on retry rather than
    failing the whole cascade task outright."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"text": "recovered"})

    client = _mock_client(handler)
    result = client.ocr_page(b"png-bytes")

    assert result.text == "recovered"
    assert calls["count"] == 2


def test_remote_client_unload_closes_http_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _mock_client(handler)
    client.unload()
    assert client._client.is_closed is True


def test_remote_client_without_api_key_has_no_auth_header() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"text": "hi"})

    settings = _settings(
        PADDLE_OCR_VL_BACKEND="remote_api",
        PADDLE_OCR_VL_API_URL="https://paddle-vl.example.com",
        PADDLE_OCR_VL_API_KEY=None,
    )
    client = pv.RemoteAPIPaddleOCRVLClient(settings, transport=httpx.MockTransport(handler))
    client.ocr_page(b"png-bytes")

    assert captured["auth"] is None


# ---------------------------------------------------------------------------
# Factory + config validation + VRAM coordination guard
# ---------------------------------------------------------------------------


def test_build_client_local_backend() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="local")
    client = pv.build_paddleocr_vl_client(settings)
    assert isinstance(client, pv.LocalPaddleOCRVLClient)


def test_build_client_remote_backend() -> None:
    settings = _settings(
        PADDLE_OCR_VL_BACKEND="remote_api",
        PADDLE_OCR_VL_API_URL="https://paddle-vl.example.com",
    )
    client = pv.build_paddleocr_vl_client(settings)
    assert isinstance(client, pv.RemoteAPIPaddleOCRVLClient)
    client.unload()


def test_build_client_unknown_backend_raises() -> None:
    fake_settings = SimpleNamespace(PADDLE_OCR_VL_BACKEND="carrier-pigeon")
    with pytest.raises(pv.PaddleOCRVLConfigError):
        pv.build_paddleocr_vl_client(fake_settings)  # type: ignore[arg-type]


def test_validate_config_unknown_backend_raises() -> None:
    fake_settings = SimpleNamespace(PADDLE_OCR_VL_BACKEND="carrier-pigeon")
    with pytest.raises(pv.PaddleOCRVLConfigError):
        pv.validate_paddleocr_vl_config_or_raise(fake_settings)  # type: ignore[arg-type]


def test_validate_config_remote_missing_url_raises() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="remote_api", PADDLE_OCR_VL_API_URL=None)
    with pytest.raises(pv.PaddleOCRVLConfigError):
        pv.validate_paddleocr_vl_config_or_raise(settings)


def test_validate_config_local_backend_ok() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="local")
    pv.validate_paddleocr_vl_config_or_raise(settings)  # should not raise


def test_require_docling_unloaded_noop_for_remote_backend() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="remote_api")
    parser = DoclingParser(device="cpu")
    parser._converter = object()  # pretend loaded
    pv.require_docling_unloaded_before_paddle_stage(parser, settings)  # no raise


def test_require_docling_unloaded_noop_when_flag_disabled() -> None:
    settings = _settings(
        PADDLE_OCR_VL_BACKEND="local", DOCLING_UNLOAD_BEFORE_PADDLE_STAGE=False
    )
    parser = DoclingParser(device="cpu")
    parser._converter = object()
    pv.require_docling_unloaded_before_paddle_stage(parser, settings)  # no raise


def test_require_docling_unloaded_noop_when_parser_none() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="local")
    pv.require_docling_unloaded_before_paddle_stage(None, settings)  # no raise


def test_require_docling_unloaded_passes_when_unloaded() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="local")
    parser = DoclingParser(device="cpu")
    assert parser.is_loaded is False
    pv.require_docling_unloaded_before_paddle_stage(parser, settings)  # no raise


def test_require_docling_unloaded_raises_when_still_loaded() -> None:
    settings = _settings(PADDLE_OCR_VL_BACKEND="local")
    parser = DoclingParser(device="cpu")
    parser._converter = object()  # pretend loaded

    with pytest.raises(pv.PaddleOCRVLConfigError, match="DoclingParser is still loaded"):
        pv.require_docling_unloaded_before_paddle_stage(parser, settings)


# ---------------------------------------------------------------------------
# run_cascade_stage orchestration
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def describe_figure(self, image_png: bytes) -> pv.VisualDescription:
        self.calls.append("describe_figure")
        return pv.VisualDescription(description="a figure")

    def reparse_table(self, image_png: bytes) -> pv.TableReparseResult:
        self.calls.append("reparse_table")
        return pv.TableReparseResult(markdown="| a |")

    def ocr_page(self, image_png: bytes) -> pv.OCRPageResult:
        self.calls.append("ocr_page")
        return pv.OCRPageResult(text="ocr text")

    def unload(self) -> None:
        self.calls.append("unload")


def _sample_pdf(tmp_path) -> str:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "hello")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_run_cascade_stage_dispatches_all_task_types(tmp_path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    element = ElementDraft(
        local_id=1,
        element_type="table",
        text=None,
        bbox={"l": 10.0, "t": 700.0, "r": 100.0, "b": 650.0},
        page_number=1,
        parent_local_id=None,
        section_path=[],
        extraction_method="docling",
        extraction_confidence=0.5,
    )
    plan = CascadePlan(
        tasks=[
            CascadeTask(
                task_type=TASK_TABLE_REPARSE, page_number=1, element_local_id=1, reason="x"
            ),
            CascadeTask(
                task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=None, reason="x"
            ),
            CascadeTask(
                task_type=TASK_FULL_PAGE_OCR, page_number=1, element_local_id=None, reason="x"
            ),
        ]
    )
    client = _FakeClient()

    results = pv.run_cascade_stage(plan, pdf_path, client, elements_by_local_id={1: element})

    assert client.calls == ["reparse_table", "describe_figure", "ocr_page"]
    assert results[0].table_reparse is not None
    assert results[1].visual_description is not None
    assert results[2].ocr_page is not None


def test_run_cascade_stage_ignores_unknown_task_type(tmp_path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    plan = CascadePlan(
        tasks=[
            CascadeTask(task_type="unknown_task", page_number=1, element_local_id=None, reason="x"),
        ]
    )
    client = _FakeClient()

    results = pv.run_cascade_stage(plan, pdf_path, client)

    assert results == []
    assert client.calls == []


def test_run_cascade_stage_element_missing_from_map_uses_full_page(tmp_path) -> None:
    pdf_path = _sample_pdf(tmp_path)
    plan = CascadePlan(
        tasks=[
            CascadeTask(
                task_type=TASK_TABLE_REPARSE, page_number=1, element_local_id=99, reason="x"
            ),
        ]
    )
    client = _FakeClient()

    results = pv.run_cascade_stage(plan, pdf_path, client)

    assert len(results) == 1
    assert results[0].table_reparse is not None
