"""Unit tests for `app/pipeline.py`.

Real model construction (`_build_pipeline`) and real inference (`_predict`,
via `_decode_image`) are `# pragma: no cover` — genuinely GPU/model-only
work, live-verified separately (see README.md for the full transcript: real
pipeline construction + real inference succeeded, ~5.98GB peak VRAM, no
`torch` dependency needed at all). Everything else (idle-unload
orchestration, response mapping) is unit tested here via monkeypatching
`_build_pipeline`.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from app import pipeline as pl
from app.config import Settings


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class _FakePaddleModel:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result if result is not None else {"markdown": {"markdown_texts": "hi"}}
        self.predict_calls = 0

    def predict(self, image: Any) -> list[Any]:
        self.predict_calls += 1
        from types import SimpleNamespace

        return [SimpleNamespace(json=self.result)]


# ---------------------------------------------------------------------------
# _first_prediction_to_dict / _extract_markdown_text
# ---------------------------------------------------------------------------


def test_first_prediction_to_dict_none() -> None:
    assert pl._first_prediction_to_dict(None) == {}


def test_first_prediction_to_dict_json_property() -> None:
    from types import SimpleNamespace

    prediction = SimpleNamespace(json={"markdown": "hi"})
    assert pl._first_prediction_to_dict(prediction) == {"markdown": "hi"}


def test_first_prediction_to_dict_markdown_attribute() -> None:
    from types import SimpleNamespace

    prediction = SimpleNamespace(markdown="hi", json=None)
    assert pl._first_prediction_to_dict(prediction) == {"markdown": "hi"}


def test_first_prediction_to_dict_plain_dict() -> None:
    assert pl._first_prediction_to_dict({"text": "hi"}) == {"text": "hi"}


def test_first_prediction_to_dict_unrecognized() -> None:
    assert pl._first_prediction_to_dict(object()) == {}


def test_first_prediction_to_dict_prefers_markdown_over_json_without_markdown_key() -> None:
    """Regression test for the SA1.2 finding: a real PaddleX `Result.json`
    is always `{"res": {...}}` (never has a top-level "markdown"/"text"
    key) — `.markdown` must be checked first, or the real output is
    silently discarded (see `_first_prediction_to_dict` docstring)."""
    from types import SimpleNamespace

    prediction = SimpleNamespace(
        json={"res": {"parsing_res_list": []}},
        markdown={"markdown_texts": "## real content", "page_index": 0},
    )
    assert pl._first_prediction_to_dict(prediction) == {
        "markdown": {"markdown_texts": "## real content", "page_index": 0}
    }


def test_extract_markdown_text_dict_markdown_texts() -> None:
    assert pl._extract_markdown_text({"markdown": {"markdown_texts": "hi"}}) == "hi"


def test_extract_markdown_text_string() -> None:
    assert pl._extract_markdown_text({"markdown": "hi"}) == "hi"


def test_extract_markdown_text_fallback_text_key() -> None:
    assert pl._extract_markdown_text({"text": "hi"}) == "hi"


def test_extract_markdown_text_nothing_found() -> None:
    assert pl._extract_markdown_text({}) is None


# ---------------------------------------------------------------------------
# PaddleOCRVLPipeline orchestration
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_builder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _fake_build(settings: Settings) -> _FakePaddleModel:
        calls.append(settings.PADDLE_OCR_VL_SERVICE_DEVICE)
        return _FakePaddleModel()

    monkeypatch.setattr(pl, "_build_pipeline", _fake_build)
    return calls


def test_pipeline_lazy_loads_once(fake_builder: list[str]) -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings())
    try:
        assert pipeline.is_loaded is False
        pipeline._ensure_loaded()
        pipeline._ensure_loaded()
        assert len(fake_builder) == 1
        assert pipeline.is_loaded is True
    finally:
        pipeline.shutdown()


def test_pipeline_describe_figure_reparse_table_ocr_page(
    fake_builder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings())
    try:
        monkeypatch.setattr(pipeline, "_predict", lambda image_base64: {"text": "hello"})

        visual = pipeline.describe_figure("base64data")
        assert visual.description == "hello"

        table = pipeline.reparse_table("base64data")
        assert table.markdown == "hello"

        ocr = pipeline.ocr_page("base64data")
        assert ocr.text == "hello"
    finally:
        pipeline.shutdown()


def test_pipeline_ocr_page_empty_defaults_to_empty_string(
    fake_builder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings())
    try:
        monkeypatch.setattr(pipeline, "_predict", lambda image_base64: {})
        assert pipeline.ocr_page("x").text == ""
    finally:
        pipeline.shutdown()


def test_maybe_unload_if_idle_noop_when_not_loaded() -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings(PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS=0))
    try:
        pipeline._maybe_unload_if_idle()
        assert pipeline.is_loaded is False
    finally:
        pipeline.shutdown()


def test_maybe_unload_if_idle_unloads_after_threshold(
    fake_builder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    released = {"called": False}
    monkeypatch.setattr(pl, "_release_gpu_memory", lambda: released.__setitem__("called", True))

    pipeline = pl.PaddleOCRVLPipeline(_settings(PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS=0))
    try:
        pipeline._ensure_loaded()
        assert pipeline.is_loaded is True
        time.sleep(0.01)
        pipeline._maybe_unload_if_idle()
        assert pipeline.is_loaded is False
        assert released["called"] is True
    finally:
        pipeline.shutdown()


def test_maybe_unload_if_idle_keeps_model_when_recently_used(fake_builder: list[str]) -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings(PADDLE_OCR_VL_SERVICE_IDLE_UNLOAD_SECONDS=300))
    try:
        pipeline._ensure_loaded()
        pipeline._maybe_unload_if_idle()
        assert pipeline.is_loaded is True
    finally:
        pipeline.shutdown()


def test_shutdown_stops_idle_watcher_thread(fake_builder: list[str]) -> None:
    pipeline = pl.PaddleOCRVLPipeline(_settings())
    assert pipeline._idle_watcher.is_alive() is True
    pipeline.shutdown()
    pipeline._idle_watcher.join(timeout=15)
    assert pipeline._idle_watcher.is_alive() is False
