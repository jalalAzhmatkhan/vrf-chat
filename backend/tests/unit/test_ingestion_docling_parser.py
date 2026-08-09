"""Unit tests for `app/ingestion/docling_parser.py` (I1.2).

`map_document_to_elements` is exercised with fake, duck-typed objects that
mimic the small slice of Docling's `DoclingDocument`/`ConfidenceReport` API
this module actually reads (`iterate_items()`, `.label`/`.text`/`.prov`/
`.self_ref`/`.captions`/`.level`, `doc.pages[n].size`) — this avoids any
GPU/model download in the unit test suite. A real end-to-end conversion
against `source-documents/` was verified manually (not part of the
automated suite, no source-documents dependency in CI) — see the I1.2
STATUS REPORT for the live-run notes (including a Windows-native
`TORCHDYNAMO_DISABLE=1` gotcha unrelated to CUDA/CPU device selection).

`DoclingParser.parse()`'s orchestration (model load caching + calling
`converter.convert()`) is tested via a fake converter injected by
monkeypatching `_build_converter`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from app.ingestion import docling_parser as dp

# ---------------------------------------------------------------------------
# Fakes mimicking the Docling object model (only fields this module reads)
# ---------------------------------------------------------------------------


@dataclass
class FakeBBox:
    l: float  # noqa: E741 — mirrors Docling's real `BoundingBox.l` (left) field name
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"


@dataclass
class FakeProvenance:
    page_no: int
    bbox: FakeBBox


@dataclass
class FakeRef:
    cref: str


@dataclass
class FakeItem:
    label: str
    self_ref: str
    text: str | None = None
    prov: list[FakeProvenance] = field(default_factory=list)
    level: int | None = None
    captions: list[FakeRef] = field(default_factory=list)
    _markdown: str | None = None

    def export_to_markdown(self, doc: Any = None) -> str | None:
        return self._markdown


def _item(
    label: str,
    ref: str,
    *,
    text: str | None = None,
    page_no: int = 1,
    bbox: tuple[float, float, float, float] | None = (50, 700, 550, 650),
    level: int | None = None,
    captions: list[str] | None = None,
    markdown: str | None = None,
) -> FakeItem:
    prov = [FakeProvenance(page_no=page_no, bbox=FakeBBox(*bbox))] if bbox is not None else []
    return FakeItem(
        label=label,
        self_ref=ref,
        text=text,
        prov=prov,
        level=level,
        captions=[FakeRef(cref=c) for c in (captions or [])],
        _markdown=markdown,
    )


class FakeDoc:
    def __init__(self, items: list[FakeItem], page_sizes: dict[int, tuple[float, float]]):
        self._items = items
        self.pages = {
            page_no: SimpleNamespace(size=SimpleNamespace(width=w, height=h))
            for page_no, (w, h) in page_sizes.items()
        }

    def iterate_items(self):
        for item in self._items:
            yield item, 1


def _page_scores(**overrides: float) -> SimpleNamespace:
    defaults = dict(
        parse_score=1.0,
        layout_score=0.95,
        table_score=math.nan,
        ocr_score=math.nan,
        mean_score=0.97,
        mean_grade="excellent",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _confidence_report(pages: dict[int, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(pages=pages)


# ---------------------------------------------------------------------------
# Mapping logic
# ---------------------------------------------------------------------------


def test_maps_basic_label_types() -> None:
    items = [
        _item("title", "#/texts/0", text="Service Manual", level=1),
        _item("section_header", "#/texts/1", text="7.3 Troubleshooting", level=2),
        _item("text", "#/texts/2", text="Press the reset button."),
        _item("list_item", "#/texts/3", text="Step 1: check power."),
        _item("table", "#/tables/0", markdown="| A | B |\n|---|---|\n| 1 | 2 |"),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    types = [e.element_type for e in result.elements]
    assert types == ["heading", "heading", "paragraph", "list", "table"]
    table_element = result.elements[-1]
    assert table_element.text == "| A | B |\n|---|---|\n| 1 | 2 |"
    # table_score is NaN in the fixture -> falls back to mean_score.
    assert table_element.extraction_confidence == pytest.approx(0.97)


def test_section_path_breadcrumb_respects_heading_level() -> None:
    items = [
        _item("title", "#/texts/0", text="Chapter 7", level=1),
        _item("section_header", "#/texts/1", text="7.3 Troubleshooting", level=2),
        _item("text", "#/texts/2", text="Body text under 7.3"),
        _item("section_header", "#/texts/3", text="7.4 Next Section", level=2),
        _item("text", "#/texts/4", text="Body text under 7.4"),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[2].section_path == ["Chapter 7", "7.3 Troubleshooting"]
    assert result.elements[4].section_path == ["Chapter 7", "7.4 Next Section"]


def test_small_picture_reclassified_as_icon_and_parented_to_preceding_text() -> None:
    """CLAUDE.md §4 — inline icon must stay associated with surrounding text."""
    items = [
        _item("text", "#/texts/0", text="Press the button", bbox=(50, 700, 400, 680)),
        _item("picture", "#/pictures/0", bbox=(410, 695, 430, 680)),  # 20x15pt -> icon-sized
        _item("text", "#/texts/1", text="to reset LOSSNAY.", bbox=(50, 670, 400, 650)),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    icon = result.elements[1]
    assert icon.element_type == "icon"
    # Parent points at the *preceding* paragraph in linear reading order.
    assert icon.parent_local_id == result.elements[0].local_id


def test_large_picture_stays_figure_not_icon() -> None:
    items = [
        _item(
            "picture",
            "#/pictures/0",
            bbox=(20, 780, 590, 100),  # near-full-page -> schematic diagram
        ),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "figure"
    assert result.elements[0].parent_local_id is None


def test_picture_without_page_size_stays_figure() -> None:
    items = [_item("picture", "#/pictures/0", bbox=(10, 10, 20, 20))]
    doc = FakeDoc(items, page_sizes={})  # no page size registered
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "figure"


def test_caption_retagged_and_parented_to_owning_figure() -> None:
    items = [
        _item(
            "picture",
            "#/pictures/0",
            bbox=(20, 780, 590, 100),
            captions=["#/texts/0"],
        ),
        _item("caption", "#/texts/0", text="Figure 7-1: Electrical schematic"),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    figure = result.elements[0]
    caption = result.elements[1]
    assert caption.element_type == "figure_caption"
    assert caption.parent_local_id == figure.local_id


def test_item_without_bbox_defaults_to_page_one() -> None:
    items = [_item("text", "#/texts/0", text="No provenance", bbox=None)]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].page_number == 1
    assert result.elements[0].bbox is None


def test_unknown_label_falls_back_to_paragraph() -> None:
    items = [_item("marker", "#/texts/0", text="???")]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "paragraph"


def test_page_confidence_missing_page_yields_all_none() -> None:
    items = [_item("text", "#/texts/0", text="hi", page_no=2)]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0), 2: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})  # page 2 missing

    result = dp.map_document_to_elements(doc, confidence, page_count=2)

    assert result.page_confidence[2].parse_score is None
    assert result.page_confidence[2].mean_grade is None
    assert result.elements[0].extraction_confidence is None


def test_table_markdown_export_exception_is_handled() -> None:
    class BrokenTable(FakeItem):
        def export_to_markdown(self, doc: Any = None) -> str | None:
            raise RuntimeError("boom")

    prov = [FakeProvenance(1, FakeBBox(0, 0, 1, 1))]
    item = BrokenTable(label="table", self_ref="#/tables/0", prov=prov)
    doc = FakeDoc([item], page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].text is None


def test_clean_score_none_stays_none() -> None:
    items = [_item("text", "#/texts/0", text="hi")]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores(ocr_score=None)})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.page_confidence[1].ocr_score is None


def test_element_page_beyond_declared_page_count_has_no_confidence() -> None:
    # Element's own bbox provenance reports a page number outside
    # `page_count` (defensive: page_confidence.get() returns None for it).
    items = [_item("text", "#/texts/0", text="stray", page_no=5)]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].extraction_confidence is None


def test_table_item_without_export_to_markdown_method() -> None:
    item = SimpleNamespace(
        label="table",
        self_ref="#/tables/0",
        prov=[FakeProvenance(1, FakeBBox(0, 0, 1, 1))],
        text=None,
    )
    doc = FakeDoc([item], page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].text is None


def test_page_item_without_size_is_skipped() -> None:
    items = [_item("text", "#/texts/0", text="hi")]
    doc = FakeDoc(items, page_sizes={})
    doc.pages[1] = SimpleNamespace(size=None)
    confidence = _confidence_report({1: _page_scores()})

    # No page size registered for page 1 -> a `picture` on that page cannot
    # be evaluated for icon-sizing and must default to `figure`.
    doc._items.append(_item("picture", "#/pictures/0", bbox=(10, 10, 20, 20)))
    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[-1].element_type == "figure"


def test_item_without_self_ref_is_not_tracked_as_last_text() -> None:
    item = SimpleNamespace(
        label="text",
        self_ref=None,
        text="no ref",
        prov=[FakeProvenance(1, FakeBBox(0, 0, 1, 1))],
        captions=[],
    )
    doc = FakeDoc([item], page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "paragraph"


def test_caption_ref_without_cref_is_ignored() -> None:
    picture = _item("picture", "#/pictures/0", bbox=(20, 780, 590, 100))
    picture.captions = [SimpleNamespace(cref=None)]
    doc = FakeDoc([picture], page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "figure"


def test_confidence_report_with_no_pages_attribute() -> None:
    items = [_item("text", "#/texts/0", text="hi")]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = SimpleNamespace()  # no `.pages` attribute at all

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.page_confidence[1].parse_score is None


# ---------------------------------------------------------------------------
# DoclingParser orchestration + unload
# ---------------------------------------------------------------------------


def test_docling_parser_rejects_invalid_device() -> None:
    with pytest.raises(dp.DoclingConfigError):
        dp.DoclingParser(device="tpu")


def test_docling_parser_parse_caches_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = []

    class FakeResult:
        document = FakeDoc(
            [_item("text", "#/texts/0", text="hi")], page_sizes={1: (609.0, 793.0)}
        )
        confidence = _confidence_report({1: _page_scores()})

    class FakeConverter:
        def convert(self, path: str) -> FakeResult:
            return FakeResult()

    def fake_build_converter(device: str) -> FakeConverter:
        build_calls.append(device)
        return FakeConverter()

    monkeypatch.setattr(dp, "_build_converter", fake_build_converter)

    parser = dp.DoclingParser(device="cpu")
    assert parser.is_loaded is False

    result1 = parser.parse("dummy.pdf")
    result2 = parser.parse("dummy.pdf")

    assert len(result1.elements) == 1
    assert len(result2.elements) == 1
    assert build_calls == ["cpu"]  # converter built once, reused across parse() calls
    assert parser.is_loaded is True
    assert parser.device == "cpu"


def _track_empty_cache(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    called = {"empty_cache": False}

    def _mark() -> None:
        called["empty_cache"] = True

    monkeypatch.setattr(dp.torch.cuda, "empty_cache", _mark)
    return called


def test_docling_parser_unload_cpu_does_not_touch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _track_empty_cache(monkeypatch)

    parser = dp.DoclingParser(device="cpu")
    parser._converter = object()  # pretend it was loaded
    parser.unload()

    assert parser.is_loaded is False
    assert called["empty_cache"] is False


def test_docling_parser_unload_cuda_empties_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    called = _track_empty_cache(monkeypatch)
    monkeypatch.setattr(dp.torch.cuda, "is_available", lambda: True)

    parser = dp.DoclingParser(device="cuda")
    parser._converter = object()
    parser.unload()

    assert parser.is_loaded is False
    assert called["empty_cache"] is True


def test_docling_parser_unload_cuda_unavailable_skips_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _track_empty_cache(monkeypatch)
    monkeypatch.setattr(dp.torch.cuda, "is_available", lambda: False)

    parser = dp.DoclingParser(device="cuda")
    parser._converter = object()
    parser.unload()

    assert called["empty_cache"] is False
