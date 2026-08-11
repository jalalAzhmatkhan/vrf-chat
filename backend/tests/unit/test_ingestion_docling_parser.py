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


def test_icon_without_text_on_same_page_falls_back_to_last_text_seen_on_earlier_page() -> None:
    """[SA1.2 fix, real bug found in document_id=3] A page consisting
    entirely of icons (e.g. a pictogram legend continuing from the previous
    page, zero paragraph/list/heading elements of its own) must NOT get
    `parent_id=NULL` — it should fall back to the last text/heading element
    seen anywhere earlier in the document, matching the real "page 8 of
    Zeggo VRV IV REYQ" scenario (6 icons, all orphaned before this fix)."""
    items = [
        _item("section_header", "#/texts/0", text="Pictograms", page_no=1, level=1),
        _item("text", "#/texts/1", text="See legend below.", page_no=1),
        # Page 2: nothing but icons — no text/list/heading element at all.
        _item("picture", "#/pictures/0", page_no=2, bbox=(10, 780, 30, 765)),
        _item("picture", "#/pictures/1", page_no=2, bbox=(40, 780, 60, 765)),
    ]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0), 2: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores(), 2: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=2)

    heading, paragraph, icon_1, icon_2 = result.elements
    assert icon_1.element_type == "icon"
    assert icon_2.element_type == "icon"
    # Falls back to the last text/heading element seen anywhere earlier in
    # the document (the page-1 paragraph), not NULL.
    assert icon_1.parent_local_id == paragraph.local_id
    assert icon_2.parent_local_id == paragraph.local_id


def test_icon_before_any_text_in_document_stays_parentless() -> None:
    """No text/heading has been seen anywhere yet (same-page AND
    document-wide fallback both empty) — `parent_local_id` correctly stays
    `None` rather than erroring."""
    items = [_item("picture", "#/pictures/0", page_no=1, bbox=(10, 780, 30, 765))]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    confidence = _confidence_report({1: _page_scores()})

    result = dp.map_document_to_elements(doc, confidence, page_count=1)

    assert result.elements[0].element_type == "icon"
    assert result.elements[0].parent_local_id is None


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
# Page-range batching equivalence (2026-08-11 OOM fix) — CLAUDE.md §4:
# batched output MUST be identical to a single unbatched pass, especially
# the icon<->text association across page/batch boundaries (SA1.2).
# ---------------------------------------------------------------------------


def _as_comparable(elements: list[dp.ElementDraft]) -> list[tuple[Any, ...]]:
    """Project the fields that must be identical between a single-pass and a
    batched-then-merged run (excludes nothing — every field matters for
    traceability/parent-linking, see module docstring)."""
    return [
        (
            e.local_id,
            e.element_type,
            e.text,
            e.page_number,
            e.parent_local_id,
            tuple(e.section_path),
        )
        for e in elements
    ]


def test_batched_parse_matches_single_pass_icon_fallback_across_batch_boundary() -> None:
    """The exact SA1.2 scenario (page of icons with no text of its own,
    falling back to the last text/heading seen earlier in the document) but
    with the batch boundary placed BETWEEN the two pages — i.e. the
    heading/paragraph are parsed in batch 1 and the icon-only page is
    parsed in batch 2, as `_run_docling_in_batches` would do it. Batched
    result must match a single unbatched pass over both pages exactly."""
    page1_items = [
        _item("section_header", "#/texts/0", text="Pictograms", page_no=1, level=1),
        _item("text", "#/texts/1", text="See legend below.", page_no=1),
    ]
    page2_items = [
        _item("picture", "#/pictures/0", page_no=2, bbox=(10, 780, 30, 765)),
        _item("picture", "#/pictures/1", page_no=2, bbox=(40, 780, 60, 765)),
    ]
    page_sizes = {1: (609.0, 793.0), 2: (609.0, 793.0)}

    # --- Single unbatched pass over both pages ---
    single_doc = FakeDoc(page1_items + page2_items, page_sizes=page_sizes)
    single_confidence = _confidence_report({1: _page_scores(), 2: _page_scores()})
    single_result = dp.map_document_to_elements(single_doc, single_confidence, page_count=2)

    # --- Batched: batch 1 = page 1 only, batch 2 = page 2 only ---
    batch1_doc = FakeDoc(page1_items, page_sizes={1: page_sizes[1]})
    batch1_confidence = _confidence_report({1: _page_scores()})
    batch1_result = dp.map_document_to_elements(
        batch1_doc, batch1_confidence, page_count=1, page_range=(1, 1)
    )

    batch2_doc = FakeDoc(page2_items, page_sizes={2: page_sizes[2]})
    batch2_confidence = _confidence_report({2: _page_scores()})
    batch2_result = dp.map_document_to_elements(
        batch2_doc,
        batch2_confidence,
        page_count=2,
        page_range=(2, 2),
        carry_state=batch1_result.carry_state,
    )

    merged_elements = batch1_result.elements + batch2_result.elements
    merged_confidence = {**batch1_result.page_confidence, **batch2_result.page_confidence}

    assert _as_comparable(merged_elements) == _as_comparable(single_result.elements)
    assert merged_confidence == single_result.page_confidence

    # The critical assertion (CLAUDE.md §4): both icons on page 2 (a
    # separate BATCH from the text that should parent them) still resolve
    # to the page-1 paragraph, not NULL.
    icon_1, icon_2 = batch2_result.elements
    paragraph = batch1_result.elements[1]
    assert icon_1.parent_local_id == paragraph.local_id
    assert icon_2.parent_local_id == paragraph.local_id


def test_batched_parse_matches_single_pass_section_path_across_batch_boundary() -> None:
    """A heading opened in batch 1 must still be the `section_path` for
    body text parsed in batch 2, with no heading of its own in that batch."""
    page1_items = [
        _item("title", "#/texts/0", text="Chapter 7", page_no=1, level=1),
        _item("section_header", "#/texts/1", text="7.3 Troubleshooting", page_no=1, level=2),
        _item("text", "#/texts/2", text="Body text under 7.3, page 1.", page_no=1),
    ]
    page2_items = [
        _item("text", "#/texts/0", text="Continued body text, page 2.", page_no=2),
    ]
    page_sizes = {1: (609.0, 793.0), 2: (609.0, 793.0)}

    single_doc = FakeDoc(page1_items + page2_items, page_sizes=page_sizes)
    single_confidence = _confidence_report({1: _page_scores(), 2: _page_scores()})
    single_result = dp.map_document_to_elements(single_doc, single_confidence, page_count=2)

    batch1_doc = FakeDoc(page1_items, page_sizes={1: page_sizes[1]})
    batch1_result = dp.map_document_to_elements(
        batch1_doc,
        _confidence_report({1: _page_scores()}),
        page_count=1,
        page_range=(1, 1),
    )
    batch2_doc = FakeDoc(page2_items, page_sizes={2: page_sizes[2]})
    batch2_result = dp.map_document_to_elements(
        batch2_doc,
        _confidence_report({2: _page_scores()}),
        page_count=2,
        page_range=(2, 2),
        carry_state=batch1_result.carry_state,
    )

    merged_elements = batch1_result.elements + batch2_result.elements
    assert _as_comparable(merged_elements) == _as_comparable(single_result.elements)
    assert batch2_result.elements[0].section_path == ["Chapter 7", "7.3 Troubleshooting"]
    # local_id stayed globally unique/monotonic across the batch boundary.
    assert batch2_result.elements[0].local_id == batch1_result.elements[-1].local_id + 1


def test_carry_state_defaults_match_pre_batching_behavior() -> None:
    """No `carry_state`/`page_range` given (the pre-2026-08-11 call shape,
    still used by every non-batched call site) behaves exactly as before:
    `local_id` starts at 1, `section_path` starts empty, no document-wide
    icon-parent fallback available yet."""
    items = [_item("picture", "#/pictures/0", page_no=1, bbox=(10, 780, 30, 765))]
    doc = FakeDoc(items, page_sizes={1: (609.0, 793.0)})
    result = dp.map_document_to_elements(doc, _confidence_report({1: _page_scores()}), page_count=1)

    assert result.elements[0].local_id == 1
    assert result.elements[0].parent_local_id is None
    assert result.carry_state.next_local_id == 2
    assert result.carry_state.section_path == []
    assert result.carry_state.last_text_local_id_doc is None


def test_docling_parser_parse_forwards_page_range_and_carry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert_calls: list[dict[str, Any]] = []

    class FakeResult:
        document = FakeDoc(
            [_item("text", "#/texts/0", text="hi", page_no=5)],
            page_sizes={5: (609.0, 793.0)},
        )
        confidence = _confidence_report({5: _page_scores()})

    class FakeConverter:
        def convert(self, path: str, **kwargs: Any) -> FakeResult:
            convert_calls.append(kwargs)
            return FakeResult()

    monkeypatch.setattr(dp, "_build_converter", lambda device: FakeConverter())

    parser = dp.DoclingParser(device="cpu")
    carry_in = dp.ParseCarryState(next_local_id=42, section_path=["Ch1"], last_text_local_id_doc=7)
    result = parser.parse("dummy.pdf", page_range=(5, 5), carry_state=carry_in)

    assert convert_calls == [{"page_range": (5, 5)}]
    assert result.elements[0].local_id == 42  # carried, not reset to 1
    assert result.elements[0].page_number == 5


def test_docling_parser_parse_without_page_range_omits_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility: the default (unbatched) call shape must not
    pass `page_range` to `converter.convert()` at all, matching the exact
    call signature used before this change (some real Docling `convert()`
    overloads/mocks in existing tests only accept a bare path)."""
    convert_calls: list[tuple[Any, ...]] = []

    class FakeResult:
        document = FakeDoc([_item("text", "#/texts/0", text="hi")], page_sizes={1: (609.0, 793.0)})
        confidence = _confidence_report({1: _page_scores()})

    class FakeConverter:
        def convert(self, path: str) -> FakeResult:  # no **kwargs on purpose
            convert_calls.append((path,))
            return FakeResult()

    monkeypatch.setattr(dp, "_build_converter", lambda device: FakeConverter())

    parser = dp.DoclingParser(device="cpu")
    result = parser.parse("dummy.pdf")

    assert convert_calls == [("dummy.pdf",)]
    assert result.elements[0].local_id == 1


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


def test_docling_parser_parse_cuda_empties_cache_after_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[2026-08-11 OOM fix] `parse()` itself (not just `unload()`) releases
    the CUDA cache after every call — this is the per-batch memory release
    `_run_docling_in_batches` relies on when `DOCLING_DEVICE=cuda`."""
    empty_cache_calls = {"count": 0}
    monkeypatch.setattr(dp.torch.cuda, "empty_cache", lambda: empty_cache_calls.__setitem__(
        "count", empty_cache_calls["count"] + 1
    ))
    monkeypatch.setattr(dp.torch.cuda, "is_available", lambda: True)

    class FakeResult:
        document = FakeDoc(
            [_item("text", "#/texts/0", text="hi")], page_sizes={1: (609.0, 793.0)}
        )
        confidence = _confidence_report({1: _page_scores()})

    class FakeConverter:
        def convert(self, path: str, **kwargs: Any) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(dp, "_build_converter", lambda device: FakeConverter())

    parser = dp.DoclingParser(device="cuda")
    parser.parse("dummy.pdf")
    parser.parse("dummy.pdf")

    assert empty_cache_calls["count"] == 2  # once per parse() call, not just on unload()


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
