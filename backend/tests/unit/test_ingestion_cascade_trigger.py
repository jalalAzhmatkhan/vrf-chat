"""Unit tests for `app/ingestion/cascade_trigger.py` (I1.3).

Builds `DoclingParseResult`/`DocumentProbe` inputs directly from the real
dataclasses (`app.ingestion.docling_parser`/`app.ingestion.native_probe`) —
these are cheap plain dataclasses, no GPU/model/PDF dependency, so real
objects are used rather than fakes.
"""

from __future__ import annotations

from app.ingestion.cascade_trigger import (
    TASK_FULL_PAGE_OCR,
    TASK_TABLE_REPARSE,
    TASK_TYPES,
    TASK_VISUAL_DESCRIPTION,
    CascadePlan,
    build_cascade_plan,
)
from app.ingestion.docling_parser import (
    DoclingParseResult,
    ElementDraft,
    PageConfidence,
)
from app.ingestion.native_probe import (
    PAGE_FLAG_LOW_TEXT_CONFIDENCE,
    PAGE_FLAG_RASTER_IMAGE,
    PAGE_FLAG_TEXT_NORMAL,
    PAGE_FLAG_VECTOR_DIAGRAM,
    DocumentProbe,
    PageProbe,
)


def _page_probe(page_number: int, flags: list[str]) -> PageProbe:
    return PageProbe(
        page_number=page_number,
        char_count=1000,
        text_density=0.005,
        vector_drawing_count=0,
        embedded_image_count=0,
        page_width=609.0,
        page_height=793.0,
        flags=flags,
    )


def _element(
    local_id: int,
    element_type: str,
    page_number: int,
    *,
    confidence: float | None = 0.9,
) -> ElementDraft:
    return ElementDraft(
        local_id=local_id,
        element_type=element_type,
        text=None,
        bbox=None,
        page_number=page_number,
        parent_local_id=None,
        section_path=[],
        extraction_method="docling",
        extraction_confidence=confidence,
    )


def _page_conf(page_number: int, parse_score: float | None) -> PageConfidence:
    return PageConfidence(
        page_number=page_number,
        parse_score=parse_score,
        layout_score=0.95,
        table_score=None,
        ocr_score=None,
        mean_score=0.9,
        mean_grade="excellent",
    )


def test_low_confidence_table_is_queued_for_reparse() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "table", page_number=1, confidence=0.5)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe, threshold_table=0.75, threshold_text=0.6)

    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.task_type == TASK_TABLE_REPARSE
    assert task.element_local_id == 1
    assert "0.5000" in task.reason
    assert "0.7500" in task.reason


def test_high_confidence_table_is_not_queued() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "table", page_number=1, confidence=0.9)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe, threshold_table=0.75, threshold_text=0.6)

    assert plan.tasks == []


def test_table_with_none_confidence_is_queued_fail_safe() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "table", page_number=1, confidence=None)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe)

    assert len(plan.tasks) == 1
    assert "None" in plan.tasks[0].reason


def test_figure_on_vector_diagram_page_queued_for_visual_description() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "figure", page_number=1, confidence=0.99)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(
        page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL, PAGE_FLAG_VECTOR_DIAGRAM])]
    )

    plan = build_cascade_plan(parse_result, probe)

    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.task_type == TASK_VISUAL_DESCRIPTION
    assert PAGE_FLAG_VECTOR_DIAGRAM in task.reason


def test_icon_on_raster_image_page_queued_for_visual_description() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "icon", page_number=1, confidence=0.99)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(
        page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL, PAGE_FLAG_RASTER_IMAGE])]
    )

    plan = build_cascade_plan(parse_result, probe)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_type == TASK_VISUAL_DESCRIPTION


def test_figure_on_plain_text_page_not_queued() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "figure", page_number=1, confidence=0.99)],
        page_confidence={1: _page_conf(1, parse_score=1.0)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe)

    assert plan.tasks == []


def test_figure_on_page_not_present_in_probe_is_skipped() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "figure", page_number=5, confidence=0.99)],
        page_confidence={5: _page_conf(5, parse_score=1.0)},
        page_count=5,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe)

    assert plan.tasks == []


def test_low_page_text_confidence_queues_full_page_ocr() -> None:
    parse_result = DoclingParseResult(
        elements=[],
        page_confidence={1: _page_conf(1, parse_score=0.2)},
        page_count=1,
    )
    probe = DocumentProbe(
        page_count=1, pages=[_page_probe(1, [PAGE_FLAG_LOW_TEXT_CONFIDENCE])]
    )

    plan = build_cascade_plan(parse_result, probe, threshold_text=0.6)

    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.task_type == TASK_FULL_PAGE_OCR
    assert task.element_local_id is None
    assert task.page_number == 1


def test_none_page_text_confidence_queues_full_page_ocr_fail_safe() -> None:
    parse_result = DoclingParseResult(
        elements=[],
        page_confidence={1: _page_conf(1, parse_score=None)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].task_type == TASK_FULL_PAGE_OCR


def test_high_page_text_confidence_not_queued() -> None:
    parse_result = DoclingParseResult(
        elements=[],
        page_confidence={1: _page_conf(1, parse_score=0.99)},
        page_count=1,
    )
    probe = DocumentProbe(page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL])])

    plan = build_cascade_plan(parse_result, probe)

    assert plan.tasks == []


def test_paragraph_elements_never_queued() -> None:
    parse_result = DoclingParseResult(
        elements=[_element(1, "paragraph", page_number=1, confidence=0.1)],
        page_confidence={1: _page_conf(1, parse_score=0.99)},
        page_count=1,
    )
    probe = DocumentProbe(
        page_count=1, pages=[_page_probe(1, [PAGE_FLAG_TEXT_NORMAL, PAGE_FLAG_VECTOR_DIAGRAM])]
    )

    plan = build_cascade_plan(parse_result, probe)

    assert plan.tasks == []


def test_multiple_independent_triggers_all_fire_for_same_page() -> None:
    parse_result = DoclingParseResult(
        elements=[
            _element(1, "table", page_number=1, confidence=0.5),
            _element(2, "figure", page_number=1, confidence=0.99),
        ],
        page_confidence={1: _page_conf(1, parse_score=0.1)},
        page_count=1,
    )
    probe = DocumentProbe(
        page_count=1, pages=[_page_probe(1, [PAGE_FLAG_VECTOR_DIAGRAM])]
    )

    plan = build_cascade_plan(parse_result, probe, threshold_table=0.75, threshold_text=0.6)

    assert len(plan.tasks) == 3
    counts = plan.task_counts()
    assert counts[TASK_TABLE_REPARSE] == 1
    assert counts[TASK_VISUAL_DESCRIPTION] == 1
    assert counts[TASK_FULL_PAGE_OCR] == 1


def test_cascade_plan_tasks_of_type_and_defaults() -> None:
    plan = CascadePlan()
    assert plan.tasks == []
    assert plan.threshold_table == 0.75
    assert plan.threshold_text == 0.6
    assert plan.tasks_of_type(TASK_TABLE_REPARSE) == []
    assert set(plan.task_counts()) == set(TASK_TYPES)
