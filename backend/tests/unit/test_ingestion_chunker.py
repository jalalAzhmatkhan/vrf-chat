"""Unit tests for `app/ingestion/chunker.py` (I1.7).

`build_chunks`/`build_entity_chunks`/`compute_content_hash` are pure
functions over plain dataclasses (`ChunkableElement`) — no DB needed.
`store_chunks` is tested against an in-memory SQLite session (same pattern
as I1.5/I1.9).
"""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.chunks import Chunk
from app.ingestion import chunker as ck


def _element(**overrides: object) -> ck.ChunkableElement:
    base = dict(
        element_id=1,
        element_type="paragraph",
        text="Hello world.",
        page_number=1,
        parent_id=None,
        section_path=["Chapter 1"],
        image_uri=None,
        visual_description=None,
        kg_candidate_entities=[],
    )
    base.update(overrides)
    return ck.ChunkableElement(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_markdown_table
# ---------------------------------------------------------------------------


def test_parse_markdown_table_valid() -> None:
    markdown = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    rows = ck._parse_markdown_table(markdown)
    assert rows == [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}]


def test_parse_markdown_table_without_separator_row() -> None:
    markdown = "| A | B |\n| 1 | 2 |"
    rows = ck._parse_markdown_table(markdown)
    assert rows == [{"A": "1", "B": "2"}]


def test_parse_markdown_table_none_returns_empty() -> None:
    assert ck._parse_markdown_table(None) == []


def test_parse_markdown_table_non_table_text_returns_empty() -> None:
    assert ck._parse_markdown_table("Just a plain sentence.") == []


def test_parse_markdown_table_single_line_returns_empty() -> None:
    assert ck._parse_markdown_table("| only header |") == []


# ---------------------------------------------------------------------------
# _parse_html_table / _parse_table_rows
# [QA phase-1-qa-report.md §1.3, F1] Real PaddleOCR-VL table-reparse output
# (extraction_method='docling+paddle_table') is HTML, not markdown —
# `_parse_markdown_table` always returned [] for it. Fixture below is a
# real (not synthetic) excerpt of `content_text` from chunk id 1499,
# document_id=3 (a "Warning" precautions table, 2 of its real 7 rows kept
# verbatim — including the real `colspan="2"` title row, `style` attributes,
# and embedded `<img>` tags — trimmed only for test-file length, not
# simplified/sanitized).
# ---------------------------------------------------------------------------

_REAL_PADDLE_TABLE_HTML = (
    "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    '<tr><td colspan="2">Warning</td></tr>'
    "<tr>"
    "<td style='text-align: center; word-wrap: break-word;'>"
    "Do not store the equipment in a room with successive fire sources "
    "(e.g., naked flame, gas appliance, electric heater)."
    "</td>"
    "<td style='text-align: center; word-wrap: break-word;'>"
    '<img src="imgs/img_in_image_box_933_105_1053_220.jpg" alt="Image"" />'
    "</td>"
    "</tr>"
    "<tr>"
    "<td style='text-align: center; word-wrap: break-word;'>"
    "Be sure to disconnect the power cable plug from the plug socket before "
    "disassembling the equipment for repair."
    "</td>"
    "<td style='text-align: center; word-wrap: break-word;'>"
    '<img src="imgs/img_in_image_box_934_268_1048_385.jpg" alt="Image"" />'
    "</td>"
    "</tr>"
    "</table>"
)


def test_parse_html_table_real_paddle_table_reparse_fixture() -> None:
    rows = ck._parse_html_table(_REAL_PADDLE_TABLE_HTML)

    # Row 0: the real colspan=2 "Warning" title row — expanded across both
    # columns (never dropped/misaligned).
    assert rows[0] == {"col_0": "Warning", "col_1": "Warning"}
    # Row 1/2: real instruction text in col_0, embedded <img> in col_1
    # contributes no text (image is not lost — content_text/markdown still
    # keeps the raw HTML; this row's structured form is correctly empty).
    assert rows[1]["col_0"].startswith("Do not store the equipment")
    assert rows[1]["col_1"] == ""
    assert rows[2]["col_0"].startswith("Be sure to disconnect")
    assert rows[2]["col_1"] == ""
    assert len(rows) == 3


def test_parse_html_table_colspan_expands_across_columns() -> None:
    html = (
        '<table><tr><td colspan="3">Merged</td></tr>'
        "<tr><td>a</td><td>b</td><td>c</td></tr></table>"
    )
    rows = ck._parse_html_table(html)
    assert rows[0] == {"col_0": "Merged", "col_1": "Merged", "col_2": "Merged"}
    assert rows[1] == {"col_0": "a", "col_1": "b", "col_2": "c"}


def test_parse_html_table_non_numeric_colspan_defaults_to_one() -> None:
    """[QA phase-1-qa-report.md follow-up, coverage gap] A non-numeric
    `colspan` attribute (malformed real-world HTML, e.g. `colspan="auto"`)
    must not crash — falls back to `colspan=1` (no expansion) rather than
    propagating the `ValueError` from `int(value)`."""
    html = '<table><tr><td colspan="auto">x</td><td>y</td></tr></table>'
    rows = ck._parse_html_table(html)
    assert rows == [{"col_0": "x", "col_1": "y"}]


def test_parse_html_table_no_rows_returns_empty() -> None:
    assert ck._parse_html_table("<table></table>") == []


def test_parse_html_table_empty_row_skipped() -> None:
    html = "<table><tr></tr><tr><td>x</td></tr></table>"
    rows = ck._parse_html_table(html)
    assert rows == [{"col_0": "x"}]


def test_parse_html_table_malformed_html_returns_empty_not_raises() -> None:
    # An unterminated <td>/<tr> (no closing tags) never fires
    # handle_endtag, so nothing is appended to a row/to `rows` — this
    # degrades safely to [] rather than raising, matching
    # `_parse_markdown_table`'s "defensive, not an error" contract for
    # anything that doesn't parse cleanly.
    assert ck._parse_html_table("<table><tr><td>unterminated") == []


def test_parse_table_rows_dispatches_html_by_content_prefix() -> None:
    rows = ck._parse_table_rows("<table><tr><td>x</td></tr></table>")
    assert rows == [{"col_0": "x"}]


def test_parse_table_rows_dispatches_markdown_by_default() -> None:
    rows = ck._parse_table_rows("| A |\n| --- |\n| 1 |")
    assert rows == [{"A": "1"}]


def test_parse_table_rows_none_returns_empty() -> None:
    assert ck._parse_table_rows(None) == []


def test_parse_table_rows_html_prefix_is_case_insensitive_and_whitespace_tolerant() -> None:
    rows = ck._parse_table_rows("  \n<TABLE><tr><td>x</td></tr></table>")
    assert rows == [{"col_0": "x"}]


def test_parse_table_rows_detects_html_with_caption_before_table() -> None:
    """[QA phase-1-qa-report.md §1.3 F1, real edge case found during fix
    verification — document_id=3 chunk id 3734] Real PaddleOCR-VL output
    sometimes prepends a caption OUTSIDE the table (`<div>...</div>` before
    `<table>`) — a strict `startswith("<table")` check misses this and
    silently falls through to the markdown parser (returning [] again)."""
    html = (
        '<div style="text-align: center;">Parameter [A]</div>\n\n'
        "<table><tr><td>x</td></tr></table>"
    )
    rows = ck._parse_table_rows(html)
    assert rows == [{"col_0": "x"}]


# ---------------------------------------------------------------------------
# build_chunks — text grouping
# ---------------------------------------------------------------------------


def test_consecutive_paragraphs_same_section_merge_into_one_chunk() -> None:
    elements = [
        _element(element_id=1, text="First paragraph.", section_path=["Ch1"]),
        _element(element_id=2, text="Second paragraph.", section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TEXT
    assert chunks[0].element_ids == [1, 2]
    assert "First paragraph." in chunks[0].content_text
    assert "Second paragraph." in chunks[0].content_text


def test_section_path_change_starts_new_chunk() -> None:
    elements = [
        _element(element_id=1, text="Under chapter 1.", section_path=["Ch1"]),
        _element(element_id=2, text="Under chapter 2.", section_path=["Ch2"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 2
    assert chunks[0].section_path == ["Ch1"]
    assert chunks[1].section_path == ["Ch2"]


def test_size_cap_forces_new_chunk() -> None:
    long_text = "x" * (ck.DEFAULT_MAX_CHUNK_CHARS + 10)
    elements = [
        _element(element_id=1, text=long_text, section_path=["Ch1"]),
        _element(element_id=2, text="More text.", section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 2
    assert chunks[0].element_ids == [1]
    assert chunks[1].element_ids == [2]


def test_page_start_end_tracked_across_group() -> None:
    elements = [
        _element(element_id=1, text="Page 1 text.", page_number=1, section_path=["Ch1"]),
        _element(element_id=2, text="Page 2 text.", page_number=2, section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


# ---------------------------------------------------------------------------
# build_chunks — icon inline association (critical requirement)
# ---------------------------------------------------------------------------


def test_icon_joins_parent_paragraph_chunk() -> None:
    elements = [
        _element(
            element_id=1, element_type="paragraph", text="Press the button", section_path=["Ch1"]
        ),
        _element(
            element_id=2,
            element_type="icon",
            text=None,
            parent_id=1,
            section_path=["Ch1"],
            image_uri="s3://icon.png",
        ),
        _element(element_id=3, element_type="paragraph", text="to reset.", section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].element_ids == [1, 2, 3]


def test_icon_joins_parent_even_across_size_cap_boundary() -> None:
    """The icon's parent chunk may already be "full" per the size cap, but
    the icon MUST still join it rather than starting a new chunk — this is
    the strongest form of the critical requirement guarantee."""
    long_text = "x" * (ck.DEFAULT_MAX_CHUNK_CHARS + 10)
    elements = [
        _element(element_id=1, element_type="paragraph", text=long_text, section_path=["Ch1"]),
        _element(element_id=2, element_type="icon", text=None, parent_id=1, section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].element_ids == [1, 2]


def test_icon_without_resolvable_parent_falls_back_to_normal_grouping() -> None:
    elements = [
        _element(
            element_id=1, element_type="icon", text=None, parent_id=None, section_path=["Ch1"]
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TEXT
    assert chunks[0].element_ids == [1]


def test_icon_with_unresolvable_parent_id_falls_back() -> None:
    elements = [
        _element(element_id=1, element_type="icon", text=None, parent_id=999, section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)
    assert chunks[0].element_ids == [1]


def test_figure_caption_joins_parent_figure_chunk() -> None:
    elements = [
        _element(
            element_id=1,
            element_type="figure",
            text=None,
            section_path=["Ch1"],
            image_uri="s3://fig.png",
        ),
        _element(
            element_id=2,
            element_type="figure_caption",
            text="Figure 7-1: Electrical schematic",
            parent_id=1,
            section_path=["Ch1"],
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_FIGURE
    assert chunks[0].element_ids == [1, 2]
    assert "Figure 7-1" in chunks[0].content_text


# ---------------------------------------------------------------------------
# build_chunks — table
# ---------------------------------------------------------------------------


def test_table_element_becomes_own_chunk_with_structured_rows() -> None:
    elements = [
        _element(
            element_id=1,
            element_type="table",
            text="| Code | Meaning |\n| --- | --- |\n| P8 | Water flow |",
            section_path=["Ch7"],
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TABLE
    assert chunks[0].content_structured["rows"] == [{"Code": "P8", "Meaning": "Water flow"}]
    assert chunks[0].element_ids == [1]


def test_table_interrupts_running_text_chunk() -> None:
    elements = [
        _element(
            element_id=1, element_type="paragraph", text="Before table.", section_path=["Ch1"]
        ),
        _element(element_id=2, element_type="table", text="| A |\n| 1 |", section_path=["Ch1"]),
        _element(element_id=3, element_type="paragraph", text="After table.", section_path=["Ch1"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 3
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TEXT
    assert chunks[1].chunk_type == ck.CHUNK_TYPE_TABLE
    assert chunks[2].chunk_type == ck.CHUNK_TYPE_TEXT
    # A new text chunk after the table, not merged with the one before it.
    assert chunks[0].element_ids == [1]
    assert chunks[2].element_ids == [3]


# ---------------------------------------------------------------------------
# build_chunks — figure
# ---------------------------------------------------------------------------


def test_figure_element_uses_visual_description_when_no_own_text() -> None:
    elements = [
        _element(
            element_id=1,
            element_type="figure",
            text=None,
            section_path=["Ch1"],
            image_uri="s3://fig.png",
            visual_description={
                "description": "Fan motor wiring diagram",
                "figure_type": "schematic",
            },
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert chunks[0].chunk_type == ck.CHUNK_TYPE_FIGURE
    assert chunks[0].content_text == "Fan motor wiring diagram"
    assert chunks[0].content_structured["image_uri"] == "s3://fig.png"


# ---------------------------------------------------------------------------
# build_chunks — procedure (list runs)
# ---------------------------------------------------------------------------


def test_consecutive_list_items_become_one_procedure_chunk_with_numbered_steps() -> None:
    elements = [
        _element(element_id=1, element_type="list", text="Turn off power.", section_path=["Ch7"]),
        _element(element_id=2, element_type="list", text="Remove cover.", section_path=["Ch7"]),
        _element(element_id=3, element_type="list", text="Check sensor.", section_path=["Ch7"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_PROCEDURE
    steps = chunks[0].content_structured["steps"]
    assert [s["step_number"] for s in steps] == [1, 2, 3]
    assert steps[0]["step_text"] == "Turn off power."
    assert steps[0]["element_id"] == 1
    assert chunks[0].element_ids == [1, 2, 3]


def test_list_run_interrupted_by_section_change_starts_new_procedure_chunk() -> None:
    elements = [
        _element(element_id=1, element_type="list", text="Step A.", section_path=["Ch7"]),
        _element(element_id=2, element_type="list", text="Step B.", section_path=["Ch8"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 2
    assert chunks[0].content_structured["steps"][0]["step_number"] == 1
    assert chunks[1].content_structured["steps"][0]["step_number"] == 1


def test_text_after_procedure_starts_fresh_chunk() -> None:
    elements = [
        _element(element_id=1, element_type="list", text="Step A.", section_path=["Ch7"]),
        _element(
            element_id=2, element_type="paragraph", text="Note: be careful.", section_path=["Ch7"]
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_PROCEDURE
    assert chunks[1].chunk_type == ck.CHUNK_TYPE_TEXT


def test_procedure_after_text_starts_fresh_chunk() -> None:
    elements = [
        _element(element_id=1, element_type="paragraph", text="Intro.", section_path=["Ch7"]),
        _element(element_id=2, element_type="list", text="Step A.", section_path=["Ch7"]),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TEXT
    assert chunks[1].chunk_type == ck.CHUNK_TYPE_PROCEDURE


# ---------------------------------------------------------------------------
# build_chunks — defensive fallback for unrecognized element types
# ---------------------------------------------------------------------------


def test_unrecognized_element_type_becomes_standalone_text_chunk() -> None:
    elements = [
        _element(
            element_id=1, element_type="diagram", text="Some diagram text.", section_path=["Ch1"]
        ),
    ]
    chunks = ck.build_chunks(elements)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == ck.CHUNK_TYPE_TEXT
    assert chunks[0].element_ids == [1]


def test_empty_elements_list_returns_no_chunks() -> None:
    assert ck.build_chunks([]) == []


# ---------------------------------------------------------------------------
# build_entity_chunks
# ---------------------------------------------------------------------------


def test_build_entity_chunks_aggregates_unique_entities() -> None:
    elements = [
        _element(
            element_id=1,
            page_number=1,
            kg_candidate_entities=[
                {
                    "name": "TH3",
                    "entity_type": "Sensor",
                    "confidence": 0.7,
                    "source_document": "doc.pdf",
                    "page": 1,
                    "element_id": 1,
                }
            ],
        ),
        _element(
            element_id=2,
            page_number=3,
            kg_candidate_entities=[
                {
                    "name": "TH3",
                    "entity_type": "Sensor",
                    "confidence": 0.9,
                    "source_document": "doc.pdf",
                    "page": 3,
                    "element_id": 2,
                }
            ],
        ),
    ]
    chunks = ck.build_entity_chunks(elements)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == ck.CHUNK_TYPE_ENTITY
    assert chunk.content_structured["name"] == "TH3"
    assert chunk.content_structured["occurrence_count"] == 2
    assert chunk.content_structured["pages"] == [1, 3]
    assert chunk.content_structured["best_confidence"] == 0.9
    assert chunk.element_ids == [1, 2]
    assert chunk.page_start == 1
    assert chunk.page_end == 3


def test_build_entity_chunks_distinguishes_by_type() -> None:
    elements = [
        _element(
            element_id=1,
            kg_candidate_entities=[
                {
                    "name": "P8",
                    "entity_type": "ErrorCode",
                    "confidence": 0.4,
                    "source_document": "doc.pdf",
                    "page": 1,
                    "element_id": 1,
                }
            ],
        ),
    ]
    chunks = ck.build_entity_chunks(elements)
    assert len(chunks) == 1
    assert chunks[0].content_structured["entity_type"] == "ErrorCode"


def test_build_entity_chunks_no_candidates_returns_empty() -> None:
    elements = [_element(element_id=1, kg_candidate_entities=[])]
    assert ck.build_entity_chunks(elements) == []


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


def test_compute_content_hash_deterministic_and_sensitive_to_content() -> None:
    draft_a = ck.ChunkDraft(
        chunk_type="text",
        section_path=["Ch1"],
        page_start=1,
        page_end=1,
        content_text="Hello",
        content_structured=None,
        element_ids=[1],
    )
    draft_b = ck.ChunkDraft(
        chunk_type="text",
        section_path=["Ch1"],
        page_start=1,
        page_end=1,
        content_text="Goodbye",
        content_structured=None,
        element_ids=[1],
    )
    assert ck.compute_content_hash(draft_a) == ck.compute_content_hash(draft_a)
    assert ck.compute_content_hash(draft_a) != ck.compute_content_hash(draft_b)


# ---------------------------------------------------------------------------
# store_chunks
# ---------------------------------------------------------------------------


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_store_chunks_persists_rows() -> None:
    db = _make_session()
    drafts = [
        ck.ChunkDraft(
            chunk_type="text",
            section_path=["Ch1"],
            page_start=1,
            page_end=1,
            content_text="Hello",
            content_structured=None,
            element_ids=[1, 2],
        )
    ]

    count = ck.store_chunks(db, document_id=1, drafts=drafts)

    assert count == 1
    stored = db.execute(select(Chunk)).scalars().all()
    assert len(stored) == 1
    assert stored[0].content_hash == ck.compute_content_hash(drafts[0])
    assert stored[0].embedding_status == "pending"
    assert stored[0].element_ids == [1, 2]


def test_store_chunks_replaces_existing_chunks_for_document() -> None:
    db = _make_session()
    first_drafts = [
        ck.ChunkDraft(
            chunk_type="text",
            section_path=["Ch1"],
            page_start=1,
            page_end=1,
            content_text="Old content",
            content_structured=None,
            element_ids=[1],
        )
    ]
    ck.store_chunks(db, document_id=1, drafts=first_drafts)

    second_drafts = [
        ck.ChunkDraft(
            chunk_type="text",
            section_path=["Ch1"],
            page_start=1,
            page_end=1,
            content_text="New content",
            content_structured=None,
            element_ids=[2],
        )
    ]
    ck.store_chunks(db, document_id=1, drafts=second_drafts)

    stored = db.execute(select(Chunk)).scalars().all()
    assert len(stored) == 1
    assert stored[0].content_text == "New content"


def test_store_chunks_scoped_to_document_id() -> None:
    db = _make_session()
    ck.store_chunks(
        db,
        document_id=1,
        drafts=[
            ck.ChunkDraft(
                chunk_type="text",
                section_path=[],
                page_start=1,
                page_end=1,
                content_text="Doc 1",
                content_structured=None,
                element_ids=[1],
            )
        ],
    )
    ck.store_chunks(
        db,
        document_id=2,
        drafts=[
            ck.ChunkDraft(
                chunk_type="text",
                section_path=[],
                page_start=1,
                page_end=1,
                content_text="Doc 2",
                content_structured=None,
                element_ids=[1],
            )
        ],
    )

    stored = db.execute(select(Chunk)).scalars().all()
    assert len(stored) == 2
    assert {row.document_id for row in stored} == {1, 2}


def test_store_chunks_expunges_created_rows_from_session() -> None:
    """[2026-08-11, round 3 memory fix] Regression guard, same pattern as
    `test_ingestion_canonical_store.py`'s analogous test — `Chunk` ORM
    objects created by `store_chunks` must not remain resident in the
    session's identity map afterward (this project's `Session` has
    `expire_on_commit=False`, `app/db/engine.py`, so a plain `db.commit()`
    alone would not release them)."""
    db = _make_session()
    drafts = [
        ck.ChunkDraft(
            chunk_type="text",
            section_path=["Ch1"],
            page_start=1,
            page_end=1,
            content_text=f"Chunk {i}",
            content_structured=None,
            element_ids=[i],
        )
        for i in range(5)
    ]

    count = ck.store_chunks(db, document_id=1, drafts=drafts)

    assert count == 5
    identity_map_classes = {type(state.object) for state in db.identity_map.all_states()}
    assert identity_map_classes == set()
