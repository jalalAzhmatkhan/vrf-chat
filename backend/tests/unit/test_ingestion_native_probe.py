"""Unit tests for `app/ingestion/native_probe.py` (I1.1).

Builds tiny synthetic single/multi-page PDFs in-memory via PyMuPDF's own
document-authoring API (no dependency on `source-documents/`, which is
read-only and reserved for pipeline-level/manual verification), so every
flag branch is deterministic and independent of real-manual content.
"""

from __future__ import annotations

import pymupdf
import pytest

from app.ingestion.native_probe import (
    PAGE_FLAG_LOW_TEXT_CONFIDENCE,
    PAGE_FLAG_RASTER_IMAGE,
    PAGE_FLAG_TEXT_NORMAL,
    PAGE_FLAG_VECTOR_DIAGRAM,
    DocumentProbe,
    PageProbe,
    probe_document,
    probe_page,
)


def _tiny_png_bytes() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 4, 4))
    pix.set_rect(pix.irect, (128,))
    return pix.tobytes("png")


def _new_doc() -> pymupdf.Document:
    return pymupdf.open()


def _insert_text(page: pymupdf.Page, text: str) -> None:
    """`insert_textbox` wraps within the rect (unlike `insert_text`, which
    silently drops characters that would overflow a single unwrapped line —
    verified empirically to under-count `page.get_text()` for long strings)."""
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)


def test_probe_page_text_normal_only() -> None:
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    _insert_text(page, "Ordinary narrative paragraph text. " * 40)

    result = probe_page(page)

    assert result.flags == [PAGE_FLAG_TEXT_NORMAL]
    assert result.vector_drawing_count == 0
    assert result.embedded_image_count == 0
    assert result.page_number == 1
    doc.close()


def test_probe_page_low_text_confidence_when_empty() -> None:
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    # No text inserted at all -> scanned-page-like / blank page.

    result = probe_page(page)

    assert result.flags == [PAGE_FLAG_LOW_TEXT_CONFIDENCE]
    doc.close()


def test_probe_page_low_text_confidence_when_below_min_chars() -> None:
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    _insert_text(page, "Hi")

    result = probe_page(page, min_text_chars=40)

    assert PAGE_FLAG_LOW_TEXT_CONFIDENCE in result.flags
    assert PAGE_FLAG_TEXT_NORMAL not in result.flags
    doc.close()


def test_probe_page_vector_diagram_flag() -> None:
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    _insert_text(page, "Electrical circuit schematic caption text here. " * 20)
    for i in range(60):
        page.draw_line((10 + i, 10), (10 + i, 100))

    result = probe_page(page, vector_drawing_threshold=50)

    assert PAGE_FLAG_VECTOR_DIAGRAM in result.flags
    assert PAGE_FLAG_TEXT_NORMAL in result.flags
    assert result.vector_drawing_count == 60
    doc.close()


def test_probe_page_below_vector_diagram_threshold_not_flagged() -> None:
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    _insert_text(page, "Normal paragraph with a couple of lines drawn.")
    for i in range(3):
        page.draw_line((10 + i, 10), (10 + i, 100))

    result = probe_page(page, vector_drawing_threshold=50)

    assert PAGE_FLAG_VECTOR_DIAGRAM not in result.flags
    doc.close()


def test_probe_page_raster_image_flag_coexists_with_text_normal() -> None:
    """Requirement (CLAUDE.md §4): inline icon in otherwise-normal text must
    NOT be classified as "not text_normal" — both flags co-occur."""
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    _insert_text(page, "Press the button icon to reset the unit. " * 20)
    page.insert_image(pymupdf.Rect(200, 200, 220, 220), stream=_tiny_png_bytes())

    result = probe_page(page)

    assert PAGE_FLAG_RASTER_IMAGE in result.flags
    assert PAGE_FLAG_TEXT_NORMAL in result.flags
    assert result.embedded_image_count == 1
    doc.close()


def test_page_probe_has_flag() -> None:
    probe = PageProbe(
        page_number=1,
        char_count=10,
        text_density=0.1,
        vector_drawing_count=0,
        embedded_image_count=0,
        page_width=600,
        page_height=800,
        flags=[PAGE_FLAG_TEXT_NORMAL],
    )
    assert probe.has_flag(PAGE_FLAG_TEXT_NORMAL) is True
    assert probe.has_flag(PAGE_FLAG_VECTOR_DIAGRAM) is False


def test_probe_document_multi_page(tmp_path) -> None:
    doc = _new_doc()
    page1 = doc.new_page(width=600, height=800)
    _insert_text(page1, "Ordinary narrative paragraph text. " * 40)
    page2 = doc.new_page(width=600, height=800)
    for i in range(60):
        page2.draw_line((10 + i, 10), (10 + i, 100))
    _insert_text(page2, "Electrical diagram caption text. " * 20)

    pdf_path = tmp_path / "synthetic.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = probe_document(pdf_path)

    assert isinstance(result, DocumentProbe)
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2

    counts = result.flag_counts()
    assert counts[PAGE_FLAG_TEXT_NORMAL] == 2
    assert counts[PAGE_FLAG_VECTOR_DIAGRAM] == 1
    assert counts[PAGE_FLAG_RASTER_IMAGE] == 0
    assert counts[PAGE_FLAG_LOW_TEXT_CONFIDENCE] == 0


def test_document_probe_page_lookup(tmp_path) -> None:
    doc = _new_doc()
    _insert_text(doc.new_page(width=600, height=800), "Page one text " * 20)
    _insert_text(doc.new_page(width=600, height=800), "Page two text " * 20)
    pdf_path = tmp_path / "synthetic.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = probe_document(pdf_path)

    found = result.page(2)
    assert found.page_number == 2

    with pytest.raises(KeyError):
        result.page(99)
