"""Unit tests for `app/agent/schemas.py` (C2.3) — the `TechnicalAnswer`/
`Citation`/`Warning` contract per
`Documentation/system-design/05-streaming-and-api-contract.md` §5."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.schemas import VISUAL_ELEMENT_TYPES, Citation, TechnicalAnswer, Warning


def test_citation_minimal_fields() -> None:
    citation = Citation(document_id="3", page=238, element_id="4961", element_type="paragraph")
    assert citation.quote is None
    assert citation.image_uri is None
    assert citation.visual_description is None


def test_citation_visual_fields() -> None:
    citation = Citation(
        document_id="3",
        page=238,
        element_id="4961",
        element_type="figure",
        image_uri="s3://bucket/fig.png",
        visual_description={"figure_type": "wiring_diagram"},
    )
    assert citation.image_uri == "s3://bucket/fig.png"
    assert citation.visual_description == {"figure_type": "wiring_diagram"}


def test_warning_default_severity_is_note() -> None:
    warning = Warning(message="check chapter 3")
    assert warning.severity == "note"


def test_warning_safety_severity() -> None:
    warning = Warning(message="high voltage", severity="safety")
    assert warning.severity == "safety"


def test_technical_answer_defaults() -> None:
    answer = TechnicalAnswer(answer="Hello.", confidence=0.9)
    assert answer.citations == []
    assert answer.warnings == []
    assert answer.related_components == []
    assert answer.related_error_codes == []
    assert answer.refused is False


def test_technical_answer_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalAnswer(answer="x", confidence=1.5)


def test_technical_answer_confidence_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalAnswer(answer="x", confidence=-0.1)


def test_visual_element_types_constant() -> None:
    assert VISUAL_ELEMENT_TYPES == {"icon", "figure", "diagram"}
