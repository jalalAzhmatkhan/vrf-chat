"""Unit tests for `app/ingestion/kg_candidate_extractor.py` (I1.9).

Pure, deterministic logic (regex + dictionary matching) — no GPU/model/DB
dependency, built directly against the real `ElementDraft`/`CascadeResult`
dataclasses (cheap plain dataclasses, same approach as I1.3's tests).
"""

from __future__ import annotations

from app.ingestion import kg_candidate_extractor as kg
from app.ingestion.cascade_trigger import TASK_TABLE_REPARSE, TASK_VISUAL_DESCRIPTION, CascadeTask
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft, PageConfidence
from app.ingestion.paddleocr_vl_cascade import CascadeResult, TableReparseResult, VisualDescription


def _element(**overrides: object) -> ElementDraft:
    base = dict(
        local_id=1,
        element_type="paragraph",
        text=None,
        bbox=None,
        page_number=1,
        parent_local_id=None,
        section_path=[],
        extraction_method="docling",
        extraction_confidence=0.9,
    )
    base.update(overrides)
    return ElementDraft(**base)  # type: ignore[arg-type]


def _page_confidence(page_number: int = 1) -> PageConfidence:
    return PageConfidence(
        page_number=page_number,
        parse_score=1.0,
        layout_score=0.95,
        table_score=None,
        ocr_score=None,
        mean_score=0.97,
        mean_grade="excellent",
    )


# ---------------------------------------------------------------------------
# extract_from_visual_description
# ---------------------------------------------------------------------------


def test_extract_from_visual_description_none_returns_empty() -> None:
    element = _element(element_type="figure")
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="x"
        ),
        visual_description=None,
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")
    assert result.entities == []
    assert result.relations == []


def test_extract_from_visual_description_components_and_connections() -> None:
    element = _element(local_id=5, element_type="figure", page_number=3)
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=3, element_local_id=5, reason="x"
        ),
        visual_description=VisualDescription(
            description="Electrical schematic",
            figure_type="schematic",
            components=["compressor", "TH3"],
            connections=["compressor -> TH3", "TH3 to PCB"],
        ),
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")

    assert len(result.entities) == 2
    assert result.entities[0].name == "compressor"
    assert result.entities[0].entity_type == "Component"
    assert result.entities[0].confidence == kg.CONFIDENCE_VLM_COMPONENT
    assert result.entities[0].source_document == "doc.pdf"
    assert result.entities[0].page == 3
    assert result.entities[0].element_id == 5

    assert len(result.relations) == 2
    assert result.relations[0].subject == "compressor"
    assert result.relations[0].predicate == kg.RELATION_CONNECTED_TO
    assert result.relations[0].object == "TH3"
    assert result.relations[1].subject == "TH3"
    assert result.relations[1].object == "PCB"


def test_extract_from_visual_description_unparseable_connection_skipped() -> None:
    element = _element(element_type="figure")
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="x"
        ),
        visual_description=VisualDescription(
            description=None, components=[], connections=["no separator here"]
        ),
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")
    assert result.relations == []


def test_parse_connection_empty_side_returns_none() -> None:
    assert kg._parse_connection(" -> TH3") is None
    assert kg._parse_connection("TH3 -> ") is None


# ---------------------------------------------------------------------------
# extract_from_text
# ---------------------------------------------------------------------------


def test_extract_from_text_none_text_returns_empty() -> None:
    element = _element(text=None)
    result = kg.extract_from_text(element, "doc.pdf")
    assert result.entities == []
    assert result.relations == []


def test_extract_from_text_component_keyword() -> None:
    element = _element(text="Check the compressor for abnormal noise.")
    result = kg.extract_from_text(element, "doc.pdf")
    names = {e.name for e in result.entities}
    assert "compressor" in names
    entity = next(e for e in result.entities if e.name == "compressor")
    assert entity.entity_type == "Component"
    assert entity.confidence == kg.CONFIDENCE_TEXT_KEYWORD


def test_extract_from_text_sensor_and_connector_ids() -> None:
    element = _element(text="Measure the resistance of TH3 connected via CN105.")
    result = kg.extract_from_text(element, "doc.pdf")

    sensor = next(e for e in result.entities if e.entity_type == "Sensor")
    assert sensor.name == "TH3"
    assert sensor.confidence == kg.CONFIDENCE_SENSOR_ID

    connector = next(e for e in result.entities if e.entity_type == "Connector")
    assert connector.name == "CN105"
    assert connector.confidence == kg.CONFIDENCE_CONNECTOR_ID


def test_extract_from_text_error_code_element_type() -> None:
    element = _element(element_type="error_code", text="P8")
    result = kg.extract_from_text(element, "doc.pdf")
    assert len(result.entities) == 1
    assert result.entities[0].entity_type == "ErrorCode"
    assert result.entities[0].name == "P8"
    assert result.entities[0].confidence == kg.CONFIDENCE_ERROR_CODE_ELEMENT_TYPE


def test_extract_from_text_error_code_pattern_in_paragraph() -> None:
    element = _element(
        element_type="paragraph", text="If error code P8 appears, check the water flow sensor."
    )
    result = kg.extract_from_text(element, "doc.pdf")
    error_codes = [e for e in result.entities if e.entity_type == "ErrorCode"]
    assert len(error_codes) == 1
    assert error_codes[0].name == "P8"
    assert error_codes[0].confidence == kg.CONFIDENCE_ERROR_CODE_PATTERN


def test_extract_from_text_no_matches_returns_empty() -> None:
    element = _element(text="This sentence has nothing relevant in it whatsoever.")
    result = kg.extract_from_text(element, "doc.pdf")
    assert result.entities == []
    assert result.relations == []


# ---------------------------------------------------------------------------
# extract_kg_candidates orchestration
# ---------------------------------------------------------------------------


def test_extract_kg_candidates_merges_text_and_vlm_sources() -> None:
    figure_element = _element(
        local_id=1, element_type="figure", text=None, page_number=1
    )
    text_element = _element(
        local_id=2, element_type="paragraph", text="Check TH3 near the compressor.", page_number=1
    )
    parse_result = DoclingParseResult(
        elements=[figure_element, text_element],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="x"
        ),
        visual_description=VisualDescription(
            description="Schematic", components=["fan motor"], connections=[]
        ),
    )

    results = kg.extract_kg_candidates(parse_result, "doc.pdf", cascade_results=[cascade_result])

    assert set(results.keys()) == {1, 2}
    assert results[1].entities[0].name == "fan motor"
    text_names = {e.name for e in results[2].entities}
    assert "TH3" in text_names
    assert "compressor" in text_names


def test_extract_kg_candidates_omits_elements_with_no_candidates() -> None:
    element = _element(local_id=1, text="Nothing of interest here.")
    parse_result = DoclingParseResult(
        elements=[element],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    results = kg.extract_kg_candidates(parse_result, "doc.pdf")
    assert results == {}


def test_extract_kg_candidates_ignores_non_matching_cascade_task_type() -> None:
    # A table_reparse cascade result for a non-figure element should not
    # contribute visual_description-based candidates (it has none).
    element = _element(local_id=1, element_type="table", text="| a | b |")
    parse_result = DoclingParseResult(
        elements=[element],
        page_confidence={1: _page_confidence(1)},
        page_count=1,
    )
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_TABLE_REPARSE, page_number=1, element_local_id=1, reason="x"
        ),
        table_reparse=TableReparseResult(markdown="| a | b |"),
    )
    results = kg.extract_kg_candidates(parse_result, "doc.pdf", cascade_results=[cascade_result])
    assert results == {}


# ---------------------------------------------------------------------------
# jsonb serialization
# ---------------------------------------------------------------------------


def test_entities_to_jsonb_and_relations_to_jsonb() -> None:
    entity = kg.KGCandidateEntity(
        name="compressor",
        entity_type="Component",
        confidence=0.5,
        source_document="doc.pdf",
        page=1,
        element_id=1,
    )
    relation = kg.KGCandidateRelation(
        subject="compressor",
        predicate="CONNECTED_TO",
        object="TH3",
        confidence=0.5,
        source_document="doc.pdf",
        page=1,
        element_id=1,
    )

    entity_dicts = kg.entities_to_jsonb([entity])
    relation_dicts = kg.relations_to_jsonb([relation])

    assert entity_dicts == [
        {
            "name": "compressor",
            "entity_type": "Component",
            "confidence": 0.5,
            "source_document": "doc.pdf",
            "page": 1,
            "element_id": 1,
        }
    ]
    assert relation_dicts == [
        {
            "subject": "compressor",
            "predicate": "CONNECTED_TO",
            "object": "TH3",
            "confidence": 0.5,
            "source_document": "doc.pdf",
            "page": 1,
            "element_id": 1,
        }
    ]
