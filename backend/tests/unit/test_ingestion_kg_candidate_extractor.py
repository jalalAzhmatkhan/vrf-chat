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
    # [KG-W1.2, K1] VLM-structured entities: extraction_method fixed,
    # canonical_name/model_family not yet populated by this module,
    # justification_span not computable (no offset into `components[]`).
    assert result.entities[0].extraction_method == kg.EXTRACTION_METHOD_VLM_COMPONENT
    assert result.entities[0].canonical_name is None
    assert result.entities[0].model_family is None
    assert result.entities[0].justification_span is None

    assert len(result.relations) == 2
    assert result.relations[0].subject == "compressor"
    assert result.relations[0].predicate == kg.RELATION_CONNECTED_TO
    assert result.relations[0].object == "TH3"
    assert result.relations[0].extraction_method == kg.EXTRACTION_METHOD_VLM_CONNECTION
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


def test_extract_from_visual_description_entities_from_description_text() -> None:
    """[KG-W1.1] Real Stage 4 output never populates `components`/
    `connections` (see module docstring finding #1) but DOES populate
    `description` (markdown/free text) — the same deterministic matchers
    used for narrative `element.text` now also mine entities from it."""
    element = _element(local_id=9, element_type="figure", page_number=5)
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=5, element_local_id=9, reason="x"
        ),
        visual_description=VisualDescription(
            description="A1P PRINTED CIRCUIT BOARD (MAIN), connected via CN105.",
            components=[],
            connections=[],
        ),
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")
    names = {e.name for e in result.entities}
    assert "printed circuit board" in names
    assert "CN105" in names
    for entity in result.entities:
        assert entity.page == 5
        assert entity.element_id == 9
    # Relations are NOT parsed from free text (would violate the
    # determinism/precision bar) — description-derived entities only.
    assert result.relations == []


def test_extract_from_visual_description_no_description_no_extra_entities() -> None:
    element = _element(local_id=1, element_type="figure")
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=1, element_local_id=1, reason="x"
        ),
        visual_description=VisualDescription(description=None, components=[], connections=[]),
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")
    assert result.entities == []


def test_has_error_code_anchor_checks_text_and_section_path() -> None:
    assert kg._has_error_code_anchor("If error code P8 appears", []) is True
    assert kg._has_error_code_anchor("P8", ["Malfunction Code Table"]) is True
    assert kg._has_error_code_anchor("Just some ordinary text with P8", []) is False
    assert kg._has_error_code_anchor(None, []) is False


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
    # [KG-W1.2, K1]
    assert entity.extraction_method == kg.EXTRACTION_METHOD_DICT_KEYWORD
    assert entity.canonical_name is None
    assert entity.model_family is None
    text = element.text
    assert text is not None
    assert entity.justification_span is not None
    start, end = entity.justification_span
    assert text[start:end].lower() == "compressor"


def test_extract_from_text_sensor_id_justification_span() -> None:
    """[KG-W1.2, K1] `justification_span` is the exact `[start, end]`
    character offset of the match within `element.text`."""
    element = _element(text="Measure the resistance of TH3 connected via CN105.")
    result = kg.extract_from_text(element, "doc.pdf")
    sensor = next(e for e in result.entities if e.entity_type == "Sensor")
    assert sensor.extraction_method == kg.EXTRACTION_METHOD_REGEX_SENSOR_ID
    text = element.text
    assert text is not None
    assert sensor.justification_span is not None
    start, end = sensor.justification_span
    assert text[start:end] == "TH3"

    connector = next(e for e in result.entities if e.entity_type == "Connector")
    assert connector.extraction_method == kg.EXTRACTION_METHOD_REGEX_CONNECTOR_ID
    assert connector.justification_span is not None
    start, end = connector.justification_span
    assert text[start:end] == "CN105"


def test_extract_from_text_error_code_extraction_method() -> None:
    element = _element(text="If error code P8 appears, check the water flow sensor.")
    result = kg.extract_from_text(element, "doc.pdf")
    error_code = next(e for e in result.entities if e.entity_type == "ErrorCode")
    assert error_code.extraction_method == kg.EXTRACTION_METHOD_REGEX_ERROR_CODE
    assert error_code.justification_span is not None


def test_extract_from_visual_description_description_entities_use_vlm_suffix_and_no_span() -> None:
    """[KG-W1.2, K1] Entities extracted from `visual_description.description`
    (not `element.text`) get an `extraction_method` suffix distinguishing the
    source, and `justification_span=None` per the K1 contract ("null untuk
    sumber VLM")."""
    element = _element(local_id=9, element_type="figure", page_number=5)
    cascade_result = CascadeResult(
        task=CascadeTask(
            task_type=TASK_VISUAL_DESCRIPTION, page_number=5, element_local_id=9, reason="x"
        ),
        visual_description=VisualDescription(
            description="Check the compressor and TH3.", components=[], connections=[]
        ),
    )
    result = kg.extract_from_visual_description(element, cascade_result, "doc.pdf")
    names = {e.name for e in result.entities}
    assert "compressor" in names
    assert "TH3" in names
    for entity in result.entities:
        assert entity.extraction_method.endswith("_vlm_description")
        assert entity.justification_span is None


def test_extract_from_text_sensor_and_connector_ids() -> None:
    element = _element(text="Measure the resistance of TH3 connected via CN105.")
    result = kg.extract_from_text(element, "doc.pdf")

    sensor = next(e for e in result.entities if e.entity_type == "Sensor")
    assert sensor.name == "TH3"
    assert sensor.confidence == kg.CONFIDENCE_SENSOR_ID

    connector = next(e for e in result.entities if e.entity_type == "Connector")
    assert connector.name == "CN105"
    assert connector.confidence == kg.CONFIDENCE_CONNECTOR_ID


def test_extract_from_text_error_code_pattern_anchored_in_own_text() -> None:
    """[KG-W1.1] An anchor keyword ("error code") in the SAME element's text
    elevates confidence above the unanchored tier — replaces the old dead
    `element_type == "error_code"` branch (confirmed never produced by
    Docling for any real document, see module docstring)."""
    element = _element(
        element_type="paragraph", text="If error code P8 appears, check the water flow sensor."
    )
    result = kg.extract_from_text(element, "doc.pdf")
    error_codes = [e for e in result.entities if e.entity_type == "ErrorCode"]
    assert len(error_codes) == 1
    assert error_codes[0].name == "P8"
    assert error_codes[0].confidence == kg.CONFIDENCE_ERROR_CODE_ANCHORED


def test_extract_from_text_error_code_anchored_via_section_path() -> None:
    """[KG-W1.1] Anchor keyword found in the heading hierarchy
    (`section_path`), not the element's own text — e.g. a table row under a
    "Malfunction Code Table" heading."""
    element = _element(
        element_type="paragraph",
        text="P8",
        section_path=["Troubleshooting", "Malfunction Code Table"],
    )
    result = kg.extract_from_text(element, "doc.pdf")
    error_codes = [e for e in result.entities if e.entity_type == "ErrorCode"]
    assert len(error_codes) == 1
    assert error_codes[0].confidence == kg.CONFIDENCE_ERROR_CODE_ANCHORED


def test_extract_from_text_error_code_table_element_anchored_highest_confidence() -> None:
    """[KG-W1.1] Table element + anchor = the heuristic replacement for the
    old dead-code high-confidence branch, per
    `09-kg-extraction-strategy.md` §5.4 ("elemen tabel + heading error
    code/malfunction code terdekat")."""
    element = _element(
        element_type="table",
        text="| Error code | Description |\n| P8 | Water flow error |",
    )
    result = kg.extract_from_text(element, "doc.pdf")
    error_codes = [e for e in result.entities if e.entity_type == "ErrorCode"]
    assert len(error_codes) == 1
    assert error_codes[0].confidence == kg.CONFIDENCE_ERROR_CODE_TABLE_ANCHORED


def test_extract_from_text_error_code_pattern_unanchored_low_confidence() -> None:
    """[KG-W1.1] No anchor keyword anywhere nearby -> lower confidence than
    the pre-fix flat value (0.4), reflecting the empirically-confirmed high
    false-positive rate of unanchored matches (document_id=3: 192/192
    ErrorCode candidates were unanchored, spanning near-alphabet-wide
    coverage inconsistent with real VRF error code allocation)."""
    element = _element(
        element_type="paragraph", text="The unit code is P8 for this configuration."
    )
    result = kg.extract_from_text(element, "doc.pdf")
    error_codes = [e for e in result.entities if e.entity_type == "ErrorCode"]
    assert len(error_codes) == 1
    assert error_codes[0].name == "P8"
    assert error_codes[0].confidence == kg.CONFIDENCE_ERROR_CODE_UNANCHORED
    assert error_codes[0].confidence < 0.4


def test_extract_from_text_no_matches_returns_empty() -> None:
    element = _element(text="This sentence has nothing relevant in it whatsoever.")
    result = kg.extract_from_text(element, "doc.pdf")
    assert result.entities == []
    assert result.relations == []


def test_extract_from_text_component_keyword_word_boundary_rejects_substring() -> None:
    """[KG-W1.1] Word-boundary fix: "fan motor" must NOT match inside "fan
    motors" (the old `keyword in text_lower` substring check would have
    incorrectly matched this)."""
    element = _element(text="Check the fan motors and controls for wear.")
    result = kg.extract_from_text(element, "doc.pdf")
    names = {e.name for e in result.entities}
    assert "fan motor" not in names


def test_extract_from_text_component_keyword_glossary_page_lower_confidence() -> None:
    """[KG-W1.1] More than `COMPONENT_KEYWORD_GLOSSARY_THRESHOLD` distinct
    keyword matches on one element (a real pattern found on document_id=3,
    14/20 keywords matched simultaneously) is treated as a likely
    glossary/spec-list page, at reduced confidence."""
    element = _element(
        text=(
            "compressor, condenser, evaporator, expansion valve, "
            "solenoid valve, fan motor, accumulator"
        )
    )
    result = kg.extract_from_text(element, "doc.pdf")
    component_entities = [e for e in result.entities if e.entity_type == "Component"]
    assert len(component_entities) == 7
    assert all(e.confidence == kg.CONFIDENCE_TEXT_KEYWORD_GLOSSARY_PAGE for e in component_entities)


def test_extract_from_text_component_keyword_below_glossary_threshold_normal_confidence() -> None:
    element = _element(text="Check the compressor and condenser for leaks.")
    result = kg.extract_from_text(element, "doc.pdf")
    component_entities = [e for e in result.entities if e.entity_type == "Component"]
    assert len(component_entities) == 2
    assert all(e.confidence == kg.CONFIDENCE_TEXT_KEYWORD for e in component_entities)


def test_extract_from_text_sensor_id_space_and_dash_variants() -> None:
    """[KG-W1.1, R-addendum] `TH 3`/`TH-3` (space/dash separator) must be
    recognized, not just `TH3` — confirmed recall bug against document_id=3
    real data (see vrf_vocabulary.py SENSOR_ID_PATTERN docstring)."""
    element = _element(text="Measure resistance between TH 3 and TH-4.")
    result = kg.extract_from_text(element, "doc.pdf")
    sensor_names = {e.name for e in result.entities if e.entity_type == "Sensor"}
    assert sensor_names == {"TH 3", "TH-4"}


def test_extract_from_text_connector_id_space_and_dash_variants() -> None:
    element = _element(text="Connectors CN 105 and CN-106 are both used.")
    result = kg.extract_from_text(element, "doc.pdf")
    connector_names = {e.name for e in result.entities if e.entity_type == "Connector"}
    assert connector_names == {"CN 105", "CN-106"}


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


def test_extract_kg_candidates_default_model_family_is_none() -> None:
    """[KG-W1.3, K2] Without an explicit `model_family` argument (the
    orchestrator.py call site today, see module docstring wiring-gap note),
    every candidate's `model_family` stays `None` — same as before this
    change, no silent behavior shift for existing callers."""
    element = _element(local_id=1, text="Check TH3 near the compressor.")
    parse_result = DoclingParseResult(
        elements=[element], page_confidence={1: _page_confidence(1)}, page_count=1
    )
    results = kg.extract_kg_candidates(parse_result, "doc.pdf")
    assert all(e.model_family is None for e in results[1].entities)


def test_extract_kg_candidates_threads_model_family_onto_every_candidate() -> None:
    """[KG-W1.3, K2] `model_family` is threaded onto every entity/relation
    produced, regardless of source (text keyword/regex, VLM structured
    components/connections, VLM description text)."""
    figure_element = _element(local_id=1, element_type="figure", text=None, page_number=1)
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
            description="Uses a solenoid valve.",
            components=["fan motor"],
            connections=["fan motor -> TH3"],
        ),
    )

    results = kg.extract_kg_candidates(
        parse_result, "doc.pdf", cascade_results=[cascade_result], model_family="VRV-IV"
    )

    for candidates in results.values():
        assert all(e.model_family == "VRV-IV" for e in candidates.entities)
        assert all(r.model_family == "VRV-IV" for r in candidates.relations)
    # sanity: this really did exercise all three sources (VLM structured,
    # VLM description text, and plain text matchers), not a vacuous pass
    assert len(results[1].entities) == 2  # "fan motor" (component) + "solenoid valve" (description)
    assert len(results[1].relations) == 1  # fan motor -> TH3
    assert len(results[2].entities) == 2  # TH3 + compressor


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
        extraction_method="dict_keyword",
        justification_span=[6, 16],
    )
    relation = kg.KGCandidateRelation(
        subject="compressor",
        predicate="CONNECTED_TO",
        object="TH3",
        confidence=0.5,
        source_document="doc.pdf",
        page=1,
        element_id=1,
        extraction_method="vlm_connection",
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
            "extraction_method": "dict_keyword",
            "canonical_name": None,
            "model_family": None,
            "justification_span": [6, 16],
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
            "extraction_method": "vlm_connection",
            "canonical_name": None,
            "model_family": None,
            "justification_span": None,
        }
    ]
