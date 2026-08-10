"""Unit tests for `app/domain/query_expansion.py` (C2.2).

`expand_query` is pure/zero-I/O — tested directly. `load_known_entities`
touches Postgres — tested against an in-memory SQLite session with real
`Document`/`ErrorCode` rows.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.documents import Document
from app.db.models.error_codes import ErrorCode
from app.domain import query_expansion as qe


def _make_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


# ---------------------------------------------------------------------------
# expand_query — synonym dictionary
# ---------------------------------------------------------------------------


def test_expand_query_matches_symptom_synonym_phrase() -> None:
    result = qe.expand_query("outdoor unit mati terus, kenapa ya?")

    assert "outdoor unit mati" in result.matched_synonym_phrases
    assert "outdoor unit stops" in result.synonyms
    assert "compressor stop" in result.synonyms
    assert "outdoor unit stops" in result.expanded_query_text


def test_expand_query_no_synonym_match_returns_empty() -> None:
    result = qe.expand_query("what is the warranty period")

    assert result.matched_synonym_phrases == ()
    assert result.synonyms == ()


def test_expand_query_multiple_synonym_phrases_matched() -> None:
    result = qe.expand_query("ac tidak dingin dan suara berisik dari outdoor")

    assert "ac tidak dingin" in result.matched_synonym_phrases
    assert "suara berisik" in result.matched_synonym_phrases
    assert "insufficient cooling" in result.synonyms
    assert "abnormal noise" in result.synonyms


# ---------------------------------------------------------------------------
# expand_query — identifier regex detection
# ---------------------------------------------------------------------------


def test_expand_query_detects_error_code_uppercase() -> None:
    result = qe.expand_query("What does error P8 mean?")
    assert result.detected_error_codes == ("P8",)
    assert "P8" in result.expanded_query_text


def test_expand_query_detects_error_code_lowercase_input() -> None:
    result = qe.expand_query("what does error p8 mean?")
    assert result.detected_error_codes == ("P8",)


def test_expand_query_detects_sensor_id() -> None:
    result = qe.expand_query("check TH3 wiring")
    assert result.detected_sensor_ids == ("TH3",)


def test_expand_query_detects_connector_id() -> None:
    result = qe.expand_query("connector CN105 is loose")
    assert result.detected_connector_ids == ("CN105",)


def test_expand_query_no_identifiers_detected() -> None:
    result = qe.expand_query("how do I clean the filter")
    assert result.detected_error_codes == ()
    assert result.detected_sensor_ids == ()
    assert result.detected_connector_ids == ()
    assert result.detected_terminal_ids == ()


def test_expand_query_detects_daikin_sensor_id() -> None:
    # SA-KG.1: R#T is the Daikin/Zeggo sensor convention — our actual
    # indexed corpus (document_id=3) is a Daikin/Zeggo manual.
    result = qe.expand_query("check R1T sensor reading")
    assert "R1T" in result.detected_sensor_ids


def test_expand_query_detects_terminal_id() -> None:
    result = qe.expand_query("check terminal X2M wiring")
    assert result.detected_terminal_ids == ("X2M",)


def test_expand_query_component_keyword_word_boundary_not_substring() -> None:
    # "fan motors" (plural) must not match via naive substring the way it
    # would with `"fan motor" in text` — word-boundary pattern only.
    result = qe.expand_query("check the fan motors assembly")
    assert "fan motor" not in result.detected_components


def test_expand_query_multiple_error_codes_sorted() -> None:
    result = qe.expand_query("difference between P8 and U4")
    assert result.detected_error_codes == ("P8", "U4")


# ---------------------------------------------------------------------------
# expand_query — component keyword detection
# ---------------------------------------------------------------------------


def test_expand_query_detects_component_keyword() -> None:
    result = qe.expand_query("the compressor is not running")
    assert "compressor" in result.detected_components


def test_expand_query_detects_multiple_components() -> None:
    result = qe.expand_query("check the pcb and thermistor readings")
    assert "pcb" in result.detected_components
    assert "thermistor" in result.detected_components


# ---------------------------------------------------------------------------
# expand_query — known_entities corroboration
# ---------------------------------------------------------------------------


def test_expand_query_known_entities_extends_error_code_detection() -> None:
    # "E9" doesn't match ERROR_CODE_PATTERN case (it does, actually) — use a
    # non-matching-by-regex-alone scenario: a code embedded without a clean
    # word boundary is still corroborated by the known list.
    known = qe.KnownEntities(error_codes=frozenset({"P8"}))
    result = qe.expand_query("tell me about code p8 please", known_entities=known)
    assert "P8" in result.detected_error_codes


def test_expand_query_known_entities_detects_model_family_substring() -> None:
    known = qe.KnownEntities(model_families=frozenset({"REYQ"}))
    result = qe.expand_query("REYQ outdoor unit troubleshooting", known_entities=known)
    assert result.detected_model_families == ("REYQ",)


def test_expand_query_known_entities_none_by_default() -> None:
    result = qe.expand_query("REYQ outdoor unit troubleshooting")
    assert result.detected_model_families == ()


def test_expand_query_known_entities_no_match_stays_empty() -> None:
    known = qe.KnownEntities(model_families=frozenset({"PUHY-P_YKB-A1"}))
    result = qe.expand_query("REYQ outdoor unit troubleshooting", known_entities=known)
    assert result.detected_model_families == ()


def test_expand_query_known_entities_empty_code_string_ignored() -> None:
    known = qe.KnownEntities(error_codes=frozenset({""}))
    result = qe.expand_query("hello", known_entities=known)
    assert result.detected_error_codes == ()


def test_expand_query_known_entities_empty_model_family_ignored() -> None:
    known = qe.KnownEntities(model_families=frozenset({""}))
    result = qe.expand_query("hello", known_entities=known)
    assert result.detected_model_families == ()


# ---------------------------------------------------------------------------
# expand_query — expanded_query_text dedup/order
# ---------------------------------------------------------------------------


def test_expand_query_synonyms_deduped_when_two_phrases_share_a_synonym() -> None:
    # "outdoor unit mati" and "compressor mati" both map to "compressor
    # stop" — it must appear only once in the deduped `synonyms` tuple.
    result = qe.expand_query("outdoor unit mati, compressor mati juga")
    assert result.synonyms.count("compressor stop") == 1


def test_expand_query_normalized_query_is_lowercased_and_stripped() -> None:
    result = qe.expand_query("  Outdoor Unit Mati  ")
    assert result.normalized_query == "outdoor unit mati"


def test_expand_query_expanded_text_starts_with_original_query() -> None:
    result = qe.expand_query("outdoor unit mati")
    assert result.expanded_query_text.startswith("outdoor unit mati")


# ---------------------------------------------------------------------------
# load_known_entities
# ---------------------------------------------------------------------------


def test_load_known_entities_collects_distinct_model_families_and_codes() -> None:
    db = _make_session()
    db.add_all(
        [
            Document(
                title="Manual A",
                filename="a.pdf",
                source_hash="hash-a",
                model_family="REYQ",
            ),
            Document(
                title="Manual B",
                filename="b.pdf",
                source_hash="hash-b",
                model_family="REYQ",
            ),
            Document(
                title="Manual C",
                filename="c.pdf",
                source_hash="hash-c",
                model_family="PUHY-P_YKB-A1",
            ),
            Document(
                title="Manual D (no model_family)",
                filename="d.pdf",
                source_hash="hash-d",
                model_family=None,
            ),
        ]
    )
    db.commit()
    db.add_all(
        [
            ErrorCode(document_id=1, code="P8"),
            ErrorCode(document_id=1, code="P8"),
            ErrorCode(document_id=1, code="U4"),
        ]
    )
    db.commit()

    known = qe.load_known_entities(db)

    assert known.model_families == frozenset({"REYQ", "PUHY-P_YKB-A1"})
    assert known.error_codes == frozenset({"P8", "U4"})


def test_load_known_entities_empty_db_returns_empty_sets() -> None:
    db = _make_session()
    known = qe.load_known_entities(db)
    assert known.model_families == frozenset()
    assert known.error_codes == frozenset()
