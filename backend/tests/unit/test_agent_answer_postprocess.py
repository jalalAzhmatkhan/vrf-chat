"""Unit tests for `app/agent/answer_postprocess.py` (C2.3) — the
non-negotiable §5.1 layer-3 marker validation gate + the "never invent"
safety net."""

from __future__ import annotations

from app.agent import answer_postprocess as pp
from app.agent.context_builder import BuiltContext, ContextElement
from app.agent.schemas import Citation, TechnicalAnswer, Warning


def _element(element_id: int, **overrides: object) -> ContextElement:
    base: dict[str, object] = dict(
        element_id=element_id,
        element_type="paragraph",
        text="Some element text",
        image_uri=None,
        visual_description=None,
        document_id=3,
        page=238,
    )
    base.update(overrides)
    return ContextElement(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# postprocess_answer — marker validation
# ---------------------------------------------------------------------------


def test_postprocess_answer_no_markers_unchanged() -> None:
    answer = TechnicalAnswer(answer="Just plain text.", confidence=0.8)
    context = BuiltContext()

    result = pp.postprocess_answer(answer, context)

    assert result.answer.answer == "Just plain text."
    assert result.stripped_element_ids == ()
    assert result.backfilled_element_ids == ()


def test_postprocess_answer_valid_marker_kept_and_citation_backfilled() -> None:
    element = _element(4961, element_type="icon", image_uri="s3://bucket/icon.png")
    context = BuiltContext(elements_by_id={4961: element})
    answer = TechnicalAnswer(answer="Press the button {{el:4961}} to continue.", confidence=0.9)

    result = pp.postprocess_answer(answer, context)

    assert "{{el:4961}}" in result.answer.answer
    assert result.stripped_element_ids == ()
    assert result.backfilled_element_ids == (4961,)
    assert len(result.answer.citations) == 1
    citation = result.answer.citations[0]
    assert citation.element_id == "4961"
    assert citation.document_id == "3"
    assert citation.page == 238
    assert citation.element_type == "icon"
    assert citation.image_uri == "s3://bucket/icon.png"


def test_postprocess_answer_invalid_marker_stripped() -> None:
    context = BuiltContext(elements_by_id={})
    answer = TechnicalAnswer(answer="See figure {{el:9999}} for details.", confidence=0.9)

    result = pp.postprocess_answer(answer, context)

    assert "{{el:9999}}" not in result.answer.answer
    assert result.stripped_element_ids == (9999,)
    assert result.answer.citations == []
    assert "See figure" in result.answer.answer
    assert "for details." in result.answer.answer


def test_postprocess_answer_mixed_valid_and_invalid_markers() -> None:
    context = BuiltContext(elements_by_id={1: _element(1)})
    answer = TechnicalAnswer(
        answer="Refer to {{el:1}} and also {{el:2}}.", confidence=0.9
    )

    result = pp.postprocess_answer(answer, context)

    assert "{{el:1}}" in result.answer.answer
    assert "{{el:2}}" not in result.answer.answer
    assert result.stripped_element_ids == (2,)
    assert result.backfilled_element_ids == (1,)


def test_postprocess_answer_marker_already_has_citation_not_duplicated() -> None:
    element = _element(1)
    context = BuiltContext(elements_by_id={1: element})
    existing_citation = Citation(
        document_id="3", page=238, element_id="1", element_type="paragraph", quote="custom quote"
    )
    answer = TechnicalAnswer(
        answer="See {{el:1}}.", confidence=0.9, citations=[existing_citation]
    )

    result = pp.postprocess_answer(answer, context)

    assert len(result.answer.citations) == 1
    assert result.answer.citations[0].quote == "custom quote"
    assert result.backfilled_element_ids == ()


def test_postprocess_answer_collapses_double_spaces_left_by_stripped_marker() -> None:
    context = BuiltContext(elements_by_id={})
    answer = TechnicalAnswer(answer="See {{el:1}} here.", confidence=0.9)

    result = pp.postprocess_answer(answer, context)

    assert "  " not in result.answer.answer


def test_postprocess_answer_non_visual_element_no_image_fields() -> None:
    element = _element(1, element_type="table")
    context = BuiltContext(elements_by_id={1: element})
    answer = TechnicalAnswer(answer="See {{el:1}}.", confidence=0.9)

    result = pp.postprocess_answer(answer, context)

    citation = result.answer.citations[0]
    assert citation.image_uri is None
    assert citation.visual_description is None


def test_postprocess_answer_element_page_none_defaults_to_zero() -> None:
    element = _element(1, page=None)
    context = BuiltContext(elements_by_id={1: element})
    answer = TechnicalAnswer(answer="See {{el:1}}.", confidence=0.9)

    result = pp.postprocess_answer(answer, context)

    assert result.answer.citations[0].page == 0


# ---------------------------------------------------------------------------
# postprocess_answer — F2-02 citations[] whitelist validation
# ---------------------------------------------------------------------------


def test_postprocess_answer_citation_with_nonexistent_element_id_dropped() -> None:
    context = BuiltContext(elements_by_id={})
    fabricated = Citation(
        document_id="3", page=282, element_id="Revision History", element_type="section"
    )
    answer = TechnicalAnswer(
        answer="The capital city of France is Paris.",
        confidence=1.0,
        citations=[fabricated],
    )

    result = pp.postprocess_answer(answer, context)

    assert result.answer.citations == []
    assert result.dropped_citation_ids == ("Revision History",)


def test_postprocess_answer_citation_with_non_numeric_element_id_dropped() -> None:
    context = BuiltContext(elements_by_id={1: _element(1)})
    fabricated = Citation(document_id="3", page=9, element_id="3.51", element_type="paragraph")
    answer = TechnicalAnswer(answer="See the manual.", confidence=0.7, citations=[fabricated])

    result = pp.postprocess_answer(answer, context)

    assert result.answer.citations == []
    assert result.dropped_citation_ids == ("3.51",)


def test_postprocess_answer_citation_metadata_overwritten_from_context() -> None:
    # Element 6127 is really an icon on page 133 — the model sends wrong
    # page/type (mirrors the exact QA repro: model claimed page=60/"text").
    element = _element(6127, element_type="icon", image_uri="s3://bucket/icon.png", page=133)
    context = BuiltContext(elements_by_id={6127: element})
    wrong_metadata_citation = Citation(
        document_id="999", page=60, element_id="6127", element_type="text", quote="model's quote"
    )
    answer = TechnicalAnswer(
        answer="Press the button.", confidence=0.9, citations=[wrong_metadata_citation]
    )

    result = pp.postprocess_answer(answer, context)

    assert len(result.answer.citations) == 1
    citation = result.answer.citations[0]
    assert citation.document_id == "3"
    assert citation.page == 133
    assert citation.element_type == "icon"
    assert citation.image_uri == "s3://bucket/icon.png"
    # `quote` is the one field left as the model provided it.
    assert citation.quote == "model's quote"
    assert result.dropped_citation_ids == ()


def test_postprocess_answer_citations_mixed_valid_and_fabricated() -> None:
    element = _element(7814, element_type="figure", page=276)
    context = BuiltContext(elements_by_id={7814: element})
    valid = Citation(document_id="3", page=276, element_id="7814", element_type="figure")
    fabricated = Citation(document_id="3", page=282, element_id="99999", element_type="section")
    answer = TechnicalAnswer(
        answer="See the wiring diagram.",
        confidence=0.85,
        citations=[valid, fabricated],
    )

    result = pp.postprocess_answer(answer, context)

    assert len(result.answer.citations) == 1
    assert result.answer.citations[0].element_id == "7814"
    assert result.dropped_citation_ids == ("99999",)


def test_postprocess_answer_out_of_scope_repro_ends_with_no_citations() -> None:
    """Deterministic reproduction of the exact QA repro (phase-2-qa-report.md
    F2-02/F2-03): an out-of-scope question answered with 2 fabricated
    citations to real-looking pages must end up with zero citations after
    validation — which is what lets `enforce_never_invent_safety_net` (F2-03)
    correctly see "no legitimate evidence" once this runs first."""
    context = BuiltContext(elements_by_id={})
    answer = TechnicalAnswer(
        answer=(
            "The capital city of France is Paris. The winner of the 2018 FIFA "
            "World Cup was France."
        ),
        confidence=1.0,
        citations=[
            Citation(
                document_id="3", page=282, element_id="Revision History", element_type="section"
            ),
            Citation(document_id="3", page=286, element_id="13", element_type="paragraph"),
        ],
    )

    result = pp.postprocess_answer(answer, context)

    assert result.answer.citations == []
    assert set(result.dropped_citation_ids) == {"Revision History", "13"}


def test_postprocess_answer_no_citations_no_markers_dropped_ids_empty() -> None:
    context = BuiltContext(elements_by_id={})
    answer = TechnicalAnswer(answer="Just plain text.", confidence=0.8)

    result = pp.postprocess_answer(answer, context)

    assert result.dropped_citation_ids == ()


# ---------------------------------------------------------------------------
# enforce_never_invent_safety_net
# ---------------------------------------------------------------------------


def test_safety_net_already_refused_left_untouched() -> None:
    answer = TechnicalAnswer(answer="I don't know.", confidence=0.0, refused=True)

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=1, any_chunks_retrieved=False
    )

    assert result is answer


def test_safety_net_no_tool_calls_conversational_turn_untouched() -> None:
    answer = TechnicalAnswer(answer="Hello! How can I help?", confidence=0.9)

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=0, any_chunks_retrieved=False
    )

    assert result.refused is False
    assert result is answer


def test_safety_net_tool_called_but_nothing_retrieved_and_no_citations_forces_refuse() -> None:
    answer = TechnicalAnswer(answer="Here is a made-up answer.", confidence=0.9)

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=1, any_chunks_retrieved=False
    )

    assert result.refused is True
    assert any(w.message == pp.NO_EVIDENCE_WARNING.message for w in result.warnings)


def test_safety_net_tool_called_and_chunks_retrieved_not_forced() -> None:
    answer = TechnicalAnswer(answer="Grounded answer.", confidence=0.9)

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=1, any_chunks_retrieved=True
    )

    assert result.refused is False


def test_safety_net_tool_called_no_chunks_but_has_citations_not_forced() -> None:
    citation = Citation(document_id="3", page=1, element_id="1", element_type="paragraph")
    answer = TechnicalAnswer(answer="Answer with citation.", confidence=0.9, citations=[citation])

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=1, any_chunks_retrieved=False
    )

    assert result.refused is False


# ---------------------------------------------------------------------------
# enforce_never_invent_safety_net — §6.1 relevance floor
# (Documentation/system-design/03-retrieval-chunking.md §6.1.4 test matrix)
# ---------------------------------------------------------------------------


def test_safety_net_below_threshold_forces_refuse_even_with_whitelisted_citation() -> None:
    """(a) — the real F2-03 case, not the empty-context version: a citation
    that legitimately passed F2-02's whitelist (its element_id really was
    retrieved this turn) must STILL be forced to `refused=True` if every
    semantic search this turn scored below MIN_RELEVANCE_SCORE — F2-02
    proves "retrieved", not "relevant"."""
    citation = Citation(document_id="3", page=282, element_id="7814", element_type="figure")
    answer = TechnicalAnswer(
        answer="The capital city of France is Paris.",
        confidence=1.0,
        citations=[citation],
    )

    result = pp.enforce_never_invent_safety_net(
        answer,
        tool_call_count=1,
        any_chunks_retrieved=True,
        max_dense_relevance_score=0.6609,  # exact round-1 out-of-scope ceiling, §6.1.3
        any_exact_evidence_found=False,
    )

    assert result.refused is True
    assert any(w.message == pp.NO_EVIDENCE_WARNING.message for w in result.warnings)


def test_safety_net_above_threshold_not_forced() -> None:
    """(b) regression guard — a genuinely relevant turn must never be forced
    to refuse, including right at the calibrated in-scope floor (0.7486,
    §6.1.3 round 2 "in_scope_awkward" minimum)."""
    citation = Citation(document_id="3", page=238, element_id="42", element_type="icon")
    answer = TechnicalAnswer(
        answer="Grounded answer.", confidence=0.9, citations=[citation]
    )

    result = pp.enforce_never_invent_safety_net(
        answer,
        tool_call_count=1,
        any_chunks_retrieved=True,
        max_dense_relevance_score=0.7486,
        any_exact_evidence_found=False,
    )

    assert result.refused is False


def test_safety_net_exactly_at_threshold_not_forced() -> None:
    """Boundary check — the gate is `< MIN_RELEVANCE_SCORE`, so a score
    exactly equal to the threshold must NOT be forced (only strictly below)."""
    answer = TechnicalAnswer(answer="Grounded answer.", confidence=0.9)

    result = pp.enforce_never_invent_safety_net(
        answer,
        tool_call_count=1,
        any_chunks_retrieved=True,
        max_dense_relevance_score=pp.MIN_RELEVANCE_SCORE,
        any_exact_evidence_found=False,
    )

    assert result.refused is False


def test_safety_net_below_threshold_but_exact_evidence_found_not_forced() -> None:
    """(c) regression guard — `any_exact_evidence_found=True` unconditionally
    exempts the turn from the relevance floor, even if a separate low-scoring
    exploratory semantic call also happened this turn."""
    citation = Citation(document_id="3", page=60, element_id="13", element_type="table")
    answer = TechnicalAnswer(
        answer="Error code P8 means transmission failure.",
        confidence=0.9,
        citations=[citation],
    )

    result = pp.enforce_never_invent_safety_net(
        answer,
        tool_call_count=1,
        any_chunks_retrieved=True,
        max_dense_relevance_score=0.4932,  # well below threshold
        any_exact_evidence_found=True,
    )

    assert result.refused is False


def test_safety_net_no_semantic_search_this_turn_relevance_branch_inapplicable() -> None:
    """`max_dense_relevance_score is None` (no semantic-search tool called
    this turn, e.g. only `tool_get_document_page`) must never trigger the
    relevance-floor branch — there is no signal to threshold against."""
    answer = TechnicalAnswer(answer="Here is page 60.", confidence=0.9)

    result = pp.enforce_never_invent_safety_net(
        answer,
        tool_call_count=1,
        any_chunks_retrieved=True,
        max_dense_relevance_score=None,
        any_exact_evidence_found=False,
    )

    assert result.refused is False


def test_safety_net_defaults_preserve_pre_6_1_behavior() -> None:
    """Sanity: omitting the two new §6.1 kwargs entirely (as every call site
    predating this change does) must behave identically to before — this is
    exactly what F2-03's original (pre-§6.1) fix relied on."""
    citation = Citation(document_id="3", page=282, element_id="99", element_type="section")
    answer = TechnicalAnswer(answer="Some answer.", confidence=0.9, citations=[citation])

    result = pp.enforce_never_invent_safety_net(
        answer, tool_call_count=1, any_chunks_retrieved=True
    )

    assert result.refused is False


def test_no_evidence_warning_severity_is_note() -> None:
    assert pp.NO_EVIDENCE_WARNING.severity == "note"


def test_warning_import_used() -> None:
    # Sanity: Warning is re-exported/usable from this module's imports.
    assert Warning(message="x").severity == "note"


# ---------------------------------------------------------------------------
# log_confidence_anomaly_if_needed — F2-09, §5.0
# ---------------------------------------------------------------------------


def test_log_confidence_anomaly_logged_when_zero_and_not_refused(caplog) -> None:
    answer = TechnicalAnswer(answer="Looks fine.", confidence=0.0, refused=False)

    with caplog.at_level("WARNING"):
        pp.log_confidence_anomaly_if_needed(answer)

    assert any(
        r.message == "agent.answer.confidence_zero_not_refused_anomaly" for r in caplog.records
    )


def test_log_confidence_anomaly_not_logged_when_refused(caplog) -> None:
    answer = TechnicalAnswer(answer="", confidence=0.0, refused=True)

    with caplog.at_level("WARNING"):
        pp.log_confidence_anomaly_if_needed(answer)

    assert not any(
        r.message == "agent.answer.confidence_zero_not_refused_anomaly" for r in caplog.records
    )


def test_log_confidence_anomaly_not_logged_when_confidence_nonzero(caplog) -> None:
    answer = TechnicalAnswer(answer="Grounded.", confidence=0.8, refused=False)

    with caplog.at_level("WARNING"):
        pp.log_confidence_anomaly_if_needed(answer)

    assert not any(
        r.message == "agent.answer.confidence_zero_not_refused_anomaly" for r in caplog.records
    )
