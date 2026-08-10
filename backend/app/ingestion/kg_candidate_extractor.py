"""KG candidate entity/relation extraction, see
`Documentation/system-design/03-retrieval-chunking.md` §5 and
`02-ingestion-pipeline.md` §5 (I1.9).

Deterministic, rule-based extraction (regex + `app/domain/vrf_vocabulary.py`
dictionary) — **not an LLM call**, consistent with the project's
"deterministic tools, not model judgment" principle
(`01-architecture-overview.md` §4) applied to the ingestion side (the same
principle `app/ingestion/cascade_trigger.py`, I1.3, already follows).

Two candidate sources per element:
1. Stage 4 VLM `visual_description` (`figure_type`/`components[]`/
   `connections[]`) for figure/icon elements that went through
   PaddleOCR-VL (`app/ingestion/paddleocr_vl_cascade.py`).
2. Lightweight regex/dictionary matching against `element.text` for
   narrative/procedural text elements — component names, sensor/connector
   identifiers, candidate error codes.

Every candidate carries explicit **provenance**
(`source_document`/`page`/`element_id`/`confidence`) per
`03-retrieval-chunking.md` §5 ("jangan membuat KG tanpa provenance").
**Not** loaded into Neo4j — Fase 3 validates + loads these candidates; this
module only produces them, for the ingestion orchestrator (I1.10) to attach
to `elements.kg_candidate_entities`/`kg_candidate_relations` (jsonb, via
`app/ingestion/canonical_store.py`, I1.5).

`element_id` in every candidate is `ElementDraft.local_id` (the same
document-scoped sequential id used throughout the ingestion pipeline before
Postgres row ids exist, see `docling_parser.py` module docstring) — the
caller remaps it to the real `elements.id` using the same
`local_id -> db_id` mapping `canonical_store.store_pages_and_elements`
already produces internally.

**[KG-W1.1, 2026-08-10 — precision/recall bugfixes]** See
`Documentation/system-design/09-kg-extraction-strategy.md` §2/§5.4/§6.2
(R4 + R-addendum) for the literature/design rationale. This revision is
additionally grounded in a **live Postgres query against `document_id=3`**
(286-page Zeggo VRV IV REYQ manual, re-run 2026-08-10 to confirm/refine
System Analyst's original findings before implementing fixes), which
confirmed:

1. **Zero `CONNECTED_TO` relations ever extracted, root cause confirmed (not
   just hypothesized) — and NOT fixable in this module.**
   `SELECT count(*) FILTER (WHERE (visual_description->>'components') !=
   '[]' OR (visual_description->>'connections') != '[]' OR
   visual_description->>'figure_type' IS NOT NULL) FROM elements WHERE
   document_id=3 AND visual_description IS NOT NULL` → **0 of 2916 rows**.
   Reading `paddleocr-vl-service/app/pipeline.py`
   `PaddleOCRVLPipeline.describe_figure()` confirms why:  it only ever sets
   `description=_extract_markdown_text(result)` — `figure_type`,
   `components`, `connections` are never populated by the service (they
   stay at their dataclass defaults, `None`/`[]`), because the VLM call
   requests free-text/markdown output, not structured JSON. This is a
   **Stage 4** behavior (`paddleocr-vl-service`) — explicitly out of scope
   for this module per Wave 1 DoD ("Jangan ubah orchestrator.py atau Stage
   1-4"). `extract_from_visual_description()` below is therefore working
   correctly against a contract (`VisualDescription.components`/
   `.connections`) that real Stage 4 output never populates — reported to
   System Analyst (see STATUS REPORT) as a Stage 4 follow-up candidate
   (structured-output prompting) rather than fixed here. Left unchanged,
   still exercised by tests using a fake `CascadeResult` that DOES populate
   `components`/`connections`, so the code path keeps working the moment
   Stage 4 starts returning them.
2. `visual_description.description` (the field Stage 4 DOES populate,
   confirmed non-empty for 381/2916 rows checked) contains real,
   substantive text (wiring-diagram legends, table markdown with component
   codes, etc. — see `elements.id=7814`/`7806` in `document_id=3`) that the
   existing deterministic text matchers (component keywords, sensor/
   connector identifiers, anchored error codes) can extract *entities*
   from, same as narrative `element.text`. `extract_from_visual_description`
   now also runs `_extract_entities_from_text` against `vd.description`
   for this reason — a real, in-scope recall improvement for the VLM path
   that does not require Stage 4 changes. **Relations remain unaddressed**
   (parsing free-text markdown into reliable `subject/predicate/object`
   triples is not attempted — would violate the determinism/precision bar;
   left to Stage 4 structured-output improvements or Tier 2 R8 GLiREL-based
   relation typing, Wave 2).

**[KG-W1.2, 2026-08-10 — K1 schema fields + K3 non-goal docstring]** See
`Documentation/system-design/09-kg-extraction-strategy.md` §6.1 K1/K3 and
§9.1 (jsonb field contract). `KGCandidateEntity`/`KGCandidateRelation` gain
four new fields (`extraction_method`, `canonical_name`, `model_family`,
`justification_span`) — **no Postgres migration**, the `elements
.kg_candidate_entities`/`kg_candidate_relations` columns stay `jsonb`
(schema-flexible by design, `06-data-schema.md`). `canonical_name` and
`model_family` are added to the *shape* here but are **not populated by
this module yet**: `canonical_name` needs an alias lookup table
(`KG-W2.1`/R6, not built yet) and `model_family` needs
`documents.model_family` joined in by the caller (`KG-W1.3`/K2, the next
branch in this chain) — both stay `None` for every candidate this module
currently produces. `extraction_method`/`justification_span` ARE populated
now (see `_extract_entities_from_text`/`EXTRACTION_METHOD_*` below).

**K3 — merge-by-similarity for relations is an explicit non-goal, not an
oversight.** Per `09-kg-extraction-strategy.md` §5.2 ("Relasi: Multigraph,
Bukan Merge"): this module (and any future caller aggregating candidates
across elements, e.g. the Wave 1 DoD re-extractor) MUST NOT collapse two
`KGCandidateRelation` records into one just because their
`(subject, predicate, object)` look alike. Two `CONNECTED_TO` claims about
the same pair of entities found on two different pages are **two separate
pieces of evidence** (different `element_id`/`page`), not duplicates — each
must stay independently traceable back to its own source location
(`CLAUDE.md` §4 traceability requirement). The **only** sanctioned form of
combining evidence is cross-source agreement on the exact SAME evidence
(`KG-W1.7`/R3 — VLM and text extraction agreeing about the same
`element_id`/page), which raises confidence on a signal field rather than
deleting/merging records. If a future change ever needs relation
deduplication for a different reason, that is a new design decision
requiring System Analyst sign-off, not something to add unilaterally here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.domain.vrf_vocabulary import (
    COMPONENT_KEYWORD_PATTERNS,
    COMPONENT_KEYWORDS,
    CONNECTOR_ID_PATTERN,
    ERROR_CODE_ANCHOR_PATTERN,
    ERROR_CODE_PATTERN,
    SENSOR_ID_PATTERN,
)
from app.ingestion.docling_parser import ELEMENT_TYPE_TABLE, DoclingParseResult, ElementDraft
from app.ingestion.paddleocr_vl_cascade import CascadeResult

# Confidence bands — deliberately conservative and documented, not tuned
# against a labeled dataset (none exists yet for KG candidates). Identifier
# regex matches (TH.../CN...) are higher confidence than free-text keyword
# matches (which can be part of unrelated prose).
CONFIDENCE_VLM_COMPONENT = 0.6
CONFIDENCE_VLM_CONNECTION = 0.5
CONFIDENCE_TEXT_KEYWORD = 0.5
CONFIDENCE_SENSOR_ID = 0.7
CONFIDENCE_CONNECTOR_ID = 0.7

# **[KG-W1.1 fix]** `element_type == "error_code"` (formerly
# `CONFIDENCE_ERROR_CODE_ELEMENT_TYPE = 0.8`) was confirmed dead code — no
# Docling `DocItemLabel` maps to `"error_code"`
# (`docling_parser._LABEL_TO_ELEMENT_TYPE`), and no Stage 3/4 code re-tags
# any element to it either, so this branch has never once fired on real
# ingested data (`document_id=3`: 0 elements with `element_type ==
# "error_code"`, despite 192 `ERROR_CODE_PATTERN` matches all landing in the
# `else` branch at the old flat `CONFIDENCE_ERROR_CODE_PATTERN = 0.4`).
# Replaced with the heuristic explicitly suggested as an alternative in
# `09-kg-extraction-strategy.md` §5.4: "elemen tabel + heading error
# code/malfunction code terdekat" — implemented as
# `_has_error_code_anchor()` below (checks `element.text` and
# `element.section_path` for `ERROR_CODE_ANCHOR_PATTERN`), with a table
# element getting the same elevated confidence the old (dead) branch used
# to grant, a non-table element with an anchor getting a middle tier, and
# an unanchored match getting a confidence *lower* than the old flat value
# (empirically, unanchored matches are mostly noise — see
# `ERROR_CODE_ANCHOR_PATTERN` docstring in `vrf_vocabulary.py`).
CONFIDENCE_ERROR_CODE_TABLE_ANCHORED = 0.75
CONFIDENCE_ERROR_CODE_ANCHORED = 0.6
CONFIDENCE_ERROR_CODE_UNANCHORED = 0.3

# **[KG-W1.1 fix]** Applied to `COMPONENT_KEYWORDS` matches on an element
# that matches MORE than this many *distinct* keywords at once — empirically
# a strong signal the element is a glossary/spec-list page (every term
# defined in one place) rather than a locally-relevant reference (confirmed:
# `document_id=3` has a real element matching 14/20 `COMPONENT_KEYWORDS`
# simultaneously). Such matches are kept (never invent-by-omission — Fase 3
# review can still see them) but at reduced confidence.
COMPONENT_KEYWORD_GLOSSARY_THRESHOLD = 5
CONFIDENCE_TEXT_KEYWORD_GLOSSARY_PAGE = 0.2

RELATION_CONNECTED_TO = "CONNECTED_TO"

_CONNECTION_SPLIT_PATTERN = re.compile(r"\s*(?:->|-{1,2}>|\bto\b)\s*", re.IGNORECASE)

# **[KG-W1.2, K1]** `extraction_method` values — identifies which matcher
# produced a candidate, see `09-kg-extraction-strategy.md` §9.1
# (`"regex_sensor_id" | "dict_keyword" | "vlm_component" | dst.`). The
# `_VLM_DESCRIPTION_SUFFIX` variants are used when `_extract_entities_from_text`
# runs against `visual_description.description` (VLM markdown) rather than
# `element.text` (Docling-native narrative text) — same matcher logic, but
# worth distinguishing because the two source fields have different
# reliability profiles (see module docstring finding #2).
EXTRACTION_METHOD_DICT_KEYWORD = "dict_keyword"
EXTRACTION_METHOD_REGEX_SENSOR_ID = "regex_sensor_id"
EXTRACTION_METHOD_REGEX_CONNECTOR_ID = "regex_connector_id"
EXTRACTION_METHOD_REGEX_ERROR_CODE = "regex_error_code"
EXTRACTION_METHOD_VLM_COMPONENT = "vlm_component"
EXTRACTION_METHOD_VLM_CONNECTION = "vlm_connection"
_VLM_DESCRIPTION_SUFFIX = "_vlm_description"


@dataclass(slots=True)
class KGCandidateEntity:
    name: str
    entity_type: str
    confidence: float
    source_document: str
    page: int
    element_id: int
    # **[KG-W1.2, K1]** Every candidate THIS MODULE produces always sets
    # `extraction_method` explicitly (see `EXTRACTION_METHOD_*` constants
    # above) — the `""` default exists only so external callers/test
    # fixtures constructing a `KGCandidateEntity` directly (e.g.
    # `canonical_store.py`'s tests) don't have to be updated just because
    # this module gained a new field; it is not a meaningful value on its
    # own and should never appear on a candidate this module's own
    # `extract_*` functions returned. `canonical_name`/`model_family` are
    # schema-only for now (always `None`, populated by later stages — see
    # module docstring); `justification_span` is `[start, end]` character
    # offsets into the SOURCE TEXT the match came from (`element.text` or
    # `visual_description.description`), `None` when not computable
    # (VLM-structured `components[]`, or no natural offset).
    extraction_method: str = ""
    canonical_name: str | None = None
    model_family: str | None = None
    justification_span: list[int] | None = None


@dataclass(slots=True)
class KGCandidateRelation:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_document: str
    page: int
    element_id: int
    # See `KGCandidateEntity` field docstring above for the rationale of
    # each of these (identical fields, same defaults/semantics).
    extraction_method: str = ""
    canonical_name: str | None = None
    model_family: str | None = None
    justification_span: list[int] | None = None


@dataclass(slots=True)
class ElementKGCandidates:
    entities: list[KGCandidateEntity] = field(default_factory=list)
    relations: list[KGCandidateRelation] = field(default_factory=list)


def _parse_connection(connection: str) -> tuple[str, str] | None:
    """Best-effort split of a VLM-produced connection string (e.g.
    `"compressor -> TH3"`, `"compressor to TH3"`) into `(subject, object)`.
    Returns `None` if no separator is recognized (kept as a
    `visual_description` free-text field only, not turned into a relation)."""
    parts = _CONNECTION_SPLIT_PATTERN.split(connection, maxsplit=1)
    if len(parts) != 2:
        return None
    subject, obj = parts[0].strip(), parts[1].strip()
    if not subject or not obj:
        return None
    return subject, obj


def _has_error_code_anchor(text: str | None, section_path: list[str]) -> bool:
    """**[KG-W1.1, R4]** True if `ERROR_CODE_ANCHOR_PATTERN` (see
    `vrf_vocabulary.py`) matches the element's own text or its heading
    hierarchy (`section_path` — e.g. a table under a "Malfunction Code
    Table" heading counts as anchored even if the word never appears in the
    table's own cells)."""
    haystack = " ".join([text or "", *section_path])
    return bool(ERROR_CODE_ANCHOR_PATTERN.search(haystack))


def _justification_span(match: re.Match[str], *, compute: bool) -> list[int] | None:
    """**[KG-W1.2, K1]** `[start, end]` character offsets of `match` within
    its source text, or `None` when `compute` is `False` (per
    `09-kg-extraction-strategy.md` §9.1: "null untuk sumber VLM" — offsets
    into `visual_description.description` are not the contract's
    `element.text` offsets, so callers extracting from VLM text pass
    `compute=False`)."""
    if not compute:
        return None
    return [match.start(), match.end()]


def _extract_entities_from_text(
    text: str,
    *,
    element_type: str,
    section_path: list[str],
    document_ref: str,
    page_number: int,
    element_id: int,
    is_vlm_description: bool = False,
) -> list[KGCandidateEntity]:
    """Deterministic component/sensor/connector/error-code entity matchers,
    shared by `extract_from_text` (`element.text`) and
    `extract_from_visual_description` (Stage 4's `visual_description
    .description` markdown, see module docstring finding #2) — the same
    identifier/keyword is recognized identically regardless of which field
    it came from. `is_vlm_description` controls the `extraction_method`
    suffix and whether `justification_span` is computed (see K1 field
    docstrings on `KGCandidateEntity`)."""
    entities: list[KGCandidateEntity] = []
    method_suffix = _VLM_DESCRIPTION_SUFFIX if is_vlm_description else ""
    compute_span = not is_vlm_description

    # **[KG-W1.1 fix]** Word-boundary match per keyword (was `keyword in
    # text_lower`, a substring check — see `COMPONENT_KEYWORD_PATTERNS`
    # docstring). `is_glossary_like`/reduced confidence: see
    # `COMPONENT_KEYWORD_GLOSSARY_THRESHOLD` docstring above.
    keyword_matches = {
        keyword: m
        for keyword in COMPONENT_KEYWORD_PATTERNS
        if (m := COMPONENT_KEYWORD_PATTERNS[keyword].search(text)) is not None
    }
    is_glossary_like = len(keyword_matches) > COMPONENT_KEYWORD_GLOSSARY_THRESHOLD
    keyword_confidence = (
        CONFIDENCE_TEXT_KEYWORD_GLOSSARY_PAGE if is_glossary_like else CONFIDENCE_TEXT_KEYWORD
    )
    for keyword, kw_match in keyword_matches.items():
        entities.append(
            KGCandidateEntity(
                name=keyword,
                entity_type=COMPONENT_KEYWORDS[keyword],
                confidence=keyword_confidence,
                source_document=document_ref,
                page=page_number,
                element_id=element_id,
                extraction_method=EXTRACTION_METHOD_DICT_KEYWORD + method_suffix,
                justification_span=_justification_span(kw_match, compute=compute_span),
            )
        )

    for match in SENSOR_ID_PATTERN.finditer(text):
        entities.append(
            KGCandidateEntity(
                name=match.group(0),
                entity_type="Sensor",
                confidence=CONFIDENCE_SENSOR_ID,
                source_document=document_ref,
                page=page_number,
                element_id=element_id,
                extraction_method=EXTRACTION_METHOD_REGEX_SENSOR_ID + method_suffix,
                justification_span=_justification_span(match, compute=compute_span),
            )
        )

    for match in CONNECTOR_ID_PATTERN.finditer(text):
        entities.append(
            KGCandidateEntity(
                name=match.group(0),
                entity_type="Connector",
                confidence=CONFIDENCE_CONNECTOR_ID,
                source_document=document_ref,
                page=page_number,
                element_id=element_id,
                extraction_method=EXTRACTION_METHOD_REGEX_CONNECTOR_ID + method_suffix,
                justification_span=_justification_span(match, compute=compute_span),
            )
        )

    if ERROR_CODE_PATTERN.search(text):
        # **[KG-W1.1 fix, R4]** Confidence now depends on contextual anchor
        # presence (see `_has_error_code_anchor`/`CONFIDENCE_ERROR_CODE_*`
        # docstrings above) instead of the old dead `element_type ==
        # "error_code"` branch / flat confidence for everything else.
        anchor_matched = _has_error_code_anchor(text, section_path)
        if element_type == ELEMENT_TYPE_TABLE and anchor_matched:
            error_code_confidence = CONFIDENCE_ERROR_CODE_TABLE_ANCHORED
        elif anchor_matched:
            error_code_confidence = CONFIDENCE_ERROR_CODE_ANCHORED
        else:
            error_code_confidence = CONFIDENCE_ERROR_CODE_UNANCHORED
        for match in ERROR_CODE_PATTERN.finditer(text):
            entities.append(
                KGCandidateEntity(
                    name=match.group(1),
                    entity_type="ErrorCode",
                    confidence=error_code_confidence,
                    source_document=document_ref,
                    page=page_number,
                    element_id=element_id,
                    extraction_method=EXTRACTION_METHOD_REGEX_ERROR_CODE + method_suffix,
                    justification_span=_justification_span(match, compute=compute_span),
                )
            )

    return entities


def extract_from_visual_description(
    element: ElementDraft, cascade_result: CascadeResult, document_ref: str
) -> ElementKGCandidates:
    result = ElementKGCandidates()
    vd = cascade_result.visual_description
    if vd is None:
        return result

    for component in vd.components:
        result.entities.append(
            KGCandidateEntity(
                name=component,
                entity_type="Component",
                confidence=CONFIDENCE_VLM_COMPONENT,
                source_document=document_ref,
                page=element.page_number,
                element_id=element.local_id,
                extraction_method=EXTRACTION_METHOD_VLM_COMPONENT,
            )
        )

    for connection in vd.connections:
        parsed = _parse_connection(connection)
        if parsed is None:
            continue
        subject, obj = parsed
        result.relations.append(
            KGCandidateRelation(
                subject=subject,
                predicate=RELATION_CONNECTED_TO,
                object=obj,
                confidence=CONFIDENCE_VLM_CONNECTION,
                source_document=document_ref,
                page=element.page_number,
                element_id=element.local_id,
                extraction_method=EXTRACTION_METHOD_VLM_CONNECTION,
            )
        )

    if vd.description:
        # **[KG-W1.1 fix]** See module docstring finding #2 — real Stage 4
        # output never populates `components`/`connections` today, but does
        # populate `description` (markdown/free text), which the same
        # deterministic matchers used for narrative `element.text` can also
        # mine for entities.
        result.entities.extend(
            _extract_entities_from_text(
                vd.description,
                element_type=element.element_type,
                section_path=element.section_path,
                document_ref=document_ref,
                page_number=element.page_number,
                element_id=element.local_id,
                is_vlm_description=True,
            )
        )

    return result


def extract_from_text(element: ElementDraft, document_ref: str) -> ElementKGCandidates:
    result = ElementKGCandidates()
    if not element.text:
        return result

    result.entities.extend(
        _extract_entities_from_text(
            element.text,
            element_type=element.element_type,
            section_path=element.section_path,
            document_ref=document_ref,
            page_number=element.page_number,
            element_id=element.local_id,
        )
    )
    return result


def extract_kg_candidates(
    parse_result: DoclingParseResult,
    document_ref: str,
    cascade_results: list[CascadeResult] | None = None,
) -> dict[int, ElementKGCandidates]:
    """Extract candidates for every element in `parse_result`, keyed by
    `ElementDraft.local_id`. Elements with no candidates found are omitted
    from the returned dict (not present with empty lists)."""
    cascade_by_local_id = {
        r.task.element_local_id: r
        for r in (cascade_results or [])
        if r.task.element_local_id is not None
    }

    results: dict[int, ElementKGCandidates] = {}
    for element in parse_result.elements:
        candidates = ElementKGCandidates()

        cascade_result = cascade_by_local_id.get(element.local_id)
        if cascade_result is not None:
            vlm_candidates = extract_from_visual_description(element, cascade_result, document_ref)
            candidates.entities.extend(vlm_candidates.entities)
            candidates.relations.extend(vlm_candidates.relations)

        text_candidates = extract_from_text(element, document_ref)
        candidates.entities.extend(text_candidates.entities)
        candidates.relations.extend(text_candidates.relations)

        if candidates.entities or candidates.relations:
            results[element.local_id] = candidates

    return results


def entities_to_jsonb(entities: list[KGCandidateEntity]) -> list[dict[str, Any]]:
    return [asdict(e) for e in entities]


def relations_to_jsonb(relations: list[KGCandidateRelation]) -> list[dict[str, Any]]:
    return [asdict(r) for r in relations]
