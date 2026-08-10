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
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.domain.vrf_vocabulary import (
    COMPONENT_KEYWORDS,
    CONNECTOR_ID_PATTERN,
    ERROR_CODE_PATTERN,
    SENSOR_ID_PATTERN,
)
from app.ingestion.docling_parser import DoclingParseResult, ElementDraft
from app.ingestion.paddleocr_vl_cascade import CascadeResult

# Confidence bands — deliberately conservative and documented, not tuned
# against a labeled dataset (none exists yet for KG candidates). Identifier
# regex matches (TH.../CN...) are higher confidence than free-text keyword
# matches (which can be part of unrelated prose); the error code pattern is
# the lowest confidence of all (see vrf_vocabulary.py ERROR_CODE_PATTERN
# docstring — single-letter prefixes collide with ordinary abbreviations).
CONFIDENCE_VLM_COMPONENT = 0.6
CONFIDENCE_VLM_CONNECTION = 0.5
CONFIDENCE_TEXT_KEYWORD = 0.5
CONFIDENCE_SENSOR_ID = 0.7
CONFIDENCE_CONNECTOR_ID = 0.7
CONFIDENCE_ERROR_CODE_ELEMENT_TYPE = 0.8
CONFIDENCE_ERROR_CODE_PATTERN = 0.4

RELATION_CONNECTED_TO = "CONNECTED_TO"

_CONNECTION_SPLIT_PATTERN = re.compile(r"\s*(?:->|-{1,2}>|\bto\b)\s*", re.IGNORECASE)


@dataclass(slots=True)
class KGCandidateEntity:
    name: str
    entity_type: str
    confidence: float
    source_document: str
    page: int
    element_id: int


@dataclass(slots=True)
class KGCandidateRelation:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_document: str
    page: int
    element_id: int


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
            )
        )

    return result


def extract_from_text(element: ElementDraft, document_ref: str) -> ElementKGCandidates:
    result = ElementKGCandidates()
    if not element.text:
        return result

    text = element.text
    text_lower = text.lower()

    for keyword, entity_type in COMPONENT_KEYWORDS.items():
        if keyword in text_lower:
            result.entities.append(
                KGCandidateEntity(
                    name=keyword,
                    entity_type=entity_type,
                    confidence=CONFIDENCE_TEXT_KEYWORD,
                    source_document=document_ref,
                    page=element.page_number,
                    element_id=element.local_id,
                )
            )

    for match in SENSOR_ID_PATTERN.finditer(text):
        result.entities.append(
            KGCandidateEntity(
                name=match.group(0),
                entity_type="Sensor",
                confidence=CONFIDENCE_SENSOR_ID,
                source_document=document_ref,
                page=element.page_number,
                element_id=element.local_id,
            )
        )

    for match in CONNECTOR_ID_PATTERN.finditer(text):
        result.entities.append(
            KGCandidateEntity(
                name=match.group(0),
                entity_type="Connector",
                confidence=CONFIDENCE_CONNECTOR_ID,
                source_document=document_ref,
                page=element.page_number,
                element_id=element.local_id,
            )
        )

    if element.element_type == "error_code":
        result.entities.append(
            KGCandidateEntity(
                name=text.strip(),
                entity_type="ErrorCode",
                confidence=CONFIDENCE_ERROR_CODE_ELEMENT_TYPE,
                source_document=document_ref,
                page=element.page_number,
                element_id=element.local_id,
            )
        )
    else:
        for match in ERROR_CODE_PATTERN.finditer(text):
            result.entities.append(
                KGCandidateEntity(
                    name=match.group(1),
                    entity_type="ErrorCode",
                    confidence=CONFIDENCE_ERROR_CODE_PATTERN,
                    source_document=document_ref,
                    page=element.page_number,
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
