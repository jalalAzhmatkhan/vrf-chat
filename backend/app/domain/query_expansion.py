"""MVP query expansion — domain dictionary + known entity list, **no LLM
call**, per `Documentation/system-design/03-retrieval-chunking.md` §6 (C2.2)
and the TTFT budget in `05-streaming-and-api-contract.md` §2 ("Query
preprocessing (heuristic query expansion, bukan LLM call) < 0.5 detik").

Everything in this module is a pure function over static dictionaries/regex
plus (optionally) a pre-loaded `KnownEntities` snapshot — there is
deliberately no synchronous LLM call and no synchronous DB call *inside*
`expand_query` itself, so it stays sub-millisecond regardless of database or
provider latency. `load_known_entities` (the one function that *does* touch
Postgres) is a separate, explicit step the caller (`app/agent/tools.py`,
C2.3) is responsible for invoking/caching on its own schedule — kept out of
`expand_query`'s hot path on purpose.

**Placement note**: `app/domain/vrf_vocabulary.py`'s module docstring
describes itself as "the intended home for the MVP query-expansion
dictionary ... when that's implemented". This module deliberately lives
alongside it instead of inside it — originally to avoid editing
`vrf_vocabulary.py` while a parallel Backend Engineer (KG Foundation
Wave 1) was actively working on it. That work has since merged into
`master` (`ERROR_CODE_PATTERN` renamed to `ERROR_CODE_MAIN_PATTERN` plus
new Daikin/Zeggo identifier patterns — this module was updated to match,
see the rebase fix commit) — the original conflict-avoidance reason no
longer applies, but this module still only *reads* `vrf_vocabulary.py`'s
identifier patterns/component keywords rather than duplicating them.
Consolidating the two modules is a reasonable non-blocking follow-up,
still not done here to keep this fix scoped to the QA-reported rebase
break rather than an unrelated restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.documents import Document
from app.db.models.error_codes import ErrorCode
from app.domain.vrf_vocabulary import (
    COMPONENT_KEYWORD_PATTERNS,
    CONNECTOR_ID_PATTERN,
    ERROR_CODE_MAIN_PATTERN,
    SENSOR_ID_PATTERN,
    SENSOR_ID_PATTERN_DAIKIN,
    TERMINAL_ID_PATTERN,
)

# Symptom phrase (lowercase, Indonesian/English as a technician would type
# it) -> tuple of manual-style English synonyms to add to the query text
# sent to the dense+sparse embedding legs (`app/retrieval/hybrid_search.py`).
# Deliberately a small, high-precision starting set covering common VRF/VRV
# troubleshooting phrasing — per `03-retrieval-chunking.md` §6, LLM-based
# expansion is the documented Fase-2-if-needed escape hatch for coverage
# gaps evaluation surfaces, not something to over-build speculatively here.
SYMPTOM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "outdoor unit mati": ("outdoor unit stops", "compressor stop", "protective operation"),
    "outdoor unit tidak mau nyala": (
        "outdoor unit does not start",
        "compressor stop",
        "protective operation",
    ),
    "outdoor unit trip": ("outdoor unit stops", "protective operation"),
    "compressor tidak jalan": ("compressor does not operate", "compressor stop"),
    "compressor mati": ("compressor stop", "compressor does not operate"),
    "unit tidak dingin": ("insufficient cooling", "cooling capacity shortage"),
    "ac tidak dingin": ("insufficient cooling", "cooling capacity shortage"),
    "tidak dingin": ("insufficient cooling", "cooling capacity shortage"),
    "kebocoran refrigerant": ("refrigerant leak", "gas leak"),
    "bocor refrigerant": ("refrigerant leak", "gas leak"),
    "suara berisik": ("abnormal noise", "noise"),
    "bunyi berisik": ("abnormal noise", "noise"),
    "remote tidak berfungsi": ("remote controller malfunction", "remote control error"),
    "remote tidak bisa": ("remote controller malfunction", "remote control error"),
    "indoor unit bocor air": ("water leak", "drain pump error"),
    "bocor air": ("water leak", "drain pump error"),
    "fan tidak berputar": ("fan motor does not rotate", "fan motor error"),
    "kipas tidak berputar": ("fan motor does not rotate", "fan motor error"),
    "tekanan tinggi": ("high pressure", "high pressure switch"),
    "tekanan rendah": ("low pressure", "low pressure switch"),
    "error code": ("error code", "malfunction code"),
    "kode error": ("error code", "malfunction code"),
}


@dataclass(slots=True)
class KnownEntities:
    """A snapshot of identifiers already indexed in Postgres — used to
    corroborate (not replace) regex-based identifier detection in
    `expand_query`. See `load_known_entities`."""

    model_families: frozenset[str] = field(default_factory=frozenset)
    error_codes: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class ExpandedQuery:
    original_query: str
    normalized_query: str
    matched_synonym_phrases: tuple[str, ...]
    synonyms: tuple[str, ...]
    detected_error_codes: tuple[str, ...]
    detected_sensor_ids: tuple[str, ...]
    detected_connector_ids: tuple[str, ...]
    detected_terminal_ids: tuple[str, ...]
    detected_components: tuple[str, ...]
    detected_model_families: tuple[str, ...]
    expanded_query_text: str


def load_known_entities(db: Session) -> KnownEntities:
    """Load the "known entity list" (indexed model families + error codes)
    from Postgres. A separate, explicit step from `expand_query` (see module
    docstring) — call this once per request (or with the caller's own
    caching strategy) rather than treating it as part of the hot,
    zero-I/O expansion path.

    Note (see `app/retrieval/hybrid_search.py` module docstring): the
    `error_codes` table is currently unpopulated by any ingestion stage
    (verified against the real document_id=3 corpus), so
    `KnownEntities.error_codes` will typically be empty today —
    `expand_query`'s regex-based detection is not degraded by this (it does
    not require a non-empty known list to detect candidate codes), it simply
    means the "corroborated by known entity list" signal isn't available
    yet for error codes specifically.
    """
    model_families = {
        row[0]
        for row in db.execute(
            select(Document.model_family).where(Document.model_family.is_not(None)).distinct()
        ).all()
        if row[0]
    }
    error_codes = {row[0] for row in db.execute(select(ErrorCode.code).distinct()).all() if row[0]}
    return KnownEntities(
        model_families=frozenset(model_families), error_codes=frozenset(error_codes)
    )


def _dedupe_preserve_order(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return tuple(ordered)


def expand_query(query: str, known_entities: KnownEntities | None = None) -> ExpandedQuery:
    """Deterministic, zero-I/O query expansion — see module docstring.

    Combines three signals, all heuristic (no LLM call):
    1. Domain dictionary synonym lookup (`SYMPTOM_SYNONYMS`) on substring
       match against the normalized (lowercased) query.
    2. Regex identifier detection (`vrf_vocabulary.ERROR_CODE_MAIN_PATTERN`/
       `SENSOR_ID_PATTERN`+`SENSOR_ID_PATTERN_DAIKIN`/`CONNECTOR_ID_PATTERN`/
       `TERMINAL_ID_PATTERN`), run case-insensitively (technicians may type
       `p8` instead of `P8`) — detected identifiers are normalized to
       uppercase, their canonical manual form. Both the Mitsubishi
       (`TH#`/`CN#`) and Daikin/Zeggo (`R#T`/`X#M`) identifier conventions
       are matched unconditionally (UNION, not per-vendor routing) — see
       `vrf_vocabulary.py`'s "SA-KG.1" module docstring section for why this
       is safe; our actual indexed corpus (`document_id=3`) is a Daikin/
       Zeggo manual, so the Daikin patterns matter in practice today.
    3. Component keyword detection (`vrf_vocabulary.COMPONENT_KEYWORD_PATTERNS`
       — word-boundary-anchored, not a naive substring check, so "fan
       motor" doesn't spuriously match inside "fan motors").

    `known_entities` (optional) is used only to corroborate/extend exact
    identifier matches already indexed (e.g. a `model_family` string that
    appears verbatim in the query but doesn't match the identifier regexes)
    — it never gates or filters out regex-detected identifiers, since a
    real identifier may legitimately not be "known" yet (a code documented
    in a *different* manual than the one loaded).
    """
    normalized_query = query.strip().lower()
    upper_query = query.upper()

    matched_synonym_phrases = tuple(
        phrase for phrase in SYMPTOM_SYNONYMS if phrase in normalized_query
    )
    synonyms = _dedupe_preserve_order(
        [syn for phrase in matched_synonym_phrases for syn in SYMPTOM_SYNONYMS[phrase]]
    )

    detected_error_codes = set(ERROR_CODE_MAIN_PATTERN.findall(upper_query))
    detected_sensor_ids = set(SENSOR_ID_PATTERN.findall(upper_query)) | set(
        SENSOR_ID_PATTERN_DAIKIN.findall(upper_query)
    )
    detected_connector_ids = set(CONNECTOR_ID_PATTERN.findall(upper_query))
    detected_terminal_ids = set(TERMINAL_ID_PATTERN.findall(upper_query))
    detected_model_families: set[str] = set()

    if known_entities is not None:
        for code in known_entities.error_codes:
            if code and code.upper() in upper_query:
                detected_error_codes.add(code.upper())
        for model_family in known_entities.model_families:
            if model_family and model_family.upper() in upper_query:
                detected_model_families.add(model_family)

    detected_components = tuple(
        keyword
        for keyword, pattern in COMPONENT_KEYWORD_PATTERNS.items()
        if pattern.search(query)
    )

    expanded_query_text = " ".join(
        _dedupe_preserve_order(
            [
                query,
                *synonyms,
                *sorted(detected_error_codes),
                *sorted(detected_sensor_ids),
                *sorted(detected_connector_ids),
                *sorted(detected_terminal_ids),
                *detected_components,
                *sorted(detected_model_families),
            ]
        )
    ).strip()

    return ExpandedQuery(
        original_query=query,
        normalized_query=normalized_query,
        matched_synonym_phrases=matched_synonym_phrases,
        synonyms=synonyms,
        detected_error_codes=tuple(sorted(detected_error_codes)),
        detected_sensor_ids=tuple(sorted(detected_sensor_ids)),
        detected_connector_ids=tuple(sorted(detected_connector_ids)),
        detected_terminal_ids=tuple(sorted(detected_terminal_ids)),
        detected_components=detected_components,
        detected_model_families=tuple(sorted(detected_model_families)),
        expanded_query_text=expanded_query_text,
    )
