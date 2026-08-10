"""VRF/VRV domain vocabulary — shared dictionary of component/sensor/
connector terminology and identifier patterns, see
`Documentation/system-design/01-architecture-overview.md` §3
(`app/domain/` = "kamus istilah VRF/VRV, tipe entity domain, helper
HVAC-specific").

Primary consumer today: `app/ingestion/kg_candidate_extractor.py` (I1.9).
Also the intended home for the MVP query-expansion dictionary
(`03-retrieval-chunking.md` §6 "domain dictionary... mapping 'outdoor unit
mati terus' -> sinonim manual") when that's implemented (Fase 2/later
retrieval work) — kept as a single source of truth rather than duplicating
VRF terminology across modules.

Entity type names match the KG node types in `03-retrieval-chunking.md` §5:
Product, Model, System, Component, PCB, Sensor, Connector, Terminal,
ErrorCode, Symptom, Procedure, Measurement, Parameter, ServiceManual,
Figure, Page.

**[KG-W1.1, 2026-08-10 — precision/recall bugfixes, see
`Documentation/system-design/09-kg-extraction-strategy.md` §5.4/§6.2 R4 +
R-addendum]** Verified directly against real ingested data (`document_id=3`,
286-page Zeggo VRV IV REYQ manual, via live Postgres query, not just the
literature-review hypothesis) — see `app/ingestion/kg_candidate_extractor.py`
module docstring for the full empirical findings this revision responds to.
"""

from __future__ import annotations

import re

# Lowercase component/part keyword -> KG node entity type. Deliberately a
# small, high-precision starting set (common VRF/VRV refrigerant-circuit and
# electrical parts) rather than an exhaustive parts list — false negatives
# (missed terms) are safer than false positives for *candidate* KG
# extraction, which downstream Fase 3 human review can expand.
COMPONENT_KEYWORDS: dict[str, str] = {
    "compressor": "Component",
    "condenser": "Component",
    "evaporator": "Component",
    "expansion valve": "Component",
    "solenoid valve": "Component",
    "fan motor": "Component",
    "accumulator": "Component",
    "four-way valve": "Component",
    "four way valve": "Component",
    "reversing valve": "Component",
    "check valve": "Component",
    "strainer": "Component",
    "muffler": "Component",
    "oil separator": "Component",
    "pcb": "PCB",
    "printed circuit board": "PCB",
    "inverter pcb": "PCB",
    "thermistor": "Sensor",
    "pressure sensor": "Sensor",
    "pressure switch": "Sensor",
    "high pressure switch": "Sensor",
    "low pressure switch": "Sensor",
    "terminal block": "Terminal",
}

# **[KG-W1.1 fix]** Word-boundary-anchored per-keyword pattern, precompiled
# once at import time (not per-call) — replaces the original `keyword in
# text_lower` substring check, which matched e.g. "fan motor" inside "fan
# motors" or "pcb" inside some unrelated longer token. `re.escape` handles
# keywords containing regex-special characters (e.g. "four-way valve"'s
# hyphen, which is not itself a regex metacharacter but escaping is cheap
# insurance). Confirmed empirically necessary (not just a theoretical
# S1/literature concern): a real element (document_id=3, page 25/27)
# matched 14/20 `COMPONENT_KEYWORDS` simultaneously under the old substring
# check — see `kg_candidate_extractor.py` `extract_from_text` glossary-page
# handling, which uses this dict's *keys* against `COMPONENT_KEYWORD_PATTERNS`
# to decide when that many simultaneous matches likely means a
# glossary/spec-list page rather than a real local reference.
COMPONENT_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    keyword: re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    for keyword in COMPONENT_KEYWORDS
}

# Identifier patterns — VRF service manuals reference specific parts by
# alphanumeric codes rather than full names (e.g. "TH3", "CN105").
#
# **[KG-W1.1 fix, R-addendum]** Extended to accept an optional single space
# or hyphen between the letter prefix and the digits (`TH 3`, `TH-3`, `CN
# 105`, `CN-105`) — the original `\bTH\d{1,3}\b`/`\bCN\d{2,4}\b` (no
# separator allowed) is a real recall bug, not just a theoretical edge case:
# confirmed via live query against `document_id=3` (286-page Zeggo VRV IV
# REYQ manual) that the un-extended pattern matched the sensor identifier
# `TH2` exactly ONCE in the entire document (`elements.text`) despite the
# manual being a wiring/troubleshooting-heavy service manual where such
# identifiers are normally referenced repeatedly.
#
# **[Further empirical finding, NOT fixed here — reported to System
# Analyst, see STATUS REPORT]** Even after this fix, `document_id=3` still
# has very low `TH`/`CN`-prefixed identifier recall, because this
# particular manual's actual electrical-symbol convention for sensors is
# `R<n>T` (e.g. `R1T`...`R14T`, confirmed against the manual's own
# "Electric symbol" legend table, `elements.id=5164`) rather than `TH<n>` —
# `TH<n>` is the convention this codebase's original vocabulary was written
# against (a different vendor/series), not a universal one. Recognizing
# additional vendor-specific identifier families (`R<n>T`, `M<n>C`/`M<n>F`,
# `Y<n>E`/`Y<n>S`, `S<n>PH`/`S<n>NPH`, ...) is a vocabulary-design decision
# (which KG node `entity_type` each maps to, how it interacts with K2's
# `(model_family, entity_type, canonical_identifier)` canonical key) —
# deliberately NOT added unilaterally in this bugfix pass; flagged for
# System Analyst / KG-W2.1 (R6 canonicalization, which already curates
# per-identifier alias tables) to decide.
SENSOR_ID_PATTERN = re.compile(r"\bTH[\s-]?\d{1,3}\b")
CONNECTOR_ID_PATTERN = re.compile(r"\bCN[\s-]?\d{2,4}\b")

# Candidate error code pattern — common VRF manufacturer prefixes (P/E/U/L/
# F/H/A/C/J followed by 1-3 digits, e.g. "P8", "U4", "E1", "L8", "H9").
# Deliberately broad/low-precision (single-letter-prefix codes collide with
# ordinary abbreviations) — extracted matches carry a LOW confidence score
# unless corroborated by a nearby contextual anchor (see
# `ERROR_CODE_ANCHOR_PATTERN` below and
# `app/ingestion/kg_candidate_extractor.py` `extract_from_text`).
#
# **[KG-W1.1 empirical finding, NOT fixed here — reported to System
# Analyst, see STATUS REPORT]** This shape (single letter + 1-3 digits) does
# NOT match every real error code format in the corpus: `document_id=3`'s
# actual error-code table (`elements.id=6548`, heading "Error code") uses
# two-letter main codes (`AH`, `AJ`, `UA`, ...) plus a dash-separated 2-digit
# subcode (`A6 - 01`, `AH - 03`, `U4 - 01`) — this pattern only matches the
# `[A-Z]\d` portion (`A6`, `U4`) and misses the two-letter forms (`AH`,
# `AJ`, `UA`) entirely, and never captures the subcode. Left unchanged here
# because broadening the character class + subcode capture is a nontrivial
# identifier-shape/entity-granularity decision (does `A6-01` and `A6-10`
# denote the same `ErrorCode` entity or two different ones?) that should be
# made deliberately, not folded into an anchor-only precision pass —
# flagged for follow-up alongside the sensor-identifier finding above.
ERROR_CODE_PATTERN = re.compile(r"\b([PEULFHACJ]\d{1,3})\b")

# **[KG-W1.1 fix, R4]** Contextual anchor keywords for `ERROR_CODE_PATTERN`
# matches — a match is only treated as reasonably-confident when one of
# these phrases appears in the same element's text or its heading
# hierarchy (`section_path`); see `kg_candidate_extractor.py`
# `_has_error_code_anchor`. Confirmed empirically necessary: with NO anchor
# requirement, `document_id=3` produced 192 `ErrorCode` candidates spanning
# 56 distinct codes across nearly the entire English alphabet
# (`A0`-`A9`,`C1`-`C9`,`E1`-`E9`,`F1`-`F6`,`H3`-`H9`,`J1`-`J9`,`L1`-`L9`,
# `P1`-`P4`,`U0`-`U9`) — a coverage pattern inconsistent with how real VRF
# error codes are actually allocated per manufacturer, i.e. mostly noise.
ERROR_CODE_ANCHOR_PATTERN = re.compile(
    r"\b(?:error code|check code|malfunction|abnormal)\b", re.IGNORECASE
)
