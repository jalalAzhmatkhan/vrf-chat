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

**[SA-KG.1, 2026-08-10 — cross-vendor UNION, see `09-kg-extraction-strategy
.md` §11.2]** System Analyst's SA-KG.1 verification scanned all **7**
`source-documents/` PDFs (not just `document_id=3`) and found **zero
meaningful collisions** between the Mitsubishi identifier conventions this
module originally targeted (`TH#`, `CN#`) and the Daikin/Zeggo conventions
found in the other 4 manuals — `R#T` never appears in any Mitsubishi manual;
`TH#` appears only 2-3 times total across the 4 Daikin manuals (consistent
with incidental noise, not a real parallel convention); `CN#` appears **zero**
times in any Daikin manual (a different connector-reference system
entirely, IEC-style `X#M`/`S#PH`/`Y#E`/etc.). **Decision: UNION, not
per-vendor** — every pattern below is tried against every document
regardless of `model_family`, rather than routing to a per-vendor pattern
subset. This is safe specifically because of two other Wave 1 decisions
already in place: K2's composite canonical key
(`(model_family, entity_type, canonical_identifier)`, see
`kg_candidate_extractor.py` module docstring) already prevents two
same-looking identifiers from different `model_family` values ever being
treated as the same entity, and the anchor+confidence-tiering discipline
`KG-W1.1` built for `ErrorCode` is extended to the new two-letter Daikin
error code forms below, not bypassed for them. See §11.2 for the reviewed
condition to revisit this (a real intra-`model_family` collision, not
predicted by current data).
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
# **[SA-KG.1 finding]** Confirmed this is a Mitsubishi-only convention —
# `TH#` appears only 2-3 times total across the 4 Daikin/Zeggo manuals
# (noise, not a real parallel convention); `CN#` appears **zero** times in
# any of them (see `TERMINAL_ID_PATTERN` below for Daikin's actual
# terminal-block convention).
SENSOR_ID_PATTERN = re.compile(r"\bTH[\s-]?\d{1,3}\b")
CONNECTOR_ID_PATTERN = re.compile(r"\bCN[\s-]?\d{2,4}\b")

# **[SA-KG.1, 2026-08-10 — §11.2, priority "Segera"]** Daikin/Zeggo sensor
# identifier convention (`R1T`...`R14T`) — the `TH<n>` convention above is
# Mitsubishi-only (see finding note there); this is its Daikin counterpart,
# confirmed empirically (168 occurrences in `document_id=3` vs `TH#`'s
# single occurrence in the same document) with zero collisions found
# against any Mitsubishi manual (§11.2 scan across all 7
# `source-documents/` PDFs). UNION, not per-vendor routing — see module
# docstring "SA-KG.1" section for why this is safe. Optional space/dash
# separator for the same recall reason as `SENSOR_ID_PATTERN` above.
SENSOR_ID_PATTERN_DAIKIN = re.compile(r"\bR[\s-]?\d{1,2}T\b")

# **[SA-KG.1, 2026-08-10 — §11.2.2, priority "Segera"]** Daikin/Zeggo
# terminal block convention (`X1M`, `X2M`, ...) — NOT a "missing connector"
# gap (Daikin's connector-reference system is entirely different from
# Mitsubishi's `CN#`, see `CONNECTOR_ID_PATTERN` note above and §11.2.2 of
# the design doc), but a precise IDENTIFIER pattern for a KG node type
# (`Terminal`) that already existed in the schema and already had a
# free-text keyword match (`"terminal block"` in `COMPONENT_KEYWORDS`
# above) — this closes the same identifier-precision gap for `Terminal`
# that `SENSOR_ID_PATTERN_DAIKIN` closes for `Sensor`. 117 occurrences in
# `document_id=3`.
TERMINAL_ID_PATTERN = re.compile(r"\bX[\s-]?\d{1,2}M\b")

# **[SA-KG.1, 2026-08-10 — §11.2, explicitly Wave 2 backlog, NOT
# implemented now]** Additional Daikin/Zeggo IEC-style electrical-reference
# identifier families found during the same §11.2 scan, each with a real
# occurrence count in `document_id=3` but deliberately NOT wired into
# extraction yet (System Analyst: "beri ke Fase 3 review dulu untuk
# validasi domain expert sebelum ekspansi lebih jauh") — captured here only
# so the regex-starting-point research doesn't need to be redone:
#   Pressure sensor (S#PH): r"\bS\d{1,2}N?PH\b" -> Sensor       (100 occ.)
#   Valve (Y#E/Y#S):        r"\bY\d{1,2}[ES]\b"  -> Component   (433 occ.)
#   Motor (M#C/M#F):        r"\bM\d{1,2}[CF]\b"  -> Component   (268 occ.)
#   Relay (K#R):             r"\bK\d{1,2}R\b"     -> Component   (189 occ.)
#   Fuse (F#U):               r"\bF\d{1,2}U\b"     -> Component   (77 occ.)
#   Breaker (Q#):             r"\bQ\d{1,2}\b"      -> Component   (50 occ.,
#     needs a context anchor before use — "Q" alone is too short/generic,
#     same risk class as the old un-anchored single-letter ErrorCode
#     pattern KG-W1.1 fixed).

# Candidate error code pattern — common VRF manufacturer prefixes (P/E/U/L/
# F/H/A/C/J followed by either 1-3 digits (Mitsubishi-style, e.g. "P8",
# "U4", "E1") or a second letter from a small, evidence-based set (Daikin
# two-letter main codes, e.g. "AH", "AJ", "UA" — see SA-KG.1 note below).
# Deliberately broad/low-precision (single-letter-prefix codes collide with
# ordinary abbreviations) — extracted matches carry a LOW confidence score
# unless corroborated by a nearby contextual anchor (see
# `ERROR_CODE_ANCHOR_PATTERN` below and
# `app/ingestion/kg_candidate_extractor.py` `extract_from_text`).
#
# **[SA-KG.1, 2026-08-10 — §11.2/§11.2.1, priority "Segera"]** Extended
# to also match Daikin/Zeggo's two-letter main error codes. The second
# letter is deliberately restricted to `{A, H, J}` (`_ERROR_CODE_SECOND_LETTER`
# below) — the three values actually observed (`AH`, `AJ`, `UA`) — rather
# than a fully generic `[A-Z]`, which would catch extremely common
# domain abbreviations like "AC" (air conditioning — arguably the single
# most frequent two-letter token in this entire corpus) as false-positive
# `ErrorCode` candidates. This is a deliberately conservative choice (own
# reasoning, not directly specified by System Analyst's regex
# starting-point) — same anchor+confidence-tiering discipline from
# `KG-W1.1` still applies to every match regardless of which branch
# matched, so this is a recall widening within an already-established
# precision safety net, not a bypass of it.
#
# **Two extraction granularities, both real `ErrorCode` candidates, per
# §11.2.1**: `ERROR_CODE_MAIN_PATTERN` matches the main code alone (`A6`,
# `AH`, `UA` — what's shown on the unit's own display, most likely to be
# asked about directly, e.g. "apa arti error A6?"); `ERROR_CODE_SUBCODE_PATTERN`
# matches main+dash-separated subcode as a SEPARATE, more diagnostically
# precise candidate (`A6-01` — service-mode-only detail). Deliberately NOT
# choosing one over the other (§11.2.1): the relationship between "A6" and
# "A6-01" (e.g. a future `HAS_SUBCODE` relation) is intentionally left
# undecided for Fase 3/R13, not assumed here.
_ERROR_CODE_SECOND_LETTER = "AHJ"
ERROR_CODE_MAIN_PATTERN = re.compile(
    rf"\b([PEULFHACJ](?:\d{{1,3}}|[{_ERROR_CODE_SECOND_LETTER}]))\b"
)
ERROR_CODE_SUBCODE_PATTERN = re.compile(
    rf"\b([PEULFHACJ](?:\d{{1,3}}|[{_ERROR_CODE_SECOND_LETTER}]))\s*-\s*(\d{{1,2}})\b"
)

# **[SA-KG.1 implementation verification, 2026-08-10 — known false-positive
# class, correctly mitigated, not fixed at the pattern level]** Re-running
# the Wave 1 DoD re-extractor against real `document_id=3` after adding
# `ERROR_CODE_SUBCODE_PATTERN` surfaced one confirmed false positive:
# `elements.id=5740`'s `visual_description.description` is a "Draft
# prevention position" TABLE (columns labeled `P0`/`P1`/`P2`/...), not an
# error-code table — flattened markdown adjacency produced spurious
# `P1-4`/`P2-4` "subcode" matches. **Correctly caught by the existing
# anchor discipline**: both landed at `CONFIDENCE_ERROR_CODE_UNANCHORED`
# (`context_anchor_matched=False`, no "error code"/"malfunction"/etc.
# keyword nearby) — the lowest-trust tier, same as any other unanchored
# match, exactly the safety net this is designed to provide. Not
# considered worth narrowing the regex further for a single confirmed
# instance (would reduce recall for genuine subcodes on real tables to fix
# a case the confidence tier already handles correctly) — noted here for
# whoever reviews Wave 1 candidate quality (R2/R10) next, not a silently
# swept-under-the-rug gap.

# **[KG-W1.1 fix, R4]** Contextual anchor keywords for `ERROR_CODE_MAIN_PATTERN`/
# `ERROR_CODE_SUBCODE_PATTERN` matches — a match is only treated as
# reasonably-confident when one of these phrases appears in the same
# element's text or its heading hierarchy (`section_path`); see
# `kg_candidate_extractor.py` `_has_error_code_anchor`. Confirmed
# empirically necessary: with NO anchor requirement, `document_id=3`
# produced 192 `ErrorCode` candidates spanning 56 distinct codes across
# nearly the entire English alphabet (`A0`-`A9`,`C1`-`C9`,`E1`-`E9`,
# `F1`-`F6`,`H3`-`H9`,`J1`-`J9`,`L1`-`L9`,`P1`-`P4`,`U0`-`U9`) — a coverage
# pattern inconsistent with how real VRF error codes are actually allocated
# per manufacturer, i.e. mostly noise.
ERROR_CODE_ANCHOR_PATTERN = re.compile(
    r"\b(?:error code|check code|malfunction|abnormal)\b", re.IGNORECASE
)
