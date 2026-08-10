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

# Identifier patterns — VRF service manuals reference specific parts by
# alphanumeric codes rather than full names (e.g. "TH3", "CN105").
SENSOR_ID_PATTERN = re.compile(r"\bTH\d{1,3}\b")
CONNECTOR_ID_PATTERN = re.compile(r"\bCN\d{2,4}\b")

# Candidate error code pattern — common VRF manufacturer prefixes (P/E/U/L/
# F/H/A/C/J followed by 1-3 digits, e.g. "P8", "U4", "E1", "L8", "H9").
# Deliberately broad/low-precision (single-letter-prefix codes collide with
# ordinary abbreviations) — extracted matches carry a LOW confidence score
# (see `app/ingestion/kg_candidate_extractor.py`) specifically because of
# this, not treated as authoritative.
ERROR_CODE_PATTERN = re.compile(r"\b([PEULFHACJ]\d{1,3})\b")
