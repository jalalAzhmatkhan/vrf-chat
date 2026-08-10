"""Document structure graph builder (R5), see
`Documentation/system-design/09-kg-extraction-strategy.md` §6.2 R5 and
`03-retrieval-chunking.md` §5 (Neo4j node/relation vocabulary for the
*semantic* KG candidate graph — this module builds a deliberately SEPARATE
node/relation namespace for pure document STRUCTURE; see "Relationship to
the semantic KG candidate graph" below for why they're kept apart).

**Deterministic, precision 100%** — built entirely from data already
persisted by Stage 1-4 (`documents.id`/`.title`/`.page_count`,
`elements.id`/`.element_type`/`.section_path`/`.parent_id`, joined to
`pages.page_number`). No KG candidate extraction involved at all — this is
why R5 is independent of `kg_candidate_extractor.py`/`vrf_vocabulary.py`
and can be built/tested in complete isolation from the rest of Wave 1.

**Not loaded into Neo4j by this module.** The Neo4j loader
(`05-phase-later-backlog.md`, explicitly waiting on Wave 1 + Wave 2) is the
eventual consumer of this graph's export (`to_nodes_jsonb`/`to_edges_jsonb`,
mirroring `kg_candidate_extractor.py`'s `entities_to_jsonb`/
`relations_to_jsonb` pattern) — this module only PRODUCES the graph, kept
independently testable without any DB/network dependency (pure function over
plain input dataclasses, same testing philosophy as the rest of
`app/ingestion/`).

## Graph model

Reads: `Document → Page → Section → Element` (+ heading hierarchy), per the
roadmap's arrow notation. Concretely:

- **`Document`** node — one per `document_id`.
- **`Page`** node — one per `page_number`, `1..page_count` (even a page with
  zero elements still gets a node, since `page_count` is passed explicitly
  rather than inferred from which pages happen to have elements).
- **`Section`** node — one per UNIQUE heading-path prefix
  (`elements.section_path`, e.g. `["Troubleshooting", "Malfunction Code
  Table"]` produces two nested `Section` nodes: `["Troubleshooting"]` and
  `["Troubleshooting", "Malfunction Code Table"]`). Scoped to the
  **document**, not a single page — `section_path` is a document-wide
  running heading stack in `docling_parser.py` (not reset per page), so a
  section legitimately spans multiple pages; which page(s) it touches is
  answerable from its `Element` children's `page_number`, not modeled as a
  separate edge (would be redundant).
- **`Element`** node — one per `elements` row. `Figure`/`Table` (and
  `icon`/`figure_caption`/etc.) are **NOT** a separate node type here — they
  are `Element` nodes whose `element_type` attribute is `"figure"`/
  `"table"`/etc. This is a deliberate modeling choice (documented so it can
  be corrected by System Analyst at `SA-KG.1` verification if wrong): the
  roadmap's "Element→Figure/Table" arrow reads as "the element hierarchy
  bottoms out with figure/table leaf elements", not as an extra node layer
  — figures/tables already ARE elements (`elements.element_type`), giving
  them a second node identity would just duplicate the same row.

Edges (own string constants, deliberately distinct from
`kg_candidate_extractor.RELATION_CONNECTED_TO`/etc. — even once both graphs
load into the same Neo4j instance, structure-graph edges must never be
confused with semantic KG candidate relations):

- `HAS_PAGE` (`Document` → `Page`)
- `HAS_SECTION` (`Document` → top-level `Section`, i.e. `section_path`
  length 1)
- `HAS_SUBSECTION` (`Section` → `Section`, nested heading hierarchy)
- `CONTAINS_ELEMENT` (`Section` → `Element`, using the DEEPEST section in
  that element's `section_path`; OR `Page` → `Element` when
  `section_path` is empty — e.g. front-matter before the first heading)
- `HAS_CHILD` (`Element` → `Element`, mirrors `elements.parent_id`) — **this
  is what preserves the `CLAUDE.md` §4 inline-icon association at the
  structure-graph level**: an `icon` element's `HAS_CHILD` source is the
  paragraph/heading it was linked to by `docling_parser.py`'s
  `_LABEL_TO_ELEMENT_TYPE`/`last_text_local_id_doc` fallback logic (SA1.2
  fix), and a `figure_caption`'s `HAS_CHILD` source is the figure/table it
  captions — both relationships are now explicit graph edges, not just an
  implicit foreign key.

## Relationship to the semantic KG candidate graph

This module and `kg_candidate_extractor.py` produce two genuinely different
kinds of graph from the same underlying `elements` rows:

- **Structure graph (this module)**: deterministic, 100% precision, answers
  "where does this content live in the document" (page/section/parent
  chain). Never wrong, because it's a direct restatement of already-verified
  ingestion output — no pattern matching, no confidence scores.
- **Semantic KG candidate graph** (`kg_candidate_extractor.py`): probabilistic
  candidates (confidence scores, extraction methods, needs Fase 3 human
  review) about domain entities (`Sensor`/`Component`/`ErrorCode`/...) and
  their relations.

Both are namespaced separately (`HAS_CHILD` vs `CONNECTED_TO`, `Section` vs
`Component`, etc.) specifically so a future Neo4j load can keep them
distinguishable — e.g. `DOCUMENTED_IN`/`HAS_ERROR_CODE` (semantic relations,
`03-retrieval-chunking.md` §5) are noted in
`09-kg-extraction-strategy.md` §6.3 (R8) as "nyaris deterministik dari
KG-W1.5/R5" — meaning a FUTURE task can derive those semantic relations FROM
this structure graph (e.g. an `ErrorCode` entity candidate whose
`element_id` sits under a `Section` node named "Malfunction Code Table"
straightforwardly implies `HAS_ERROR_CODE`/`DOCUMENTED_IN`), but that
derivation is explicitly out of scope here — R5's job is only to build the
structure graph itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# **[R5]** Structure-graph relation type constants — see module docstring
# "Edges" section. Deliberately a separate namespace from
# `kg_candidate_extractor.RELATION_CONNECTED_TO`/etc.
HAS_PAGE = "HAS_PAGE"
HAS_SECTION = "HAS_SECTION"
HAS_SUBSECTION = "HAS_SUBSECTION"
CONTAINS_ELEMENT = "CONTAINS_ELEMENT"
HAS_CHILD = "HAS_CHILD"

NODE_TYPE_DOCUMENT = "Document"
NODE_TYPE_PAGE = "Page"
NODE_TYPE_SECTION = "Section"
NODE_TYPE_ELEMENT = "Element"

_LABEL_SNIPPET_MAX_LEN = 80


@dataclass(slots=True)
class StructureElementInput:
    """One `elements` row, as already persisted by Stage 1-4 (I1.5's
    `canonical_store.store_pages_and_elements`) — plain dataclass, NOT the
    SQLAlchemy `Element` model, so this module has zero DB/ORM dependency
    and is fully testable with plain Python objects. Callers (e.g. the
    Wave 1 re-extractor module, or a future orchestrator/Neo4j-loader
    wiring) are responsible for mapping their real row source into this
    shape."""

    id: int
    element_type: str
    page_number: int
    parent_id: int | None
    section_path: list[str]
    text: str | None = None


@dataclass(slots=True)
class StructureNode:
    node_id: str
    node_type: str
    label: str
    document_id: int
    page_number: int | None = None
    element_type: str | None = None
    element_id: int | None = None
    section_path: list[str] | None = None


@dataclass(slots=True)
class StructureEdge:
    relation: str
    from_node_id: str
    to_node_id: str


@dataclass(slots=True)
class DocumentStructureGraph:
    nodes: list[StructureNode] = field(default_factory=list)
    edges: list[StructureEdge] = field(default_factory=list)


def _document_node_id(document_id: int) -> str:
    return f"document:{document_id}"


def _page_node_id(document_id: int, page_number: int) -> str:
    return f"page:{document_id}:{page_number}"


def _section_node_id(document_id: int, section_path: list[str]) -> str:
    # json.dumps (not a plain "> "-joined string) deliberately avoids any
    # collision/ambiguity risk from heading text that happens to contain
    # whatever separator a naive join would pick.
    return f"section:{document_id}:{json.dumps(section_path)}"


def _element_node_id(element_id: int) -> str:
    # `elements.id` is already a globally unique Postgres primary key
    # (unique across every document), no `document_id` prefix needed.
    return f"element:{element_id}"


def _element_label(element: StructureElementInput) -> str:
    if element.text:
        snippet = element.text.strip().replace("\n", " ")
        if len(snippet) > _LABEL_SNIPPET_MAX_LEN:
            snippet = snippet[:_LABEL_SNIPPET_MAX_LEN].rstrip() + "..."
        return snippet
    return f"[{element.element_type}]"


def build_document_structure_graph(
    *,
    document_id: int,
    document_title: str,
    page_count: int,
    elements: list[StructureElementInput],
) -> DocumentStructureGraph:
    """Build the full structure graph for one document — see module
    docstring for the node/edge model. Deterministic: identical input always
    produces identical output (node/edge ordering follows `elements` input
    order, section nodes created in first-seen order), no randomness/model
    inference anywhere.
    """
    graph = DocumentStructureGraph()

    document_node_id = _document_node_id(document_id)
    graph.nodes.append(
        StructureNode(
            node_id=document_node_id,
            node_type=NODE_TYPE_DOCUMENT,
            label=document_title,
            document_id=document_id,
        )
    )

    page_node_ids: dict[int, str] = {}
    for page_number in range(1, page_count + 1):
        page_node_id = _page_node_id(document_id, page_number)
        page_node_ids[page_number] = page_node_id
        graph.nodes.append(
            StructureNode(
                node_id=page_node_id,
                node_type=NODE_TYPE_PAGE,
                label=f"Page {page_number}",
                document_id=document_id,
                page_number=page_number,
            )
        )
        graph.edges.append(
            StructureEdge(relation=HAS_PAGE, from_node_id=document_node_id, to_node_id=page_node_id)
        )

    section_node_ids: dict[tuple[str, ...], str] = {}

    def _ensure_section(section_path: list[str]) -> str:
        """Create (if missing) every `Section` node along `section_path`,
        `HAS_SECTION`/`HAS_SUBSECTION` edges included, returning the deepest
        (leaf) section's `node_id`. Idempotent — a `section_path` prefix
        already created by an earlier element is reused, not duplicated."""
        node_id = ""
        for depth in range(1, len(section_path) + 1):
            prefix = tuple(section_path[:depth])
            if prefix in section_node_ids:
                node_id = section_node_ids[prefix]
                continue
            node_id = _section_node_id(document_id, list(prefix))
            section_node_ids[prefix] = node_id
            graph.nodes.append(
                StructureNode(
                    node_id=node_id,
                    node_type=NODE_TYPE_SECTION,
                    label=prefix[-1],
                    document_id=document_id,
                    section_path=list(prefix),
                )
            )
            if depth == 1:
                graph.edges.append(
                    StructureEdge(
                        relation=HAS_SECTION, from_node_id=document_node_id, to_node_id=node_id
                    )
                )
            else:
                parent_node_id = section_node_ids[tuple(section_path[: depth - 1])]
                graph.edges.append(
                    StructureEdge(
                        relation=HAS_SUBSECTION, from_node_id=parent_node_id, to_node_id=node_id
                    )
                )
        return node_id

    element_node_ids: dict[int, str] = {}
    for elem in elements:
        element_node_id = _element_node_id(elem.id)
        element_node_ids[elem.id] = element_node_id
        graph.nodes.append(
            StructureNode(
                node_id=element_node_id,
                node_type=NODE_TYPE_ELEMENT,
                label=_element_label(elem),
                document_id=document_id,
                page_number=elem.page_number,
                element_type=elem.element_type,
                element_id=elem.id,
            )
        )

        parent_container_id: str | None
        if elem.section_path:
            parent_container_id = _ensure_section(elem.section_path)
        else:
            # No heading seen yet for this element — attach directly to its
            # Page (e.g. front-matter/cover content before the first
            # heading), per module docstring.
            parent_container_id = page_node_ids.get(elem.page_number)

        if parent_container_id is not None:
            graph.edges.append(
                StructureEdge(
                    relation=CONTAINS_ELEMENT,
                    from_node_id=parent_container_id,
                    to_node_id=element_node_id,
                )
            )

        if elem.parent_id is not None:
            parent_element_node_id = element_node_ids.get(elem.parent_id)
            if parent_element_node_id is not None:
                graph.edges.append(
                    StructureEdge(
                        relation=HAS_CHILD,
                        from_node_id=parent_element_node_id,
                        to_node_id=element_node_id,
                    )
                )

    return graph


def to_nodes_jsonb(nodes: list[StructureNode]) -> list[dict[str, Any]]:
    return [asdict(n) for n in nodes]


def to_edges_jsonb(edges: list[StructureEdge]) -> list[dict[str, Any]]:
    return [asdict(e) for e in edges]
