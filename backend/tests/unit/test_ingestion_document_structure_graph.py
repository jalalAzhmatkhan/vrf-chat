"""Unit tests for `app/ingestion/document_structure_graph.py` (KG-W1.5, R5).

Pure, deterministic logic over plain dataclasses — no GPU/model/DB
dependency (see module docstring), same approach as the rest of
`app/ingestion/`'s unit tests.
"""

from __future__ import annotations

from app.ingestion import document_structure_graph as dsg


def _elem(
    id: int,
    element_type: str,
    page_number: int,
    section_path: list[str] | None = None,
    parent_id: int | None = None,
    text: str | None = None,
) -> dsg.StructureElementInput:
    return dsg.StructureElementInput(
        id=id,
        element_type=element_type,
        page_number=page_number,
        parent_id=parent_id,
        section_path=section_path or [],
        text=text,
    )


# ---------------------------------------------------------------------------
# Document/Page nodes
# ---------------------------------------------------------------------------


def test_build_graph_creates_document_and_all_page_nodes_even_with_no_elements() -> None:
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Zeggo VRV IV REYQ", page_count=3, elements=[]
    )
    document_nodes = [n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_DOCUMENT]
    page_nodes = [n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_PAGE]

    assert len(document_nodes) == 1
    assert document_nodes[0].label == "Zeggo VRV IV REYQ"
    assert document_nodes[0].document_id == 1

    assert len(page_nodes) == 3
    assert {n.page_number for n in page_nodes} == {1, 2, 3}

    has_page_edges = [e for e in graph.edges if e.relation == dsg.HAS_PAGE]
    assert len(has_page_edges) == 3
    assert all(e.from_node_id == document_nodes[0].node_id for e in has_page_edges)


# ---------------------------------------------------------------------------
# Element attachment (no section_path -> attaches to Page)
# ---------------------------------------------------------------------------


def test_element_without_section_path_attaches_to_page() -> None:
    elements = [_elem(101, "paragraph", page_number=1, text="Cover page text")]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    element_node = next(n for n in graph.nodes if n.element_id == 101)
    assert element_node.node_type == dsg.NODE_TYPE_ELEMENT
    assert element_node.element_type == "paragraph"
    assert element_node.page_number == 1
    assert element_node.label == "Cover page text"

    contains_edges = [e for e in graph.edges if e.relation == dsg.CONTAINS_ELEMENT]
    assert len(contains_edges) == 1
    page_node = next(n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_PAGE)
    assert contains_edges[0].from_node_id == page_node.node_id
    assert contains_edges[0].to_node_id == element_node.node_id


def test_element_label_falls_back_to_bracketed_element_type_when_no_text() -> None:
    elements = [_elem(101, "figure", page_number=1, text=None)]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    element_node = next(n for n in graph.nodes if n.element_id == 101)
    assert element_node.label == "[figure]"


def test_element_label_truncates_long_text_and_strips_newlines() -> None:
    long_text = "A" * 200 + "\nmore text"
    elements = [_elem(101, "paragraph", page_number=1, text=long_text)]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    element_node = next(n for n in graph.nodes if n.element_id == 101)
    assert len(element_node.label) <= dsg._LABEL_SNIPPET_MAX_LEN + 3  # + "..."
    assert element_node.label.endswith("...")
    assert "\n" not in element_node.label


# ---------------------------------------------------------------------------
# Section hierarchy
# ---------------------------------------------------------------------------


def test_element_with_section_path_creates_nested_sections() -> None:
    elements = [
        _elem(
            101,
            "table",
            page_number=5,
            section_path=["Troubleshooting", "Malfunction Code Table"],
        )
    ]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=5, elements=elements
    )
    section_nodes = {n.label: n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_SECTION}
    assert set(section_nodes.keys()) == {"Troubleshooting", "Malfunction Code Table"}
    assert section_nodes["Troubleshooting"].section_path == ["Troubleshooting"]
    assert section_nodes["Malfunction Code Table"].section_path == [
        "Troubleshooting",
        "Malfunction Code Table",
    ]

    document_node = next(n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_DOCUMENT)
    has_section_edges = [e for e in graph.edges if e.relation == dsg.HAS_SECTION]
    assert len(has_section_edges) == 1
    assert has_section_edges[0].from_node_id == document_node.node_id
    assert has_section_edges[0].to_node_id == section_nodes["Troubleshooting"].node_id

    has_subsection_edges = [e for e in graph.edges if e.relation == dsg.HAS_SUBSECTION]
    assert len(has_subsection_edges) == 1
    assert has_subsection_edges[0].from_node_id == section_nodes["Troubleshooting"].node_id
    assert has_subsection_edges[0].to_node_id == section_nodes["Malfunction Code Table"].node_id

    contains_edges = [e for e in graph.edges if e.relation == dsg.CONTAINS_ELEMENT]
    assert len(contains_edges) == 1
    assert contains_edges[0].from_node_id == section_nodes["Malfunction Code Table"].node_id


def test_section_prefix_reused_not_duplicated_across_elements() -> None:
    """[R5] Two elements sharing a section_path PREFIX must reuse the same
    Section node for that prefix, not create duplicates — the whole point
    of a heading hierarchy graph."""
    elements = [
        _elem(101, "paragraph", page_number=5, section_path=["Troubleshooting", "Overview"]),
        _elem(102, "table", page_number=6, section_path=["Troubleshooting", "Codes"]),
    ]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=6, elements=elements
    )
    troubleshooting_nodes = [
        n
        for n in graph.nodes
        if n.node_type == dsg.NODE_TYPE_SECTION and n.label == "Troubleshooting"
    ]
    assert len(troubleshooting_nodes) == 1

    has_section_edges = [e for e in graph.edges if e.relation == dsg.HAS_SECTION]
    assert len(has_section_edges) == 1  # not duplicated per element

    subsection_nodes = {
        n.label for n in graph.nodes if n.node_type == dsg.NODE_TYPE_SECTION
    } - {"Troubleshooting"}
    assert subsection_nodes == {"Overview", "Codes"}


def test_section_spans_multiple_pages_single_node() -> None:
    """[R5] section_path is a document-wide heading stack (not reset per
    page, see docling_parser.py) — a section with elements on two different
    pages is still ONE Section node."""
    elements = [
        _elem(101, "paragraph", page_number=5, section_path=["Wiring Diagrams"]),
        _elem(102, "figure", page_number=6, section_path=["Wiring Diagrams"]),
    ]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=6, elements=elements
    )
    section_nodes = [n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_SECTION]
    assert len(section_nodes) == 1
    contains_edges = [e for e in graph.edges if e.relation == dsg.CONTAINS_ELEMENT]
    assert len(contains_edges) == 2
    assert all(e.from_node_id == section_nodes[0].node_id for e in contains_edges)


# ---------------------------------------------------------------------------
# HAS_CHILD (inline icon / caption association — CLAUDE.md §4)
# ---------------------------------------------------------------------------


def test_child_element_creates_has_child_edge_to_parent() -> None:
    """[R5, CLAUDE.md §4] An icon's parent_id (set by docling_parser.py to
    the surrounding paragraph) becomes an explicit HAS_CHILD edge here."""
    elements = [
        _elem(101, "paragraph", page_number=1, text="Press the button to enter menu."),
        _elem(102, "icon", page_number=1, parent_id=101),
    ]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    has_child_edges = [e for e in graph.edges if e.relation == dsg.HAS_CHILD]
    assert len(has_child_edges) == 1
    paragraph_node = next(n for n in graph.nodes if n.element_id == 101)
    icon_node = next(n for n in graph.nodes if n.element_id == 102)
    assert has_child_edges[0].from_node_id == paragraph_node.node_id
    assert has_child_edges[0].to_node_id == icon_node.node_id


def test_child_element_with_unresolvable_parent_id_no_crash_no_edge() -> None:
    """[R5] Defensive: a parent_id pointing at an element not present in
    the input list (shouldn't happen given canonical_store.py's insert
    ordering, but must not crash) produces no HAS_CHILD edge."""
    elements = [_elem(102, "icon", page_number=1, parent_id=999)]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    assert [e for e in graph.edges if e.relation == dsg.HAS_CHILD] == []


def test_element_referencing_out_of_range_page_no_contains_element_edge() -> None:
    """[R5] Defensive: an element whose `page_number` is outside
    `1..page_count` (shouldn't happen given `page_count`/`elements` both
    come from the same already-validated ingestion output, but must not
    crash) gets no `CONTAINS_ELEMENT` edge — there's no Page node to attach
    to."""
    elements = [_elem(101, "paragraph", page_number=99, text="Orphaned.")]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    assert [e for e in graph.edges if e.relation == dsg.CONTAINS_ELEMENT] == []
    # the Element node itself is still created — just unattached
    assert any(n.element_id == 101 for n in graph.nodes)


def test_element_with_no_parent_id_no_has_child_edge() -> None:
    elements = [_elem(101, "paragraph", page_number=1, text="Standalone.")]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    assert [e for e in graph.edges if e.relation == dsg.HAS_CHILD] == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_nodes_jsonb_and_to_edges_jsonb() -> None:
    elements = [_elem(101, "paragraph", page_number=1, text="Hello")]
    graph = dsg.build_document_structure_graph(
        document_id=1, document_title="Manual", page_count=1, elements=elements
    )
    node_dicts = dsg.to_nodes_jsonb(graph.nodes)
    edge_dicts = dsg.to_edges_jsonb(graph.edges)

    assert all(isinstance(d, dict) for d in node_dicts)
    assert all(isinstance(d, dict) for d in edge_dicts)
    assert len(node_dicts) == len(graph.nodes)
    assert len(edge_dicts) == len(graph.edges)
    element_dict = next(d for d in node_dicts if d.get("element_id") == 101)
    assert element_dict["label"] == "Hello"
    assert element_dict["node_type"] == dsg.NODE_TYPE_ELEMENT


# ---------------------------------------------------------------------------
# End-to-end shape sanity
# ---------------------------------------------------------------------------


def test_build_graph_full_scenario_node_and_edge_counts() -> None:
    """One document, 2 pages, one nested section, one icon-in-paragraph
    association, one front-matter element with no section — exercise every
    edge type at once."""
    elements = [
        _elem(1, "paragraph", page_number=1, text="Front matter, no heading yet."),
        _elem(
            2,
            "paragraph",
            page_number=2,
            section_path=["Troubleshooting", "Codes"],
            text="Press the button.",
        ),
        _elem(3, "icon", page_number=2, section_path=["Troubleshooting", "Codes"], parent_id=2),
    ]
    graph = dsg.build_document_structure_graph(
        document_id=7, document_title="Manual", page_count=2, elements=elements
    )

    assert len([n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_DOCUMENT]) == 1
    assert len([n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_PAGE]) == 2
    assert len([n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_SECTION]) == 2
    assert len([n for n in graph.nodes if n.node_type == dsg.NODE_TYPE_ELEMENT]) == 3

    relation_counts = {}
    for edge in graph.edges:
        relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
    assert relation_counts == {
        dsg.HAS_PAGE: 2,
        dsg.HAS_SECTION: 1,
        dsg.HAS_SUBSECTION: 1,
        dsg.CONTAINS_ELEMENT: 3,
        dsg.HAS_CHILD: 1,
    }
