"""Evidence graph service.

Two defects drive this file:

1. `recompute_kge_features` returned `np.random.rand(128)` per node as
   "embeddings", and `run_kge_analysis` returned `np.random.random()` as a
   "confidence" with the explanation "Similar embedding space to
   {target}" -- a fabricated justification attached to a random number.
2. Node ids came from Python's built-in `hash()`, which is salted per
   process, so the same condition got a different id on every run and an
   upsert added a duplicate node rather than merging into the existing one.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from src.evidence_graph_service.main import (
    EvidenceGraphService,
    GraphEdge,
    GraphNode,
    entity_id,
)


@pytest.fixture
def service() -> EvidenceGraphService:
    """Two studies. Study 1 links diabetes to metformin and ACE inhibitors;
    study 2 links breast cancer to trastuzumab."""
    service = EvidenceGraphService()
    elements = service.extract_entities_relations([
        {
            "id": "evidence_1",
            "source": "PubMed",
            "type": "publication",
            "entities": {
                "conditions": ["diabetes mellitus"],
                "interventions": ["metformin", "ACE inhibitors"],
                "outcomes": ["HbA1c reduction"],
            },
            "timestamp": "2023-10-01T00:00:00Z",
        },
        {
            "id": "evidence_2",
            "source": "ClinicalTrials.gov",
            "type": "clinical_trial",
            "entities": {
                "conditions": ["breast cancer"],
                "interventions": ["trastuzumab"],
                "outcomes": ["overall survival"],
            },
            "timestamp": "2023-10-01T00:00:00Z",
        },
    ])
    service.upsert_graph_nodes_edges(elements)
    return service


# --------------------------------------------------------------------------
# Stable identity
# --------------------------------------------------------------------------

def test_entity_id_depends_only_on_the_name():
    assert entity_id("condition", "diabetes mellitus") == entity_id(
        "condition", "diabetes mellitus")
    assert entity_id("condition", "diabetes mellitus") != entity_id(
        "condition", "breast cancer")


def test_entity_id_normalises_case_and_whitespace():
    """The same condition written two ways must merge to one node."""
    assert entity_id("condition", "  Diabetes Mellitus ") == entity_id(
        "condition", "diabetes mellitus")


def test_entity_id_is_namespaced_by_label():
    assert entity_id("condition", "x") != entity_id("intervention", "x")
    assert entity_id("condition", "x").startswith("condition_")


def test_entity_id_is_stable_across_processes():
    """The regression guard. Python salts str.__hash__ per process, so the
    previous scheme produced a different id for the same entity on every
    run and the graph never merged anything.
    """
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from src.evidence_graph_service.main import entity_id;"
        "print(entity_id('condition', 'diabetes mellitus'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }

    assert len(runs) == 1, f"id varied across hash seeds: {runs}"


def test_re_extracting_the_same_evidence_does_not_duplicate_nodes(service):
    node_count = len(service.nodes)

    repeat = service.extract_entities_relations([{
        "id": "evidence_1",
        "source": "PubMed",
        "type": "publication",
        "entities": {
            "conditions": ["diabetes mellitus"],
            "interventions": ["metformin", "ACE inhibitors"],
            "outcomes": ["HbA1c reduction"],
        },
        "timestamp": "2023-10-01T00:00:00Z",
    }])
    service.upsert_graph_nodes_edges(repeat)

    assert len(service.nodes) == node_count


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def test_kge_features_are_refused_rather_than_generated(service):
    """Random vectors are not embeddings, and anything computed from them
    is a property of the RNG. Untrained must be distinguishable from
    trained."""
    with pytest.raises(NotImplementedError, match="require a model trained"):
        service.recompute_kge_features()


# --------------------------------------------------------------------------
# Link suggestion
# --------------------------------------------------------------------------

def test_suggestions_come_from_actual_co_occurrence(service):
    diabetes = entity_id("condition", "diabetes mellitus")

    names = {s["name"] for s in service.suggest_related_entities(diabetes)}

    # Both appear in the same study as diabetes.
    assert names == {"metformin", "ACE inhibitors"}
    # Trastuzumab shares no study with diabetes, so it must not appear.
    assert "trastuzumab" not in names


def test_an_unrelated_condition_gets_its_own_suggestions(service):
    breast_cancer = entity_id("condition", "breast cancer")

    names = {s["name"] for s in service.suggest_related_entities(breast_cancer)}

    assert names == {"trastuzumab"}


def test_every_suggestion_names_the_evidence_behind_it(service):
    """Explainability by design: a score you cannot trace is a black box."""
    diabetes = entity_id("condition", "diabetes mellitus")

    for suggestion in service.suggest_related_entities(diabetes):
        assert suggestion["shared_neighbours"] == ["evidence_1"]
        assert suggestion["scoring_method"] == "adamic_adar"


def test_the_score_is_not_called_a_confidence(service):
    """It is an unbounded ranking signal, not a calibrated probability."""
    diabetes = entity_id("condition", "diabetes mellitus")

    suggestion = service.suggest_related_entities(diabetes)[0]

    assert "confidence" not in suggestion
    assert suggestion["score"] > 0


def test_suggestions_are_deterministic(service):
    """Regression guard: the score used to be np.random.random(), so the
    same graph produced different suggestions on every call."""
    diabetes = entity_id("condition", "diabetes mellitus")

    first = service.suggest_related_entities(diabetes)
    second = service.suggest_related_entities(diabetes)

    assert first == second


def test_already_linked_entities_are_not_suggested(service):
    """There is nothing to suggest about an edge that already exists."""
    metformin = entity_id("intervention", "metformin")
    service.add_edge(GraphEdge(
        id="direct", source=metformin,
        target=entity_id("condition", "diabetes mellitus"),
        relationship="TREATS", properties={},
    ))

    names = {s["name"] for s in service.suggest_related_entities(
        entity_id("condition", "diabetes mellitus"))}

    assert "metformin" not in names


def test_a_promiscuous_study_contributes_less_than_a_focused_one():
    """Adamic-Adar down-weights hubs: co-occurring in a review that lists
    fifty interventions is weaker evidence than co-occurring in a trial
    that tests two."""
    service = EvidenceGraphService()
    service.add_node(GraphNode(id="cond", label="Condition", properties={"name": "c"}))
    service.add_node(GraphNode(id="focused", label="Trial", properties={}))
    service.add_node(GraphNode(id="broad", label="Publication", properties={}))
    service.add_node(GraphNode(id="drug_focused", label="Intervention", properties={"name": "focused_drug"}))
    service.add_node(GraphNode(id="drug_broad", label="Intervention", properties={"name": "broad_drug"}))

    edges = [("cond", "focused"), ("drug_focused", "focused"),
             ("cond", "broad"), ("drug_broad", "broad")]
    # Pad the broad study with many other interventions to raise its degree.
    for i in range(20):
        service.add_node(GraphNode(id=f"other_{i}", label="Intervention", properties={"name": f"o{i}"}))
        edges.append((f"other_{i}", "broad"))

    for source, target in edges:
        service.add_edge(GraphEdge(
            id=f"e_{source}_{target}", source=source, target=target,
            relationship="MENTIONS", properties={}))

    scores = {s["name"]: s["score"] for s in service.suggest_related_entities("cond")}

    assert scores["focused_drug"] > scores["broad_drug"]


def test_suggesting_for_an_unknown_node_is_an_error(service):
    with pytest.raises(KeyError, match="not in the graph"):
        service.suggest_related_entities("no_such_node")


def test_a_node_with_no_shared_neighbours_yields_nothing():
    service = EvidenceGraphService()
    service.add_node(GraphNode(id="lonely", label="Condition", properties={"name": "c"}))
    service.add_node(GraphNode(id="other", label="Intervention", properties={"name": "i"}))

    assert service.suggest_related_entities("lonely") == []
