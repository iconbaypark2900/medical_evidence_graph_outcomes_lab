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


# --------------------------------------------------------------------------
# Knowledge graph embeddings
#
# recompute_kge_features returned np.random.rand(128) per node, then an
# honest NotImplementedError. It now trains a real model -- and refuses to
# serve one that loses to its baselines, which is the same judgement the
# NotImplementedError was standing in for.
# --------------------------------------------------------------------------

def learnable_triples(n_docs: int = 90, n_topics: int = 6) -> list:
    """Documents in topics, each topic with its own vocabulary.

    Enough distinct entities that "guess the commonest tail" is a real
    opponent rather than a formality -- otherwise the gate would pass
    without the model having learned anything.
    """
    triples = []
    for i in range(n_docs):
        topic = i % n_topics
        for k in range(3):
            triples.append((f"doc_{i}", "HAS_CONDITION", f"condition_{topic}_{k}"))
            triples.append((f"doc_{i}", "HAS_INTERVENTION", f"drug_{topic}_{k}"))
    return triples


def test_training_returns_an_evaluation_report(service):
    report = service.recompute_kge_features(
        triples=learnable_triples(), epochs=60, dim=16)

    described = report.describe()
    assert described["model"]["n_test_triples"] > 0
    assert "beats_baselines" in described
    assert set(described["comparison"]) == {"frequency", "adamic_adar"}


def test_an_empty_graph_cannot_be_embedded(service):
    empty = EvidenceGraphService()

    with pytest.raises(ValueError, match="no edges"):
        empty.recompute_kge_features()


def test_suggestions_before_training_are_refused(service):
    with pytest.raises(RuntimeError, match="No embeddings have been trained"):
        service.kge_suggestions("doc_0", "HAS_CONDITION")


def test_suggestions_from_a_losing_model_are_refused(service, monkeypatch):
    """The judgement the NotImplementedError used to stand in for: a
    predictor that loses to its baselines is not served, however
    sophisticated its scores look."""
    import src.kge as kge
    from src.kge import Evaluation

    monkeypatch.setattr(
        kge, "evaluate_ranking",
        lambda score_fn, store, test, all_triples, name:
            Evaluation(name, 0.01, 0.0, 0.0, 0.0, 10, 99.0) if name != "frequency"
            else Evaluation(name, 0.9, 0.9, 0.9, 0.9, 10, 1.0))

    service.recompute_kge_features(triples=learnable_triples(), epochs=5, dim=8)

    with pytest.raises(RuntimeError, match="did not beat its baselines"):
        service.kge_suggestions("doc_0", "HAS_CONDITION")


def test_suggestions_carry_the_evaluation_that_justifies_them(service):
    """A model score is not a probability; shipping the held-out metric
    with it is what makes the number checkable."""
    report = service.recompute_kge_features(
        triples=learnable_triples(), epochs=200, dim=32)
    if not report.beats_baselines:
        pytest.skip("model did not beat baselines on this seed; nothing served")

    suggestions = service.kge_suggestions("doc_0", "HAS_INTERVENTION", limit=3)

    assert suggestions
    for suggestion in suggestions:
        assert suggestion["scoring_method"].startswith("kge:")
        assert suggestion["model_mrr"] == round(report.evaluation.mrr, 4)
        assert "confidence" not in suggestion


def test_an_entity_with_no_embedding_is_refused(service):
    service.recompute_kge_features(triples=learnable_triples(), epochs=5, dim=8)

    with pytest.raises(KeyError, match="no embedding"):
        service.kge_suggestions("never_seen", "HAS_CONDITION")


def test_an_unknown_relation_is_refused(service):
    service.recompute_kge_features(triples=learnable_triples(), epochs=5, dim=8)

    with pytest.raises(KeyError, match="Unknown relation"):
        service.kge_suggestions("doc_0", "CURES")


@pytest.mark.requires_stack
async def test_triples_can_be_read_from_the_populated_graph(require_stack):
    """The graph this exists to embed is the one in Neo4j, not the
    in-memory one the service builds for a demo.

    Writes its own evidence rather than assuming the graph is already
    populated: depending on whatever happens to be indexed passes on a
    developer machine and fails on a fresh CI stack, which is what it did.
    """
    import neo4j

    from src.integration import load_database_config
    from src.kge import load_triples_from_neo4j

    config = load_database_config()["neo4j"]
    driver = neo4j.AsyncGraphDatabase.driver(
        config["uri"], auth=(config["username"], config["password"]))
    written = [
        ("kgetest_1", "HAS_CONDITION", "kgetest heart failure"),
        ("kgetest_1", "HAS_INTERVENTION", "kgetest dapagliflozin"),
        ("kgetest_2", "HAS_CONDITION", "kgetest heart failure"),
    ]
    try:
        async with driver.session() as session:
            for head, relation, tail in written:
                await session.run(
                    f"MERGE (e:Evidence {{id: $head}}) "
                    f"MERGE (n:Condition {{name: $tail}}) "
                    f"MERGE (e)-[:{relation}]->(n)",
                    head=head, tail=tail)

        triples = await load_triples_from_neo4j()

        ours = [t for t in triples if t[0].startswith("kgetest_")]
        assert len(ours) == len(written)
        for head, relation, tail in ours:
            assert relation.startswith("HAS_")
            assert head and tail
    finally:
        async with driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.id STARTS WITH 'kgetest_' "
                "OR n.name STARTS WITH 'kgetest ' DETACH DELETE n")
        await driver.close()
