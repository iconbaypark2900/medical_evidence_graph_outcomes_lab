"""Graph and embedding endpoints.

The graph service was reachable only in-process: fourth instance of a
working capability with no way to invoke it. Services with no endpoint,
endpoints with no page, and then a whole module with neither.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api_backend as backend
from src.api_backend import app
from src.evidence_graph_service.main import GraphEdge, GraphNode, RELATION_TARGET_LABEL


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_graph():
    service = backend.graph_service
    service.nodes, service.edges = {}, {}
    service.graph_source = "empty"
    for attribute in ("kge_model", "kge_store", "kge_report", "kge_edge_count"):
        if hasattr(service, attribute):
            delattr(service, attribute)
    yield service


def load_learnable_graph(service, n_docs: int = 90, n_topics: int = 6) -> None:
    for i in range(n_docs):
        topic = i % n_topics
        service.add_node(GraphNode(id=f"doc_{i}", label="Evidence", properties={}))
        for k in range(3):
            for relation, prefix in (("HAS_CONDITION", "condition"),
                                     ("HAS_INTERVENTION", "drug")):
                name = f"{prefix}_{topic}_{k}"
                service.add_node(GraphNode(
                    id=name, label=RELATION_TARGET_LABEL[relation],
                    properties={"name": name}))
                service.add_edge(GraphEdge(
                    id=f"e_{i}_{name}", source=f"doc_{i}", target=name,
                    relationship=relation, properties={}))
    service.graph_source = "memory"


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def test_an_empty_graph_reports_itself_as_empty(client):
    body = client.get("/api/graph").json()

    assert body["graph"]["source"] == "empty"
    assert body["embeddings"]["trained"] is False
    assert "/api/graph/embeddings/train" in body["embeddings"]["note"]


def test_the_status_counts_nodes_by_label(client, fresh_graph):
    load_learnable_graph(fresh_graph, n_docs=6, n_topics=2)

    graph = client.get("/api/graph").json()["graph"]

    assert graph["edges"] > 0
    assert graph["by_label"]["Evidence"] == 6
    assert "Intervention" in graph["by_label"]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def test_training_without_a_graph_is_refused(client):
    response = client.post("/api/graph/embeddings/train", json={"epochs": 10})

    assert response.status_code == 409
    assert "/api/graph/reload" in response.json()["detail"]


def test_training_returns_the_evaluation_and_whether_it_is_served(client, fresh_graph):
    load_learnable_graph(fresh_graph)

    body = client.post("/api/graph/embeddings/train",
                       json={"model": "distmult", "dim": 32, "epochs": 150}).json()

    assert "served" in body
    assert body["model"]["n_test_triples"] > 0
    assert set(body["comparison"]) == {"frequency", "adamic_adar"}
    assert body["served"] is body["beats_baselines"]


def test_a_losing_model_is_reported_as_not_served(client, fresh_graph, monkeypatch):
    """The report comes back either way; `served` says which happened."""
    import src.kge as kge
    from src.kge import Evaluation

    monkeypatch.setattr(
        kge, "evaluate_ranking",
        lambda score_fn, store, test, all_triples, name:
            Evaluation(name, 0.01, 0.0, 0.0, 0.0, 10, 99.0) if name != "frequency"
            else Evaluation(name, 0.9, 0.9, 0.9, 0.9, 10, 1.0))
    load_learnable_graph(fresh_graph, n_docs=30, n_topics=3)

    body = client.post("/api/graph/embeddings/train", json={"epochs": 10}).json()

    assert body["served"] is False
    assert body["beats_baselines"] is False


def test_a_graph_too_small_to_evaluate_is_refused(client, fresh_graph):
    fresh_graph.add_node(GraphNode(id="d", label="Evidence", properties={}))
    fresh_graph.add_node(GraphNode(id="c", label="Condition", properties={"name": "c"}))
    fresh_graph.add_edge(GraphEdge(
        id="e", source="d", target="c", relationship="HAS_CONDITION", properties={}))

    response = client.post("/api/graph/embeddings/train", json={"epochs": 10})

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Suggestions
# --------------------------------------------------------------------------

def test_suggestions_fall_back_to_the_structural_scorer(client, fresh_graph):
    """`auto` says which predictor ran rather than blurring the two."""
    load_learnable_graph(fresh_graph, n_docs=12, n_topics=2)

    body = client.get("/api/graph/suggestions", params={
        "entity": "doc_0", "relation": "HAS_INTERVENTION", "method": "auto"}).json()

    assert body["method_used"] == "structural"


def test_asking_for_embeddings_that_are_not_served_is_an_error(client, fresh_graph):
    """`auto` may fall back; `kge` explicitly must not pretend."""
    load_learnable_graph(fresh_graph, n_docs=12, n_topics=2)

    response = client.get("/api/graph/suggestions", params={
        "entity": "doc_0", "relation": "HAS_INTERVENTION", "method": "kge"})

    assert response.status_code == 409


def test_an_unknown_entity_is_a_404(client, fresh_graph):
    load_learnable_graph(fresh_graph, n_docs=12, n_topics=2)

    response = client.get("/api/graph/suggestions", params={
        "entity": "no_such_node", "relation": "HAS_CONDITION"})

    assert response.status_code == 404


def test_an_unknown_relation_is_refused(client, fresh_graph):
    load_learnable_graph(fresh_graph, n_docs=12, n_topics=2)

    response = client.get("/api/graph/suggestions", params={
        "entity": "doc_0", "relation": "CURES"})

    assert response.status_code == 422
    assert "HAS_CONDITION" in response.json()["detail"]


def test_an_unknown_method_is_refused(client, fresh_graph):
    response = client.get("/api/graph/suggestions", params={
        "entity": "doc_0", "relation": "HAS_CONDITION", "method": "vibes"})

    assert response.status_code == 422


def test_the_response_says_scores_are_not_probabilities(client, fresh_graph):
    load_learnable_graph(fresh_graph, n_docs=12, n_topics=2)

    body = client.get("/api/graph/suggestions", params={
        "entity": "doc_0", "relation": "HAS_INTERVENTION"}).json()

    assert "not probabilities" in body["note"]


# --------------------------------------------------------------------------
# Authentication and audit
# --------------------------------------------------------------------------

def test_the_graph_endpoints_are_protected(monkeypatch, fresh_graph):
    monkeypatch.setattr(backend, "API_KEYS", {"k"})
    keyed = TestClient(app)

    unprotected = []
    if keyed.get("/api/graph").status_code != 401:
        unprotected.append("GET /api/graph")
    if keyed.post("/api/graph/reload").status_code != 401:
        unprotected.append("POST /api/graph/reload")
    if keyed.post("/api/graph/embeddings/train", json={}).status_code != 401:
        unprotected.append("POST /api/graph/embeddings/train")
    if keyed.get("/api/graph/suggestions",
                 params={"entity": "x"}).status_code != 401:
        unprotected.append("GET /api/graph/suggestions")

    assert unprotected == []


def test_training_is_audited(client, fresh_graph, tmp_path, monkeypatch):
    from src.audit import AuditLog

    log = AuditLog(tmp_path / "audit.jsonl")
    monkeypatch.setattr(backend, "audit_log", log)
    load_learnable_graph(fresh_graph, n_docs=30, n_topics=3)

    client.post("/api/graph/embeddings/train", json={"epochs": 20, "dim": 16})

    events = [e for e in log.read() if e["action"] == "graph.embeddings.train"]
    assert events and "mrr" in events[0] and "served" in events[0]
