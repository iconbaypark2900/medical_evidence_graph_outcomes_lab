"""Outcomes and pathway endpoints, and the join between the two halves.

OutcomesAnalyticsService and PathwayGuidelineService were both working and
tested, and neither was reachable: the API exposed ten endpoints and none
of them touched either. The project is named Medical Evidence Graph &
Outcomes Insight Lab and the two halves did not meet -- retrieval could
say what the literature holds, the outcomes side could compare cohorts,
and nothing joined them.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api_backend as backend
from src.api_backend import EvidenceSearchResponse, EvidenceSearcher, app, get_evidence_searcher


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_services():
    backend.pathway_service.guidelines = {}
    backend.pathway_service.pathway_graphs = {}
    backend.outcomes_service.cohorts = {}
    yield


def patient(index: int, age: int = 60, days: float = 100.0, event: bool = True,
            sex: str = "male") -> dict:
    return {
        "patient_id": f"pt_{index}",
        "demographics": {"age": age, "sex": sex},
        "clinical_indicators": {"baseline_risk_score": 0.5, "comorbidity_count": 1},
        "follow_up": {"observed_time_days": days, "event_observed": event},
    }


GUIDELINE = {
    "id": "dm2",
    "name": "Type 2 Diabetes Management",
    "condition": "type 2 diabetes",
    "steps": [
        {"name": "HbA1c measurement", "type": "test"},
        {"name": "Metformin initiation", "type": "intervention"},
        {"name": "Sulfonylurea add-on", "type": "intervention", "recommended": False},
    ],
    "decision_points": [{"question": "HbA1c above target?", "options": ["yes", "no"]}],
}


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------

def test_a_cohort_is_built_from_the_supplied_patients(client):
    body = client.post("/api/outcomes/cohort", json={
        "cohort_id": "c1",
        "patients": [patient(i, days=10.0 * (i + 1)) for i in range(6)],
    }).json()

    assert body["size"] == 6
    assert body["survival"]["events"] == 6
    assert body["survival"]["median_survival_days"] == 30.0


def test_inclusion_criteria_actually_exclude(client):
    patients = [patient(i, age=30 + i * 20) for i in range(4)]  # 30, 50, 70, 90

    body = client.post("/api/outcomes/cohort", json={
        "cohort_id": "c1", "patients": patients,
        "criteria": {"inclusion": {"age": [18, 80]}},
    }).json()

    assert body["supplied"] == 4
    assert body["size"] == 3
    assert body["excluded_by_criteria"] == 1


def test_a_criterion_naming_an_absent_field_is_refused(client):
    """Skipping it would produce a cohort broader than the one defined."""
    response = client.post("/api/outcomes/cohort", json={
        "cohort_id": "c1", "patients": [patient(i) for i in range(3)],
        "criteria": {"inclusion": {"pregnancy": False}},
    })

    assert response.status_code == 422
    assert "pregnancy" in response.json()["detail"]


def test_survival_intervals_are_returned_with_the_curve(client):
    body = client.post("/api/outcomes/cohort", json={
        "cohort_id": "c1",
        "patients": [patient(i, days=10.0 * (i + 1)) for i in range(8)],
    }).json()

    survival = body["survival"]
    assert len(survival["confidence_intervals"]) == len(survival["survival_probabilities"])
    for probability, (lower, upper) in zip(
            survival["survival_probabilities"][1:], survival["confidence_intervals"][1:]):
        assert lower <= probability <= upper


def test_a_cohort_matching_nobody_is_refused(client):
    response = client.post("/api/outcomes/cohort", json={
        "cohort_id": "c1", "patients": [patient(i) for i in range(3)],
        "criteria": {"inclusion": {"age": [200, 300]}},
    })

    assert response.status_code == 422
    assert "No patients met the criteria" in response.json()["detail"]


# --------------------------------------------------------------------------
# Comparative effectiveness
# --------------------------------------------------------------------------

def comparative_payload(n: int = 60, seed: int = 5) -> dict:
    gen = np.random.default_rng(seed)
    patients, groups = [], []
    for i in range(n):
        treated = i % 2 == 0
        # Treated patients survive markedly longer.
        days = float(gen.exponential(400 if treated else 120) + 5)
        patients.append(patient(i, days=round(days, 1), event=True))
        groups.append("treatment" if treated else "control")
    return {"cohort_id": "c1", "patients": patients, "groups": groups,
            "follow_up_period_days": 3000}


def test_a_real_survival_difference_is_detected(client):
    body = client.post(
        "/api/outcomes/comparative-effectiveness", json=comparative_payload()).json()

    assert body["test"] == "logrank"
    assert body["p_value"] < 0.01
    assert set(body["group_sizes"]) == {"treatment", "control"}


def test_the_p_value_arrives_with_its_denominators(client):
    """The same p can come from a large effect in a small cohort or a
    trivial one in a huge cohort."""
    body = client.post(
        "/api/outcomes/comparative-effectiveness", json=comparative_payload()).json()

    assert sum(body["group_sizes"].values()) == 60
    assert sum(body["group_events"].values()) == 60
    assert body["degrees_of_freedom"] == 1


def test_arm_labels_must_match_the_patients(client):
    payload = comparative_payload()
    payload["groups"] = payload["groups"][:5]

    response = client.post("/api/outcomes/comparative-effectiveness", json=payload)

    assert response.status_code == 422
    assert "arm labels" in response.json()["detail"]


def test_the_comparison_states_its_confounding_caveat(client):
    body = client.post(
        "/api/outcomes/comparative-effectiveness", json=comparative_payload()).json()

    assert any("confounding is not addressed" in note for note in body["notes"])


# --------------------------------------------------------------------------
# Guidelines and adherence
# --------------------------------------------------------------------------

def test_a_guideline_can_be_registered_and_listed(client):
    registered = client.post("/api/pathways/guidelines", json=GUIDELINE).json()

    assert registered["guideline_id"] == "dm2"
    assert registered["required_steps"] == 2  # the sulfonylurea step is optional

    listed = client.get("/api/pathways/guidelines").json()["guidelines"]
    assert [g["id"] for g in listed] == ["dm2"]


def test_adherence_is_computed_from_the_observed_steps(client):
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    body = client.post("/api/pathways/adherence", json={
        "guideline_id": "dm2", "patient_id": "pt_1", "condition": "type 2 diabetes",
        "steps": [{"name": "HbA1c measurement"}],
    }).json()

    assert body["comparison"]["adherence_score"] == pytest.approx(0.5)
    assert body["comparison"]["missing_steps"] == ["Metformin initiation"]
    assert body["comparison"]["n_required"] == 2


def test_full_adherence_raises_no_opportunities(client):
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    body = client.post("/api/pathways/adherence", json={
        "guideline_id": "dm2", "patient_id": "pt_1", "condition": "type 2 diabetes",
        "steps": [{"name": "HbA1c measurement"}, {"name": "Metformin initiation"}],
    }).json()

    assert body["comparison"]["adherence_score"] == 1.0
    assert body["opportunities"] == []


def test_a_guideline_with_a_decision_point_does_not_crash(client):
    """It used to: node['recommended'] raised KeyError on decision nodes,
    which have no such key, so the comparison never completed on any
    guideline containing one."""
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    response = client.post("/api/pathways/adherence", json={
        "guideline_id": "dm2", "patient_id": "pt_1", "condition": "type 2 diabetes",
        "steps": [],
    })

    assert response.status_code == 200
    assert response.json()["comparison"]["adherence_score"] == 0.0


def test_adherence_against_an_unregistered_guideline_is_a_404(client):
    response = client.post("/api/pathways/adherence", json={
        "guideline_id": "nope", "patient_id": "pt_1", "condition": "x", "steps": [],
    })

    assert response.status_code == 404


# --------------------------------------------------------------------------
# The join: evidence for a guideline's steps
# --------------------------------------------------------------------------

class ScriptedSearcher(EvidenceSearcher):
    """Returns evidence for some queries and nothing for others."""

    def __init__(self, supported: set):
        self.supported = supported
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        hit = any(term.lower() in query.lower() for term in self.supported)
        results = [{
            "id": "pubmed_1", "citation": "PMID:1", "title": f"Evidence for {query}",
            "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        }] if hit else []
        return EvidenceSearchResponse(
            results=results, mode="index" if hit else "index_then_live")


@pytest.fixture
def scripted(request):
    searcher = ScriptedSearcher(getattr(request, "param", {"Metformin"}))
    app.dependency_overrides[get_evidence_searcher] = lambda: searcher
    yield searcher
    app.dependency_overrides.pop(get_evidence_searcher, None)


def test_each_recommended_step_becomes_a_retrieval_query(client, scripted):
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    body = client.get("/api/pathways/guidelines/dm2/evidence").json()

    assert body["steps_examined"] == 2  # optional step is not examined
    assert scripted.queries == [
        "type 2 diabetes HbA1c measurement",
        "type 2 diabetes Metformin initiation",
    ]


def test_a_step_with_no_evidence_in_the_corpus_is_reported(client, scripted):
    """The finding worth seeing: a recommendation this corpus cannot
    support. Not the same as one the literature cannot support."""
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    body = client.get("/api/pathways/guidelines/dm2/evidence").json()

    assert body["steps_with_no_evidence_in_corpus"] == ["HbA1c measurement"]
    assert "not a statement about the literature" in body["note"]


def test_supporting_evidence_is_returned_with_citations(client, scripted):
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    findings = client.get("/api/pathways/guidelines/dm2/evidence").json()["findings"]

    supported = next(f for f in findings if f["step"] == "Metformin initiation")
    assert supported["supporting_records"] == 1
    assert supported["evidence"][0]["citation"] == "PMID:1"
    assert supported["retrieval_mode"] == "index"


def test_the_retrieval_mode_is_reported_per_step(client, scripted):
    client.post("/api/pathways/guidelines", json=GUIDELINE)

    modes = {f["step"]: f["retrieval_mode"]
             for f in client.get("/api/pathways/guidelines/dm2/evidence").json()["findings"]}

    assert modes["Metformin initiation"] == "index"
    assert modes["HbA1c measurement"] == "index_then_live"


def test_evidence_for_an_unregistered_guideline_is_a_404(client, scripted):
    assert client.get("/api/pathways/guidelines/nope/evidence").status_code == 404


# --------------------------------------------------------------------------
# Authentication covers the new surface too
# --------------------------------------------------------------------------

def test_the_new_endpoints_are_protected(monkeypatch):
    monkeypatch.setattr(backend, "API_KEYS", {"k"})
    keyed = TestClient(app)

    unprotected = []
    for method, path, payload in [
        ("post", "/api/outcomes/cohort", {"cohort_id": "c", "patients": [patient(0), patient(1)]}),
        ("post", "/api/outcomes/comparative-effectiveness", comparative_payload(4)),
        ("post", "/api/pathways/guidelines", GUIDELINE),
        ("post", "/api/pathways/adherence",
         {"guideline_id": "dm2", "patient_id": "p", "condition": "c", "steps": []}),
    ]:
        if getattr(keyed, method)(path, json=payload).status_code != 401:
            unprotected.append(path)
    if keyed.get("/api/pathways/guidelines").status_code != 401:
        unprotected.append("GET /api/pathways/guidelines")
    if keyed.get("/api/pathways/guidelines/dm2/evidence").status_code != 401:
        unprotected.append("GET /api/pathways/guidelines/{id}/evidence")

    assert unprotected == []
