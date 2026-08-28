"""FastAPI surface.

Every analysis endpoint now takes the observed outcome as input. The tests
that used to be `known_defect` xfails -- proving each endpoint discarded
its input and invented replacement data with `np.random` -- are ordinary
tests here, and the determinism checks among them are the regression
guard: a deterministic analysis of a fixed cohort must return a fixed
answer.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api_backend import (
    EvidenceSearcher,
    EvidenceSearchResponse,
    GraphRAGEvidenceSearcher,
    app,
    get_evidence_searcher,
    median_survival_time,
    risk_model_registry,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is process-global; no test may inherit another's model."""
    risk_model_registry.models = {}
    risk_model_registry.version = None
    risk_model_registry.trained_at = None
    yield
    risk_model_registry.models = {}
    risk_model_registry.version = None
    risk_model_registry.trained_at = None


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------

def patient(index: int = 0, age: int = 60, risk: float = 0.5, sex: str = "male") -> dict:
    return {
        "patient_id": f"patient_{index}",
        "demographics": {"age": age, "sex": sex},
        "clinical_indicators": {
            "baseline_risk_score": risk,
            "comorbidity_count": index % 5,
        },
    }


def record(index: int = 0, days: float = 100.0, event: bool = True, **kwargs) -> dict:
    """A patient with observed follow-up."""
    return {
        **patient(index, **kwargs),
        "follow_up": {"observed_time_days": days, "event_observed": event},
    }


def treated(index: int, treatment: bool, outcome: float, **kwargs) -> dict:
    return {
        **patient(index, **kwargs),
        "treatment_assigned": treatment,
        "outcome_value": outcome,
    }


def member(index: int, outcomes: dict, **kwargs) -> dict:
    return {**patient(index, **kwargs), "outcomes": outcomes}


def survival_cohort_payload(n: int = 40, seed: int = 0) -> list[dict]:
    gen = np.random.default_rng(seed)
    return [
        record(
            i,
            days=float(round(gen.uniform(10, 400), 2)),
            event=bool(i % 3),
            age=40 + (i % 40),
            risk=round((i % 9) / 10 + 0.1, 2),
            sex="male" if i % 2 else "female",
        )
        for i in range(n)
    ]


def labelled_cohort(n: int = 80, seed: int = 1) -> list[dict]:
    """Training data where mortality genuinely tracks the risk score."""
    gen = np.random.default_rng(seed)
    cohort = []
    for i in range(n):
        risk = round(float(gen.uniform(0, 1)), 3)
        mortality = float(gen.binomial(1, 0.08 + 0.84 * risk))
        cohort.append(
            member(
                i,
                {"mortality": mortality},
                age=int(gen.integers(30, 90)),
                risk=risk,
                sex="male" if i % 2 else "female",
            )
        )
    return cohort


# --------------------------------------------------------------------------
# Service surface
# --------------------------------------------------------------------------

def test_health_reports_risk_assessment_as_not_ready_before_training(client):
    body = client.get("/api/health").json()

    assert body["status"] == "healthy"
    assert body["models_ready"]["survival_analysis"] is True
    assert body["models_ready"]["risk_assessment"] is False
    assert body["risk_model_version"] is None


def test_health_is_also_served_at_the_path_the_frontend_uses(client):
    """src/frontend_interface.py polls /health, not /api/health."""
    assert client.get("/health").status_code == 200


def test_root_identifies_the_service(client):
    body = client.get("/").json()

    assert body["status"] == "running"
    assert body["risk_model_trained"] is False


def test_openapi_schema_lists_the_analysis_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert {
        "/api/patients/risk-assessment",
        "/api/models/risk/train",
        "/api/survival-analysis/kaplan-meier",
        "/api/survival-analysis/cox-regression",
        "/api/causal-inference/ate-estimation",
        "/api/cohorts/compare",
        "/api/evidence/search",
    } <= set(paths)


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------

def test_age_outside_the_permitted_range_is_rejected(client):
    bad = patient()
    bad["demographics"]["age"] = 200

    assert client.post("/api/patients/risk-assessment", json=[bad]).status_code == 422


def test_unknown_sex_value_is_rejected(client):
    assert client.post(
        "/api/patients/risk-assessment", json=[patient(sex="unspecified")]
    ).status_code == 422


def test_survival_analysis_rejects_patients_without_follow_up(client):
    """The schema is the guard: without an outcome there is nothing to
    analyse, and the endpoint must not supply one."""
    response = client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [patient(0), patient(1)]},
    )

    assert response.status_code == 422
    assert "follow_up" in response.text


def test_causal_analysis_rejects_patients_without_treatment_and_outcome(client):
    response = client.post(
        "/api/causal-inference/ate-estimation",
        json={
            "patient_data": [patient(i) for i in range(4)],
            "treatment_variable": "statin",
            "outcome_variable": "mortality",
        },
    )

    assert response.status_code == 422
    assert "treatment_assigned" in response.text


def test_negative_follow_up_time_is_rejected(client):
    bad = record(0, days=-5.0)

    response = client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [bad, record(1)]},
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Risk models
# --------------------------------------------------------------------------

def test_risk_assessment_is_unavailable_until_a_model_is_trained(client):
    """Previously this trained on np.random labels inside the request and
    returned a mortality risk of 0.0 for every patient, with HTTP 200."""
    response = client.post("/api/patients/risk-assessment", json=[patient(0)])

    assert response.status_code == 503
    assert "/api/models/risk/train" in response.json()["detail"]


def test_training_reports_held_out_performance(client):
    body = client.post(
        "/api/models/risk/train", json={"training_data": labelled_cohort()}
    ).json()

    mortality = body["models"]["mortality"]
    assert mortality["task"] == "classification"
    assert mortality["holdout_metric"] == "roc_auc"
    # The label genuinely tracks the risk score, so a fitted model should
    # beat chance on data it never saw.
    assert mortality["holdout_value"] > 0.7
    assert mortality["n_train"] + mortality["n_test"] == 80
    assert body["model_version"]


def test_risk_assessment_returns_real_scores_once_trained(client):
    client.post("/api/models/risk/train", json={"training_data": labelled_cohort()})

    body = client.post(
        "/api/patients/risk-assessment",
        json=[patient(0, risk=0.05), patient(1, risk=0.95)],
    ).json()

    scores = [a["risks"]["mortality"]["score"] for a in body["risk_assessments"]]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert any(s > 0.0 for s in scores)
    # The score must respond to the patient, not just to the cohort size.
    assert scores[1] > scores[0]
    assert body["model_version"] == risk_model_registry.version


def test_every_risk_score_carries_its_holdout_metric(client):
    """Explainability by design: a score with no accompanying evidence of
    how well the model generalises is an unexplained recommendation."""
    client.post("/api/models/risk/train", json={"training_data": labelled_cohort()})

    body = client.post("/api/patients/risk-assessment", json=[patient(0)]).json()

    risk = body["risk_assessments"][0]["risks"]["mortality"]
    assert risk["holdout_metric"] == "roc_auc"
    assert risk["holdout_value"] > 0.7
    assert risk["imputed_features"] == []


def test_health_reports_the_model_once_trained(client):
    client.post("/api/models/risk/train", json={"training_data": labelled_cohort()})

    body = client.get("/api/health").json()

    assert body["models_ready"]["risk_assessment"] is True
    assert body["risk_model_version"] == risk_model_registry.version


def test_training_rejects_a_constant_outcome(client):
    cohort = [member(i, {"mortality": 0.0}) for i in range(30)]

    response = client.post("/api/models/risk/train", json={"training_data": cohort})

    assert response.status_code == 422
    assert "constant outcome" in response.json()["detail"]


def test_training_skips_a_constant_outcome_but_keeps_the_rest(client):
    cohort = labelled_cohort()
    for entry in cohort:
        entry["outcomes"]["everyone_survived"] = 0.0

    body = client.post("/api/models/risk/train", json={"training_data": cohort}).json()

    assert "mortality" in body["models"]
    assert "everyone_survived" in body["skipped_outcomes"]


def test_training_rejects_outcomes_recorded_for_only_some_patients(client):
    cohort = labelled_cohort()
    cohort[0]["outcomes"]["readmission"] = 1.0

    response = client.post("/api/models/risk/train", json={"training_data": cohort})

    assert response.status_code == 422
    assert "readmission" in response.json()["detail"]


def test_imputed_features_are_reported_back(client):
    """A prediction made from partly-filled-in data says so."""
    cohort = labelled_cohort()
    for entry in cohort:
        entry["clinical_indicators"]["severity_score"] = 5.0

    client.post("/api/models/risk/train", json={"training_data": cohort})

    body = client.post("/api/patients/risk-assessment", json=[patient(0)]).json()

    assert body["risk_assessments"][0]["risks"]["mortality"]["imputed_features"] == [
        "severity_score"
    ]


# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------

def test_kaplan_meier_endpoint_is_deterministic_for_a_fixed_cohort(client):
    """Regression guard. The curve used to come from np.random.exponential,
    so the same request returned a different survival curve every time."""
    request = {"patient_data": survival_cohort_payload(40), "time_horizon_days": 365}

    first = client.post("/api/survival-analysis/kaplan-meier", json=request).json()
    second = client.post("/api/survival-analysis/kaplan-meier", json=request).json()

    assert first["survival_curve"] == second["survival_curve"]
    assert first["stats"] == second["stats"]


def test_kaplan_meier_curve_comes_from_the_submitted_follow_up(client):
    """Four patients, all events, at 10/20/30/40 days."""
    request = {
        "patient_data": [
            record(i, days=10.0 * (i + 1), event=True) for i in range(4)
        ],
        "time_horizon_days": 365,
    }

    body = client.post("/api/survival-analysis/kaplan-meier", json=request).json()

    curve = body["survival_curve"]
    assert curve["time_points"] == [0.0, 10.0, 20.0, 30.0, 40.0]
    np.testing.assert_allclose(
        curve["survival_probability"], [1.0, 0.75, 0.5, 0.25, 0.0], atol=1e-12
    )
    assert body["stats"]["events_occurred"] == 4
    assert body["stats"]["censored"] == 0
    assert body["stats"]["median_survival_days"] == 20.0


def test_kaplan_meier_reports_survival_at_the_requested_horizon(client):
    request = {
        "patient_data": [record(i, days=10.0 * (i + 1), event=True) for i in range(4)],
        "time_horizon_days": 25,
    }

    body = client.post("/api/survival-analysis/kaplan-meier", json=request).json()

    assert body["stats"]["survival_at_horizon"] == pytest.approx(0.5)


def test_kaplan_meier_counts_censoring_separately(client):
    request = {
        "patient_data": [
            record(0, days=10.0, event=True),
            record(1, days=20.0, event=False),
            record(2, days=30.0, event=True),
        ],
        "time_horizon_days": 365,
    }

    body = client.post("/api/survival-analysis/kaplan-meier", json=request).json()

    assert body["stats"]["events_occurred"] == 2
    assert body["stats"]["censored"] == 1


def test_kaplan_meier_rejects_a_cohort_with_no_events(client):
    request = {
        "patient_data": [record(i, event=False) for i in range(4)],
        "time_horizon_days": 365,
    }

    response = client.post("/api/survival-analysis/kaplan-meier", json=request)

    assert response.status_code == 422
    assert "No events observed" in response.json()["detail"]


def test_median_survival_is_none_when_the_curve_never_reaches_a_half():
    """"Not reached" is a real answer and must not be replaced by a number."""
    assert median_survival_time([0.0, 10.0, 20.0], [1.0, 0.9, 0.8]) is None
    assert median_survival_time([0.0, 10.0, 20.0], [1.0, 0.6, 0.4]) == 20.0


# --------------------------------------------------------------------------
# Cox regression
# --------------------------------------------------------------------------

def test_cox_endpoint_is_deterministic_for_a_fixed_cohort(client):
    """Regression guard: observed_time and event used to be drawn per row
    from np.random, so the hazard ratios were noise."""
    request = {"patient_data": survival_cohort_payload(60), "time_horizon_days": 365}

    first = client.post("/api/survival-analysis/cox-regression", json=request).json()
    second = client.post("/api/survival-analysis/cox-regression", json=request).json()

    assert first["hazard_ratios"] == second["hazard_ratios"]
    assert first["p_values"] == second["p_values"]


def test_cox_hazard_ratios_respond_to_the_cohort(client):
    a = client.post(
        "/api/survival-analysis/cox-regression",
        json={"patient_data": survival_cohort_payload(60, seed=1)},
    ).json()
    b = client.post(
        "/api/survival-analysis/cox-regression",
        json={"patient_data": survival_cohort_payload(60, seed=2)},
    ).json()

    assert a["hazard_ratios"] != b["hazard_ratios"]


def test_cox_reports_confidence_intervals_that_bracket_the_estimate(client):
    body = client.post(
        "/api/survival-analysis/cox-regression",
        json={"patient_data": survival_cohort_payload(120)},
    ).json()

    assert body["confidence_intervals"]
    for name, (lower, upper) in body["confidence_intervals"].items():
        assert lower < body["hazard_ratios"][name] < upper, name
        assert (lower, upper) != (0, 0)


def test_cox_labels_its_c_index_as_in_sample(client):
    """The figure is computed on the fitted rows. Saying so in the payload
    keeps a reader from reading it as generalisation performance."""
    body = client.post(
        "/api/survival-analysis/cox-regression",
        json={"patient_data": survival_cohort_payload(60)},
    ).json()

    assert body["c_index_is_in_sample"] is True


def test_cox_drops_constant_covariates_and_says_which(client):
    cohort = [
        record(i, days=10.0 * (i + 1), event=bool(i % 2), age=60, risk=0.5, sex="male")
        for i in range(20)
    ]
    for entry in cohort:
        entry["clinical_indicators"]["comorbidity_count"] = 2

    body = client.post(
        "/api/survival-analysis/cox-regression", json={"patient_data": cohort}
    )

    # Age, sex, risk and comorbidity count are all constant here.
    assert body.status_code == 422
    assert "constant" in body.json()["detail"]


# --------------------------------------------------------------------------
# Causal inference
# --------------------------------------------------------------------------

def ate_payload(effect: float = 2.0, n: int = 200, seed: int = 3) -> dict:
    gen = np.random.default_rng(seed)
    patients = []
    for i in range(n):
        treatment = bool(gen.binomial(1, 0.5))
        outcome = 1.0 + effect * treatment + float(gen.normal(0, 1))
        patients.append(
            treated(
                i,
                treatment,
                round(outcome, 4),
                age=int(gen.integers(40, 85)),
                risk=round(float(gen.uniform(0, 1)), 3),
                sex="male" if i % 2 else "female",
            )
        )
    return {
        "patient_data": patients,
        "treatment_variable": "statin",
        "outcome_variable": "ldl_change",
    }


def test_ate_endpoint_is_deterministic_for_a_fixed_cohort(client):
    """Regression guard: the outcome used to be synthesised as
    np.random.binomial(1, 0.1 + 0.1 * treatment), so the reported effect
    was the 0.1 constant in that line."""
    request = ate_payload()

    first = client.post("/api/causal-inference/ate-estimation", json=request).json()
    second = client.post("/api/causal-inference/ate-estimation", json=request).json()

    assert first["ate_estimates"] == second["ate_estimates"]


def test_ate_recovers_an_effect_present_in_the_submitted_outcomes(client):
    body = client.post("/api/causal-inference/ate-estimation", json=ate_payload(effect=2.0)).json()

    assert body["ate_estimates"]["ate_simple"] == pytest.approx(2.0, abs=0.35)
    assert body["ate_estimates"]["ate_regression"] == pytest.approx(2.0, abs=0.35)
    assert body["cohort"]["treated"] + body["cohort"]["control"] == 200


def test_ate_reports_no_effect_when_there_is_none(client):
    body = client.post("/api/causal-inference/ate-estimation", json=ate_payload(effect=0.0)).json()

    assert body["ate_estimates"]["ate_simple"] == pytest.approx(0.0, abs=0.35)


def test_ate_states_its_observational_caveat(client):
    body = client.post("/api/causal-inference/ate-estimation", json=ate_payload()).json()

    assert "unmeasured confounding" in body["caveat"]


def test_ate_rejects_a_single_treatment_arm(client):
    request = ate_payload()
    for entry in request["patient_data"]:
        entry["treatment_assigned"] = True

    response = client.post("/api/causal-inference/ate-estimation", json=request)

    assert response.status_code == 422
    assert "same treatment arm" in response.json()["detail"]


def test_ate_rejects_an_unavailable_confounder(client):
    request = ate_payload()
    request["confounders"] = ["smoking_pack_years"]

    response = client.post("/api/causal-inference/ate-estimation", json=request)

    assert response.status_code == 422
    assert "smoking_pack_years" in response.json()["detail"]


# --------------------------------------------------------------------------
# Cohort comparison
# --------------------------------------------------------------------------

def test_identical_cohorts_are_not_reported_as_different(client):
    """Regression guard: cohorts_have_difference and the p-value were
    literals (True, 0.05), so a cohort compared against a copy of itself
    still came back as significantly different."""
    same = [member(i, {"mortality": float(i % 4 == 0)}) for i in range(40)]

    body = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": same, "comparator_cohort": same},
    ).json()["cohort_comparison"]

    assert body["outcomes"]["mortality"]["p_value"] > 0.05
    assert body["outcomes"]["mortality"]["significant"] is False
    assert body["comparison_metrics"]["cohorts_have_difference"] is False


def test_a_real_difference_between_cohorts_is_detected(client):
    low = [member(i, {"mortality": float(i < 2)}) for i in range(40)]
    high = [member(i, {"mortality": float(i < 32)}) for i in range(40)]

    outcome = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": low, "comparator_cohort": high},
    ).json()["cohort_comparison"]["outcomes"]["mortality"]

    assert outcome["p_value"] < 0.001
    assert outcome["significant"] is True
    assert outcome["cohort_1_rate"] == pytest.approx(0.05)
    assert outcome["cohort_2_rate"] == pytest.approx(0.80)
    assert outcome["risk_difference"] == pytest.approx(0.75)


def test_cohort_comparison_names_the_test_it_ran(client):
    """A p-value with no named test cannot be checked by a reviewer."""
    low = [member(i, {"mortality": float(i < 5)}) for i in range(40)]
    high = [member(i, {"mortality": float(i < 25)}) for i in range(40)]

    outcome = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": low, "comparator_cohort": high},
    ).json()["cohort_comparison"]["outcomes"]["mortality"]

    assert outcome["test"] in {"chi2_contingency", "fisher_exact"}


def test_small_cells_use_fishers_exact_test(client):
    low = [member(i, {"mortality": 0.0}) for i in range(10)]
    high = [member(i, {"mortality": float(i < 1)}) for i in range(10)]

    outcome = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": low, "comparator_cohort": high},
    ).json()["cohort_comparison"]["outcomes"]["mortality"]

    assert outcome["test"] == "fisher_exact"


def test_continuous_outcomes_use_welchs_t_test(client):
    gen = np.random.default_rng(7)
    low = [member(i, {"los_days": float(gen.normal(4, 1))}) for i in range(50)]
    high = [member(i, {"los_days": float(gen.normal(9, 1))}) for i in range(50)]

    outcome = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": low, "comparator_cohort": high},
    ).json()["cohort_comparison"]["outcomes"]["los_days"]

    assert outcome["test"] == "welch_t"
    assert outcome["mean_difference"] == pytest.approx(5.0, abs=0.6)
    assert outcome["significant"] is True


def test_cohort_comparison_summarises_both_cohorts(client):
    a = [member(i, {"mortality": float(i % 2)}, age=50) for i in range(10)]
    b = [member(i, {"mortality": float(i % 2)}, age=80) for i in range(10)]

    body = client.post(
        "/api/cohorts/compare", json={"patient_cohort": a, "comparator_cohort": b}
    ).json()["cohort_comparison"]

    assert body["cohort_1"]["characteristics"]["mean_age"] == 50
    assert body["cohort_2"]["characteristics"]["mean_age"] == 80
    assert sum(body["cohort_1"]["characteristics"]["gender_distribution"].values()) == 10


def test_cohort_comparison_requires_a_comparator(client):
    response = client.post(
        "/api/cohorts/compare",
        json={"patient_cohort": [member(i, {"mortality": 0.0}) for i in range(4)]},
    )

    assert response.status_code == 422
    assert "comparator cohort is required" in response.json()["detail"]


def test_cohort_comparison_rejects_an_outcome_missing_from_one_side(client):
    a = [member(i, {"mortality": float(i % 2)}) for i in range(10)]
    b = [member(i, {"readmission": float(i % 2)}) for i in range(10)]

    response = client.post(
        "/api/cohorts/compare", json={"patient_cohort": a, "comparator_cohort": b}
    )

    assert response.status_code == 422


def test_cohort_comparison_warns_about_multiple_comparisons(client):
    a = [member(i, {"mortality": float(i % 2), "readmission": float(i % 3)}) for i in range(20)]
    b = [member(i, {"mortality": float(i % 2), "readmission": float(i % 4)}) for i in range(20)]

    metrics = client.post(
        "/api/cohorts/compare", json={"patient_cohort": a, "comparator_cohort": b}
    ).json()["cohort_comparison"]["comparison_metrics"]

    assert metrics["outcomes_tested"] == 2
    assert "multiple comparisons" in metrics["note"]


# --------------------------------------------------------------------------
# Evidence search
# --------------------------------------------------------------------------

class StubSearcher(EvidenceSearcher):
    def __init__(self, results=None, error=None, mode="live"):
        self.results = results or []
        self.error = error
        self.mode = mode
        self.calls = []

    async def search(self, query, limit):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return EvidenceSearchResponse(
            results=self.results[:limit], mode=self.mode, note="stub")


@pytest.fixture
def stub_searcher():
    """Swap the live PubMed/ClinicalTrials searcher out for the test."""
    stub = StubSearcher()
    app.dependency_overrides[get_evidence_searcher] = lambda: stub
    yield stub
    app.dependency_overrides.pop(get_evidence_searcher, None)


ARTICLE = {
    "id": "pubmed_31234567",
    "title": "Dapagliflozin in Patients with Heart Failure",
    "abstract": "CONCLUSIONS: the risk of worsening heart failure was lower.",
    "source": "PubMed",
    "pub_date": "2019",
    "authors": ["John J V McMurray"],
    "mesh_terms": ["Heart Failure"],
    "url": "https://pubmed.ncbi.nlm.nih.gov/31234567/",
}


def test_evidence_search_reports_which_mode_produced_the_results(client, stub_searcher):
    """"Not in our corpus" and "not in the literature" are different
    claims; the payload has to distinguish them."""
    stub_searcher.results = [ARTICLE]
    stub_searcher.mode = "index"

    body = client.get("/api/evidence/search", params={"query": "heart failure"}).json()

    assert body["retrieval_mode"] == "index"
    assert "graph_context" in body


def test_evidence_search_returns_what_the_searcher_found(client, stub_searcher):
    """Regression guard: results used to be templated strings built from the
    query ('Medical Evidence Title 0: <query>')."""
    stub_searcher.results = [ARTICLE]

    body = client.get("/api/evidence/search", params={"query": "heart failure"}).json()

    assert body["results"] == [ARTICLE]
    assert body["total_results"] == 1
    assert stub_searcher.calls == [("heart failure", 10)]


def test_evidence_search_does_not_echo_the_query_back_as_a_result(client, stub_searcher):
    stub_searcher.results = []

    body = client.get(
        "/api/evidence/search", params={"query": "zzqx nonexistent condition"}
    ).json()

    assert body["results"] == []
    assert body["total_results"] == 0


def test_evidence_search_is_a_get(client, stub_searcher):
    """It has no request body, and this is the verb the Streamlit frontend
    has always used -- the previous POST declaration answered it with 405."""
    assert client.get("/api/evidence/search", params={"query": "hf"}).status_code == 200
    assert client.post("/api/evidence/search", params={"query": "hf"}).status_code == 405


def test_evidence_search_passes_the_limit_through(client, stub_searcher):
    stub_searcher.results = [ARTICLE] * 10

    body = client.get(
        "/api/evidence/search", params={"query": "heart failure", "limit": 3}
    ).json()

    assert stub_searcher.calls == [("heart failure", 3)]
    assert len(body["results"]) == 3


def test_evidence_search_rejects_malformed_filters(client, stub_searcher):
    response = client.get(
        "/api/evidence/search", params={"query": "heart failure", "filters": "{not json"}
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_evidence_search_rejects_an_empty_query(client, stub_searcher):
    assert client.get("/api/evidence/search", params={"query": ""}).status_code == 422


def test_evidence_search_surfaces_an_upstream_failure(client, stub_searcher):
    """A 429 from PubMed must not read as 'no evidence exists'."""
    stub_searcher.error = RuntimeError("PubMed search returned HTTP 429")

    response = client.get("/api/evidence/search", params={"query": "heart failure"})

    assert response.status_code == 502
    assert "429" in response.json()["detail"]


# --------------------------------------------------------------------------
# Graph-RAG backed search
# --------------------------------------------------------------------------

class StubFused:
    def __init__(self, doc_id, title, pmid=None):
        self.id = doc_id
        self.title = title
        self.content = f"Abstract of {doc_id}"
        self.source = "PubMed"
        self.citation = f"PMID:{pmid}" if pmid else doc_id
        self.fused_score = 0.05
        self.found_by = {"bm25": 1, "vector": 2}
        self.metadata = {"pmid": pmid, "journal": "NEJM", "pub_date": "2019"}


class StubAnswer:
    def __init__(self, results, graph_context=None):
        self.results = results
        self.graph_context = graph_context or []
        self.coverage = {"note": "retrieval consensus, not correctness"}


class StubRagService:
    def __init__(self, answer):
        self.answer = answer
        self.queries = []

    async def answer_query(self, query, limit):
        self.queries.append((query, limit))
        return self.answer


class StubLive(EvidenceSearcher):
    def __init__(self, results):
        self.results = results
        self.fetched = []

    async def fetch(self, query, limit):
        self.fetched.append((query, limit))
        return self.results


async def test_graph_rag_search_uses_the_index_and_reports_graph_context():
    """The whole point of indexing: search the local corpus, not PubMed."""
    service = StubRagService(StubAnswer(
        [StubFused("pubmed_1", "Dapagliflozin in heart failure", pmid="31234567")],
        graph_context=[{"entity": "Heart Failure", "entity_type": "Condition"}],
    ))
    live = StubLive([{"id": "should_not_be_used"}])

    response = await GraphRAGEvidenceSearcher(service, live).search("heart failure", 5)

    assert response.mode == "index"
    assert response.results[0]["citation"] == "PMID:31234567"
    assert response.graph_context[0]["entity"] == "Heart Failure"
    assert live.fetched == [], "live retrieval should not run when the index answers"


async def test_graph_rag_results_expose_how_they_were_ranked():
    """A fused ranking a reader cannot inspect is a black box."""
    service = StubRagService(StubAnswer([StubFused("pubmed_1", "T", pmid="1")]))

    response = await GraphRAGEvidenceSearcher(service, StubLive([])).search("hf", 5)

    assert response.results[0]["found_by"] == {"bm25": 1, "vector": 2}
    assert response.results[0]["fused_score"] == 0.05


async def test_an_empty_index_falls_back_to_live_and_says_so():
    """Silently blending the two would let "our corpus lacks this" pass as
    "the literature lacks this"."""
    service = StubRagService(StubAnswer([]))
    live = StubLive([{"id": "pubmed_99", "title": "Found live"}])

    response = await GraphRAGEvidenceSearcher(service, live).search("rare disease", 5)

    assert response.mode == "index_then_live"
    assert response.results == [{"id": "pubmed_99", "title": "Found live"}]
    assert "not absence from the literature" in response.note
    assert live.fetched == [("rare disease", 5)]


async def test_the_fallback_does_not_write_to_the_index():
    """Ingesting on read would let the corpus reshape itself around
    whatever people happen to search for."""
    service = StubRagService(StubAnswer([]))

    await GraphRAGEvidenceSearcher(service, StubLive([])).search("rare disease", 5)

    assert not hasattr(service, "stored")


async def test_the_searcher_degrades_to_live_when_the_stack_is_down(monkeypatch):
    """A missing docker-compose stack must not take the API down with it."""
    import src.api_backend as backend

    class Boom:
        def __init__(self, *a, **k):
            raise ConnectionError("Cannot reach Neo4j")

    monkeypatch.setattr(
        "src.graph_rag_service.main.GraphRAGService", Boom, raising=True)

    searcher = await backend.build_evidence_searcher()

    assert isinstance(searcher, backend.LiteratureEvidenceSearcher)


# --------------------------------------------------------------------------
# Authentication
#
# There was none, and CORS was allow_origins=["*"] with credentials, which
# lets any page on the internet make authenticated requests on a viewer's
# behalf against a patient-analysis API.
# --------------------------------------------------------------------------

import src.api_backend as backend


@pytest.fixture
def keyed_client(monkeypatch):
    """A client against an API with keys configured."""
    monkeypatch.setattr(backend, "API_KEYS", {"secret-key-one", "secret-key-two"})
    return TestClient(app)


def test_health_states_whether_authentication_is_on(client):
    """An unauthenticated deployment should be visible to whoever looks."""
    assert client.get("/api/health").json()["authentication"] == "disabled"


def test_health_states_when_authentication_is_on(keyed_client):
    assert keyed_client.get("/api/health").json()["authentication"] == "api_key"


def test_an_analysis_endpoint_rejects_a_request_with_no_key(keyed_client):
    response = keyed_client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [record(i) for i in range(3)]})

    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_an_analysis_endpoint_rejects_a_wrong_key(keyed_client):
    response = keyed_client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [record(i) for i in range(3)]},
        headers={"X-API-Key": "not-a-real-key"})

    assert response.status_code == 401


def test_a_valid_key_is_accepted(keyed_client):
    response = keyed_client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [record(i, days=10.0 * (i + 1)) for i in range(3)]},
        headers={"X-API-Key": "secret-key-one"})

    assert response.status_code == 200


def test_every_analysis_endpoint_is_protected(keyed_client):
    """A single unprotected analysis route defeats the whole thing."""
    unprotected = []
    for method, path, payload in [
        ("post", "/api/models/risk/train", {"training_data": labelled_cohort(20)}),
        ("post", "/api/patients/risk-assessment", [patient(0)]),
        ("post", "/api/survival-analysis/kaplan-meier", {"patient_data": [record(0), record(1)]}),
        ("post", "/api/survival-analysis/cox-regression", {"patient_data": [record(0), record(1)]}),
        ("post", "/api/causal-inference/ate-estimation", ate_payload(n=8)),
        ("post", "/api/cohorts/compare",
         {"patient_cohort": [member(i, {"m": 0.0}) for i in range(2)],
          "comparator_cohort": [member(i, {"m": 1.0}) for i in range(2)]}),
    ]:
        if getattr(keyed_client, method)(path, json=payload).status_code != 401:
            unprotected.append(path)

    assert unprotected == [], f"reachable without a key: {unprotected}"


def test_evidence_search_is_protected(keyed_client, stub_searcher):
    assert keyed_client.get(
        "/api/evidence/search", params={"query": "hf"}).status_code == 401
    assert keyed_client.get(
        "/api/evidence/search", params={"query": "hf"},
        headers={"X-API-Key": "secret-key-two"}).status_code == 200


def test_health_and_root_stay_open_for_probes(keyed_client):
    """A load balancer cannot present a key."""
    assert keyed_client.get("/api/health").status_code == 200
    assert keyed_client.get("/health").status_code == 200
    assert keyed_client.get("/").status_code == 200


def test_cors_no_longer_allows_every_origin():
    """allow_origins=["*"] with allow_credentials lets any page on the
    internet make authenticated requests on a viewer's behalf."""
    assert "*" not in backend.ALLOWED_ORIGINS
    assert backend.ALLOWED_ORIGINS


def test_api_keys_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MEG_API_KEYS", " key-a , key-b ,, ")

    assert backend.load_api_keys() == {"key-a", "key-b"}


def test_allowed_origins_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MEG_ALLOWED_ORIGINS", "https://clinic.example")

    assert backend.load_allowed_origins() == ["https://clinic.example"]


def test_a_missing_config_file_does_not_silently_grant_access(tmp_path):
    """An unreadable config must yield no keys, not an empty allowlist that
    happens to authorise everything."""
    assert backend.load_api_keys(str(tmp_path / "nope.json")) == set()
