"""Metrics and optional experiment tracking."""
from __future__ import annotations

import pytest

from src.observability import (
    ExperimentTracker,
    Metrics,
    MLflowConfigurationError,
    RequestTimer,
    tracker_from_config,
)


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


def rendered(metrics: Metrics) -> str:
    return metrics.render()[0].decode()


def test_two_instances_do_not_collide():
    """Its own registry per instance: the global default would raise on the
    second collector of the same name, at import, confusingly."""
    Metrics()
    Metrics()


def test_requests_are_counted_by_endpoint_and_status(metrics):
    metrics.observe_request("GET", "/api/health", 200, 0.01)
    metrics.observe_request("GET", "/api/health", 200, 0.02)
    metrics.observe_request("POST", "/api/audit", 401, 0.001)

    body = rendered(metrics)
    assert 'meg_http_requests_total{endpoint="/api/health",method="GET",status="200"} 2.0' in body
    assert 'status="401"' in body


def test_request_durations_are_observed(metrics):
    metrics.observe_request("GET", "/api/health", 200, 0.25)

    assert "meg_http_request_duration_seconds_count" in rendered(metrics)


def test_analyses_and_patient_counts_are_tracked(metrics):
    metrics.observe_analysis("survival.cox", 40)
    metrics.observe_analysis("survival.cox", 10)

    body = rendered(metrics)
    assert 'meg_analyses_total{analysis="survival.cox"} 2.0' in body
    assert 'meg_patients_analysed_total{analysis="survival.cox"} 50.0' in body


def test_a_ready_risk_model_shows_its_holdout_metric(metrics):
    class Model:
        metric_name, metric_value = "roc_auc", 0.83

    metrics.observe_risk_model({"mortality": Model()})

    body = rendered(metrics)
    assert "meg_risk_model_ready 1.0" in body
    assert 'meg_risk_model_holdout_metric{metric="roc_auc",outcome="mortality"} 0.83' in body


def test_no_model_reads_as_not_ready(metrics):
    metrics.observe_risk_model({})

    assert "meg_risk_model_ready 0.0" in rendered(metrics)


def test_a_model_with_an_unestimable_metric_is_not_reported_as_zero(metrics):
    """None means the holdout could not support the metric. Recording 0.0
    would read as a model that discriminates worse than chance."""
    class Model:
        metric_name, metric_value = "roc_auc", None

    metrics.observe_risk_model({"mortality": Model()})

    # The HELP/TYPE header is always emitted; what must be absent is a
    # sample line carrying a value.
    samples = [line for line in rendered(metrics).splitlines()
               if line.startswith("meg_risk_model_holdout_metric{")]
    assert samples == []


def test_the_timer_measures_elapsed_time():
    import time

    with RequestTimer() as timer:
        time.sleep(0.02)

    assert timer.seconds >= 0.02


# --------------------------------------------------------------------------
# Experiment tracking
# --------------------------------------------------------------------------

def test_tracking_is_off_by_default():
    tracker = ExperimentTracker()

    assert tracker.enabled is False
    assert tracker.log_training("v1", {}) is False


def test_a_disabled_tracker_says_what_happens_instead():
    described = ExperimentTracker().describe

    assert described["enabled"] is False
    assert "persist with their held-out metrics" in described["note"]


def test_enabling_tracking_without_mlflow_raises():
    """Same rule as Presidio: a silent no-op is how a config comes to claim
    something it is not doing."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        with pytest.raises(MLflowConfigurationError, match="not installed"):
            ExperimentTracker(enabled=True)
    else:
        pytest.skip("mlflow is installed; the failure path cannot be shown")


def test_config_without_an_mlflow_block_yields_a_disabled_tracker():
    assert tracker_from_config({}).enabled is False
    assert tracker_from_config(None).enabled is False


# --------------------------------------------------------------------------
# Through the API
# --------------------------------------------------------------------------

def test_the_metrics_endpoint_serves_prometheus_text():
    from fastapi.testclient import TestClient

    from src.api_backend import app

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "meg_http_requests_total" in response.text


def test_the_metrics_endpoint_is_open_like_health():
    """A scraper cannot present an API key, and the series are counts and
    durations, never patient content."""
    import src.api_backend as backend
    from fastapi.testclient import TestClient

    original = backend.API_KEYS
    backend.API_KEYS = {"a-key"}
    try:
        assert TestClient(app_of(backend)).get("/metrics").status_code == 200
    finally:
        backend.API_KEYS = original


def app_of(backend):
    return backend.app


def test_requests_are_labelled_by_route_not_by_path():
    """Labelling by concrete path creates a series per guideline id, which
    is how a metrics endpoint becomes the heaviest thing in the service."""
    from fastapi.testclient import TestClient

    from src.api_backend import app, metrics

    client = TestClient(app)
    client.get("/api/pathways/guidelines/aaa/evidence")
    client.get("/api/pathways/guidelines/bbb/evidence")

    body = metrics.render()[0].decode()
    assert "{guideline_id}" in body
    assert "aaa" not in body and "bbb" not in body
