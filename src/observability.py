"""Metrics, and optional experiment tracking.

`config/settings.json` declared an mlflow_tracking_uri, langfuse keys and
a prometheus_endpoint while no code referenced any of them. That is the
same defect as the `presidio.enabled: true` that was fixed with no
Presidio in the repository: a config asserting a capability answers the
question nobody then asks again.

Prometheus is implemented here, because it is cheap and useful. MLflow is
optional and follows the Presidio pattern -- enabling it without the
package installed raises at startup rather than quietly tracking nothing.

Langfuse is gone rather than implemented. It traces LLM and RAG
generation, and this system has no generation in it: retrieval returns
cited records and stops. Configuring a tracer for a component that does
not exist is the same false claim in a different direction.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


logger = logging.getLogger(__name__)


class MLflowConfigurationError(RuntimeError):
    """Experiment tracking was requested in a form that cannot be provided."""


class Metrics:
    """Prometheus collectors for this service.

    Its own registry rather than the global default: two instances in one
    process (the app and a test) would otherwise collide on collector
    names, which raises at import and is a confusing way to discover it.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        self.requests = Counter(
            "meg_http_requests_total", "HTTP requests handled",
            ["method", "endpoint", "status"], registry=self.registry)
        self.request_seconds = Histogram(
            "meg_http_request_duration_seconds", "Request duration",
            ["endpoint"], registry=self.registry)
        self.analyses = Counter(
            "meg_analyses_total", "Analyses run, by kind",
            ["analysis"], registry=self.registry)
        self.patients_analysed = Counter(
            "meg_patients_analysed_total",
            "Patient records passed through an analysis",
            ["analysis"], registry=self.registry)
        self.phi_rejections = Counter(
            "meg_phi_rejections_total",
            "Requests rejected for carrying direct identifiers",
            registry=self.registry)
        self.risk_model_ready = Gauge(
            "meg_risk_model_ready",
            "1 when a risk model is loaded and servable",
            registry=self.registry)
        self.risk_model_holdout = Gauge(
            "meg_risk_model_holdout_metric",
            "Held-out metric of the loaded risk model, by outcome",
            ["outcome", "metric"], registry=self.registry)

    def observe_request(self, method: str, endpoint: str, status: int,
                        seconds: float) -> None:
        self.requests.labels(method, endpoint, str(status)).inc()
        self.request_seconds.labels(endpoint).observe(seconds)

    def observe_analysis(self, analysis: str, n_patients: int = 0) -> None:
        self.analyses.labels(analysis).inc()
        if n_patients:
            self.patients_analysed.labels(analysis).inc(n_patients)

    def observe_risk_model(self, models: Dict[str, Any]) -> None:
        self.risk_model_ready.set(1 if models else 0)
        for outcome, model in models.items():
            if model.metric_value is not None:
                self.risk_model_holdout.labels(
                    outcome, model.metric_name).set(model.metric_value)

    def render(self) -> tuple:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class ExperimentTracker:
    """Optional MLflow tracking for risk-model training.

    Models already persist with their held-out metrics, so this adds a
    history across trainings rather than the metrics themselves. Off by
    default; enabling it without mlflow installed raises here rather than
    letting the service run while recording nothing.
    """

    def __init__(self, enabled: bool = False, tracking_uri: Optional[str] = None,
                 experiment: str = "risk-models"):
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self._mlflow = None

        if not self.enabled:
            return

        try:
            import mlflow
        except ImportError as e:
            raise MLflowConfigurationError(
                f"Experiment tracking is enabled but mlflow is not installed "
                f"({e}). Install the `tracking` extra, or set "
                f"observability.mlflow.enabled to false -- but do not leave "
                f"it claiming to record runs it is not recording.") from e

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_training(self, model_version: str, models: Dict[str, Any],
                     params: Optional[Dict[str, Any]] = None) -> bool:
        """Record one training run. Returns whether anything was logged."""
        if not self.enabled:
            return False

        try:
            with self._mlflow.start_run(run_name=model_version):
                self._mlflow.log_params(params or {})
                self._mlflow.set_tag("model_version", model_version)
                for outcome, model in models.items():
                    if model.metric_value is not None:
                        self._mlflow.log_metric(
                            f"{outcome}_{model.metric_name}", model.metric_value)
                    self._mlflow.log_param(f"{outcome}_n_train", model.n_train)
            return True
        except Exception as e:
            # Tracking is observability, not the product. It must not fail
            # a training run, and it must not fail silently either.
            logger.error(f"Could not log training run to MLflow: {e}")
            return False

    @property
    def describe(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False,
                    "note": ("Risk models persist with their held-out metrics; "
                             "training runs are recorded in the audit log. No "
                             "experiment tracker is configured.")}
        return {"enabled": True, "tracking_uri": self.tracking_uri,
                "experiment": self.experiment}


def tracker_from_config(config: Optional[Dict[str, Any]]) -> ExperimentTracker:
    mlflow_config = (config or {}).get("observability", {}).get("mlflow", {})
    return ExperimentTracker(
        enabled=bool(mlflow_config.get("enabled", False)),
        tracking_uri=mlflow_config.get("tracking_uri"),
        experiment=mlflow_config.get("experiment", "risk-models"),
    )


class RequestTimer:
    """Context manager measuring wall time in seconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
