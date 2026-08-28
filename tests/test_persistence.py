"""Persistence for trained models and registered guidelines.

Both registries were in-memory dicts. A restart silently discarded every
trained model -- risk assessment reverting to 503 "no risk model has been
trained" -- and every registered guideline, with nothing anywhere saying
they had ever existed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.api_backend import GuidelineStore, RiskModel, RiskModelRegistry


def fitted_model() -> RiskModel:
    gen = np.random.default_rng(0)
    X = gen.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    return RiskModel(
        outcome="mortality", model=forest, task="classification",
        feature_columns=["a", "b", "c"],
        feature_medians={"a": 0.0, "b": 0.0, "c": 0.0},
        metric_name="roc_auc", metric_value=0.91, n_train=30, n_test=10)


@pytest.fixture
def registry(tmp_path) -> RiskModelRegistry:
    registry = RiskModelRegistry()
    registry.__init_store__(tmp_path)
    return registry


# --------------------------------------------------------------------------
# Risk models
# --------------------------------------------------------------------------

def test_a_fresh_registry_has_nothing_to_load(registry):
    assert registry.load() is False
    assert registry.is_ready() is False


def test_a_trained_model_survives_a_restart(registry, tmp_path):
    registry.replace({"mortality": fitted_model()})
    version = registry.version

    restarted = RiskModelRegistry()
    restarted.__init_store__(tmp_path)

    assert restarted.load() is True
    assert restarted.is_ready()
    assert restarted.version == version
    assert restarted.models["mortality"].metric_value == 0.91


def test_the_restored_model_still_predicts_the_same(registry, tmp_path):
    """A model that loads but scores differently is worse than none: the
    response still carries the held-out AUC measured before."""
    original = fitted_model()
    registry.replace({"mortality": original})
    features = np.random.default_rng(1).normal(size=(5, 3))
    before = original.model.predict_proba(features)[:, 1]

    restarted = RiskModelRegistry()
    restarted.__init_store__(tmp_path)
    restarted.load()
    after = restarted.models["mortality"].model.predict_proba(features)[:, 1]

    np.testing.assert_allclose(before, after)


def test_the_feature_contract_survives_too(registry, tmp_path):
    """Columns and medians are what align incoming patients to the model;
    losing them would silently change what is being scored."""
    registry.replace({"mortality": fitted_model()})

    restarted = RiskModelRegistry()
    restarted.__init_store__(tmp_path)
    restarted.load()

    restored = restarted.models["mortality"]
    assert restored.feature_columns == ["a", "b", "c"]
    assert restored.feature_medians == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_a_model_from_another_sklearn_is_refused(registry, tmp_path, monkeypatch):
    """Unpickling an estimator across scikit-learn versions is undefined:
    it usually works, and when it does not it yields wrong predictions
    rather than an error. Refuse, do not warn."""
    import joblib

    registry.replace({"mortality": fitted_model()})
    state = joblib.load(registry._path)
    state["sklearn_version"] = "0.0.1-from-the-past"
    joblib.dump(state, registry._path)

    restarted = RiskModelRegistry()
    restarted.__init_store__(tmp_path)

    assert restarted.load() is False
    assert restarted.is_ready() is False


def test_an_unreadable_store_does_not_take_the_api_down(registry, tmp_path):
    registry._path.parent.mkdir(parents=True, exist_ok=True)
    registry._path.write_bytes(b"not a joblib file")

    assert registry.load() is False


def test_training_without_persisting_leaves_no_file(registry):
    registry.replace({"mortality": fitted_model()}, persist=False)

    assert not registry._path.exists()
    assert registry.is_ready()


# --------------------------------------------------------------------------
# Guidelines
# --------------------------------------------------------------------------

GUIDELINE = {
    "id": "dm2", "name": "Diabetes", "condition": "type 2 diabetes",
    "version": "1.0",
    "steps": [{"name": "HbA1c measurement", "type": "test", "recommended": True,
               "description": "", "timing": "immediate", "evidence_level": "unknown"}],
    "decision_points": [],
}


def test_guidelines_round_trip(tmp_path):
    store = GuidelineStore(tmp_path)
    store.save({"dm2": GUIDELINE})

    assert GuidelineStore(tmp_path).load()["dm2"]["condition"] == "type 2 diabetes"


def test_an_empty_store_loads_as_empty(tmp_path):
    assert GuidelineStore(tmp_path).load() == {}


def test_guidelines_are_stored_as_readable_json(tmp_path):
    """A guideline is plain data; a format a human can read and correct
    matters more here than convenience."""
    store = GuidelineStore(tmp_path)
    store.save({"dm2": GUIDELINE})

    text = store.path.read_text()
    assert "HbA1c measurement" in text
    assert store.path.suffix == ".json"


def test_a_corrupt_guideline_store_does_not_take_the_api_down(tmp_path):
    store = GuidelineStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json")

    assert store.load() == {}
