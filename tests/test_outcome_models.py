"""Feature engineering, multi-task outcome models, and the deep survival model."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch

import src.enhanced_ml_models as ml_models
from src.enhanced_ml_models import DeepSurvivalModel, EnhancedOutcomeModels, TorchRiskModel


@pytest.fixture
def outcome_models() -> EnhancedOutcomeModels:
    return EnhancedOutcomeModels()


@pytest.fixture
def small_patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [70.0, 50.0, 82.0, 44.0],
            "gender": ["M", "F", "F", "M"],
            "baseline_risk_score": [0.9, 0.1, 0.7, 0.2],
            "comorbidity_count": [2, 3, 5, 0],
        }
    )


# --------------------------------------------------------------------------
# Feature building
# --------------------------------------------------------------------------

def test_build_patient_features_derives_expected_columns(outcome_models, small_patients):
    features = outcome_models.build_patient_features(small_patients)

    assert list(features.columns) == [
        "age",
        "age_squared",
        "is_elderly",
        "is_male",
        "is_female",
        "baseline_risk_score",
        "comorbidity_count",
    ]


def test_build_patient_features_keeps_severity_score(outcome_models, small_patients):
    """The API schema accepts severity_score; it used to be dropped here."""
    patients = small_patients.assign(severity_score=[1.0, 4.0, 9.0, 2.0])

    features = outcome_models.build_patient_features(patients)

    assert features["severity_score"].tolist() == [1.0, 4.0, 9.0, 2.0]
    assert len(features) == len(small_patients)
    assert not features.isna().any().any()

    np.testing.assert_allclose(features["age_squared"], small_patients["age"] ** 2)
    assert features["is_elderly"].tolist() == [1, 0, 1, 0]
    assert features["is_male"].tolist() == [1, 0, 0, 1]
    assert features["is_female"].tolist() == [0, 1, 1, 0]


def test_build_patient_features_omits_absent_inputs(outcome_models):
    """Callers may pass a partial frame; the builder must not invent columns."""
    features = outcome_models.build_patient_features(pd.DataFrame({"age": [30.0, 60.0]}))

    assert list(features.columns) == ["age", "age_squared", "is_elderly"]
    assert "is_male" not in features


def test_build_patient_features_fills_missing_values(outcome_models):
    patients = pd.DataFrame({"age": [40.0, np.nan, 60.0], "gender": ["M", "F", "M"]})

    features = outcome_models.build_patient_features(patients)

    assert not features.isna().any().any()
    assert features["age"].iloc[1] == pytest.approx(50.0)  # median of 40 and 60


# --------------------------------------------------------------------------
# Multi-task training
# --------------------------------------------------------------------------

def test_train_multi_task_model_labels_each_outcome_by_task_type(
    outcome_models, survival_cohort
):
    patients = survival_cohort.head(300)
    outcomes = {
        "mortality": patients["event"].to_numpy(),
        "length_of_stay": patients["observed_time"].to_numpy(),
    }

    models = outcome_models.train_multi_task_model(patients, outcomes)

    assert models["mortality"]["type"] == "classification"
    assert models["length_of_stay"]["type"] == "regression"
    assert outcome_models.is_fitted


def test_train_multi_task_model_skips_constant_outcomes(outcome_models, small_patients):
    """An outcome with no variation carries no signal and must not be modelled."""
    outcomes = {
        "everyone_survived": np.zeros(len(small_patients), dtype=int),
        "mixed": np.array([0, 1, 1, 0]),
    }

    models = outcome_models.train_multi_task_model(small_patients, outcomes)

    assert "everyone_survived" not in models
    assert "mixed" in models


def test_train_multi_task_model_skips_length_mismatched_outcomes(
    outcome_models, small_patients
):
    outcomes = {"wrong_length": np.array([0, 1])}

    models = outcome_models.train_multi_task_model(small_patients, outcomes)

    assert models == {}


def test_predict_risk_returns_probabilities_for_classifiers(outcome_models, survival_cohort):
    patients = survival_cohort.head(300)
    models = outcome_models.train_multi_task_model(
        patients, {"mortality": patients["event"].to_numpy()}
    )

    features = outcome_models.build_patient_features(patients)
    risk = outcome_models.predict_risk(models["mortality"]["model"], features)

    assert risk.shape == (len(patients),)
    assert ((risk >= 0.0) & (risk <= 1.0)).all()


def test_predict_risk_rejects_objects_that_cannot_predict(outcome_models, small_patients):
    with pytest.raises(ValueError, match="doesn't have predict"):
        outcome_models.predict_risk(object(), small_patients)


# --------------------------------------------------------------------------
# Deep survival model
# --------------------------------------------------------------------------

def test_torch_risk_model_outputs_probabilities_by_default():
    torch.manual_seed(0)
    model = TorchRiskModel(input_dim=5)

    output = model(torch.randn(8, 5))

    assert output.shape == (8, 1)
    assert ((output >= 0.0) & (output <= 1.0)).all()


def test_torch_risk_model_can_emit_an_unbounded_log_risk_score():
    """Required by the Cox partial likelihood, which is defined on the
    log-hazard scale -- a sigmoid would cap the achievable hazard ratio."""
    torch.manual_seed(0)
    model = TorchRiskModel(input_dim=5, final_activation="none")

    output = model(torch.randn(256, 5) * 10)

    assert output.shape == (256, 1)
    assert bool((output < 0.0).any()), "output never goes negative; still squashed"


def test_torch_risk_model_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="final_activation must be one of"):
        TorchRiskModel(input_dim=5, final_activation="softmax")


def test_deep_survival_model_refuses_to_predict_before_training():
    model = DeepSurvivalModel(input_dim=3)

    with pytest.raises(ValueError, match="must be trained before prediction"):
        model.predict_risk(pd.DataFrame(np.zeros((2, 3))))


def test_deep_survival_model_trains_and_predicts(survival_cohort):
    torch.manual_seed(0)
    patients = survival_cohort.head(300)
    features = EnhancedOutcomeModels().build_patient_features(patients)

    model = DeepSurvivalModel(input_dim=features.shape[1])
    losses = model.train(
        features, patients["observed_time"].to_numpy(), patients["event"].to_numpy(), epochs=40
    )

    risk = model.predict_risk(features)

    assert model.is_fitted
    assert risk.shape == (len(patients),)
    assert np.isfinite(risk).all()
    # Log-risk scores are unbounded; a sigmoid-squashed output would sit in
    # [0, 1] and could not express a hazard ratio.
    assert risk.std() > 0
    assert losses[-1] < losses[0], "partial likelihood did not improve"


def test_deep_survival_model_learns_to_discriminate(survival_cohort):
    """A c-index above chance on data with a real hazard signal."""
    torch.manual_seed(0)
    patients = survival_cohort.head(600)
    features = EnhancedOutcomeModels().build_patient_features(patients)
    times = patients["observed_time"].to_numpy()
    events = patients["event"].to_numpy()

    model = DeepSurvivalModel(input_dim=features.shape[1])
    model.train(features, times, events, epochs=120)

    assert model.concordance(features, times, events) > 0.55


def test_partial_hazard_is_the_exponentiated_risk_score(survival_cohort):
    torch.manual_seed(0)
    patients = survival_cohort.head(80)
    features = EnhancedOutcomeModels().build_patient_features(patients)

    model = DeepSurvivalModel(input_dim=features.shape[1])
    model.train(features, patients["observed_time"].to_numpy(), patients["event"].to_numpy(), epochs=3)

    np.testing.assert_allclose(
        model.predict_partial_hazard(features),
        np.exp(model.predict_risk(features)),
        rtol=1e-6,
    )


def test_cox_partial_log_likelihood_matches_a_hand_computation():
    """Three patients, all events, all risk scores zero.

    Sorted by descending duration the risk sets have sizes 1, 2, 3, so the
    contributions are -log(1), -log(2), -log(3) and the mean negative
    partial log-likelihood is (log 2 + log 3) / 3.
    """
    log_risk = torch.zeros(3)
    durations = torch.tensor([1.0, 2.0, 3.0])
    events = torch.tensor([1.0, 1.0, 1.0])

    loss = DeepSurvivalModel.cox_partial_log_likelihood(log_risk, durations, events)

    assert float(loss) == pytest.approx((np.log(2) + np.log(3)) / 3, abs=1e-6)


def test_cox_partial_log_likelihood_rejects_a_fully_censored_batch():
    """With no observed event the partial likelihood has no terms at all."""
    with pytest.raises(ValueError, match="every patient in this batch is censored"):
        DeepSurvivalModel.cox_partial_log_likelihood(
            torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3)
        )


def test_deep_survival_model_rejects_mismatched_input_lengths(survival_cohort):
    features = EnhancedOutcomeModels().build_patient_features(survival_cohort.head(10))
    model = DeepSurvivalModel(input_dim=features.shape[1])

    with pytest.raises(ValueError, match="Length mismatch"):
        model.train(features, np.ones(5), np.ones(10))


# --------------------------------------------------------------------------
# Known defects -- proven here, not yet fixed.
# --------------------------------------------------------------------------

def test_comorbidity_count_is_the_comorbidity_count(outcome_models, small_patients):
    """Previously this summed baseline_risk_score in too, because the column
    filter matched any name containing 'score'."""
    features = outcome_models.build_patient_features(small_patients)

    assert features["comorbidity_count"].tolist() == [2, 3, 5, 0]


def test_deep_survival_model_uses_survival_times(survival_cohort):
    """The fit must depend on WHEN events happened, not just whether.

    The earlier implementation minimised BCE on the event indicator alone,
    so permuting the durations left the fit untouched. Note that comparing
    two *constant* duration vectors would not detect this: under Cox every
    risk set is then identical regardless of the constant.
    """
    patients = survival_cohort.head(200)
    features = EnhancedOutcomeModels().build_patient_features(patients)
    events = patients["event"].to_numpy()
    times = patients["observed_time"].to_numpy()

    def fit_with(durations):
        torch.manual_seed(0)
        model = DeepSurvivalModel(input_dim=features.shape[1])
        model.train(features, durations, events, epochs=20)
        return model.predict_risk(features)

    as_observed = fit_with(times)
    reversed_order = fit_with(np.sort(times)[::-1][np.argsort(np.argsort(times))].copy())

    assert not np.allclose(as_observed, reversed_order), (
        "reversing the survival ordering had no effect on the fit"
    )


def test_comorbidity_count_is_the_comorbidity_count(outcome_models, small_patients):
    """Previously this summed baseline_risk_score in too, because the column
    filter matched any name containing 'score'."""
    features = outcome_models.build_patient_features(small_patients)

    assert features["comorbidity_count"].tolist() == [2, 3, 5, 0]


def test_no_duplicate_definitions_in_enhanced_ml_models():
    """EnhancedOutcomeModels and DeepSurvivalModel.predict_risk were each
    defined twice, the later definition silently shadowing the earlier."""
    source = inspect.getsource(ml_models)

    assert source.count("class EnhancedOutcomeModels") == 1
    assert inspect.getsource(DeepSurvivalModel).count("def predict_risk") == 1
