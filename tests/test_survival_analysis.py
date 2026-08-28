"""Survival analysis: Kaplan-Meier and Cox proportional hazards.

The recovery tests below are the point of this file. The cohort in
`survival_cohort` is simulated from an exponential proportional-hazards
model with hazard ratios we chose, so "did Cox work?" has a checkable
answer rather than an eyeballed one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.enhanced_ml_models import SurvivalAnalysisModels, SurvivalAnalysisResult
from tests.conftest import TRUE_BETA_AGE_Z, TRUE_BETA_TREATMENT


@pytest.fixture
def models() -> SurvivalAnalysisModels:
    return SurvivalAnalysisModels()


# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------

def test_kaplan_meier_matches_hand_computed_curve(models):
    """With no censoring, KM reduces to the empirical survival function."""
    duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 1, 1, 1, 1])

    times, survival = models.kaplan_meier_analysis(duration, event)

    assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(survival, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], atol=1e-12)


def test_kaplan_meier_censoring_does_not_drop_survival(models):
    """A censored observation removes a patient from the risk set; it is not a death."""
    duration = np.array([1.0, 2.0, 3.0])
    all_events = models.kaplan_meier_analysis(duration, np.array([1, 1, 1]))[1]
    censored_middle = SurvivalAnalysisModels().kaplan_meier_analysis(
        duration, np.array([1, 0, 1])
    )[1]

    # Censoring at t=2 means the curve does not step down there.
    assert all_events[2] == pytest.approx(1 / 3)
    assert censored_middle[2] == pytest.approx(2 / 3)


def test_kaplan_meier_curve_is_monotone_and_bounded(models, survival_cohort):
    times, survival = models.kaplan_meier_analysis(
        survival_cohort["observed_time"].to_numpy(),
        survival_cohort["event"].to_numpy(),
    )

    assert survival[0] == pytest.approx(1.0)
    assert times == sorted(times)
    assert all(0.0 <= s <= 1.0 for s in survival)
    assert all(later <= earlier + 1e-12 for earlier, later in zip(survival, survival[1:]))


def test_kaplan_meier_raises_rather_than_returning_an_empty_curve(models):
    """Regression guard for commit 54e4b8f.

    An empty curve renders as a blank chart, which a clinician reads as a
    result. A failed fit must raise.
    """
    with pytest.raises(RuntimeError, match="Kaplan-Meier fit failed on 0 observations"):
        models.kaplan_meier_analysis(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# Cox proportional hazards
# --------------------------------------------------------------------------

def test_cox_recovers_the_simulated_hazard_ratios(models, survival_cohort):
    """The estimator is checked against the coefficients used to simulate the data."""
    result = models.cox_regression_analysis(
        survival_cohort, "observed_time", "event", ["treatment", "age_z"]
    )

    assert isinstance(result, SurvivalAnalysisResult)
    assert result.hazard_ratios["treatment"] == pytest.approx(
        np.exp(TRUE_BETA_TREATMENT), rel=0.10
    )
    assert result.hazard_ratios["age_z"] == pytest.approx(
        np.exp(TRUE_BETA_AGE_Z), rel=0.10
    )
    # Both effects are strong at n=4000; p-values should be decisive.
    assert result.p_values["treatment"] < 1e-6
    assert result.p_values["age_z"] < 1e-6


def test_cox_encodes_string_covariates_rather_than_failing(models, survival_cohort):
    """`gender` arrives as 'M'/'F'; the model label-encodes it internally."""
    result = models.cox_regression_analysis(
        survival_cohort, "observed_time", "event", ["treatment", "gender"]
    )

    assert set(result.hazard_ratios) == {"treatment", "gender"}
    # Gender was not in the data-generating process, so it should be null.
    assert result.p_values["gender"] > 0.01


def test_cox_c_index_is_in_the_plausible_range(models, survival_cohort):
    """0.5 is chance. A c-index near 1.0 on observational data means leakage.

    PHASE_2_SUMMARY.md reports "C-index of 0.980 (excellent discrimination)".
    On data simulated from a real PH model with noise, the achievable
    c-index is far lower; 0.98 indicates the outcome was a near-deterministic
    function of the covariates, not that the model discriminates well.
    """
    result = models.cox_regression_analysis(
        survival_cohort, "observed_time", "event", ["treatment", "age_z"]
    )

    assert 0.55 < result.c_index < 0.80


def test_cox_raises_keyerror_when_a_covariate_is_missing(models):
    """Regression guard for commit 54e4b8f: no empty result object.

    The type matters: a missing column is a mistake at the call site, and a
    caller can only distinguish that from an estimator failure if KeyError
    survives instead of being flattened into RuntimeError.
    """
    data = pd.DataFrame({"t": [1.0, 2.0, 3.0], "e": [1, 0, 1]})

    with pytest.raises(KeyError) as excinfo:
        models.cox_regression_analysis(data, "t", "e", ["nonexistent_covariate"])

    assert "nonexistent_covariate" in str(excinfo.value)


def test_cox_raises_when_no_rows_survive_cleaning(models):
    data = pd.DataFrame(
        {"t": [np.nan, np.nan], "e": [1, 0], "x": [1.0, 2.0]}
    )

    with pytest.raises(ValueError, match="No rows remain"):
        models.cox_regression_analysis(data, "t", "e", ["x"])


def test_cox_raises_when_a_covariate_is_not_numeric(models):
    data = pd.DataFrame(
        {"t": [1.0, 2.0, 3.0], "e": [1, 0, 1], "x": [1.0, 2.0, 3.0]}
    )

    with pytest.raises(RuntimeError):
        # A single-row-per-stratum frame cannot support a Cox fit.
        models.cox_regression_analysis(data.head(1), "t", "e", ["x"])


# --------------------------------------------------------------------------
# Result completeness
#
# Both assertions below were silently false until the wrong-column-name bugs
# in cox_regression_analysis were fixed.
# --------------------------------------------------------------------------

def test_cox_returns_real_confidence_intervals(models, survival_cohort):
    """Intervals must bracket the point estimate and cover the true value.

    These were previously always (0, 0) -- the assertion "the hazard ratio
    is exactly zero, with certainty" -- because the lookup used the
    exp(coef) column names against confidence_intervals_, which names them
    '95% lower-bound'/'95% upper-bound'.
    """
    result = models.cox_regression_analysis(
        survival_cohort, "observed_time", "event", ["treatment", "age_z"]
    )

    for param, true_beta in [("treatment", TRUE_BETA_TREATMENT), ("age_z", TRUE_BETA_AGE_Z)]:
        lower, upper = result.confidence_intervals[param]
        assert lower < result.hazard_ratios[param] < upper
        assert lower < np.exp(true_beta) < upper, f"{param} interval missed the truth"


def test_cox_returns_a_baseline_survival_curve(models, survival_cohort):
    """Previously always empty: the column was read as 'baseline survival_'
    (the real name has no trailing underscore) behind a bare `except:`."""
    result = models.cox_regression_analysis(
        survival_cohort, "observed_time", "event", ["treatment", "age_z"]
    )

    assert len(result.baseline_survival) > 0
    assert len(result.time_points) == len(result.baseline_survival)
    assert len(result.survival_probabilities) == len(result.baseline_survival)
    assert result.baseline_survival[0] == pytest.approx(1.0, abs=0.05)
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(result.baseline_survival, result.baseline_survival[1:])
    )
