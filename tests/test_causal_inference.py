"""Causal inference: propensity score matching and ATE estimation.

`estimate_ate` is the highest-stakes function in the codebase -- an
average treatment effect is a direct claim about whether a treatment
helps. These tests check it against a known effect under both randomised
and confounded assignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.enhanced_ml_models import CausalInferenceModels
from tests.conftest import TRUE_ATE


@pytest.fixture
def causal() -> CausalInferenceModels:
    return CausalInferenceModels()


# --------------------------------------------------------------------------
# Recovery of a known effect
# --------------------------------------------------------------------------

def test_ate_recovers_the_known_effect_under_randomisation(causal, unconfounded_cohort):
    """With random assignment all three estimators should agree with the truth."""
    results = causal.estimate_ate(
        unconfounded_cohort,
        unconfounded_cohort["treatment"].to_numpy(),
        unconfounded_cohort["outcome"].to_numpy(),
        ["x1", "x2"],
    )

    assert set(results) == {"ate_simple", "ate_matched", "ate_regression"}
    for method, estimate in results.items():
        assert estimate == pytest.approx(TRUE_ATE, abs=0.15), f"{method} was {estimate}"


def test_unadjusted_ate_is_biased_when_assignment_is_confounded(causal, confounded_cohort):
    """The naive difference in means must NOT be trusted here -- and isn't.

    x1 raises both the chance of treatment and the outcome, so the simple
    difference overstates the effect substantially. This test exists so
    that a future "simplification" of estimate_ate down to the simple
    difference fails loudly.
    """
    results = causal.estimate_ate(
        confounded_cohort,
        confounded_cohort["treatment"].to_numpy(),
        confounded_cohort["outcome"].to_numpy(),
        ["x1", "x2"],
    )

    assert results["ate_simple"] > TRUE_ATE + 1.0


def test_adjustment_removes_most_of_the_confounding_bias(causal, confounded_cohort):
    results = causal.estimate_ate(
        confounded_cohort,
        confounded_cohort["treatment"].to_numpy(),
        confounded_cohort["outcome"].to_numpy(),
        ["x1", "x2"],
    )

    naive_bias = abs(results["ate_simple"] - TRUE_ATE)
    for adjusted in ("ate_matched", "ate_regression"):
        assert abs(results[adjusted] - TRUE_ATE) < naive_bias / 2, (
            f"{adjusted} did not improve on the unadjusted estimate"
        )

    assert results["ate_regression"] == pytest.approx(TRUE_ATE, abs=0.15)


# --------------------------------------------------------------------------
# Propensity score matching
# --------------------------------------------------------------------------

def test_matching_returns_index_and_treatment_arrays_of_equal_length(
    causal, confounded_cohort
):
    matched_idx, matched_treatment = causal.propensity_score_matching(
        confounded_cohort, confounded_cohort["treatment"].to_numpy(), ["x1", "x2"]
    )

    assert len(matched_idx) == len(matched_treatment)
    assert len(matched_idx) > 0
    # Matched sets are constructed in pairs, so the result is balanced by design.
    assert matched_treatment.sum() * 2 == len(matched_treatment)


def test_matching_improves_covariate_balance(causal, confounded_cohort):
    """The purpose of matching is balance on x1; check it actually delivers."""
    treatment = confounded_cohort["treatment"].to_numpy()
    x1 = confounded_cohort["x1"].to_numpy()

    unmatched_gap = abs(x1[treatment == 1].mean() - x1[treatment == 0].mean())

    matched_idx, matched_treatment = causal.propensity_score_matching(
        confounded_cohort, treatment, ["x1", "x2"]
    )
    matched_x1 = x1[matched_idx]
    matched_gap = abs(
        matched_x1[matched_treatment == 1].mean() - matched_x1[matched_treatment == 0].mean()
    )

    assert matched_gap < unmatched_gap


def test_matching_raises_when_no_one_is_treated(causal, unconfounded_cohort):
    """A single-class treatment vector cannot fit a propensity model."""
    with pytest.raises(RuntimeError, match="Propensity score matching failed"):
        causal.propensity_score_matching(
            unconfounded_cohort, np.zeros(len(unconfounded_cohort), dtype=int), ["x1", "x2"]
        )


# --------------------------------------------------------------------------
# No-fabrication contract (regression guard for commit 54e4b8f)
# --------------------------------------------------------------------------

def test_ate_raises_rather_than_reporting_an_effect_of_zero(causal, unconfounded_cohort):
    """An ATE of 0.0 is the assertion 'this treatment does nothing'.

    It must never be produced by a failure path.
    """
    with pytest.raises(RuntimeError, match="ATE estimation failed"):
        causal.estimate_ate(
            unconfounded_cohort,
            unconfounded_cohort["treatment"].to_numpy(),
            unconfounded_cohort["outcome"].to_numpy(),
            ["a_covariate_that_does_not_exist"],
        )


def test_estimate_ate_requires_a_treatment_column_in_X(causal, unconfounded_cohort):
    """Undocumented coupling, pinned here so it is not discovered in production.

    The signature takes `treatment` as a separate array, which reads as
    "X holds covariates only". But the regression-adjustment step does
    `X[covariates + ['treatment']]`, so X must ALSO carry a 'treatment'
    column -- whose values are then immediately overwritten by the array.
    A caller who passes covariates only gets a RuntimeError.
    """
    covariates_only = unconfounded_cohort.drop(columns=["treatment"])

    with pytest.raises(RuntimeError, match="ATE estimation failed"):
        causal.estimate_ate(
            covariates_only,
            unconfounded_cohort["treatment"].to_numpy(),
            unconfounded_cohort["outcome"].to_numpy(),
            ["x1", "x2"],
        )


def test_treatment_column_in_X_is_ignored_in_favour_of_the_argument(causal, unconfounded_cohort):
    """Corollary of the above: the column's values do not affect the result."""
    truth = unconfounded_cohort["treatment"].to_numpy()

    scrambled = unconfounded_cohort.copy()
    scrambled["treatment"] = 1 - scrambled["treatment"]

    from_original = causal.estimate_ate(
        unconfounded_cohort, truth, unconfounded_cohort["outcome"].to_numpy(), ["x1", "x2"]
    )
    from_scrambled = CausalInferenceModels().estimate_ate(
        scrambled, truth, unconfounded_cohort["outcome"].to_numpy(), ["x1", "x2"]
    )

    assert from_original["ate_regression"] == pytest.approx(
        from_scrambled["ate_regression"], abs=1e-9
    )
