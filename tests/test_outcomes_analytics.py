"""Outcomes analytics service.

Two things this file is really testing:

1. The service analyses a cohort it is given and refuses to invent one.
   `create_cohort` used to ignore its own criteria and generate 1000
   patients with np.random, so every rate it reported described patients
   that did not exist.
2. Its p-values come from a test. They were previously
   `np.random.uniform(0.001, 0.05)` -- drawn entirely from below the
   significance threshold, so a comparative-effectiveness run reported a
   significant result every single time, whatever the data said.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.outcomes_analytics_service.main import (
    CohortDefinition,
    CohortError,
    OutcomesAnalyticsService,
    synthetic_demo_cohort,
)


@pytest.fixture
def service() -> OutcomesAnalyticsService:
    return OutcomesAnalyticsService()


@pytest.fixture
def definition() -> CohortDefinition:
    return CohortDefinition(
        id="c1",
        name="Test cohort",
        inclusion_criteria={},
        exclusion_criteria={},
        follow_up_period=1825,
        outcome_definition={"primary": "death"},
    )


@pytest.fixture
def patients() -> pd.DataFrame:
    """20 patients with two arms and a real difference in event rate."""
    return pd.DataFrame({
        "patient_id": [f"pt_{i}" for i in range(20)],
        "age": [25, 30, 45, 55, 60, 65, 70, 75, 85, 90] * 2,
        "gender": ["M", "F"] * 10,
        "diagnosis": ["type_2_diabetes"] * 18 + ["type_1_diabetes"] * 2,
        "treatment_group": ["treatment_A"] * 10 + ["usual_care"] * 10,
        "survival_time": [float(100 * (i + 1)) for i in range(20)],
        "event_status": [0] * 8 + [1] * 2 + [1] * 8 + [0] * 2,
    })


# --------------------------------------------------------------------------
# Cohort construction
# --------------------------------------------------------------------------

def test_create_cohort_requires_patient_data(service, definition):
    """Regression guard: this used to generate 1000 synthetic patients."""
    with pytest.raises(CohortError, match="does not generate one"):
        service.create_cohort(definition, pd.DataFrame())


def test_create_cohort_requires_the_outcome_columns(service, definition):
    frame = pd.DataFrame({"patient_id": ["a"], "age": [50]})

    with pytest.raises(CohortError, match="survival_time"):
        service.create_cohort(definition, frame)


def test_inclusion_criteria_are_actually_applied(service, definition, patients):
    definition.inclusion_criteria = {"diagnosis": "type_2_diabetes"}

    cohort = service.create_cohort(definition, patients)

    assert len(cohort) == 18
    assert set(cohort["diagnosis"]) == {"type_2_diabetes"}


def test_a_range_criterion_is_applied_as_a_range(service, definition, patients):
    definition.inclusion_criteria = {"age": (18, 80)}

    cohort = service.create_cohort(definition, patients)

    assert cohort["age"].between(18, 80).all()
    assert len(cohort) == 16  # the four patients aged 85 and 90 are excluded


def test_a_list_criterion_is_applied_as_membership(service, definition, patients):
    definition.inclusion_criteria = {"treatment_group": ["treatment_A"]}

    cohort = service.create_cohort(definition, patients)

    assert set(cohort["treatment_group"]) == {"treatment_A"}


def test_exclusion_criteria_remove_patients(service, definition, patients):
    definition.exclusion_criteria = {"gender": "M"}

    cohort = service.create_cohort(definition, patients)

    assert set(cohort["gender"]) == {"F"}


def test_a_criterion_naming_an_absent_column_is_refused(service, definition, patients):
    """Skipping it silently would widen the cohort past its own definition."""
    definition.inclusion_criteria = {"pregnancy": False}

    with pytest.raises(CohortError, match="'pregnancy' is not a column"):
        service.create_cohort(definition, patients)


def test_a_cohort_that_matches_nobody_is_refused(service, definition, patients):
    definition.inclusion_criteria = {"diagnosis": "not_a_diagnosis"}

    with pytest.raises(CohortError, match="No patients met the criteria"):
        service.create_cohort(definition, patients)


def test_follow_up_horizon_censors_rather_than_dropping(service, definition, patients):
    """A patient event-free at the horizon is censored, not event-free forever."""
    definition.follow_up_period = 1000
    beyond_horizon = patients["survival_time"] > 1000

    cohort = service.create_cohort(definition, patients)

    assert len(cohort) == len(patients)  # nobody is dropped
    assert (cohort.loc[beyond_horizon, "survival_time"] == 1000).all()
    assert (cohort.loc[beyond_horizon, "event_status"] == 0).all()
    # An event observed exactly at the horizon is still an event.
    at_horizon = patients["survival_time"] == 1000
    assert cohort.loc[at_horizon, "event_status"].tolist() == \
           patients.loc[at_horizon, "event_status"].tolist()


def test_extract_population_data_requires_a_created_cohort(service, definition):
    with pytest.raises(CohortError, match="call create_cohort first"):
        service.extract_population_data(definition)


# --------------------------------------------------------------------------
# Survival analysis
# --------------------------------------------------------------------------

def test_survival_analysis_reports_counts_and_intervals(service, definition, patients):
    cohort = service.create_cohort(definition, patients)

    result = service.run_survival_analysis(cohort)

    assert result.n_patients == 20
    assert result.n_events + result.n_censored == 20
    assert len(result.confidence_intervals) == len(result.survival_probabilities)
    for survival, (lower, upper) in zip(
            result.survival_probabilities[1:], result.confidence_intervals[1:]):
        assert lower <= survival <= upper


def test_a_single_survival_curve_carries_no_p_value(service, definition, patients):
    """A p-value needs a comparison. One curve has nothing to be tested
    against, so the field is gone rather than filled in."""
    cohort = service.create_cohort(definition, patients)

    result = service.run_survival_analysis(cohort)

    assert not hasattr(result, "p_value")


def test_survival_analysis_refuses_a_cohort_with_no_events(service, definition, patients):
    frame = patients.assign(event_status=0)
    cohort = service.create_cohort(definition, frame)

    with pytest.raises(CohortError, match="No events observed"):
        service.run_survival_analysis(cohort)


def test_median_survival_is_none_when_never_reached(service, definition, patients):
    frame = patients.assign(event_status=[1] + [0] * 19)
    cohort = service.create_cohort(definition, frame)

    result = service.run_survival_analysis(cohort)

    assert result.median_survival is None


# --------------------------------------------------------------------------
# Comparative effectiveness
# --------------------------------------------------------------------------

def test_comparison_refuses_to_invent_treatment_arms(service, definition, patients):
    """Regression guard. When the column was missing this used to assign
    arms with np.random.choice, making the comparison a comparison of the
    random number generator."""
    cohort = service.create_cohort(definition, patients.drop(columns=["treatment_group"]))

    with pytest.raises(CohortError, match="cannot be inferred"):
        service.run_comparative_effectiveness_analysis(cohort)


def test_comparison_is_deterministic_for_a_fixed_cohort(service, definition, patients):
    """The p-value used to be np.random.uniform(0.001, 0.05), so the same
    cohort produced a different answer on every run."""
    cohort = service.create_cohort(definition, patients)

    first = service.run_comparative_effectiveness_analysis(cohort)
    second = service.run_comparative_effectiveness_analysis(cohort)

    assert first.comparison.p_value == second.comparison.p_value
    assert first.comparison.test_statistic == second.comparison.test_statistic


def test_a_real_difference_between_arms_is_detected(service, definition):
    frame = synthetic_demo_cohort(600, seed=3)
    cohort = service.create_cohort(definition, frame)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert result.comparison.test == "logrank"
    assert result.comparison.p_value < 0.01
    # treatment_A halves the hazard in the generator.
    assert (result.group_outcomes["treatment_A"]["event_rate"]
            < result.group_outcomes["usual_care"]["event_rate"])


def test_identical_arms_are_not_reported_as_different(service, definition):
    """The decisive test. A p-value drawn from U(0.001, 0.05) cannot pass
    this, because it is always below 0.05."""
    gen = np.random.default_rng(21)
    n = 500
    frame = pd.DataFrame({
        "patient_id": [f"pt_{i}" for i in range(2 * n)],
        "age": gen.integers(40, 80, 2 * n),
        "gender": gen.choice(["M", "F"], 2 * n),
        # Both arms drawn from the same distribution.
        "treatment_group": ["arm_a"] * n + ["arm_b"] * n,
        "survival_time": gen.exponential(500, 2 * n),
        "event_status": gen.binomial(1, 0.6, 2 * n),
    })
    cohort = service.create_cohort(definition, frame)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert result.comparison.p_value > 0.05


def test_the_test_that_produced_the_p_value_is_named(service, definition, patients):
    cohort = service.create_cohort(definition, patients)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert result.comparison.test in {"logrank", "multivariate_logrank"}
    assert result.comparison.degrees_of_freedom == 1


def test_number_needed_to_treat_comes_from_the_observed_risk_difference(
    service, definition, patients
):
    cohort = service.create_cohort(definition, patients)

    result = service.run_comparative_effectiveness_analysis(cohort)

    rates = {g: o["event_rate"] for g, o in result.group_outcomes.items()}
    expected = rates["usual_care"] - rates["treatment_A"]
    assert result.risk_difference == pytest.approx(expected)
    assert result.number_needed_to_treat == pytest.approx(1 / abs(expected))


def test_number_needed_to_treat_is_undefined_when_rates_match(service, definition, patients):
    definition.follow_up_period = 10_000  # no administrative censoring here
    frame = patients.assign(event_status=[1, 0] * 10)
    cohort = service.create_cohort(definition, frame)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert result.risk_difference == 0
    assert result.number_needed_to_treat is None
    assert any("undefined" in note for note in result.notes)


def test_three_arms_use_the_multivariate_test_and_omit_nnt(service, definition):
    gen = np.random.default_rng(4)
    n = 300
    frame = pd.DataFrame({
        "patient_id": [f"pt_{i}" for i in range(3 * n)],
        "age": gen.integers(40, 80, 3 * n),
        "gender": gen.choice(["M", "F"], 3 * n),
        "treatment_group": ["a"] * n + ["b"] * n + ["c"] * n,
        "survival_time": np.concatenate([
            gen.exponential(300, n), gen.exponential(600, n), gen.exponential(1200, n)]),
        "event_status": np.ones(3 * n, dtype=int),
    })
    cohort = service.create_cohort(definition, frame)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert result.comparison.test == "multivariate_logrank"
    assert result.comparison.degrees_of_freedom == 2
    assert result.number_needed_to_treat is None
    assert any("two-group comparison only" in note for note in result.notes)


def test_the_comparison_states_its_confounding_caveat(service, definition, patients):
    cohort = service.create_cohort(definition, patients)

    result = service.run_comparative_effectiveness_analysis(cohort)

    assert any("confounding is not addressed" in note for note in result.notes)


# --------------------------------------------------------------------------
# Subgroups
# --------------------------------------------------------------------------

def test_subgroup_analysis_reports_rates_with_their_denominators(
    service, definition, patients
):
    cohort = service.create_cohort(definition, patients)

    results = service.run_subgroup_analysis(cohort, "gender")

    assert set(results) == {"M", "F"}
    for subgroup in results.values():
        assert subgroup["n_events"] <= subgroup["n_patients"]
        assert subgroup["event_rate"] == pytest.approx(
            subgroup["n_events"] / subgroup["n_patients"])


def test_subgroup_analysis_carries_no_p_value(service, definition, patients):
    """Subgroup analyses are where multiple comparisons do the most damage;
    an unadjusted p per subgroup invites a reading it cannot support."""
    cohort = service.create_cohort(definition, patients)

    results = service.run_subgroup_analysis(cohort, "gender")

    assert all("p_value" not in subgroup for subgroup in results.values())


def test_subgroup_analysis_refuses_an_absent_column(service, definition, patients):
    cohort = service.create_cohort(definition, patients)

    with pytest.raises(CohortError, match="'smoking' is not in the data"):
        service.run_subgroup_analysis(cohort, "smoking")


# --------------------------------------------------------------------------
# Metrics payload
# --------------------------------------------------------------------------

def test_metrics_payload_carries_the_test_and_its_denominators(
    service, definition, patients
):
    cohort = service.create_cohort(definition, patients)
    survival = service.run_survival_analysis(cohort)
    comparative = service.run_comparative_effectiveness_analysis(cohort)

    metrics = service.store_and_return_metrics(survival, comparative, definition)

    comparison = metrics["comparative_effectiveness"]
    assert comparison["test"] == "logrank"
    assert comparison["degrees_of_freedom"] == 1
    assert 0.0 <= comparison["p_value"] <= 1.0
    assert metrics["survival_analysis"]["n_events"] + \
           metrics["survival_analysis"]["n_censored"] == 20
