"""
Main entry point for the Outcomes Analytics Service.

This service analyses a cohort you give it. It does not manufacture one,
and it does not manufacture the statistics it reports about one.

Both of those used to happen. `create_cohort` ignored its own inclusion
and exclusion criteria and generated 1000 patients with `np.random`;
`run_survival_analysis` attached `np.random.uniform(0.001, 0.1)` to the
result as a p-value; and `run_comparative_effectiveness_analysis` invented
treatment arms with `np.random.choice` when the column was missing and
reported `np.random.uniform(0.001, 0.05)` as its p-value — drawn entirely
from below the significance threshold, so every comparative-effectiveness
run reported a significant result, whatever the data said.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime

from src.enhanced_ml_models import (
    SurvivalAnalysisModels,
    SurvivalComparison,
)


REQUIRED_COLUMNS = ("patient_id", "survival_time", "event_status")


@dataclass
class CohortDefinition:
    """Defines a patient cohort for analysis.

    `inclusion_criteria` and `exclusion_criteria` are applied for real by
    `create_cohort`. Each is a mapping of column name to either a scalar
    (exact match), a list (membership), or a (min, max) tuple for ranges.
    """
    id: str
    name: str
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    follow_up_period: int  # in days
    outcome_definition: Dict[str, Any]


@dataclass
class SurvivalResult:
    """Kaplan-Meier estimate for a single cohort.

    There is deliberately no p-value here. A single survival curve has
    nothing to be tested against; a p-value requires a comparison, which
    is what `run_comparative_effectiveness_analysis` produces.
    """
    time_points: List[float]
    survival_probabilities: List[float]
    confidence_intervals: List[Tuple[float, float]]
    n_patients: int
    n_events: int
    n_censored: int
    median_survival: Optional[float]


@dataclass
class ComparativeEffectivenessResult:
    """Between-group comparison, with the test that produced the p-value."""
    group_outcomes: Dict[str, Dict[str, float]]
    comparison: SurvivalComparison
    risk_difference: Optional[float] = None
    number_needed_to_treat: Optional[float] = None
    reference_group: Optional[str] = None
    comparison_group: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class CohortError(ValueError):
    """The supplied data cannot support the requested analysis."""


def _matches(series: pd.Series, criterion: Any) -> pd.Series:
    """Apply one criterion to one column."""
    if isinstance(criterion, tuple) and len(criterion) == 2:
        low, high = criterion
        return series.between(low, high)
    if isinstance(criterion, (list, set)):
        return series.isin(list(criterion))
    return series == criterion


class OutcomesAnalyticsService:
    def __init__(self):
        self.cohorts: Dict[str, CohortDefinition] = {}
        self.population_data = pd.DataFrame()
        self.survival_models = SurvivalAnalysisModels()

    # -----------------------------------------------------------------
    # Cohort construction
    # -----------------------------------------------------------------

    def create_cohort(self, definition: CohortDefinition, patient_data: pd.DataFrame):
        """Build a cohort by applying the definition's criteria to real data.

        `patient_data` is required. Generating a cohort here would mean the
        service reports outcomes for patients that do not exist, which is
        what it previously did whenever it was called.
        """
        if patient_data is None or patient_data.empty:
            raise CohortError(
                f"Cohort {definition.id!r} needs patient data. This service "
                f"analyses a cohort you supply; it does not generate one.")

        missing = [c for c in REQUIRED_COLUMNS if c not in patient_data.columns]
        if missing:
            raise CohortError(
                f"Patient data is missing {missing}. Required: "
                f"{list(REQUIRED_COLUMNS)}; supplied: {list(patient_data.columns)}")

        selected = patient_data.copy()
        applied = []

        for column, criterion in (definition.inclusion_criteria or {}).items():
            if column not in selected.columns:
                # Silently skipping an unmet criterion would produce a cohort
                # broader than the one that was defined, and every rate
                # computed from it would be wrong without any sign of it.
                raise CohortError(
                    f"Inclusion criterion {column!r} is not a column in the "
                    f"supplied data, so it cannot be applied. Columns: "
                    f"{list(selected.columns)}")
            selected = selected[_matches(selected[column], criterion)]
            applied.append(f"include {column}={criterion!r}")

        for column, criterion in (definition.exclusion_criteria or {}).items():
            if column not in selected.columns:
                raise CohortError(
                    f"Exclusion criterion {column!r} is not a column in the "
                    f"supplied data, so it cannot be applied. Columns: "
                    f"{list(selected.columns)}")
            selected = selected[~_matches(selected[column], criterion)]
            applied.append(f"exclude {column}={criterion!r}")

        if selected.empty:
            raise CohortError(
                f"No patients met the criteria for cohort {definition.id!r} "
                f"({'; '.join(applied) or 'no criteria'}). "
                f"{len(patient_data)} patients were supplied.")

        # Administrative censoring at the end of the follow-up window: a
        # patient still event-free then is censored, not event-free forever.
        selected = selected.copy()
        beyond_window = selected['survival_time'] > definition.follow_up_period
        selected.loc[beyond_window, 'event_status'] = 0
        selected.loc[beyond_window, 'survival_time'] = definition.follow_up_period
        selected['cohort_id'] = definition.id

        self.population_data = selected
        self.cohorts[definition.id] = definition

        print(
            f"Cohort {definition.name!r}: {len(selected)} of "
            f"{len(patient_data)} patients met the criteria "
            f"({int(selected['event_status'].sum())} events, "
            f"{int(beyond_window.sum())} censored at the follow-up horizon)")
        return selected

    def load_cohort_definition(self, cohort_id: str) -> CohortDefinition:
        if cohort_id not in self.cohorts:
            raise CohortError(
                f"Cohort {cohort_id!r} not found. Known: {sorted(self.cohorts)}")
        return self.cohorts[cohort_id]

    def extract_population_data(self, cohort_definition: CohortDefinition) -> pd.DataFrame:
        if 'cohort_id' not in self.population_data.columns:
            raise CohortError(
                f"No stored data for cohort {cohort_definition.id!r}; call "
                f"create_cohort first.")
        cohort_data = self.population_data[
            self.population_data['cohort_id'] == cohort_definition.id].copy()
        if cohort_data.empty:
            raise CohortError(
                f"No stored data for cohort {cohort_definition.id!r}; call "
                f"create_cohort first.")
        return cohort_data

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    def run_survival_analysis(self, population_data: pd.DataFrame) -> SurvivalResult:
        """Kaplan-Meier estimate with Greenwood confidence intervals.

        Replaces a hand-rolled product-limit loop whose intervals used the
        binomial standard error sqrt(S(1-S)/n), which ignores censoring and
        so understates uncertainty in the tail of the curve.
        """
        missing = [c for c in ('survival_time', 'event_status') if c not in population_data.columns]
        if missing:
            raise CohortError(f"Survival analysis needs columns {missing}")

        duration = population_data['survival_time'].to_numpy()
        event = population_data['event_status'].to_numpy()

        if event.sum() == 0:
            raise CohortError(
                f"No events observed among {len(event)} patients; the survival "
                f"curve would be flat at 1.0 and carries no information")

        times, survival, lower, upper = (
            self.survival_models.kaplan_meier_with_confidence_intervals(duration, event))

        # Tolerance, not a bare <= 0.5. The product-limit estimate is a
        # running product of floats, so a curve that reaches exactly one
        # half lands on 0.5000000000000001 as often as on 0.5 --
        # (5/6)(4/5)(3/4) evaluates to the former -- and a bare comparison
        # then reports the next event time as the median.
        median = next(
            (float(t) for t, s in zip(times, survival) if s <= 0.5 + 1e-9), None)

        print(f"Survival analysis: {len(times)} time points, "
              f"median survival "
              f"{f'{median:.1f}' if median is not None else 'not reached'}")

        return SurvivalResult(
            time_points=times,
            survival_probabilities=survival,
            confidence_intervals=list(zip(lower, upper)),
            n_patients=len(duration),
            n_events=int(event.sum()),
            n_censored=int(len(event) - event.sum()),
            median_survival=median,
        )

    def run_comparative_effectiveness_analysis(
        self,
        population_data: pd.DataFrame,
        group_col: str = 'treatment_group',
    ) -> ComparativeEffectivenessResult:
        """Compare outcomes between treatment groups with a log-rank test.

        `group_col` must already exist. Assigning patients to arms here —
        which this method used to do with np.random.choice when the column
        was absent — turns the comparison into a comparison of the RNG.
        """
        if group_col not in population_data.columns:
            raise CohortError(
                f"Column {group_col!r} is not in the data, so there are no "
                f"groups to compare. Supply the treatment assignment; it "
                f"cannot be inferred. Columns: {list(population_data.columns)}")

        groups = population_data[group_col].to_numpy()
        duration = population_data['survival_time'].to_numpy()
        event = population_data['event_status'].to_numpy()

        comparison = self.survival_models.compare_survival_curves(duration, event, groups)

        group_outcomes = {}
        for group in sorted(set(map(str, groups))):
            mask = groups.astype(str) == group
            group_outcomes[group] = {
                'event_rate': float(event[mask].mean()),
                'median_survival': float(np.median(duration[mask])),
                'n_patients': int(mask.sum()),
                'n_events': int(event[mask].sum()),
            }

        notes = []
        risk_difference = None
        nnt = None
        reference = comparison_group = None

        names = sorted(group_outcomes)
        if len(names) == 2:
            reference, comparison_group = names
            risk_difference = (
                group_outcomes[comparison_group]['event_rate']
                - group_outcomes[reference]['event_rate'])
            if risk_difference != 0:
                # NNT is 1 / |absolute risk difference|, kept as a float:
                # rounding 1/0.007 down to 142 states more precision than the
                # estimate supports, and int() truncates rather than rounds.
                nnt = float(1.0 / abs(risk_difference))
            else:
                notes.append(
                    "Event rates are identical between groups, so the number "
                    "needed to treat is undefined")
        else:
            notes.append(
                f"{len(names)} groups present; risk difference and NNT are "
                f"defined for a two-group comparison only")

        notes.append(
            "Unadjusted comparison. The log-rank test assumes the groups "
            "differ only by arm; with observational data, confounding is not "
            "addressed here.")

        print(f"Comparative effectiveness across {len(names)} groups: "
              f"{comparison.test} p={comparison.p_value:.4g}")

        return ComparativeEffectivenessResult(
            group_outcomes=group_outcomes,
            comparison=comparison,
            risk_difference=risk_difference,
            number_needed_to_treat=nnt,
            reference_group=reference,
            comparison_group=comparison_group,
            notes=notes,
        )

    def run_subgroup_analysis(
        self, population_data: pd.DataFrame, subgroup_col: str = 'gender'
    ) -> Dict[str, Dict[str, float]]:
        """Event rates by subgroup.

        Descriptive only, and deliberately carries no p-value: subgroup
        analyses are where multiple comparisons do the most damage, and an
        unadjusted p per subgroup invites exactly the reading it cannot
        support.
        """
        if subgroup_col not in population_data.columns:
            raise CohortError(
                f"Column {subgroup_col!r} is not in the data. Columns: "
                f"{list(population_data.columns)}")

        results = {}
        for subgroup in sorted(set(map(str, population_data[subgroup_col]))):
            subgroup_data = population_data[
                population_data[subgroup_col].astype(str) == subgroup]
            results[subgroup] = {
                'event_rate': float(subgroup_data['event_status'].mean()),
                'n_patients': int(len(subgroup_data)),
                'n_events': int(subgroup_data['event_status'].sum()),
            }
        return results

    def store_and_return_metrics(
        self,
        survival_result: SurvivalResult,
        comparative_results: ComparativeEffectivenessResult,
        cohort_definition: CohortDefinition,
    ) -> Dict[str, Any]:
        """Format results for storage or transport."""
        return {
            "cohort_id": cohort_definition.id,
            "cohort_name": cohort_definition.name,
            "total_patients": survival_result.n_patients,
            "survival_analysis": {
                "time_points": survival_result.time_points,
                "survival_probabilities": survival_result.survival_probabilities,
                "confidence_intervals": survival_result.confidence_intervals,
                "n_time_points": len(survival_result.time_points),
                "n_events": survival_result.n_events,
                "n_censored": survival_result.n_censored,
                "median_survival": survival_result.median_survival,
            },
            "comparative_effectiveness": {
                "group_outcomes": comparative_results.group_outcomes,
                "test": comparative_results.comparison.test,
                "test_statistic": comparative_results.comparison.test_statistic,
                "p_value": comparative_results.comparison.p_value,
                "degrees_of_freedom": comparative_results.comparison.degrees_of_freedom,
                "risk_difference": comparative_results.risk_difference,
                "number_needed_to_treat": comparative_results.number_needed_to_treat,
                "notes": comparative_results.notes,
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }


def synthetic_demo_cohort(n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic patients for the demo in `main` below.

    This lives outside the service on purpose. Generating a cohort is a
    legitimate thing for a demo to do and an illegitimate thing for an
    analytics service to do, and keeping the two in the same function is
    how the distinction got lost.
    """
    gen = np.random.default_rng(seed)

    ages = np.clip(gen.normal(55, 15, n_patients), 18, 90)
    genders = gen.choice(['M', 'F'], n_patients, p=[0.45, 0.55])
    arms = gen.choice(['treatment_A', 'usual_care'], n_patients, p=[0.5, 0.5])

    # A real effect to recover: treatment_A roughly halves the hazard.
    log_hazard = (
        (ages - 55) / 40
        + np.where(genders == 'M', 0.10, -0.10)
        + np.where(arms == 'treatment_A', -0.70, 0.0)
    )
    hazards = 0.0006 * np.exp(log_hazard)
    survival_times = gen.exponential(1 / hazards)
    dropout = gen.uniform(200, 3000, n_patients)

    observed = np.minimum(survival_times, dropout)
    events = (survival_times <= dropout).astype(int)

    return pd.DataFrame({
        'patient_id': [f"pt_{i}" for i in range(n_patients)],
        'age': ages,
        'gender': genders,
        'diagnosis': 'type_2_diabetes',
        'treatment_group': arms,
        'survival_time': observed,
        'event_status': events,
    })


async def main():
    """Run the service over a synthetic cohort supplied from outside."""
    print("Starting Outcomes Analytics Service...")

    service = OutcomesAnalyticsService()

    cohort_def = CohortDefinition(
        id="diabetes_cohort_1",
        name="Type 2 Diabetes Patients",
        inclusion_criteria={"diagnosis": "type_2_diabetes", "age": (18, 80)},
        exclusion_criteria={},
        follow_up_period=1825,  # 5 years in days
        outcome_definition={
            "primary": "cardiovascular_event",
            "secondary": ["all_cause_mortality", "diabetes_complications"],
        },
    )

    service.create_cohort(cohort_def, synthetic_demo_cohort())

    loaded_def = service.load_cohort_definition("diabetes_cohort_1")
    pop_data = service.extract_population_data(loaded_def)

    survival_result = service.run_survival_analysis(pop_data)
    comp_effect_result = service.run_comparative_effectiveness_analysis(pop_data)
    subgroup_result = service.run_subgroup_analysis(pop_data)
    metrics = service.store_and_return_metrics(
        survival_result, comp_effect_result, loaded_def)

    survival = metrics['survival_analysis']
    comparative = metrics['comparative_effectiveness']

    print(f"\n{'='*60}")
    print("OUTCOMES ANALYTICS SUMMARY")
    print(f"{'='*60}")
    print(f"Cohort: {metrics['cohort_name']}")
    print(f"Patients: {metrics['total_patients']} "
          f"({survival['n_events']} events, {survival['n_censored']} censored)")
    median = survival['median_survival']
    print(f"Median survival: "
          f"{f'{median:.0f} days' if median is not None else 'not reached'}")
    print(f"\nComparative effectiveness ({comparative['test']}, "
          f"{comparative['degrees_of_freedom']} df):")
    print(f"  test statistic: {comparative['test_statistic']:.2f}")
    print(f"  p-value: {comparative['p_value']:.4g}")
    if comparative['risk_difference'] is not None:
        print(f"  risk difference: {comparative['risk_difference']:+.3f}")
    if comparative['number_needed_to_treat'] is not None:
        print(f"  number needed to treat: "
              f"{comparative['number_needed_to_treat']:.1f}")

    print(f"\nGroup outcomes:")
    for group, outcome in comparative['group_outcomes'].items():
        print(f"  {group}: event rate {outcome['event_rate']:.3f} "
              f"({outcome['n_events']}/{outcome['n_patients']}), "
              f"median follow-up {outcome['median_survival']:.0f} days")

    print(f"\nSubgroup analysis (descriptive only):")
    for sub, result in subgroup_result.items():
        print(f"  {sub}: event rate {result['event_rate']:.3f} "
              f"({result['n_events']}/{result['n_patients']})")

    for note in comparative['notes']:
        print(f"\nNote: {note}")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
