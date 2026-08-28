"""Pathway & guideline service.

The one service that was honest from the start: adherence is a genuine
set intersection, recomputed from the observed steps rather than echoed
back from the input, and reported with its denominators. It had no tests,
which made it the only substantive module in src/ with no coverage.

The tests below pin the arithmetic, and the two at the bottom pin the
properties that matter most: that a claimed adherence score cannot
influence the computed one, and that a "perfect adherence" default is
reachable only when there is genuinely nothing to adhere to.
"""
from __future__ import annotations

import pytest

from src.pathway_guideline_service.main import (
    GuidelinePathway,
    ObservedPathway,
    PathwayGuidelineService,
)


GUIDELINE = {
    "id": "dm2_2023",
    "name": "Type 2 Diabetes Management",
    "condition": "type_2_diabetes",
    "version": "2.1",
    "steps": [
        {"name": "HbA1c measurement", "type": "test", "recommended": True},
        {"name": "Metformin initiation", "type": "intervention", "recommended": True},
        {"name": "Retinal screening", "type": "test", "recommended": True},
        {"name": "Sulfonylurea add-on", "type": "intervention", "recommended": False},
    ],
    "decision_points": [
        {"question": "HbA1c above target?", "description": "Escalate if above 7%",
         "options": ["yes", "no"]},
    ],
}


@pytest.fixture
def service() -> PathwayGuidelineService:
    return PathwayGuidelineService()


@pytest.fixture
def pathway(service) -> GuidelinePathway:
    return service.represent_guideline_as_pathway(GUIDELINE)


def observed(*step_names: str, adherence_score: float = 0.0) -> ObservedPathway:
    return ObservedPathway(
        patient_id="pt_1",
        condition="type_2_diabetes",
        steps=[{"name": name} for name in step_names],
        timestamps=["2026-01-01T00:00:00Z"] * len(step_names),
        outcomes=[],
        adherence_score=adherence_score,
    )


# --------------------------------------------------------------------------
# Representing a guideline
# --------------------------------------------------------------------------

def test_every_step_becomes_a_pathway_node(pathway):
    step_nodes = [n for n in pathway.nodes if n["type"] != "decision"]

    assert [n["name"] for n in step_nodes] == [s["name"] for s in GUIDELINE["steps"]]


def test_step_defaults_are_applied(service):
    minimal = {**GUIDELINE, "steps": [{"name": "Bare step"}], "decision_points": []}

    result = service.represent_guideline_as_pathway(minimal)

    node = result.nodes[0]
    assert node["type"] == "intervention"
    assert node["recommended"] is True
    assert node["timing"] == "immediate"
    assert node["evidence_level"] == "unknown"


def test_consecutive_steps_are_chained(pathway):
    follows = [e for e in pathway.edges if e["type"] == "follows"]

    assert len(follows) == len(GUIDELINE["steps"]) - 1
    assert follows[0]["source"] == "step_0"
    assert follows[0]["target"] == "step_1"


def test_decision_points_become_decision_nodes(pathway):
    decisions = [n for n in pathway.nodes if n["type"] == "decision"]

    assert len(decisions) == 1
    assert decisions[0]["name"] == "HbA1c above target?"
    assert decisions[0]["options"] == ["yes", "no"]


def test_the_pathway_is_registered_and_retrievable(service, pathway):
    assert service.guidelines["dm2_2023"] is pathway
    assert "dm2_2023" in service.pathway_graphs


def test_a_networkx_graph_mirrors_the_pathway(service, pathway):
    graph = service.pathway_graphs["dm2_2023"]

    assert graph.number_of_nodes() == len(pathway.nodes)
    assert graph.number_of_edges() == len(pathway.edges)
    assert graph.has_edge("step_0", "step_1")


# --------------------------------------------------------------------------
# Adherence
# --------------------------------------------------------------------------

def test_adherence_is_the_share_of_required_steps_performed(service, pathway):
    """Three steps are recommended. Two were performed, so 2/3."""
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Metformin initiation"), "dm2_2023")

    assert result["adherence_score"] == pytest.approx(2 / 3)
    assert result["n_required"] == 3
    assert result["n_performed"] == 2


def test_full_adherence_scores_one(service, pathway):
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Metformin initiation", "Retinal screening"),
        "dm2_2023")

    assert result["adherence_score"] == pytest.approx(1.0)
    assert result["missing_steps"] == []


def test_no_adherence_scores_zero(service, pathway):
    result = service.compare_observed_to_recommended(observed(), "dm2_2023")

    assert result["adherence_score"] == 0.0
    assert result["n_missing"] == 3


def test_missing_steps_are_named(service, pathway):
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement"), "dm2_2023")

    assert set(result["missing_steps"]) == {"Metformin initiation", "Retinal screening"}


def test_a_non_recommended_step_counts_as_extra_not_required(service, pathway):
    """Sulfonylurea is in the guideline but not recommended, so performing
    it is a variance rather than adherence."""
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Sulfonylurea add-on"), "dm2_2023")

    assert "Sulfonylurea add-on" not in result["required_steps"]
    assert result["extra_steps"] == ["Sulfonylurea add-on"]
    assert result["adherence_score"] == pytest.approx(1 / 3)


def test_a_step_outside_the_guideline_is_ignored(service, pathway):
    """Care the guideline says nothing about is not a guideline variance."""
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Acupuncture"), "dm2_2023")

    assert "Acupuncture" not in result["extra_steps"]
    assert result["n_performed"] == 1


def test_decision_points_are_not_treated_as_care_steps(service, pathway):
    result = service.compare_observed_to_recommended(observed(), "dm2_2023")

    assert "HbA1c above target?" not in result["required_steps"]


def test_the_result_carries_its_denominators(service, pathway):
    """A rate without its denominator cannot be judged: 2/3 and 200/300 are
    the same number and very different evidence."""
    result = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Metformin initiation"), "dm2_2023")

    for key in ("n_required", "n_performed", "n_missing", "n_extra"):
        assert key in result
    assert result["n_required"] == result["n_performed"] + result["n_missing"]


def test_comparing_against_an_unknown_guideline_is_an_error(service):
    with pytest.raises(ValueError, match="not found"):
        service.compare_observed_to_recommended(observed(), "no_such_guideline")


def test_the_claimed_adherence_score_does_not_affect_the_computed_one(service, pathway):
    """The property that makes this service trustworthy.

    ObservedPathway carries an `adherence_score` field. If the comparison
    read it instead of recomputing, a caller could assert their own
    adherence and have it reported back as a finding.
    """
    honest = service.compare_observed_to_recommended(
        observed("HbA1c measurement", adherence_score=0.0), "dm2_2023")
    inflated = service.compare_observed_to_recommended(
        observed("HbA1c measurement", adherence_score=0.99), "dm2_2023")

    assert honest["adherence_score"] == inflated["adherence_score"] == pytest.approx(1 / 3)


def test_perfect_adherence_by_default_needs_an_empty_guideline(service):
    """`adherence_score = 1.0` when there are no required steps is a real
    edge case, not a fallback: it must not be reachable while requirements
    exist and go unmet."""
    empty = {**GUIDELINE, "id": "empty",
             "steps": [{"name": "Optional review", "recommended": False}],
             "decision_points": []}
    service.represent_guideline_as_pathway(empty)

    result = service.compare_observed_to_recommended(observed(), "empty")

    assert result["n_required"] == 0
    assert result["adherence_score"] == 1.0


# --------------------------------------------------------------------------
# Optimization opportunities
# --------------------------------------------------------------------------

def test_missing_steps_raise_an_opportunity(service, pathway):
    comparison = service.compare_observed_to_recommended(
        observed("HbA1c measurement"), "dm2_2023")

    opportunities = service.highlight_optimization_opportunities(comparison)

    missing = next(o for o in opportunities if o["type"] == "missing_recommended")
    assert set(missing["steps"]) == {"Metformin initiation", "Retinal screening"}
    assert missing["priority"] == "medium"  # two missing; >2 would be high


def test_more_than_two_missing_steps_is_high_priority(service):
    big = {**GUIDELINE, "id": "big", "decision_points": [],
           "steps": [{"name": f"Step {i}"} for i in range(5)]}
    service.represent_guideline_as_pathway(big)
    comparison = service.compare_observed_to_recommended(observed(), "big")

    opportunities = service.highlight_optimization_opportunities(comparison)

    assert opportunities[0]["priority"] == "high"


def test_non_recommended_care_raises_an_opportunity(service, pathway):
    comparison = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Metformin initiation",
                 "Retinal screening", "Sulfonylurea add-on"), "dm2_2023")

    opportunities = service.highlight_optimization_opportunities(comparison)

    assert any(o["type"] == "unnecessary_steps" for o in opportunities)


def test_perfect_adherence_raises_no_opportunities(service, pathway):
    comparison = service.compare_observed_to_recommended(
        observed("HbA1c measurement", "Metformin initiation", "Retinal screening"),
        "dm2_2023")

    assert service.highlight_optimization_opportunities(comparison) == []


def test_the_timing_opportunity_is_not_about_timing(service):
    """Characterisation, not endorsement.

    The rule is `abs(n_performed - n_required) > 2`, a difference in
    counts with no time component anywhere in it. The description
    ("variance in care steps performed vs. recommended") is accurate; the
    `type` label is not. Neither ObservedPathway.timestamps nor the
    pathway's `timing` field is consulted.
    """
    big = {**GUIDELINE, "id": "big", "decision_points": [],
           "steps": [{"name": f"Step {i}"} for i in range(5)]}
    service.represent_guideline_as_pathway(big)
    comparison = service.compare_observed_to_recommended(observed(), "big")

    timing = next(
        o for o in service.highlight_optimization_opportunities(comparison)
        if o["type"] == "timing_optimization")

    assert "variance in care steps" in timing["description"]
    assert timing["steps"] == []


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def test_visualising_an_unknown_pathway_says_so(service):
    assert service.generate_pathway_visualization("nope") == "Pathway not found"
