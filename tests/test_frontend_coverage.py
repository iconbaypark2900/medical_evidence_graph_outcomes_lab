"""Every API endpoint has a way to reach it.

This file exists because the same defect happened twice. Six working
services were unreachable from the API; that was fixed by adding six
endpoints; and those six endpoints were then unreachable from the
frontend. A capability nobody can invoke is not delivered, at whichever
layer the gap sits.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.api_backend import app


FRONTEND = Path("src/frontend_interface.py").read_text()

# Reachable without a UI by design: probes and scrape targets. /metrics is
# consumed by Prometheus, not browsed by a person; the health aliases and
# the generated docs are the same shape of exemption. Anything else that
# ends up here should be argued for, not added quietly.
INFRASTRUCTURE_ROUTES = {"/", "/health", "/api/health", "/metrics",
                         "/openapi.json", "/docs", "/docs/oauth2-redirect",
                         "/redoc"}


def api_routes() -> set:
    return {
        route.path for route in app.routes
        if getattr(route, "methods", None) and route.path not in INFRASTRUCTURE_ROUTES
    }


def route_is_called(path: str) -> bool:
    """Whether the frontend calls this path, allowing for f-string params."""
    if "{" in path:
        # e.g. /api/pathways/guidelines/{id}/evidence -> match the literal parts
        head, _, tail = path.partition("{")
        tail = tail.split("}", 1)[1]
        return head in FRONTEND and tail in FRONTEND
    return path in FRONTEND


def test_every_api_endpoint_is_reachable_from_the_frontend():
    unreachable = sorted(p for p in api_routes() if not route_is_called(p))

    assert unreachable == [], (
        f"endpoints with no way to invoke them from the UI: {unreachable}")


def test_the_evidence_join_has_an_interface():
    """The endpoint the project is named for. It shipped with no UI."""
    assert "/api/pathways/guidelines/" in FRONTEND
    assert "/evidence" in FRONTEND
    assert "Evidence for a Guideline" in FRONTEND


def frontend_module():
    """Import the frontend, which needs the optional `ui` extra.

    Most of this file reads the source text and needs no import; only the
    two checks below actually load the module.
    """
    pytest.importorskip("streamlit", reason="the ui extra is not installed")
    pytest.importorskip("plotly", reason="the ui extra is not installed")
    import src.frontend_interface as frontend

    return frontend


@pytest.mark.parametrize("page", [
    "Cohort Builder",
    "Treatment Effect",
    "Audit Trail",
    "Time-to-Event Comparison",
    "Compare Outcome Rates",
    "Guidelines & Adherence",
    "Evidence for a Guideline",
])
def test_the_new_pages_are_registered(page):
    assert page in frontend_module().PAGES


def test_the_outcomes_service_is_reachable_from_the_ui():
    """Cohort criteria and the log-rank test were only in the API."""
    assert "/api/outcomes/cohort" in FRONTEND
    assert "/api/outcomes/comparative-effectiveness" in FRONTEND


def test_every_registered_page_has_a_callable():
    for name, handler in frontend_module().PAGES.items():
        assert callable(handler), name


def test_the_p_value_is_never_shown_without_its_denominators():
    """A p-value alone cannot be judged: the same value can come from a
    large effect in a small cohort or a trivial one in a huge one."""
    page = FRONTEND[FRONTEND.index("def page_comparative_effectiveness"):
                    FRONTEND.index("def page_guidelines")]

    assert "p_value" in page
    assert "group_sizes" in page
    assert "group_events" in page


def test_adherence_is_shown_with_its_denominator():
    page = FRONTEND[FRONTEND.index("def page_guidelines"):
                    FRONTEND.index("def page_guideline_evidence")]

    assert "n_required" in page
    assert "n_performed" in page


def test_the_corpus_coverage_caveat_reaches_the_reader():
    """"This corpus does not cover it" must not read as "the literature
    does not support it"; the API says so and the UI has to show it."""
    page = FRONTEND[FRONTEND.index("def page_guideline_evidence"):]

    assert 'result["note"]' in page
    assert "No supporting record" in page


# --------------------------------------------------------------------------
# Telling the two comparison pages apart
# --------------------------------------------------------------------------

def test_the_two_comparison_pages_are_named_for_what_they_do():
    """"Cohort Analysis" and "Comparative Effectiveness" sat side by side
    doing different analyses, and the names did not distinguish them."""
    frontend = frontend_module()

    assert "Time-to-Event Comparison" in frontend.PAGES
    assert "Compare Outcome Rates" in frontend.PAGES
    assert "Cohort Analysis" not in frontend.PAGES
    assert "Comparative Effectiveness" not in frontend.PAGES


def test_each_comparison_page_points_at_the_other():
    """A reader on the wrong one should be told where the right one is."""
    rates = FRONTEND[FRONTEND.index("def page_cohort_analysis"):]
    rates = rates[:rates.index("def ", 10)]
    survival = FRONTEND[FRONTEND.index("def page_comparative_effectiveness"):]
    survival = survival[:survival.index("def ", 10)]

    assert "Time-to-Event Comparison" in rates
    assert "Compare Outcome Rates" in survival


def test_every_page_has_a_caption():
    frontend = frontend_module()

    missing = [name for name in frontend.PAGES if not frontend.PAGE_CAPTIONS.get(name)]
    assert missing == [], f"pages with no sidebar caption: {missing}"
