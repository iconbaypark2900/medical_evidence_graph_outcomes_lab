"""Rendering tests for the Streamlit clinical frontend.

`src/frontend_interface.py` is the largest module in the project and had
7% coverage: the existing tests in test_frontend_coverage.py read the file
as text and assert substrings appear in it, which cannot tell a rendered
page from a syntactically valid one. The banner that warns an operator the
API is unauthenticated sat dead for a while -- it tested for a health state
that had been renamed -- and nothing noticed.

These drive the real app with streamlit.testing.v1.AppTest and assert on
what it renders. The API is faked at the `requests` layer, not by patching
the module's own helpers: Streamlit executes the file as a fresh __main__
module, so an attribute patched on the imported `src.frontend_interface`
is not the one the running app calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "streamlit", reason="the ui extra is not installed; `pip install -e '.[ui]'`")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute, because AppTest.from_file resolves a relative path against the
# file that calls it -- from tests/ the literal "src/frontend_interface.py"
# becomes tests/src/frontend_interface.py and raises FileNotFoundError.
APP = str(Path(__file__).resolve().parent.parent / "src" / "frontend_interface.py")

HEALTHY = {
    "status": "healthy",
    "timestamp": "2026-08-30T12:00:00Z",
    "models_ready": {"survival_analysis": True, "causal_inference": True,
                     "risk_assessment": False},
    "risk_model_version": None,
    "risk_model_trained_at": None,
    "risk_models_persisted": False,
    "guidelines_registered": 0,
    "authentication": "api_key",
    "phi_detection": {"enabled": True, "backend": "patterns"},
    "experiment_tracking": {"enabled": False},
    "metrics_endpoint": "/metrics",
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def health(**overrides):
    return {**HEALTHY, **overrides}


@pytest.fixture
def api(monkeypatch):
    """Fake the API at the requests layer.

    Two entry points, not one: call_api goes through requests.request and
    fetch_health goes through requests.get. Patching only the first leaves
    the dashboard making a real call to localhost:8000.
    """
    import requests

    routes = {"health": health()}
    calls = []

    def route(url):
        for key, value in routes.items():
            if key in url:
                return value
        return None

    def _get(url, *a, **kw):
        calls.append(("GET", url))
        payload = route(url)
        return FakeResponse(payload) if payload is not None else FakeResponse({}, 404)

    def _request(method, url, *a, **kw):
        calls.append((method, url))
        payload = route(url)
        if payload is None:
            return FakeResponse({"detail": "no route"}, 404)
        if isinstance(payload, tuple):
            body, status = payload
            return FakeResponse(body, status)
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", _get)
    monkeypatch.setattr(requests, "request", _request)
    return type("Api", (), {"routes": routes, "calls": calls})()


def run(page, connected=True):
    """Render one page of the real app."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api_connected"] = connected
    at.session_state["api_url"] = "http://testserver"
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def texts(elements):
    return [e.value for e in elements]


# ---------------------------------------------------------------------------
# The page registry
# ---------------------------------------------------------------------------

def test_every_registered_page_renders_without_raising(api):
    """Each of the 13 pages renders, connected, with the API answering."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api_connected"] = True
    at.session_state["api_url"] = "http://testserver"
    at.run()

    pages = at.sidebar.radio[0].options
    assert len(pages) == 13, f"expected 13 pages, got {pages}"

    for page in pages:
        rendered = at.sidebar.radio[0].set_value(page).run()
        assert not rendered.exception, f"{page} raised: {rendered.exception}"


def test_a_page_stops_when_the_api_is_not_connected(api):
    """Twelve of the thirteen pages refuse to render without a connection.

    Asserting the warning alone is not enough, and this test originally did
    exactly that: deleting the st.stop() left the warning in place and the
    page rendered on underneath it, and the test still passed. The absence
    of the page's own widgets is the half that pins the behaviour.
    """
    at = run("Survival Analysis", connected=False)

    assert not at.exception
    assert any("Connect to the API using the sidebar" in w for w in texts(at.warning))
    assert not at.file_uploader, (
        "the page rendered past require_connection: st.stop() did not stop it")


def test_the_dashboard_renders_while_disconnected(api):
    """One page is exempt, deliberately: it is how you see the API is down."""
    at = run("Dashboard", connected=False)

    assert not at.exception
    assert not any("Connect to the API using the sidebar" in w
                   for w in texts(at.warning))


# ---------------------------------------------------------------------------
# The dashboard's honesty banners
#
# These are the reason the frontend is worth testing at all: they are the
# only thing telling an operator the API in front of them is open.
# ---------------------------------------------------------------------------

def test_the_dashboard_warns_when_the_api_serves_anonymously(api):
    api.routes["health"] = health(authentication="anonymous")

    at = run("Dashboard")

    assert any("no authentication configured" in w for w in texts(at.warning)), (
        f"expected the unauthenticated warning, got {texts(at.warning)}")


def test_the_dashboard_reports_an_api_that_is_refusing(api):
    api.routes["health"] = health(authentication="refusing")

    at = run("Dashboard")

    banner = " ".join(texts(at.error))
    assert "refusing every analysis endpoint" in banner
    assert "MEG_API_KEYS" in banner and "MEG_ALLOW_ANONYMOUS" in banner


def test_the_dashboard_says_nothing_when_authentication_is_on(api):
    """The banner has to be absent when it does not apply.

    A warning that is always shown carries no information, and this is the
    half of the assertion that catches a banner wired to the wrong state.
    """
    api.routes["health"] = health(authentication="api_key")

    at = run("Dashboard")

    assert not any("no authentication configured" in w for w in texts(at.warning))
    assert not any("refusing" in e for e in texts(at.error))


def test_the_dashboard_reports_an_unreachable_api(api):
    import requests

    def explode(*a, **kw):
        raise requests.RequestException("connection refused")

    api.routes.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["api_connected"] = True
    at.session_state["api_url"] = "http://testserver"
    at.run()
    at.sidebar.radio[0].set_value("Dashboard").run()

    assert any("not responding" in e for e in texts(at.error))


def test_the_dashboard_distinguishes_no_model_from_a_trained_one(api):
    api.routes["health"] = health(risk_model_version=None)
    at = run("Dashboard")
    assert any("No risk model has been trained" in i for i in texts(at.info))

    api.routes["health"] = health(risk_model_version="abc123",
                                  models_ready={"survival_analysis": True,
                                                "causal_inference": True,
                                                "risk_assessment": True})
    at = run("Dashboard")
    assert any("abc123" in s for s in texts(at.success))


# ---------------------------------------------------------------------------
# Where the evidence came from
#
# "our corpus does not cover this" and "the literature does not cover this"
# look identical once you are reading the list, so the source is rendered
# before the results. It is an st.caption, which does not appear in
# at.markdown -- a test looking there would find nothing and conclude the
# label was missing.
# ---------------------------------------------------------------------------

def _evidence(mode, note=""):
    return {"results": [{"title": "A trial", "source": "pubmed",
                         "citation": "PMID:1", "abstract": "text"}],
            "retrieval_mode": mode, "note": note}


def search(api, mode, note=""):
    """Route the search, render the page, and click Search Evidence."""
    api.routes["evidence/search"] = _evidence(mode, note)
    at = run("Evidence Search")
    at.button[0].click().run()
    return at


def test_evidence_search_names_the_indexed_corpus_as_its_source(api):
    # The search runs on the button, not on page load -- rendering the page
    # and reading captions finds only the global footer.
    at = search(api, "index")

    assert any("Indexed corpus" in c for c in texts(at.caption)), texts(at.caption)


def test_evidence_search_flags_a_fallback_to_live_retrieval(api):
    """The corpus missing something must not look like the literature missing it."""
    at = search(api, "index_then_live",
                note="The indexed corpus had no match for this query.")

    assert any("Live" in c for c in texts(at.caption)), texts(at.caption)
    assert any("indexed corpus had no match" in i for i in texts(at.info))


# ---------------------------------------------------------------------------
# The embedding verdict
# ---------------------------------------------------------------------------

def _graph(served, loses_on=None):
    model = {"model": "distmult", "mrr": 0.553, "hits_at_1": 0.469,
             "hits_at_10": 0.75}
    return {
        "graph": {"source": "neo4j", "nodes": 239, "edges": 460,
                  "by_label": {"Condition": 71}},
        "embeddings": {
            "trained": True, "served": served, "model": model,
            "baselines": [{"model": "frequency", "mrr": 0.510,
                           "hits_at_1": 0.313, "hits_at_10": 0.938}],
            "loses_to_a_baseline_on": loses_on or [],
            "note": "Filtered tail-prediction on held-out triples.",
        },
    }


def test_a_refused_model_is_rendered_as_refused(api):
    """The page must show the verdict, not a score in isolation."""
    api.routes["api/graph"] = _graph(served=False, loses_on=["hits_at_10"])

    at = run("Graph & Embeddings")

    warning = " ".join(texts(at.warning))
    assert "not measurably better" in warning
    assert "not served" in warning
    # Scoped, not bare: the sidebar always emits st.success("Connected to
    # API") when connected, so `not at.success` can never hold.
    assert not any("is being served" in s for s in texts(at.success))


def test_a_served_model_is_rendered_as_served(api):
    api.routes["api/graph"] = _graph(served=True)

    at = run("Graph & Embeddings")

    assert any("is being served" in s for s in texts(at.success))
    assert not any("not served" in w for w in texts(at.warning))


def test_the_metrics_a_model_loses_on_are_shown(api):
    """Reported rather than averaged away -- and in a caption, not markdown."""
    api.routes["api/graph"] = _graph(served=False,
                                     loses_on=["hits_at_3", "hits_at_10"])

    at = run("Graph & Embeddings")

    losses = [c for c in texts(at.caption) if "Loses to a baseline on" in c]
    assert losses, texts(at.caption)
    assert "hits_at_3" in losses[0] and "hits_at_10" in losses[0]


def test_an_untrained_graph_says_so_rather_than_showing_zeros(api):
    api.routes["api/graph"] = {
        "graph": {"source": "empty", "nodes": 0, "edges": 0, "by_label": {}},
        "embeddings": {"trained": False,
                       "note": "No embeddings have been trained."},
    }

    at = run("Graph & Embeddings")

    assert any("No embeddings have been trained" in i for i in texts(at.info))


# ---------------------------------------------------------------------------
# Refusing to score without a model
# ---------------------------------------------------------------------------

def test_risk_assessment_refuses_before_a_model_is_trained(api):
    """503-until-trained is the API's rule; the page has to honour it."""
    api.routes["health"] = health(
        models_ready={"survival_analysis": True, "causal_inference": True,
                      "risk_assessment": False})

    at = run("Patient Risk Assessment")

    assert any("No risk model has been trained" in w for w in texts(at.warning))
    assert not at.number_input, (
        "the patient form rendered anyway: the page did not return")
