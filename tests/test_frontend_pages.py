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
