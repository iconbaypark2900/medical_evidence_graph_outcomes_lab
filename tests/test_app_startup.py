"""Importing the application must not run it.

`import src.api_backend` used to restore a trained risk model from disk,
re-register persisted guidelines (printing as it went), and read the
config four times. Six test files import this module, so every one
inherited whatever happened to be in .model_store/ and .audit/ on the
machine running them -- the same ambient-state dependence that made a
Neo4j test pass locally and fail in CI, but structural rather than one
test's mistake.

It was never a speed problem. The 3.4 second import is torch and
scikit-learn; the file reads are hundredths of a second. What it was is
state nobody asked for.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

import src.api_backend as backend
from src.api_backend import app


def test_importing_the_module_loads_no_persisted_state():
    """Run in a subprocess: this process has already been configured by
    other tests, so asking the current interpreter proves nothing."""
    script = (
        "import sys; sys.path.insert(0, '.');"
        "import src.api_backend as b;"
        "print(bool(b.API_KEYS), b.phi_scanner.enabled,"
        " b.risk_model_registry.version is not None,"
        " len(b.pathway_service.guidelines))"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    keys, phi, models, guidelines = result.stdout.strip().split()
    assert keys == "False"
    assert phi == "False"
    assert models == "False"
    assert guidelines == "0"


def test_importing_the_module_prints_nothing_to_stdout():
    """It used to announce "Representing guideline ... as pathway" during
    test collection."""
    script = (
        "import sys; sys.path.insert(0, '.');"
        "import src.api_backend  # noqa"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=300)

    assert result.stdout.strip() == ""


def test_configure_is_what_loads_state():
    restored = backend.configure()

    assert set(restored) == {"api_keys", "risk_models", "guidelines"}
    assert backend.phi_scanner.enabled is True


def test_configure_is_idempotent():
    first = backend.configure()
    second = backend.configure()

    assert first.keys() == second.keys()
    assert backend.phi_scanner.enabled is True


def test_the_lifespan_configures_the_app():
    """A TestClient used as a context manager runs startup, which is how a
    served app gets its configuration."""
    with TestClient(app) as client:
        body = client.get("/api/health").json()

    assert body["phi_detection"]["enabled"] is True


def test_a_bare_test_client_does_not_run_startup(monkeypatch):
    """The property the tests rely on: an unconfigured app is available to
    anything that wants one."""
    monkeypatch.setattr(backend, "phi_scanner", backend.PHIScanner())

    body = TestClient(app).get("/api/health").json()

    assert body["phi_detection"]["enabled"] is False


def test_the_suite_points_persistent_state_at_a_throwaway_directory():
    """Set in conftest before the application is imported, so no test reads
    the developer's real store."""
    import os

    assert "meg-test-state-" in os.environ["MEG_MODEL_STORE"]
    assert "meg-test-state-" in os.environ["MEG_AUDIT_LOG"]
    assert str(backend.MODEL_STORE).startswith(os.environ["MEG_MODEL_STORE"][:20])
