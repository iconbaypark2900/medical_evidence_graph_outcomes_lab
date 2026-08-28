"""System launcher.

Two defects, and the second only became dangerous once API-key
authentication landed with keys disabled by default:

- `streamlit run "src.frontend_interface:main"` is rejected outright --
  Streamlit takes a file path -- so the frontend could never start. The
  error handler reported that as "normal during initial load" and returned
  the process as a success.
- Both services bound to 0.0.0.0 unconditionally, so the one-command
  launcher published an unauthenticated patient-analysis API to every
  interface on the machine.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import start_system


# --------------------------------------------------------------------------
# Refusing an unsafe bind
# --------------------------------------------------------------------------

def test_a_public_bind_without_keys_is_refused(monkeypatch, capsys):
    monkeypatch.setattr("src.api_backend.load_api_keys", lambda *a, **k: set())

    assert start_system.main(["--host", "0.0.0.0"]) == 1
    assert "Refusing to bind" in capsys.readouterr().out


def test_a_public_bind_is_refused_before_anything_starts(monkeypatch):
    """The refusal must come before the subprocesses, or by the time anyone
    reads the message the API is already listening."""
    monkeypatch.setattr("src.api_backend.load_api_keys", lambda *a, **k: set())
    monkeypatch.setattr(start_system, "start_api_server",
                        lambda *a: pytest.fail("API started despite the refusal"))

    assert start_system.main(["--host", "1.2.3.4"]) == 1


def test_loopback_needs_no_keys(monkeypatch):
    """Requiring keys to develop locally would just get them hard-coded."""
    monkeypatch.setattr("src.api_backend.load_api_keys", lambda *a, **k: set())
    monkeypatch.setattr(start_system, "port_is_free", lambda *a, **k: True)
    monkeypatch.setattr(start_system, "start_api_server", lambda *a: None)

    # 1 because the stubbed API failed to start, not because of the bind.
    assert start_system.main([]) == 1


def test_the_default_host_is_loopback():
    assert start_system.LOOPBACK == "127.0.0.1"


# --------------------------------------------------------------------------
# The Streamlit target
# --------------------------------------------------------------------------

class FakeProcess:
    def poll(self):
        return None

    def terminate(self):
        pass


def test_the_frontend_is_launched_by_file_path(monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start_system, "wait_until_healthy", lambda *a, **k: True)

    start_system.start_frontend("127.0.0.1", 8501, "http://127.0.0.1:8000")

    assert "src/frontend_interface.py" in captured["command"]
    assert not any(part.endswith(":main") for part in captured["command"])


def test_streamlit_rejects_a_module_target():
    """Pins the actual reason the previous invocation could not work.

    Needs Streamlit itself, which is the optional `ui` extra -- the point
    is what Streamlit does with the argument, so a stub would prove
    nothing.
    """
    pytest.importorskip("streamlit", reason="the ui extra is not installed")

    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src.frontend_interface:main"],
        capture_output=True, text=True, timeout=180)

    assert result.returncode != 0
    assert "raw Python" in (result.stderr + result.stdout)


def test_the_api_url_is_passed_to_the_frontend(monkeypatch):
    """So the frontend follows the port the API was actually started on."""
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start_system, "wait_until_healthy", lambda *a, **k: True)

    start_system.start_frontend("127.0.0.1", 8501, "http://127.0.0.1:9999")

    assert captured["env"]["MEG_API_URL"] == "http://127.0.0.1:9999"


def test_the_api_output_is_not_piped_into_a_buffer_nobody_reads(monkeypatch):
    """A piped stream nobody drains fills and blocks the child, which for
    uvicorn means it stops serving with no indication why."""
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start_system, "wait_until_healthy", lambda *a, **k: True)

    start_system.start_api_server("127.0.0.1", 8000, reload=False)

    assert "stdout" not in captured
    assert "stderr" not in captured


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------

def test_readiness_gives_up_when_the_process_dies():
    class DeadProcess:
        def poll(self):
            return 1

    assert start_system.wait_until_healthy(
        "http://127.0.0.1:1/health", DeadProcess(), timeout=30) is False


def test_readiness_polls_rather_than_checking_once(monkeypatch):
    """The API imports torch and scikit-learn, which takes longer than the
    three seconds the previous version waited before its single attempt."""
    attempts = {"n": 0}

    class LiveProcess:
        def poll(self):
            return None

    def flaky_get(url, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise start_system.requests.RequestException("not up yet")
        return type("Response", (), {"status_code": 200})()

    monkeypatch.setattr(start_system.requests, "get", flaky_get)

    assert start_system.wait_until_healthy("http://x/health", LiveProcess(), timeout=30)
    assert attempts["n"] == 3


def test_the_launcher_is_a_python_script():
    """It began with a `#!/bin/bash` shebang while containing Python."""
    first_line = Path("start_system.py").read_text().splitlines()[0]

    assert "python" in first_line
    assert "bash" not in first_line
