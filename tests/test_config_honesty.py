"""The config must describe what the code does.

Twice now a configuration key has asserted a capability that did not
exist: `presidio.enabled: true` with no Presidio in the repository, then
an OIDC issuer, an OPA url, a Vault url, an MLflow tracking uri and
Langfuse keys with no code referencing any of them. A config claiming a
control is worse than one admitting its absence, because it answers the
question nobody then asks again.

This file is the guard. Every capability-describing key is either backed
by code or listed under `_not_implemented`, and the tests check the
claims against what actually runs rather than against a list of names.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


CONFIG = json.loads(Path("config/settings.json").read_text())


def source_mentions(term: str) -> bool:
    """Whether any module under src/ references a term, ignoring comments."""
    result = subprocess.run(
        ["grep", "-rin", term, "--include=*.py", "src/"],
        capture_output=True, text=True)
    for line in result.stdout.splitlines():
        code = line.split(":", 2)[-1].strip()
        if code.startswith("#") or code.startswith('"""') or code.startswith("*"):
            continue
        if term.lower() in code.lower():
            return True
    return False


# --------------------------------------------------------------------------
# Removed claims stay removed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("removed", ["presidio", "opa", "vault"])
def test_security_no_longer_claims_what_it_cannot_do(removed):
    assert removed not in CONFIG["security"], (
        f"security.{removed} asserts a capability with no implementation")


def test_observability_no_longer_claims_langfuse():
    """It traces LLM and RAG generation; this system has no generation."""
    assert "langfuse_secret_key" not in CONFIG["observability"]
    assert "langfuse_public_key" not in CONFIG["observability"]
    assert any("Langfuse" in note
               for note in CONFIG["observability"]["_not_implemented"])


def test_the_auth_type_names_the_mechanism_that_runs():
    """It said "oidc_sso" while a shared API key was the actual gate."""
    from src.api_backend import api_key_header

    assert CONFIG["security"]["auth"]["type"] == "api_key"
    assert CONFIG["security"]["auth"]["header"] == api_key_header.model.name


def test_unimplemented_auth_capabilities_are_named_as_such():
    not_implemented = " ".join(CONFIG["security"]["auth"]["_not_implemented"])

    assert "OIDC" in not_implemented
    assert "Open Policy Agent" in not_implemented


def test_the_plaintext_key_list_says_it_is_plaintext():
    """Vault's url previously implied managed secrets while this list sat
    in the same file."""
    assert "plaintext" in CONFIG["security"]["_secrets_note"].lower()
    assert "MEG_API_KEYS" in CONFIG["security"]["_secrets_note"]


# --------------------------------------------------------------------------
# Remaining claims are backed by code
# --------------------------------------------------------------------------

def test_prometheus_is_claimed_and_implemented():
    from src.api_backend import app

    assert CONFIG["observability"]["prometheus"]["enabled"] is True
    endpoint = CONFIG["observability"]["prometheus"]["endpoint"]
    assert any(getattr(route, "path", None) == endpoint for route in app.routes)


def test_mlflow_is_claimed_off_and_is_off():
    from src.api_backend import experiment_tracker

    assert CONFIG["observability"]["mlflow"]["enabled"] is False
    assert experiment_tracker.enabled is False
    assert source_mentions("mlflow"), "the key is kept, so code must read it"


def test_phi_settings_match_the_running_scanner():
    from src.api_backend import phi_scanner

    declared = CONFIG["security"]["phi"]
    assert phi_scanner.enabled == declared["enabled"]
    assert phi_scanner.backend == declared["backend"]
    assert phi_scanner.on_detection == declared["on_detection"]


def test_the_audit_path_is_the_one_actually_written():
    from src.audit import DEFAULT_AUDIT_PATH

    assert CONFIG["observability"]["audit"]["path"] == str(DEFAULT_AUDIT_PATH)


def test_every_configured_service_is_reachable_in_code():
    """A key naming a host or a URI implies something contacts it."""
    services = CONFIG["services"]
    for name in services:
        module = Path("src") / name / "main.py"
        assert module.exists(), f"config declares {name} with no module"


# --------------------------------------------------------------------------
# The general rule
# --------------------------------------------------------------------------

def test_no_block_names_a_url_nothing_contacts():
    """The shape the removed claims all had: a plausible endpoint for a
    service no code ever opened a connection to."""
    suspicious = []
    for section in ("security", "observability"):
        for key, value in CONFIG[section].items():
            if not isinstance(value, dict):
                continue
            for inner_key, inner in value.items():
                if inner_key.startswith("_") or not isinstance(inner, str):
                    continue
                if inner.startswith(("http://", "https://")):
                    # Keeping a URL means something reads that block.
                    if not source_mentions(key):
                        suspicious.append(f"{section}.{key}.{inner_key} = {inner}")

    assert suspicious == [], (
        f"config names endpoints nothing contacts: {suspicious}")
