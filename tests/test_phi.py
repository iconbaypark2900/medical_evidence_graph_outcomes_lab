"""PHI detection.

`config/settings.json` declared `presidio.enabled: true` while no Presidio
code existed anywhere in the repository. A config asserting a safety
control that is not implemented is worse than one admitting it is absent:
it answers the question nobody then asks again.
"""
from __future__ import annotations

import pytest

from src.phi import (
    PHIConfigurationError,
    PHIDetected,
    PHIScanner,
    scanner_from_config,
)


@pytest.fixture
def scanner() -> PHIScanner:
    return PHIScanner(enabled=True)


# --------------------------------------------------------------------------
# What it catches
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,kind", [
    ("patient 123-45-6789", "national_id"),
    ("reach me at jane.doe@clinic.example", "email"),
    ("call 555-867-5309", "phone"),
    ("born 1954-03-02", "date_of_birth"),
    ("MRN 4029381", "record_number"),
])
def test_direct_identifiers_are_found(scanner, value, kind):
    findings = scanner.scan_text(value, "patient_id")

    assert [f.kind for f in findings] == [kind]


@pytest.mark.parametrize("value", [
    "pt_001", "White", "metformin 500mg", "Cohort A", "", "HbA1c",
])
def test_ordinary_clinical_text_is_not_flagged(scanner, value):
    """A detector that fires on 'metformin 500mg' would be turned off."""
    assert scanner.scan_text(value, "notes") == []


def test_a_national_id_is_reported_once_not_twice(scanner):
    """It also matches the long-digit-run rule; report the specific one."""
    findings = scanner.scan_text("123-45-6789", "patient_id")

    assert len(findings) == 1
    assert findings[0].kind == "national_id"


def test_the_matched_text_is_never_retained(scanner):
    """Storing it would copy the identifier into logs and error payloads --
    the thing detection exists to prevent."""
    finding = scanner.scan_text("MRN 4029381", "patient_id")[0]

    assert not hasattr(finding, "text")
    assert "4029381" not in str(vars(finding))


def test_the_error_message_names_the_field_not_the_value(scanner):
    with pytest.raises(PHIDetected) as excinfo:
        scanner.enforce({"patient_id": "123-45-6789"})

    message = str(excinfo.value)
    assert "patient_id" in message
    assert "national_id" in message
    assert "123-45-6789" not in message


# --------------------------------------------------------------------------
# Where it looks
# --------------------------------------------------------------------------

def test_it_scans_inside_lists(scanner):
    findings = scanner.scan({"medication_list": ["aspirin", "call 555-867-5309"]})

    assert [f.kind for f in findings] == ["phone"]


def test_it_scans_dictionary_keys(scanner):
    """lab_values keys are free-form; the values are floats."""
    findings = scanner.scan({"labs": {"creatinine": 1.1, "mrn 4029381": 2.0}})

    assert [f.kind for f in findings] == ["record_number"]


def test_a_disabled_scanner_finds_nothing(scanner):
    off = PHIScanner(enabled=False)

    assert off.scan_text("123-45-6789", "patient_id") == []


# --------------------------------------------------------------------------
# Configuration honesty
# --------------------------------------------------------------------------

def test_requesting_presidio_without_presidio_raises():
    """Falling back quietly is exactly how a config comes to claim a
    control it does not have."""
    pytest.importorskip  # noqa: B018 - readability
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        with pytest.raises(PHIConfigurationError, match="not installed"):
            PHIScanner(enabled=True, backend="presidio")
    else:
        pytest.skip("presidio is installed; the failure path cannot be shown")


def test_an_unknown_backend_is_refused():
    with pytest.raises(PHIConfigurationError, match="Unknown PHI backend"):
        PHIScanner(enabled=True, backend="magic")


def test_an_unknown_action_is_refused():
    with pytest.raises(PHIConfigurationError, match="Unknown on_detection"):
        PHIScanner(enabled=True, on_detection="ignore")


def test_a_disabled_scanner_does_not_validate_its_backend():
    """Nothing runs, so nothing needs to be installed."""
    assert PHIScanner(enabled=False, backend="presidio").enabled is False


def test_the_description_states_what_is_actually_done(scanner):
    described = scanner.describe

    assert described["enabled"] is True
    assert described["backend"] == "patterns"
    # The limits are stated, not implied.
    assert "not detected" in described["note"]


def test_a_disabled_scanner_says_so(scanner):
    described = PHIScanner(enabled=False).describe

    assert described["enabled"] is False
    assert "No PHI detection is performed" in described["note"]


def test_config_without_a_phi_block_yields_a_disabled_scanner():
    """Absent means off, and the health payload will say off."""
    assert scanner_from_config({}).enabled is False
    assert scanner_from_config(None).enabled is False


def test_the_shipped_config_is_truthful():
    """The point of the whole file: what the config claims is what runs."""
    import json
    from pathlib import Path

    import src.api_backend as backend

    # configure() is what a running app does at startup; the module-level
    # scanner is deliberately inert until then.
    backend.configure()
    phi_scanner = backend.phi_scanner

    config = json.loads(Path("config/settings.json").read_text())
    assert "presidio" not in config["security"], (
        "the presidio block asserted a control with no implementation")

    declared = config["security"]["phi"]
    assert phi_scanner.enabled == declared["enabled"]
    assert phi_scanner.backend == declared["backend"]


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

def test_redaction_removes_the_identifier(scanner):
    redacted = PHIScanner(enabled=True, on_detection="redact").redact(
        "contact jane@clinic.example about MRN 4029381")

    assert "jane@clinic.example" not in redacted
    assert "4029381" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_mode_does_not_reject(scanner):
    """Configurable, but not the default: silently altering the caller's
    data changes what is analysed without saying so."""
    lenient = PHIScanner(enabled=True, on_detection="redact")

    findings = lenient.enforce({"patient_id": "123-45-6789"})

    assert [f.kind for f in findings] == ["national_id"]
