"""Audit trail.

The README promises "full audit trails of ingested datasets,
transformations, and queries" and there was no audit code at all. It
became both more tractable and more conspicuous once API keys landed:
there is now an identity to attribute an action to.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api_backend as backend
from src.api_backend import app
from src.audit import AuditLog, actor_id


@pytest.fixture
def log(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl")


# --------------------------------------------------------------------------
# Actors
# --------------------------------------------------------------------------

def test_the_actor_is_a_hash_not_the_key():
    """The log must not become a list of live credentials."""
    identifier = actor_id("super-secret-key")

    assert "super-secret-key" not in identifier
    assert identifier.startswith("key:")


def test_the_same_key_always_gives_the_same_actor():
    """Enough to say two actions came from the same caller."""
    assert actor_id("k") == actor_id("k")
    assert actor_id("k") != actor_id("other")


def test_an_unauthenticated_caller_is_recorded_as_anonymous():
    assert actor_id(None) == "anonymous"
    assert actor_id("") == "anonymous"


# --------------------------------------------------------------------------
# What may be recorded
# --------------------------------------------------------------------------

def test_metadata_is_recorded(log):
    event = log.record("risk_model.train", actor="key:abc", n_patients=80,
                       outcomes=["mortality"])

    assert event["action"] == "risk_model.train"
    assert event["n_patients"] == 80
    assert event["timestamp"].endswith("+00:00")


def test_counts_named_like_payloads_are_still_allowed(log):
    """A count called `results` is metadata; the check is about shape."""
    assert log.record("evidence.search", query="metformin", results=3)


@pytest.mark.parametrize("details", [
    {"patients": [{"id": "x"}]},
    {"patient_data": [{"age": 70}]},
    {"anything_at_all": [{"age": 70}]},
    {"payload": {"nested": 1}},
])
def test_structured_records_are_refused(log, details):
    """An audit log that accumulates the data it audits becomes the largest
    copy of that data in the system, in the file least likely to be
    access-controlled."""
    with pytest.raises(ValueError, match="metadata, not the data"):
        log.record("something", **details)


def test_the_refusal_is_by_shape_not_only_by_name(log):
    """The key a payload arrives under is the caller's choice."""
    with pytest.raises(ValueError):
        log.record("x", innocuously_named=[{"mrn": "4029381"}])


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------

def test_events_append_rather_than_replace(log):
    log.record("a")
    log.record("b")

    assert len(log.path.read_text().splitlines()) == 2


def test_each_line_is_valid_json(log):
    log.record("a", n=1)
    log.record("b", n=2)

    for line in log.path.read_text().splitlines():
        json.loads(line)


def test_reading_returns_newest_first(log):
    log.record("first")
    log.record("second")

    assert [e["action"] for e in log.read()] == ["second", "first"]


def test_reading_can_filter_by_action(log):
    log.record("keep")
    log.record("drop")

    assert [e["action"] for e in log.read(action="keep")] == ["keep"]


def test_a_missing_log_reads_as_empty(tmp_path):
    assert AuditLog(tmp_path / "nothing.jsonl").read() == []


def test_one_damaged_line_does_not_hide_the_rest(log):
    log.record("good")
    with log.path.open("a") as handle:
        handle.write("{not json\n")
    log.record("also_good")

    assert [e["action"] for e in log.read()] == ["also_good", "good"]


def test_an_unwritable_log_does_not_take_the_api_down(tmp_path):
    """It must not pass unnoticed either, which is why it logs an error."""
    blocked = tmp_path / "file"
    blocked.write_text("i am a file, not a directory")

    unwritable = AuditLog(blocked / "audit.jsonl")

    assert unwritable.record("still returns") is not None


# --------------------------------------------------------------------------
# Through the API
# --------------------------------------------------------------------------

@pytest.fixture
def audited(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "audit_log", AuditLog(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(backend, "API_KEYS", {"caller-one"})
    return backend.audit_log, TestClient(app)


def test_an_analysis_is_attributed_to_its_caller(audited):
    log, client = audited
    client.post(
        "/api/survival-analysis/kaplan-meier",
        json={"patient_data": [
            {"patient_id": f"p{i}",
             "demographics": {"age": 60, "sex": "male"},
             "clinical_indicators": {"baseline_risk_score": 0.5, "comorbidity_count": 1},
             "follow_up": {"observed_time_days": 10.0 * (i + 1), "event_observed": True}}
            for i in range(3)]},
        headers={"X-API-Key": "caller-one"})

    events = log.read()
    assert events[0]["action"] == "survival.kaplan_meier"
    assert events[0]["actor"] == actor_id("caller-one")
    assert events[0]["n_patients"] == 3


def test_a_rejected_request_records_nothing(audited):
    """No key, no analysis, nothing to attribute."""
    log, client = audited
    client.post("/api/survival-analysis/kaplan-meier",
                json={"patient_data": []})

    assert log.read() == []


def test_the_audit_endpoint_reads_back_what_was_recorded(audited):
    log, client = audited
    log.record("evidence.search", actor=actor_id("caller-one"), query="metformin")

    body = client.get("/api/audit", headers={"X-API-Key": "caller-one"}).json()

    assert body["events"][0]["action"] == "evidence.search"
    assert "not a list of live credentials" in body["note"]


def test_the_audit_endpoint_needs_a_key(audited):
    _, client = audited

    assert client.get("/api/audit").status_code == 401


def test_no_patient_content_can_be_read_back_out(audited):
    """Nothing is recorded, so nothing can be retrieved."""
    log, client = audited
    client.post(
        "/api/patients/risk-assessment",
        json=[{"patient_id": "pt_0",
               "demographics": {"age": 71, "sex": "female", "race": "Asian"},
               "clinical_indicators": {"baseline_risk_score": 0.9, "comorbidity_count": 4}}],
        headers={"X-API-Key": "caller-one"})

    dumped = json.dumps(log.read())
    for value in ("pt_0", "Asian", "0.9", "71"):
        assert value not in dumped, f"{value!r} reached the audit log"
