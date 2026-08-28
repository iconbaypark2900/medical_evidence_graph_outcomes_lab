"""Cohort CSV -> API payload conversion.

The rule these tests enforce: a malformed cohort file raises with a
message naming the row and the column. It is never silently coerced or
dropped, because a dropped row changes the denominator of every rate
computed from the cohort without saying so.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.cohort_io import (
    CohortValidationError,
    describe_api_error,
    frame_to_cohort_members,
    frame_to_patient_records,
    frame_to_patients,
    frame_to_treatment_records,
    outcome_candidate_columns,
)


@pytest.fixture
def cohort_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": ["a", "b", "c"],
        "age": [70, 55, 81],
        "sex": ["M", "female", "f"],
        "baseline_risk_score": [0.8, 0.2, 0.6],
        "comorbidity_count": [3, 0, 5],
        "observed_time_days": [120.0, 400.0, 30.0],
        "event_observed": [1, 0, "true"],
        "mortality": [1, 0, 1],
        "readmission": [0, 0, 1],
    })


# --------------------------------------------------------------------------
# Patients
# --------------------------------------------------------------------------

def test_frame_to_patients_maps_every_row(cohort_frame):
    patients = frame_to_patients(cohort_frame)

    assert len(patients) == 3
    assert [p["patient_id"] for p in patients] == ["a", "b", "c"]
    assert patients[0]["demographics"] == {"age": 70, "sex": "male"}
    assert patients[0]["clinical_indicators"]["baseline_risk_score"] == 0.8
    assert patients[0]["clinical_indicators"]["comorbidity_count"] == 3


def test_sex_accepts_the_common_spellings(cohort_frame):
    patients = frame_to_patients(cohort_frame)

    assert [p["demographics"]["sex"] for p in patients] == ["male", "female", "female"]


def test_patient_ids_are_generated_when_absent(cohort_frame):
    patients = frame_to_patients(cohort_frame.drop(columns=["patient_id"]))

    assert [p["patient_id"] for p in patients] == ["row_1", "row_2", "row_3"]


def test_lab_columns_become_lab_values(cohort_frame):
    frame = cohort_frame.assign(lab_creatinine=[1.1, 0.9, 2.3], lab_hba1c=[7.2, 5.4, 8.8])

    patients = frame_to_patients(frame)

    assert patients[0]["clinical_indicators"]["lab_values"] == {
        "creatinine": 1.1, "hba1c": 7.2
    }


def test_severity_score_is_passed_through_when_present(cohort_frame):
    frame = cohort_frame.assign(severity_score=[4.0, 1.0, 9.0])

    patients = frame_to_patients(frame)

    assert patients[2]["clinical_indicators"]["severity_score"] == 9.0


def test_missing_required_column_names_what_is_missing(cohort_frame):
    with pytest.raises(CohortValidationError, match=r"\['baseline_risk_score'\]"):
        frame_to_patients(cohort_frame.drop(columns=["baseline_risk_score"]))


def test_unrecognised_sex_names_the_row(cohort_frame):
    frame = cohort_frame.copy()
    frame.loc[1, "sex"] = "unknown"

    with pytest.raises(CohortValidationError, match="Row 2: sex 'unknown'"):
        frame_to_patients(frame)


def test_a_missing_value_is_refused_rather_than_filled(cohort_frame):
    """Imputing here would change the cohort without the caller knowing."""
    frame = cohort_frame.copy()
    frame.loc[0, "baseline_risk_score"] = float("nan")

    with pytest.raises(CohortValidationError, match="Row 1: baseline_risk_score"):
        frame_to_patients(frame)


def test_an_empty_file_is_refused(cohort_frame):
    with pytest.raises(CohortValidationError, match="no rows"):
        frame_to_patients(cohort_frame.iloc[:0])


# --------------------------------------------------------------------------
# Follow-up
# --------------------------------------------------------------------------

def test_frame_to_patient_records_attaches_follow_up(cohort_frame):
    records = frame_to_patient_records(cohort_frame)

    assert records[0]["follow_up"] == {"observed_time_days": 120.0, "event_observed": True}
    assert records[1]["follow_up"] == {"observed_time_days": 400.0, "event_observed": False}
    assert records[2]["follow_up"]["event_observed"] is True


def test_follow_up_must_be_positive(cohort_frame):
    frame = cohort_frame.copy()
    frame.loc[2, "observed_time_days"] = 0.0

    with pytest.raises(CohortValidationError, match="Row 3: observed_time_days"):
        frame_to_patient_records(frame)


def test_an_unparseable_event_flag_is_refused(cohort_frame):
    frame = cohort_frame.copy()
    frame["event_observed"] = frame["event_observed"].astype(object)
    frame.loc[1, "event_observed"] = "maybe"

    with pytest.raises(CohortValidationError, match="Row 2: event_observed is 'maybe'"):
        frame_to_patient_records(frame)


def test_missing_follow_up_columns_are_named(cohort_frame):
    with pytest.raises(CohortValidationError, match="survival cohort"):
        frame_to_patient_records(cohort_frame.drop(columns=["event_observed"]))


# --------------------------------------------------------------------------
# Treatment and outcomes
# --------------------------------------------------------------------------

def test_frame_to_treatment_records_attaches_assignment_and_outcome(cohort_frame):
    frame = cohort_frame.assign(
        treatment_assigned=[True, False, True], outcome_value=[2.5, 1.0, 3.5]
    )

    records = frame_to_treatment_records(frame)

    assert [r["treatment_assigned"] for r in records] == [True, False, True]
    assert [r["outcome_value"] for r in records] == [2.5, 1.0, 3.5]


def test_treatment_columns_can_be_renamed(cohort_frame):
    frame = cohort_frame.assign(on_statin=[1, 0, 1], ldl_change=[-30.0, -2.0, -25.0])

    records = frame_to_treatment_records(frame, "on_statin", "ldl_change")

    assert records[0]["treatment_assigned"] is True
    assert records[0]["outcome_value"] == -30.0


def test_frame_to_cohort_members_attaches_named_outcomes(cohort_frame):
    members = frame_to_cohort_members(cohort_frame, ["mortality", "readmission"])

    assert members[0]["outcomes"] == {"mortality": 1.0, "readmission": 0.0}
    assert members[2]["outcomes"] == {"mortality": 1.0, "readmission": 1.0}


def test_cohort_members_require_at_least_one_outcome(cohort_frame):
    with pytest.raises(CohortValidationError, match="at least one outcome"):
        frame_to_cohort_members(cohort_frame, [])


def test_outcome_candidates_exclude_covariates_and_follow_up(cohort_frame):
    assert outcome_candidate_columns(cohort_frame) == ["mortality", "readmission"]


def test_outcome_candidates_exclude_lab_columns(cohort_frame):
    frame = cohort_frame.assign(lab_creatinine=[1.1, 0.9, 2.3])

    assert "lab_creatinine" not in outcome_candidate_columns(frame)


# --------------------------------------------------------------------------
# API error rendering
# --------------------------------------------------------------------------

def test_a_validation_error_is_rendered_field_by_field():
    """FastAPI 422 bodies are a list; str() on them is unreadable."""
    body = {"detail": [
        {"loc": ["body", 0, "follow_up"], "msg": "Field required"},
        {"loc": ["body", 1, "demographics", "age"], "msg": "Input should be <= 120"},
    ]}

    message = describe_api_error(422, body)

    assert "0 -> follow_up: Field required" in message
    assert "1 -> demographics -> age: Input should be <= 120" in message


def test_a_service_unavailable_error_keeps_its_instruction():
    message = describe_api_error(503, {"detail": "POST to /api/models/risk/train first."})

    assert "not ready" in message
    assert "/api/models/risk/train" in message


def test_a_plain_detail_string_is_passed_through():
    assert "No events observed" in describe_api_error(422, {"detail": "No events observed"})


def test_an_error_with_no_body_still_reports_the_status():
    assert "HTTP 500" in describe_api_error(500, None)
