"""Cohort CSV -> API payload conversion.

Kept free of Streamlit so it can be imported and tested directly; the
frontend is a thin layer over these functions.

Everything here fails loudly. A cohort file with a missing column, an
unparseable sex value or a negative follow-up time raises
`CohortValidationError` naming the problem, rather than being silently
coerced, dropped, or filled with a default. A dropped row changes the
denominator of every rate computed downstream.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PATIENT_COLUMNS = ("age", "sex", "baseline_risk_score", "comorbidity_count")
FOLLOW_UP_COLUMNS = ("observed_time_days", "event_observed")

_SEX_ALIASES = {
    "m": "male", "male": "male", "1": "male",
    "f": "female", "female": "female", "0": "female",
}

_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}


class CohortValidationError(ValueError):
    """A cohort file cannot be turned into a valid request payload."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], purpose: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise CohortValidationError(
            f"{purpose} needs column(s) {missing}. The file has: "
            f"{list(frame.columns)}")


def _parse_sex(value: Any, row: int) -> str:
    key = str(value).strip().lower()
    if key not in _SEX_ALIASES:
        raise CohortValidationError(
            f"Row {row}: sex {value!r} is not recognised. Use one of "
            f"{sorted(set(_SEX_ALIASES))}.")
    return _SEX_ALIASES[key]


def _parse_bool(value: Any, column: str, row: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    key = str(value).strip().lower()
    if key in _TRUE_VALUES:
        return True
    if key in _FALSE_VALUES:
        return False
    raise CohortValidationError(
        f"Row {row}: {column} is {value!r}, which is neither true nor false. "
        f"Use one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}.")


def _parse_number(value: Any, column: str, row: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CohortValidationError(
            f"Row {row}: {column} is {value!r}, which is not a number.") from None
    if not np.isfinite(number):
        raise CohortValidationError(
            f"Row {row}: {column} is {value!r}; a missing value cannot be "
            f"filled in without changing the result.")
    return number


def frame_to_patients(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Covariates only. Suitable for scoring, not for analysis."""
    if frame.empty:
        raise CohortValidationError("The cohort file has no rows.")
    _require_columns(frame, PATIENT_COLUMNS, "A patient cohort")

    lab_columns = [c for c in frame.columns if c.startswith("lab_")]

    patients = []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        indicators: Dict[str, Any] = {
            "baseline_risk_score": _parse_number(
                row["baseline_risk_score"], "baseline_risk_score", position),
            "comorbidity_count": int(
                _parse_number(row["comorbidity_count"], "comorbidity_count", position)),
        }
        if "severity_score" in frame.columns and pd.notna(row["severity_score"]):
            indicators["severity_score"] = _parse_number(
                row["severity_score"], "severity_score", position)
        if lab_columns:
            indicators["lab_values"] = {
                column.removeprefix("lab_"): _parse_number(row[column], column, position)
                for column in lab_columns
                if pd.notna(row[column])
            }

        patients.append({
            "patient_id": str(row["patient_id"]) if "patient_id" in frame.columns
                          else f"row_{position}",
            "demographics": {
                "age": int(_parse_number(row["age"], "age", position)),
                "sex": _parse_sex(row["sex"], position),
            },
            "clinical_indicators": indicators,
        })
    return patients


def frame_to_patient_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Patients with observed follow-up, for time-to-event analysis."""
    _require_columns(frame, FOLLOW_UP_COLUMNS, "A survival cohort")

    records = frame_to_patients(frame)
    for position, (record, (_, row)) in enumerate(
            zip(records, frame.iterrows()), start=1):
        days = _parse_number(row["observed_time_days"], "observed_time_days", position)
        if days <= 0:
            raise CohortValidationError(
                f"Row {position}: observed_time_days is {days}. Follow-up must "
                f"be positive; a patient observed for no time contributes "
                f"nothing to a survival estimate.")
        record["follow_up"] = {
            "observed_time_days": days,
            "event_observed": _parse_bool(row["event_observed"], "event_observed", position),
        }
    return records


def frame_to_treatment_records(
    frame: pd.DataFrame,
    treatment_column: str = "treatment_assigned",
    outcome_column: str = "outcome_value",
) -> List[Dict[str, Any]]:
    """Patients with a treatment assignment and an observed outcome."""
    _require_columns(frame, (treatment_column, outcome_column), "A causal analysis")

    records = frame_to_patients(frame)
    for position, (record, (_, row)) in enumerate(
            zip(records, frame.iterrows()), start=1):
        record["treatment_assigned"] = _parse_bool(
            row[treatment_column], treatment_column, position)
        record["outcome_value"] = _parse_number(
            row[outcome_column], outcome_column, position)
    return records


def frame_to_cohort_members(
    frame: pd.DataFrame, outcome_columns: Iterable[str]
) -> List[Dict[str, Any]]:
    """Patients with named observed outcomes, for cohort comparison."""
    outcome_columns = list(outcome_columns)
    if not outcome_columns:
        raise CohortValidationError(
            "Name at least one outcome column to compare.")
    _require_columns(frame, outcome_columns, "A cohort comparison")

    records = frame_to_patients(frame)
    for position, (record, (_, row)) in enumerate(
            zip(records, frame.iterrows()), start=1):
        record["outcomes"] = {
            column: _parse_number(row[column], column, position)
            for column in outcome_columns
        }
    return records


def outcome_candidate_columns(frame: pd.DataFrame) -> List[str]:
    """Numeric columns that are not patient covariates -- plausible outcomes."""
    reserved = set(PATIENT_COLUMNS) | set(FOLLOW_UP_COLUMNS) | {
        "patient_id", "severity_score", "treatment_assigned", "outcome_value"}
    return [
        column for column in frame.columns
        if column not in reserved
        and not column.startswith("lab_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def describe_api_error(status_code: int, body: Optional[Dict[str, Any]]) -> str:
    """Turn an API error into something a reader can act on.

    FastAPI's 422 body is a list of per-field errors; flattening it to
    str(body) produces an unreadable wall, and swallowing it produces the
    'something went wrong' message that sends people to the logs.
    """
    detail = (body or {}).get("detail")

    if status_code == 503:
        return f"The service is not ready: {detail}"

    if isinstance(detail, list):
        problems = []
        for item in detail:
            location = " -> ".join(str(p) for p in item.get("loc", []) if p != "body")
            problems.append(f"{location}: {item.get('msg', 'invalid')}")
        return "The request was rejected:\n- " + "\n- ".join(problems)

    if detail:
        return f"The request was rejected: {detail}"

    return f"The request failed with HTTP {status_code} and no explanation."
