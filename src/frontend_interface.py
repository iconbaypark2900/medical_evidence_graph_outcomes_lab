"""
Streamlit frontend for Medical Evidence Graph & Outcomes Insight Lab
Clinical Decision Support Interface

Every number shown here comes from the API, which computes it from data
the user supplied. Nothing on this page is generated locally to fill a
gap.

That was not previously true. The dashboard reported an "Avg C-index" of
0.89 and a feed of named clinicians performing assessments that never
happened; the survival curve was drawn from `exp(-0.001 * t) * (1 + 0.2 *
sin(t / 50))`, a closed-form expression with no patient input; and the
cohort page generated its own mortality and readmission rates with
`np.random.binomial`, then printed p-values chosen by an if-statement on
the size of the difference and told the reader the result was
"statistically significant". Where the data needed for an analysis is not
available, this page now says so and asks for it.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.cohort_io import (
    CohortValidationError,
    frame_to_patient_records,
    frame_to_treatment_records,
    describe_api_error,
    frame_to_cohort_members,
    frame_to_patient_records,
    frame_to_patients,
    outcome_candidate_columns,
)


st.set_page_config(
    page_title="Medical Evidence Graph & Outcomes Insight Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'api_url' not in st.session_state:
    # start_system.py passes this so the frontend follows the port the API
    # was actually started on.
    st.session_state.api_url = os.environ.get("MEG_API_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def call_api(method: str, path: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Call the API, surfacing failures instead of returning {}.

    An empty dict was previously returned on error, which the display
    functions rendered as "no data available" -- the same thing they show
    for a genuinely empty result.
    """
    url = f"{st.session_state.api_url}{path}"
    headers = {**kwargs.pop("headers", {}), **_auth_headers()}
    try:
        response = requests.request(method, url, timeout=60, headers=headers, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {url}: {exc}")
        return None

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        st.error(describe_api_error(response.status_code, body))
        return None

    return response.json()


def _auth_headers() -> Dict[str, str]:
    """API key from the environment, if the API requires one."""
    key = os.environ.get("MEG_API_KEY", "")
    return {"X-API-Key": key} if key else {}


def fetch_health() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(
            f"{st.session_state.api_url}/api/health", timeout=5,
            headers=_auth_headers())
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        return None


def connect_to_api():
    api_url = st.sidebar.text_input("API URL", value=st.session_state.api_url)
    if st.sidebar.button("Connect to API"):
        st.session_state.api_url = api_url
        if fetch_health() is not None:
            st.session_state.api_connected = True
            st.sidebar.success(f"Connected to API at {api_url}")
        else:
            st.session_state.api_connected = False
            st.sidebar.error(f"No API responding at {api_url}")


def require_connection() -> None:
    if not st.session_state.api_connected:
        st.warning("Connect to the API using the sidebar to use this page.")
        st.stop()


def load_cohort_file(label: str, key: str) -> Optional[pd.DataFrame]:
    uploaded = st.file_uploader(label, type=["csv"], key=key)
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_survival_curve(survival_data: Dict, title: str = "Kaplan-Meier Survival Curve"):
    curve = survival_data["survival_curve"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["time_points"],
        y=curve["survival_probability"],
        mode='lines',
        line_shape='hv',  # A KM estimate is a step function, not a smooth line.
        name='Survival Probability',
        line=dict(color='#1f77b4', width=3),
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Days since index date',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    stats = survival_data["stats"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", stats["total_patients"])
    col2.metric("Events", stats["events_occurred"])
    col3.metric("Censored", stats["censored"])
    median = stats["median_survival_days"]
    # "Not reached" is the honest answer when the curve never crosses 0.5.
    col4.metric(
        "Median survival",
        f"{median:.0f} days" if median is not None else "Not reached",
    )


def display_cox_results(cox_data: Dict):
    st.subheader("Cox proportional hazards")

    rows = []
    for covariate, hazard_ratio in cox_data["hazard_ratios"].items():
        lower, upper = cox_data["confidence_intervals"][covariate]
        rows.append({
            "Covariate": covariate,
            "Hazard ratio": round(hazard_ratio, 3),
            "95% CI": f"{lower:.3f} – {upper:.3f}",
            "p": f"{cox_data['p_values'][covariate]:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(
        f"C-index {cox_data['c_index']:.3f}, computed on the same patients the "
        f"model was fitted on. It describes fit, not how the model would "
        f"discriminate on new patients."
    )
    if cox_data["covariates_dropped_as_constant"]:
        st.info(
            "Dropped as constant in this cohort: "
            + ", ".join(cox_data["covariates_dropped_as_constant"])
        )


MODE_LABELS = {
    "index": "🗂️ Indexed corpus (BM25 + vector + graph)",
    "live": "🌐 Live PubMed / ClinicalTrials.gov",
    "index_then_live": "🌐 Live — the indexed corpus had no match",
}


def display_evidence_results(evidence_data: Dict):
    results = evidence_data["results"]
    mode = evidence_data.get("retrieval_mode", "live")

    # Where the results came from is shown before the results themselves:
    # "our corpus does not cover this" and "the literature does not cover
    # this" look identical once you are reading the list.
    st.caption(f"Source: {MODE_LABELS.get(mode, mode)}")
    if mode == "index_then_live":
        st.info(evidence_data.get("note", ""))

    if not results:
        st.warning(
            f"No evidence found for “{evidence_data['query']}” in the "
            f"indexed corpus or upstream."
        )
        return

    if evidence_data.get("graph_context"):
        entities = sorted({row["entity"] for row in evidence_data["graph_context"]})
        with st.expander(f"Graph context ({len(entities)} entities)"):
            st.write(", ".join(entities))

    st.caption(f"{evidence_data['total_results']} result(s)")
    for evidence in results:
        with st.container():
            if evidence.get("url"):
                st.markdown(f"### [{evidence['title']}]({evidence['url']})")
            else:
                st.markdown(f"### {evidence['title']}")

            line = f"**Source:** {evidence['source']}"
            if evidence.get("pub_date"):
                line += f" | **Date:** {evidence['pub_date']}"
            if evidence.get("authors"):
                shown = ", ".join(evidence["authors"][:3])
                if len(evidence["authors"]) > 3:
                    shown += f", and {len(evidence['authors']) - 3} more"
                line += f" | **Authors:** {shown}"
            st.write(line)

            if evidence.get("abstract"):
                st.write(evidence["abstract"])
            if evidence.get("mesh_terms"):
                st.write("**MeSH terms:** " + ", ".join(evidence["mesh_terms"]))
            if evidence.get("found_by"):
                # The fused ranking, made inspectable: which retrievers
                # surfaced this result and at what rank.
                found = ", ".join(
                    f"{name} #{rank}" for name, rank in sorted(evidence["found_by"].items()))
                st.caption(f"Retrieved by {found} · fused score "
                           f"{evidence.get('fused_score', 0):.4f}")
            st.divider()


def display_cohort_comparison(comparison: Dict):
    cohort_1 = comparison["cohort_1"]
    cohort_2 = comparison["cohort_2"]

    col1, col2 = st.columns(2)
    for column, cohort, name in ((col1, cohort_1, "Cohort 1"), (col2, cohort_2, "Cohort 2")):
        with column:
            st.subheader(name)
            st.metric("Patients", cohort["size"])
            characteristics = cohort["characteristics"]
            st.write(f"Mean age: {characteristics['mean_age']:.1f}")
            st.write(f"Mean baseline risk: {characteristics['mean_baseline_risk_score']:.2f}")
            st.write(f"Mean comorbidities: {characteristics['mean_comorbidity_count']:.2f}")
            st.write(f"Sex: {characteristics['gender_distribution']}")

    st.subheader("Outcomes")
    rows = []
    for outcome, result in comparison["outcomes"].items():
        if result["test"] is None:
            rows.append({
                "Outcome": outcome, "Test": "not applicable",
                "Cohort 1": "-", "Cohort 2": "-", "Difference": "-",
                "p": "-", "Significant": result["reason"],
            })
            continue

        if result["test"] == "welch_t":
            value_1, value_2 = result["cohort_1_mean"], result["cohort_2_mean"]
            difference = result["mean_difference"]
            fmt = "{:.3f}"
        else:
            value_1, value_2 = result["cohort_1_rate"], result["cohort_2_rate"]
            difference = result["risk_difference"]
            fmt = "{:.1%}"

        rows.append({
            "Outcome": outcome,
            "Test": result["test"],
            "Cohort 1": fmt.format(value_1),
            "Cohort 2": fmt.format(value_2),
            "Difference": fmt.format(difference),
            "p": f"{result['p_value']:.4f}",
            "Significant": "yes" if result["significant"] else "no",
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    metrics = comparison["comparison_metrics"]
    st.caption(f"α = {metrics['alpha']}. {metrics['note']}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard():
    st.header("🏥 Service status")

    health = fetch_health()
    if health is None:
        st.error("The API is not responding.")
        return

    ready = health["models_ready"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Survival analysis", "ready" if ready["survival_analysis"] else "not ready")
    col2.metric("Causal inference", "ready" if ready["causal_inference"] else "not ready")
    col3.metric("Risk assessment", "ready" if ready["risk_assessment"] else "not trained")

    if health.get("authentication") == "disabled":
        st.warning(
            "This API has no authentication configured. Anyone who can reach "
            "its port can run analyses and read the corpus."
        )

    if health["risk_model_version"]:
        st.success(f"Risk model version `{health['risk_model_version']}`")
    else:
        st.info(
            "No risk model has been trained. Use the **Train Risk Model** page "
            "to fit one on labelled patients before requesting risk scores."
        )

    st.caption(f"Reported by the API at {health['timestamp']}")


def page_train_risk_model():
    st.header("🎓 Train risk model")
    require_connection()

    st.write(
        "Upload a CSV of patients whose outcomes are already known. Required "
        "columns: `age`, `sex`, `baseline_risk_score`, `comorbidity_count`, "
        "plus one column per outcome to model."
    )

    frame = load_cohort_file("Labelled patient cohort (CSV)", "train_upload")
    if frame is None:
        return

    st.dataframe(frame.head(), use_container_width=True)

    candidates = outcome_candidate_columns(frame)
    if not candidates:
        st.error("No numeric outcome columns found in this file.")
        return

    outcomes = st.multiselect("Outcome columns to model", candidates, default=candidates[:1])
    test_size = st.slider("Hold-out fraction", 0.1, 0.5, 0.25, 0.05)

    if st.button("Train") and outcomes:
        try:
            members = frame_to_cohort_members(frame, outcomes)
        except CohortValidationError as exc:
            st.error(str(exc))
            return

        result = call_api(
            "POST", "/api/models/risk/train",
            json={"training_data": members, "test_size": test_size},
        )
        if result is None:
            return

        st.success(f"Trained model version `{result['model_version']}`")
        st.dataframe(
            pd.DataFrame(result["models"].values()), hide_index=True,
            use_container_width=True)
        st.caption(
            "Metrics are measured on the held-out fraction, not on the rows "
            "the model was fitted on."
        )
        if result["skipped_outcomes"]:
            st.warning(f"Skipped: {result['skipped_outcomes']}")


def create_patient_input_form() -> Optional[Dict]:
    with st.form("patient_input_form"):
        st.subheader("Enter Patient Information")
        col1, col2 = st.columns(2)

        with col1:
            patient_id = st.text_input("Patient ID (optional)", value="")
            age = st.number_input("Age", min_value=0, max_value=120, value=65)
            sex = st.selectbox("Sex", ["male", "female"])
            weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0, step=0.1)

        with col2:
            height = st.number_input("Height (cm)", min_value=0.0, value=170.0, step=0.1)
            baseline_risk = st.slider("Baseline Risk Score", 0.0, 10.0, 2.5, 0.1)
            comorbidities = st.number_input("Number of Comorbidities", min_value=0, value=1)
            severity = st.slider("Severity Score", 0.0, 10.0, 1.0, 0.1)

        if st.form_submit_button("Assess Risk"):
            return {
                "patient_id": patient_id or f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "demographics": {
                    "age": age, "sex": sex,
                    "weight_kg": weight, "height_cm": height,
                },
                "clinical_indicators": {
                    "baseline_risk_score": baseline_risk,
                    "comorbidity_count": comorbidities,
                    "severity_score": severity,
                },
            }
    return None


def page_risk_assessment():
    st.header("🩺 Patient Risk Assessment")
    require_connection()

    health = fetch_health()
    if health and not health["models_ready"]["risk_assessment"]:
        st.warning(
            "No risk model has been trained, so no risk can be estimated. "
            "Train one on the **Train Risk Model** page first."
        )
        return

    patient_data = create_patient_input_form()
    if not patient_data:
        return

    with st.spinner("Scoring patient..."):
        results = call_api("POST", "/api/patients/risk-assessment", json=[patient_data])
    if results is None:
        return

    assessment = results["risk_assessments"][0]
    st.success(
        f"Scored patient {assessment['patient_id']} with model "
        f"`{results['model_version']}`"
    )

    columns = st.columns(max(1, len(assessment["risks"])))
    for column, (outcome, risk) in zip(columns, assessment["risks"].items()):
        score = risk["score"]
        with column:
            st.metric(outcome.replace("_", " ").title(), f"{score:.3f}")
            # A score is only as trustworthy as the model's held-out
            # performance, so the two are always shown together.
            if risk["holdout_value"] is None:
                st.caption(f"{risk['holdout_metric']}: not estimable")
            else:
                st.caption(f"Held-out {risk['holdout_metric']}: {risk['holdout_value']:.3f}")
            if risk["imputed_features"]:
                st.caption(f"⚠️ Imputed: {', '.join(risk['imputed_features'])}")


def page_survival_analysis():
    st.header("📈 Survival Analysis")
    require_connection()

    st.write(
        "Upload a cohort with observed follow-up. Required columns: `age`, "
        "`sex`, `baseline_risk_score`, `comorbidity_count`, "
        "`observed_time_days`, `event_observed`."
    )
    st.caption(
        "`event_observed` false means the patient was censored at "
        "`observed_time_days` — event-free when last seen, not event-free "
        "forever."
    )

    frame = load_cohort_file("Survival cohort (CSV)", "survival_upload")
    if frame is None:
        return

    st.dataframe(frame.head(), use_container_width=True)
    horizon = st.number_input("Reporting horizon (days)", min_value=1, value=365)

    if not st.button("Run analysis"):
        return

    try:
        records = frame_to_patient_records(frame)
    except CohortValidationError as exc:
        st.error(str(exc))
        return

    payload = {"patient_data": records, "time_horizon_days": int(horizon)}

    km = call_api("POST", "/api/survival-analysis/kaplan-meier", json=payload)
    if km is not None:
        display_survival_curve(km)
        horizon_survival = km["stats"]["survival_at_horizon"]
        if horizon_survival is not None:
            st.write(
                f"Survival at {horizon} days: **{horizon_survival:.1%}**"
            )

    cox = call_api("POST", "/api/survival-analysis/cox-regression", json=payload)
    if cox is not None:
        display_cox_results(cox)


def page_evidence_search():
    st.header("🔍 Medical Evidence Search")
    require_connection()

    query = st.text_input("Enter your clinical question or topic:", "diabetes treatment")
    limit = st.slider("Maximum results", 1, 50, 10)

    if query and st.button("Search Evidence"):
        with st.spinner("Searching PubMed and ClinicalTrials.gov..."):
            results = call_api(
                "GET", "/api/evidence/search",
                params={"query": query, "limit": limit},
            )
        if results is not None:
            display_evidence_results(results)


def page_cohort_analysis():
    st.header("🧬 Cohort Analysis & Comparison")
    require_connection()

    st.write(
        "Upload two cohorts with their observed outcomes. Both files need "
        "`age`, `sex`, `baseline_risk_score`, `comorbidity_count`, plus the "
        "outcome columns to compare."
    )

    col1, col2 = st.columns(2)
    with col1:
        frame_1 = load_cohort_file("Cohort 1 (CSV)", "cohort_1_upload")
    with col2:
        frame_2 = load_cohort_file("Cohort 2 (CSV)", "cohort_2_upload")

    if frame_1 is None or frame_2 is None:
        return

    shared = [c for c in outcome_candidate_columns(frame_1)
              if c in outcome_candidate_columns(frame_2)]
    if not shared:
        st.error("The two files share no numeric outcome column to compare.")
        return

    outcomes = st.multiselect("Outcomes to compare", shared, default=shared)
    alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)

    if st.button("Compare cohorts") and outcomes:
        try:
            cohort_1 = frame_to_cohort_members(frame_1, outcomes)
            cohort_2 = frame_to_cohort_members(frame_2, outcomes)
        except CohortValidationError as exc:
            st.error(str(exc))
            return

        result = call_api(
            "POST", "/api/cohorts/compare",
            json={
                "patient_cohort": cohort_1,
                "comparator_cohort": cohort_2,
                "outcome_variables": outcomes,
                "alpha": alpha,
            },
        )
        if result is not None:
            display_cohort_comparison(result["cohort_comparison"])


def page_cohort_builder():
    st.header("👥 Cohort Builder")
    require_connection()

    st.write(
        "Apply inclusion and exclusion criteria to a cohort with observed "
        "follow-up, and see the Kaplan-Meier curve for who is left. Required "
        "columns: `age`, `sex`, `baseline_risk_score`, `comorbidity_count`, "
        "`observed_time_days`, `event_observed`."
    )

    frame = load_cohort_file("Cohort with follow-up (CSV)", "builder_upload")
    if frame is None:
        return

    st.dataframe(frame.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        min_age, max_age = st.slider("Include ages", 0, 120, (18, 90))
    with col2:
        horizon = st.number_input("Follow-up horizon (days)", min_value=1, value=1825)

    exclude_field = st.selectbox(
        "Exclude on (optional)", ["(none)"] + list(frame.columns))
    exclude_value = None
    if exclude_field != "(none)":
        exclude_value = st.selectbox(
            f"Exclude rows where {exclude_field} equals",
            sorted(frame[exclude_field].astype(str).unique()))

    if not st.button("Build cohort"):
        return

    try:
        patients = frame_to_patient_records(frame)
    except CohortValidationError as exc:
        st.error(str(exc))
        return

    criteria = {"inclusion": {"age": [min_age, max_age]}, "exclusion": {}}
    if exclude_value is not None:
        criteria["exclusion"][exclude_field] = exclude_value

    result = call_api("POST", "/api/outcomes/cohort", json={
        "cohort_id": "ui_cohort", "name": "Cohort from upload",
        "patients": patients, "criteria": criteria,
        "follow_up_period_days": int(horizon),
    })
    if result is None:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Supplied", result["supplied"])
    col2.metric("Met criteria", result["size"])
    col3.metric("Excluded", result["excluded_by_criteria"])

    survival = result["survival"]
    fig = go.Figure()
    lower = [ci[0] for ci in survival["confidence_intervals"]]
    upper = [ci[1] for ci in survival["confidence_intervals"]]
    # Greenwood band, drawn: it widens as the risk set shrinks, and a curve
    # shown without it invites reading the tail as firmly as the head.
    fig.add_trace(go.Scatter(
        x=survival["time_points"] + survival["time_points"][::-1],
        y=upper + lower[::-1], fill="toself", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=survival["time_points"], y=survival["survival_probabilities"],
        mode="lines", line_shape="hv", line=dict(color="#1f77b4", width=3),
        name="Survival"))
    fig.update_layout(
        title="Kaplan-Meier with Greenwood confidence band",
        xaxis_title="Days", yaxis_title="Survival probability",
        yaxis=dict(range=[0, 1]), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    median = survival["median_survival_days"]
    st.write(
        f"Events {survival['events']}, censored {survival['censored']}, "
        f"median survival "
        f"**{f'{median:.0f} days' if median is not None else 'not reached'}**"
    )


def page_comparative_effectiveness():
    st.header("⚖️ Comparative Effectiveness")
    require_connection()

    st.write(
        "Compare treatment arms with a log-rank test. The file needs the "
        "follow-up columns plus a column naming each patient's arm."
    )
    st.caption(
        "Arm assignment has to come from the data. Inferring it here would "
        "make the comparison a comparison of whatever rule did the inferring."
    )

    frame = load_cohort_file("Cohort with arms (CSV)", "ce_upload")
    if frame is None:
        return

    st.dataframe(frame.head(), use_container_width=True)

    arm_column = st.selectbox("Column naming the arm", list(frame.columns))
    if not st.button("Compare arms"):
        return

    arms = frame[arm_column].astype(str).tolist()
    if len(set(arms)) < 2:
        st.error(f"`{arm_column}` has only one distinct value; nothing to compare.")
        return

    try:
        patients = frame_to_patient_records(frame)
    except CohortValidationError as exc:
        st.error(str(exc))
        return

    result = call_api("POST", "/api/outcomes/comparative-effectiveness", json={
        "cohort_id": "ui_ce", "patients": patients, "groups": arms,
        "follow_up_period_days": 10_000,
    })
    if result is None:
        return

    rows = [
        {"Arm": arm, "Patients": o["n_patients"], "Events": o["n_events"],
         "Event rate": f"{o['event_rate']:.1%}",
         "Median follow-up (days)": f"{o['median_survival']:.0f}"}
        for arm, o in result["group_outcomes"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(result["test"], f"p = {result['p_value']:.4g}")
    col2.metric("Test statistic", f"{result['test_statistic']:.2f}")
    if result["number_needed_to_treat"] is not None:
        col3.metric("NNT", f"{result['number_needed_to_treat']:.1f}")

    # The p-value never appears without its denominators.
    st.caption(
        f"{result['degrees_of_freedom']} df · sizes {result['group_sizes']} · "
        f"events {result['group_events']}"
    )
    for note in result["notes"]:
        st.info(note)


def page_causal_inference():
    st.header("🎯 Treatment Effect")
    require_connection()

    st.write(
        "Estimate an average treatment effect from observed assignments and "
        "outcomes. The file needs the patient columns plus a treatment column "
        "and an outcome column."
    )

    frame = load_cohort_file("Cohort with treatment and outcome (CSV)", "ate_upload")
    if frame is None:
        return

    st.dataframe(frame.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        treatment_column = st.selectbox("Treatment column", list(frame.columns))
    with col2:
        outcome_column = st.selectbox("Outcome column", list(frame.columns))

    confounders = st.multiselect(
        "Adjust for", ["age", "gender_encoded", "baseline_risk_score",
                       "comorbidity_count"],
        default=["age", "baseline_risk_score", "comorbidity_count"])

    if not st.button("Estimate effect"):
        return

    try:
        patients = frame_to_treatment_records(frame, treatment_column, outcome_column)
    except CohortValidationError as exc:
        st.error(str(exc))
        return

    result = call_api("POST", "/api/causal-inference/ate-estimation", json={
        "patient_data": patients,
        "treatment_variable": treatment_column,
        "outcome_variable": outcome_column,
        "confounders": confounders,
    })
    if result is None:
        return

    estimates = result["ate_estimates"]
    columns = st.columns(len(estimates))
    for column, (method, value) in zip(columns, estimates.items()):
        column.metric(method.replace("ate_", "").replace("_", " ").title(),
                      f"{value:.3f}")

    cohort = result["cohort"]
    st.caption(
        f"{cohort['treated']} treated, {cohort['control']} control · "
        f"adjusted for {', '.join(result['confounders_adjusted_for'])}"
    )
    if result["confounders_dropped_as_constant"]:
        st.info("Dropped as constant: "
                + ", ".join(result["confounders_dropped_as_constant"]))

    # The unadjusted and adjusted numbers differing is the finding, so the
    # caveat has to sit with them rather than in documentation somewhere.
    st.warning(result["caveat"])


def page_guidelines():
    st.header("📋 Guidelines & Adherence")
    require_connection()

    registered = call_api("GET", "/api/pathways/guidelines")
    if registered is None:
        return
    guidelines = registered["guidelines"]

    register_tab, adherence_tab = st.tabs(["Register a guideline", "Score adherence"])

    with register_tab:
        st.write("Steps, one per line. Prefix with `?` to mark a step optional.")
        with st.form("guideline_form"):
            guideline_id = st.text_input("Guideline ID", "dm2_2026")
            name = st.text_input("Name", "Type 2 Diabetes Management")
            condition = st.text_input("Condition", "type 2 diabetes")
            steps_text = st.text_area(
                "Steps",
                "\n".join([
                    "HbA1c measurement",
                    "Metformin initiation",
                    "Retinal screening",
                    "?Sulfonylurea add-on",
                ]))
            if st.form_submit_button("Register"):
                steps = []
                for line in filter(None, (l.strip() for l in steps_text.splitlines())):
                    optional = line.startswith("?")
                    steps.append({"name": line.lstrip("?").strip(),
                                  "recommended": not optional})
                result = call_api("POST", "/api/pathways/guidelines", json={
                    "id": guideline_id, "name": name, "condition": condition,
                    "steps": steps, "decision_points": [],
                })
                if result is not None:
                    st.success(
                        f"Registered `{result['guideline_id']}`: "
                        f"{result['required_steps']} of {result['steps']} steps required")
                    st.rerun()

    with adherence_tab:
        if not guidelines:
            st.info("Register a guideline first.")
            return

        chosen = st.selectbox(
            "Guideline", [g["id"] for g in guidelines],
            format_func=lambda i: next(g["name"] for g in guidelines if g["id"] == i))
        patient_id = st.text_input("Patient ID", "pt_1")
        performed = st.text_area("Steps actually performed, one per line", "")

        if st.button("Score adherence"):
            result = call_api("POST", "/api/pathways/adherence", json={
                "guideline_id": chosen, "patient_id": patient_id,
                "condition": next(g["condition"] for g in guidelines if g["id"] == chosen),
                "steps": [{"name": l.strip()}
                          for l in performed.splitlines() if l.strip()],
            })
            if result is None:
                return

            comparison = result["comparison"]
            score = comparison["adherence_score"]
            # Always with its denominator: 2/3 and 200/300 are the same
            # number and very different evidence.
            st.metric(
                "Adherence", f"{score:.0%}",
                f"{comparison['n_performed']} of {comparison['n_required']} required steps")

            if comparison["missing_steps"]:
                st.warning("Missing: " + ", ".join(comparison["missing_steps"]))
            if comparison["extra_steps"]:
                st.info("Not recommended: " + ", ".join(comparison["extra_steps"]))
            for opportunity in result["opportunities"]:
                st.write(
                    f"**{opportunity['type']}** ({opportunity['priority']}): "
                    f"{opportunity['description']} — {opportunity['suggestion']}")


def page_guideline_evidence():
    st.header("🔗 Evidence for a Guideline")
    require_connection()

    st.write(
        "Each recommended step is used as a retrieval query against the "
        "indexed corpus. This is where the evidence side and the pathway "
        "side meet."
    )

    registered = call_api("GET", "/api/pathways/guidelines")
    if registered is None:
        return
    guidelines = registered["guidelines"]
    if not guidelines:
        st.info("Register a guideline on the **Guidelines & Adherence** page first.")
        return

    chosen = st.selectbox(
        "Guideline", [g["id"] for g in guidelines],
        format_func=lambda i: next(g["name"] for g in guidelines if g["id"] == i))
    per_step = st.slider("Records per step", 1, 10, 3)

    if not st.button("Check evidence"):
        return

    result = call_api(
        "GET", f"/api/pathways/guidelines/{chosen}/evidence",
        params={"per_step": per_step})
    if result is None:
        return

    unsupported = result["steps_with_no_evidence_in_corpus"]
    if unsupported:
        st.warning(
            f"{len(unsupported)} of {result['steps_examined']} steps have no "
            f"supporting record in the corpus: " + ", ".join(unsupported))
    else:
        st.success(f"All {result['steps_examined']} steps have supporting records.")

    for finding in result["findings"]:
        with st.expander(
                f"{finding['step']} — {finding['supporting_records']} record(s)",
                expanded=bool(finding["supporting_records"])):
            st.caption(
                f"query: “{finding['query']}” · mode: {finding['retrieval_mode']}")
            if not finding["evidence"]:
                st.write("No supporting record in the indexed corpus.")
            for item in finding["evidence"]:
                if item.get("url"):
                    st.markdown(f"- [{item['citation']}]({item['url']}) {item['title']}")
                else:
                    st.markdown(f"- {item['citation']} {item['title']}")

    # The distinction the whole endpoint exists to preserve.
    st.info(result["note"])


PAGES = {
    "Dashboard": page_dashboard,
    "Train Risk Model": page_train_risk_model,
    "Patient Risk Assessment": page_risk_assessment,
    "Survival Analysis": page_survival_analysis,
    "Cohort Builder": page_cohort_builder,
    "Comparative Effectiveness": page_comparative_effectiveness,
    "Treatment Effect": page_causal_inference,
    "Guidelines & Adherence": page_guidelines,
    "Evidence for a Guideline": page_guideline_evidence,
    "Evidence Search": page_evidence_search,
    "Cohort Analysis": page_cohort_analysis,
}


def main():
    st.title("🏥 Medical Evidence Graph & Outcomes Insight Lab")
    st.subheader("Clinical Decision Support System")

    with st.sidebar:
        st.header("API Connection")
        connect_to_api()

        st.header("Navigation")
        page = st.radio("Select a page:", list(PAGES))

        if st.session_state.api_connected:
            st.success("✅ Connected to API")
        else:
            st.error("❌ Not connected to API")

    PAGES[page]()

    st.divider()
    st.caption(
        "Medical Evidence Graph & Outcomes Insight Lab — every figure on this "
        "page is computed by the API from data you supplied."
    )


if __name__ == "__main__":
    main()
