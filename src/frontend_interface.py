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
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.cohort_io import (
    CohortValidationError,
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
    st.session_state.api_url = "http://localhost:8000"


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
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
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


def fetch_health() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{st.session_state.api_url}/api/health", timeout=5)
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


PAGES = {
    "Dashboard": page_dashboard,
    "Train Risk Model": page_train_risk_model,
    "Patient Risk Assessment": page_risk_assessment,
    "Survival Analysis": page_survival_analysis,
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
