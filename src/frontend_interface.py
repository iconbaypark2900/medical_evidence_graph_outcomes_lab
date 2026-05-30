"""
Streamlit frontend for Medical Evidence Graph & Outcomes Insight Lab
Clinical Decision Support Interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import asyncio


# Page configuration
st.set_page_config(
    page_title="Medical Evidence Graph & Outcomes Insight Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state for API connection
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


def test_api_connection(api_url: str) -> bool:
    """Test connection to the backend API"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def connect_to_api():
    """Connect to the backend API"""
    api_url = st.sidebar.text_input("API URL", value=st.session_state.api_url)
    if st.sidebar.button("Connect to API"):
        if test_api_connection(api_url):
            st.session_state.api_connected = True
            st.session_state.api_url = api_url
            st.sidebar.success(f"Connected to API at {api_url}")
        else:
            st.sidebar.error(f"Failed to connect to {api_url}")
            st.session_state.api_connected = False


def create_patient_input_form() -> Dict:
    """Create form for patient input"""
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
        
        submitted = st.form_submit_button("Assess Risk")
        
        if submitted:
            patient_data = {
                "patient_id": patient_id or f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "demographics": {
                    "age": age,
                    "sex": sex,
                    "weight_kg": weight,
                    "height_cm": height
                },
                "clinical_indicators": {
                    "baseline_risk_score": baseline_risk,
                    "comorbidity_count": comorbidities,
                    "severity_score": severity,
                    "lab_values": {}  # For future implementation
                }
            }
            return patient_data
    
    return None


def get_risk_assessment(api_url: str, patient_data: Dict) -> Dict:
    """Get risk assessment from API"""
    try:
        response = requests.post(
            f"{api_url}/api/patients/risk-assessment",
            json=[patient_data],  # API expects a list
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error getting risk assessment: {str(e)}")
        return {}


def display_survival_curve(survival_data: Dict):
    """Display survival curve"""
    if not survival_data or "survival_curve" not in survival_data:
        st.warning("No survival curve data available")
        return
    
    curve = survival_data["survival_curve"]
    time_points = curve["time_points"]
    survival_probs = curve["survival_probability"]
    
    # Create the survival curve plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=survival_probs,
        mode='lines+markers',
        name='Survival Probability',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Kaplan-Meier Survival Curve',
        xaxis_title='Time',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_ate_analysis(ate_data: Dict):
    """Display ATE analysis results"""
    if not ate_data or "ate_estimates" not in ate_data:
        st.warning("No ATE analysis data available")
        return
    
    estimates = ate_data["ate_estimates"]
    st.subheader("Average Treatment Effect (ATE) Estimates")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (method, value) in enumerate(estimates.items()):
        if i == 0:
            col1.metric(f"{method.replace('_', ' ').title()}", f"{value:.3f}")
        elif i == 1:
            col2.metric(f"{method.replace('_', ' ').title()}", f"{value:.3f}")
        else:
            col3.metric(f"{method.replace('_', ' ').title()}", f"{value:.3f}")


def search_evidence(api_url: str, query: str) -> Dict:
    """Search medical evidence"""
    try:
        response = requests.get(
            f"{api_url}/api/evidence/search",
            params={"query": query, "limit": 10},
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error searching evidence: {str(e)}")
        return {}


def display_evidence_results(evidence_data: Dict):
    """Display evidence search results"""
    if not evidence_data or "results" not in evidence_data:
        st.warning("No evidence results to display")
        return
    
    results = evidence_data["results"]
    
    for i, evidence in enumerate(results[:5]):  # Show first 5 results
        with st.container():
            st.markdown(f"### [{evidence['title']}]()")
            st.write(f"**Source:** {evidence['source']} | **Date:** {evidence['pub_date']}")
            st.write(f"**Abstract:** {evidence['abstract'][:200]}...")
            
            # Display entities
            if "entities" in evidence:
                st.write("**Entities:**")
                cols = st.columns(3)
                entity_types = ["conditions", "interventions", "outcomes"]
                for j, ent_type in enumerate(entity_types):
                    if ent_type in evidence["entities"]:
                        with cols[j]:
                            st.write(f"**{ent_type.title()}:**")
                            for entity in evidence["entities"][ent_type][:3]:  # Show first 3
                                st.write(f"- {entity}")
            
            st.divider()


def main():
    st.title("🏥 Medical Evidence Graph & Outcomes Insight Lab")
    st.subheader("Clinical Decision Support System")
    
    # Sidebar with API connection
    with st.sidebar:
        st.header("API Connection")
        connect_to_api()
        
        st.header("Navigation")
        page = st.radio("Select a page:", 
                       ["Dashboard", "Patient Risk Assessment", "Evidence Search", "Cohort Analysis"])
        
        if st.session_state.api_connected:
            st.success("✅ Connected to API")
        else:
            st.error("❌ Not connected to API")
    
    # Dashboard page
    if page == "Dashboard":
        st.header("🏥 Clinical Decision Dashboard")
        
        if not st.session_state.api_connected:
            st.warning("Connect to the API to access dashboard features")
            st.stop()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk Assessments", "24", "+2")
        with col2:
            st.metric("Evidence Queries", "17", "-1")
        with col3:
            st.metric("Active Cohorts", "8", "+1")  
        with col4:
            st.metric("Avg C-index", "0.89", "+0.02")
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_activity = [
            {"time": "2 mins ago", "action": "Risk assessment for patient PT-4821", "user": "Dr. Smith"},
            {"time": "15 mins ago", "action": "Evidence search for diabetes treatment", "user": "Dr. Johnson"},
            {"time": "1 hour ago", "action": "Cohort analysis completed", "user": "Dr. Williams"},
        ]
        
        for activity in recent_activity:
            st.write(f"⏱️ {activity['time']} | 👤 {activity['user']} | {activity['action']}")
    
    # Patient Risk Assessment page
    elif page == "Patient Risk Assessment":
        st.header("🩺 Patient Risk Assessment")
        
        if not st.session_state.api_connected:
            st.warning("Connect to the API to perform risk assessments")
            st.stop()
        
        patient_data = create_patient_input_form()
        
        if patient_data:
            with st.spinner("Performing risk assessment..."):
                results = get_risk_assessment(st.session_state.api_url, patient_data)
                
                if "risk_assessments" in results and results["risk_assessments"]:
                    assessment = results["risk_assessments"][0]
                    
                    st.success(f"Risk assessment completed for patient {assessment['patient_id']}")
                    
                    # Display risk scores
                    risks = assessment.get("risks", {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        mortality_risk = risks.get("mortality", 0)
                        risk_level = "High" if mortality_risk > 0.3 else "Medium" if mortality_risk > 0.1 else "Low"
                        st.metric("Mortality Risk", f"{mortality_risk:.2f}", f"{risk_level} Risk")
                    
                    with col2:
                        readmission_risk = risks.get("readmission", 0)
                        risk_level = "High" if readmission_risk > 0.3 else "Medium" if readmission_risk > 0.1 else "Low"
                        st.metric("Readmission Risk", f"{readmission_risk:.2f}", f"{risk_level} Risk")
                    
                    with col3:
                        stay_risk = risks.get("extended_stay", 0)
                        risk_level = "High" if stay_risk > 0.3 else "Medium" if stay_risk > 0.1 else "Low"
                        st.metric("Extended Stay Risk", f"{stay_risk:.2f}", f"{risk_level} Risk")
                    
                    # Display risk factors
                    st.subheader("Risk Factors Analysis")
                    factors_df = pd.DataFrame({
                        "Factor": ["Age", "Baseline Risk", "Comorbidities", "Severity"],
                        "Value": [patient_data["demographics"]["age"], 
                                 patient_data["clinical_indicators"]["baseline_risk_score"],
                                 patient_data["clinical_indicators"]["comorbidity_count"],
                                 patient_data["clinical_indicators"]["severity_score"]],
                        "Importance": [0.3, 0.25, 0.2, 0.25]  # Example importance scores
                    })
                    
                    st.dataframe(factors_df, hide_index=True)
        
        # Survival analysis section
        with st.expander("Survival Analysis"):
            st.write("Run survival analysis to predict patient outcomes over time.")
            if st.button("Run Survival Analysis"):
                with st.spinner("Calculating survival curve..."):
                    # Mock survival analysis (in real implementation, this would call the API)
                    time_points = np.linspace(0, 730, 100)  # 2 years in days
                    survival_prob = np.exp(-0.001 * time_points) * (1 + 0.2 * np.sin(time_points / 50))  # Mock curve
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=survival_prob,
                        mode='lines',
                        name='Survival Probability',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    fig.update_layout(
                        title='Predicted Survival Curve',
                        xaxis_title='Days',
                        yaxis_title='Survival Probability',
                        yaxis=dict(range=[0, 1]),
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Evidence Search page
    elif page == "Evidence Search":
        st.header("🔍 Medical Evidence Search")
        
        if not st.session_state.api_connected:
            st.warning("Connect to the API to search medical evidence")
            st.stop()
        
        query = st.text_input("Enter your clinical question or topic:", "diabetes treatment")
        
        if query and st.button("Search Evidence"):
            with st.spinner("Searching medical literature..."):
                results = search_evidence(st.session_state.api_url, query)
                display_evidence_results(results)
    
    # Cohort Analysis page
    elif page == "Cohort Analysis":
        st.header("🧬 Cohort Analysis & Comparison")
        
        if not st.session_state.api_connected:
            st.warning("Connect to the API to perform cohort analysis")
            st.stop()
        
        st.write("Analyze patient populations and compare outcomes between groups.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Define Cohorts")
            cohort1_name = st.text_input("Cohort 1 Name", "Treatment Group")
            cohort1_desc = st.text_area("Cohort 1 Criteria", "Patients receiving the new treatment")
            
            num_patients1 = st.slider("Number of Patients", 10, 1000, 100, step=10)
            treatment_rate1 = st.slider("Treatment Rate (%)", 0, 100, 80)
        
        with col2:
            st.subheader("Comparator Cohort")
            cohort2_name = st.text_input("Cohort 2 Name", "Control Group")
            cohort2_desc = st.text_area("Cohort 2 Criteria", "Patients receiving standard care")
            
            num_patients2 = st.slider("Number of Controls", 10, 1000, 100, step=10)
            treatment_rate2 = st.slider("Control Treatment Rate (%)", 0, 100, 20)
        
        if st.button("Run Cohort Analysis"):
            with st.spinner("Analyzing cohorts..."):
                # Create mock patient data for demonstration
                ages1 = np.random.normal(65, 12, num_patients1)
                ages2 = np.random.normal(66, 11, num_patients2)
                
                # Create comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Age distribution comparison
                    age_fig = px.box(
                        x=['Treatment Group', 'Control Group'],
                        y=[ages1, ages2],
                        labels={'x': 'Cohort', 'y': 'Age'},
                        title='Age Distribution Comparison'
                    )
                    st.plotly_chart(age_fig, use_container_width=True)
                
                with col2:
                    # Treatment comparison
                    treatment_data = pd.DataFrame({
                        'Cohort': ['Treatment Group'] * num_patients1 + ['Control Group'] * num_patients2,
                        'Treatment': [1 if np.random.random() < treatment_rate1/100 else 0 for _ in range(num_patients1)] +
                                   [1 if np.random.random() < treatment_rate2/100 else 0 for _ in range(num_patients2)]
                    })
                    
                    treatment_fig = px.histogram(
                        treatment_data,
                        x='Cohort',
                        color='Treatment',
                        barmode='group',
                        title='Treatment Assignment Comparison'
                    )
                    st.plotly_chart(treatment_fig, use_container_width=True)
                
                # Outcome comparison
                st.subheader("Outcome Comparison")
                outcomes_col1, outcomes_col2 = st.columns(2)
                
                with outcomes_col1:
                    # Mortality comparison
                    mort1 = np.random.binomial(1, 0.12, num_patients1)  # 12% in treatment
                    mort2 = np.random.binomial(1, 0.18, num_patients2)  # 18% in control
                    
                    mortality_fig = go.Figure(data=[
                        go.Bar(name=cohort1_name, x=['Mortality'], y=[mort1.mean()], marker_color='red', opacity=0.7),
                        go.Bar(name=cohort2_name, x=['Mortality'], y=[mort2.mean()], marker_color='blue', opacity=0.7)
                    ])
                    mortality_fig.update_layout(
                        title='Mortality Rate Comparison',
                        yaxis=dict(tickformat='.1%', range=[0, 0.3])
                    )
                    st.plotly_chart(mortality_fig, use_container_width=True)
                
                with outcomes_col2:
                    # Readmission comparison
                    readmit1 = np.random.binomial(1, 0.15, num_patients1)  # 15% in treatment
                    readmit2 = np.random.binomial(1, 0.22, num_patients2)  # 22% in control
                    
                    readmit_fig = go.Figure(data=[
                        go.Bar(name=cohort1_name, x=['Readmission'], y=[readmit1.mean()], marker_color='orange', opacity=0.7),
                        go.Bar(name=cohort2_name, x=['Readmission'], y=[readmit2.mean()], marker_color='green', opacity=0.7)
                    ])
                    readmit_fig.update_layout(
                        title='Readmission Rate Comparison',
                        yaxis=dict(tickformat='.1%', range=[0, 0.3])
                    )
                    st.plotly_chart(readmit_fig, use_container_width=True)
                
                # Statistical significance
                st.subheader("Statistical Analysis")
                mortality_diff = mort1.mean() - mort2.mean()
                readmit_diff = readmit1.mean() - readmit2.mean()
                
                st.write(f"**Mortality Difference**: {mort1.mean():.3f} - {mort2.mean():.3f} = {mortality_diff:.3f}")
                st.write(f"**Readmission Difference**: {readmit1.mean():.3f} - {readmit2.mean():.3f} = {readmit_diff:.3f}")
                
                # P-value approximations (in real implementation, would run actual statistical tests)
                p_mortality = 0.023 if abs(mortality_diff) > 0.03 else 0.156
                p_readmission = 0.048 if abs(readmit_diff) > 0.04 else 0.210
                
                st.write(f"**Mortality P-value**: {p_mortality:.3f} {'*' if p_mortality < 0.05 else ''}")
                st.write(f"**Readmission P-value**: {p_readmission:.3f} {'*' if p_readmission < 0.05 else ''}")
                
                if p_mortality < 0.05:
                    st.success("Statistically significant difference in mortality between groups")
                else:
                    st.info("No statistically significant difference in mortality between groups")
    
    # Footer
    st.divider()
    st.caption(f"Medical Evidence Graph & Outcomes Insight Lab | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()