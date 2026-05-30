"""
FastAPI backend for Medical Evidence Graph & Outcomes Insight Lab
Provides RESTful APIs for the clinical decision support system
"""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import UUID, uuid4
import json
from enum import Enum
import asyncio
import logging

# Import the enhanced ML models from Phase 2
from src.enhanced_ml_models import (
    SurvivalAnalysisModels,
    CausalInferenceModels,
    EnhancedOutcomeModels,
    DeepSurvivalModel
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Medical Evidence Graph & Outcomes Insight Lab API",
    description="API for clinical decision support and evidence-based medicine",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data Models
class SexEnum(str, Enum):
    male = "male"
    female = "female"


class PatientDemographics(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: SexEnum = Field(..., description="Patient biological sex")
    race: Optional[str] = Field(None, description="Patient race/ethnicity")
    weight_kg: Optional[float] = Field(None, ge=0, description="Patient weight in kg")
    height_cm: Optional[float] = Field(None, ge=0, description="Patient height in cm")


class ClinicalIndicators(BaseModel):
    baseline_risk_score: float = Field(..., ge=0, le=10, description="Baseline risk score")
    comorbidity_count: int = Field(0, ge=0, description="Count of comorbidities")
    severity_score: Optional[float] = Field(None, ge=0, le=10, description="Overall severity score")
    lab_values: Optional[Dict[str, float]] = Field(None, description="Laboratory values")


class TreatmentHistory(BaseModel):
    previous_treatments: List[str] = Field([], description="List of previous treatments")
    current_treatments: List[str] = Field([], description="Currently prescribed treatments")
    medication_list: List[str] = Field([], description="Current medication list")
    treatment_response: Optional[Dict[str, float]] = Field(None, description="Prior treatment responses")


class PatientInput(BaseModel):
    patient_id: Optional[str] = Field(None, description="Unique patient identifier")
    demographics: PatientDemographics
    clinical_indicators: ClinicalIndicators
    treatment_history: Optional[TreatmentHistory] = Field(None)
    admission_date: Optional[str] = Field(datetime.now().isoformat(), description="Date of admission")


class SurvivalAnalysisRequest(BaseModel):
    patient_data: List[PatientInput]
    time_horizon_days: int = Field(365, description="Time horizon for survival analysis")


class CausalAnalysisRequest(BaseModel):
    patient_data: List[PatientInput]
    treatment_variable: str = Field(..., description="Variable representing the treatment")
    outcome_variable: str = Field(..., description="Variable representing the outcome")
    confounders: List[str] = Field([], description="Potential confounding variables")


class CohortAnalysisRequest(BaseModel):
    patient_cohort: List[PatientInput]
    comparator_cohort: Optional[List[PatientInput]] = Field(None, description="Comparator group for comparison")
    outcome_variables: List[str] = Field([], description="Outcomes to analyze")


# Initialize model instances
survival_models = SurvivalAnalysisModels()
causal_models = CausalInferenceModels()
enhanced_models = EnhancedOutcomeModels()


@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {
        "message": "Medical Evidence Graph & Outcomes Insight Lab API",
        "status": "running",
        "version": "1.0.0",
        "models_loaded": True
    }


@app.post("/api/patients/risk-assessment")
async def assess_patient_risk(patients: List[PatientInput]):
    """Assess risk for one or more patients using enhanced ML models"""
    logger.info(f"Assessing risk for {len(patients)} patients")
    
    try:
        # Convert patient data to features for ML models
        patient_data = []
        for patient in patients:
            patient_dict = {
                'age': patient.demographics.age,
                'sex': patient.demographics.sex.value,
                'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
                'comorbidity_count': patient.clinical_indicators.comorbidity_count,
            }
            
            # Add optional fields
            if patient.clinical_indicators.severity_score:
                patient_dict['severity_score'] = patient.clinical_indicators.severity_score
            if patient.clinical_indicators.lab_values:
                for lab_name, lab_value in patient.clinical_indicators.lab_values.items():
                    patient_dict[f'lab_{lab_name}'] = lab_value
            
            patient_data.append(patient_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(patient_data)
        
        # Prepare outcomes for multi-task modeling
        # For demonstration, we'll create synthetic outcomes
        n_patients = len(df)
        outcomes = {
            'mortality': np.random.binomial(1, 0.1, n_patients),  # 10% mortality rate
            'readmission': np.random.binomial(1, 0.15, n_patients),  # 15% readmission rate
            'extended_stay': np.random.binomial(1, 0.2, n_patients)  # 20% extended stay
        }
        
        # Train multi-task models
        trained_models = enhanced_models.train_multi_task_model(df, outcomes)
        
        # Generate risk predictions
        risk_assessments = []
        for i, patient in enumerate(patients):
            patient_risks = {
                'patient_id': patient.patient_id or f"patient_{i}",
                'assessed_at': datetime.utcnow().isoformat(),
                'risks': {}
            }
            
            # Calculate risks for each outcome
            if len(df) > i:
                patient_row = df.iloc[[i]]  # Select just this patient's row
                for outcome_name, model_info in trained_models.items():
                    if outcome_name in outcomes:
                        model = model_info['model']
                        try:
                            risk_score = enhanced_models.predict_risk(model, patient_row)[0]
                            patient_risks['risks'][outcome_name] = float(risk_score)
                        except Exception as e:
                            logger.warning(f"Could not predict risk for outcome {outcome_name}: {str(e)}")
                            patient_risks['risks'][outcome_name] = 0.0
            else:
                logger.warning(f"No feature data available for patient {i}")
                
            risk_assessments.append(patient_risks)
        
        logger.info(f"Risk assessment completed for {len(risk_assessments)} patients")
        return {
            "status": "success",
            "risk_assessments": risk_assessments,
            "total_patients": len(risk_assessments)
        }
    
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/survival-analysis/kaplan-meier")
async def kaplan_meier_analysis(request: SurvivalAnalysisRequest):
    """Perform Kaplan-Meier survival analysis"""
    logger.info(f"Running Kaplan-Meier analysis for {len(request.patient_data)} patients")
    
    try:
        # Generate synthetic survival data for demonstration
        n_patients = len(request.patient_data)
        duration = np.random.exponential(2, n_patients)  # Time to event (years)
        event = np.random.binomial(1, 0.4, n_patients)   # Whether event occurred
        
        # Perform Kaplan-Meier analysis
        time_points, survival_probs = survival_models.kaplan_meier_analysis(duration, event)
        
        result = {
            "status": "success",
            "time_horizon_days": request.time_horizon_days,
            "survival_curve": {
                "time_points": time_points,
                "survival_probability": survival_probs
            },
            "stats": {
                "median_survival": np.median(time_points) if time_points else None,
                "total_patients": n_patients,
                "events_occurred": int(sum(event))
            }
        }
        
        logger.info("Kaplan-Meier analysis completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error in Kaplan-Meier analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/survival-analysis/cox-regression")
async def cox_regression_analysis(request: SurvivalAnalysisRequest):
    """Perform Cox proportional hazards regression"""
    logger.info(f"Running Cox regression for {len(request.patient_data)} patients")
    
    try:
        # Prepare data from patient inputs
        data_rows = []
        for patient in request.patient_data:
            row = {
                'age': patient.demographics.age,
                'gender_encoded': 1 if patient.demographics.sex == SexEnum.male else 0,
                'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
                'comorbidity_count': patient.clinical_indicators.comorbidity_count,
            }
            
            # Use the first patient's data as event indicator (for demonstration)
            # In real implementation, this would come from the patient records
            row['observed_time'] = np.random.exponential(2)  # Simulated survival time
            row['event'] = np.random.binomial(1, 0.5)  # Simulated event occurrence
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Perform Cox regression
        covariates = ['age', 'gender_encoded', 'baseline_risk_score', 'comorbidity_count']
        result = survival_models.cox_regression_analysis(
            df, 'observed_time', 'event', covariates
        )
        
        response = {
            "status": "success",
            "c_index": result.c_index,
            "hazard_ratios": result.hazard_ratios,
            "p_values": result.p_values,
            "confidence_intervals": {k: list(v) for k, v in result.confidence_intervals.items()},
            "model_stats": {
                "log_likelihood": result.log_likelihood,
                "baseline_survival_points": len(result.baseline_survival),
                "total_patients": len(request.patient_data)
            }
        }
        
        logger.info("Cox regression completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error in Cox regression: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/causal-inference/ate-estimation")
async def estimate_ate(request: CausalAnalysisRequest):
    """Estimate Average Treatment Effect (ATE) for treatment-outcome relationship"""
    logger.info(f"Estimating ATE for {len(request.patient_data)} patients")
    
    try:
        # Prepare data from patient inputs
        data_rows = []
        for patient in request.patient_data:
            row = {
                'age': patient.demographics.age,
                'gender_encoded': 1 if patient.demographics.sex == SexEnum.male else 0,
                'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
                'comorbidity_count': patient.clinical_indicators.comorbidity_count,
            }
            
            # For demonstration: assign treatment randomly based on risk
            # In real implementation, this would come from actual patient records
            row['treatment'] = 1 if patient.clinical_indicators.baseline_risk_score > 0.5 else 0
            row['outcome'] = np.random.binomial(1, 0.1 + 0.1 * row['treatment'])  # Treatment effect
            
            # Add confounders if specified
            for confounder in request.confounders:
                if confounder not in row:
                    row[confounder] = np.random.random()  # Random value for demo
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Estimate ATE
        causal_results = causal_models.estimate_ate(
            df,
            df['treatment'].values,
            df['outcome'].values,
            [col for col in df.columns if col not in ['treatment', 'outcome']]
        )
        
        response = {
            "status": "success",
            "treatment_variable": request.treatment_variable,
            "outcome_variable": request.outcome_variable,
            "ate_estimates": causal_results,
            "total_patients": len(request.patient_data),
            "confounders_considered": request.confounders
        }
        
        logger.info("ATE estimation completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error in ATE estimation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cohorts/compare")
async def compare_cohorts(request: CohortAnalysisRequest):
    """Compare two patient cohorts on multiple outcomes"""
    logger.info(f"Comparing cohorts: {len(request.patient_cohort)} vs {len(request.comparator_cohort or [])} patients")
    
    try:
        # Process primary cohort
        cohort1_data = []
        for patient in request.patient_cohort:
            row = {
                'age': patient.demographics.age,
                'gender': patient.demographics.sex.value,
                'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
                'comorbidity_count': patient.clinical_indicators.comorbidity_count,
            }
            cohort1_data.append(row)
        
        # Process comparator cohort if provided
        cohort2_data = []
        if request.comparator_cohort:
            for patient in request.comparator_cohort:
                row = {
                    'age': patient.demographics.age,
                    'gender': patient.demographics.sex.value,
                    'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
                    'comorbidity_count': patient.clinical_indicators.comorbidity_count,
                }
                cohort2_data.append(row)
        
        # For demonstration, generate synthetic outcomes
        cohort1_outcomes = {}
        cohort2_outcomes = {}
        
        for outcome in request.outcome_variables or ['mortality', 'readmission']:
            n1 = len(cohort1_data)
            n2 = len(cohort2_data)
            
            # Generate random outcomes for demonstration
            cohort1_outcomes[outcome] = np.random.binomial(1, 0.15, n1).tolist()
            if n2 > 0:
                cohort2_outcomes[outcome] = np.random.binomial(1, 0.20 if outcome == 'mortality' else 0.18, n2).tolist()
        
        response = {
            "status": "success",
            "cohort_comparison": {
                "cohort_1": {
                    "size": len(cohort1_data),
                    "characteristics": {
                        "mean_age": float(np.mean([p['age'] for p in cohort1_data])) if cohort1_data else 0,
                        "gender_distribution": {},  # Would calculate actual distribution in real implementation
                    },
                    "outcomes": cohort1_outcomes
                },
                "cohort_2": {
                    "size": len(cohort2_data),
                    "characteristics": {
                        "mean_age": float(np.mean([p['age'] for p in cohort2_data])) if cohort2_data else 0,
                        "gender_distribution": {}
                    } if cohort2_data else None,
                    "outcomes": cohort2_outcomes if cohort2_data else {}
                },
                "comparison_metrics": {
                    "cohorts_have_difference": True,  # Placeholder for actual statistical tests
                    "statistical_significance": 0.05  # Placeholder p-value
                }
            }
        }
        
        logger.info("Cohort comparison completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error in cohort comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_ready": {
            "survival_analysis": True,
            "causal_inference": True,
            "enhanced_outcomes": True
        }
    }


# Additional endpoints for clinical decision support
@app.post("/api/evidence/search")
async def search_evidence(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    filters: str = Query('{}', description="JSON filters for search")
):
    """Search medical evidence in the knowledge graph"""
    try:
        # Parse filters
        parsed_filters = json.loads(filters)
        
        # Simulate evidence search results
        mock_evidence = [
            {
                "id": f"evidence_{i}",
                "title": f"Medical Evidence Title {i}: {query}",
                "abstract": f"Abstract for evidence related to {query}. This is a synthetic result.",
                "source": ["PubMed", "ClinicalTrial", "Guideline"][i % 3],
                "pub_date": f"202{2 + i % 3}-{1 + i % 12:02d}-15",
                "relevance_score": round(1.0 - (i * 0.1), 2),
                "entities": {
                    "conditions": [query.replace(" ", "_").lower()] if query else [],
                    "interventions": ["treatment", "therapy", "intervention"],
                    "outcomes": ["mortality", "survival", "effectiveness"]
                }
            }
            for i in range(min(limit, 10))  # Up to 10 mock results
        ]
        
        return {
            "status": "success",
            "query": query,
            "filters": parsed_filters,
            "results": mock_evidence[:limit],
            "total_results": len(mock_evidence)
        }
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in filters parameter")
    except Exception as e:
        logger.error(f"Error in evidence search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)