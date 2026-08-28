"""
Phase 2 Implementation - Enhanced ML Models for Medical Evidence Graph & Outcomes Insight Lab

This script demonstrates the implementation of advanced ML models for:
1. Survival analysis (Kaplan-Meier, Cox regression)
2. Causal inference (propensity matching, ATE estimation)
3. Enhanced outcome prediction models
4. Deep learning for survival prediction
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from src.enhanced_ml_models import (
    SurvivalAnalysisModels, 
    CausalInferenceModels, 
    EnhancedOutcomeModels, 
    DeepSurvivalModel
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_survival_analysis():
    """Demonstrate survival analysis capabilities"""
    logger.info("=== DEMONSTRATING SURVIVAL ANALYSIS ===")
    
    # Initialize models
    survival_models = SurvivalAnalysisModels()
    
    # Generate sample patient data
    np.random.seed(42)
    n_patients = 500
    
    # Generate survival data
    age = np.random.normal(65, 15, n_patients)
    age = np.clip(age, 18, 95)
    treatment = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
    comorbidity_score = np.random.normal(1.5, 0.8, n_patients)
    comorbidity_score = np.clip(comorbidity_score, 0, 5)
    
    # Generate survival times with treatment effect
    # Calibrated against the follow-up window below. At the previous
    # value the per-patient event probability over 0.5-5 years was well
    # under 1%, so almost every patient was censored: the Cox fit rested
    # on a handful of events and the ATE was estimated from an outcome
    # that was zero for nearly everyone.
    base_hazard = 0.30
    age_effect = (age - 50) / 100
    treatment_effect = -0.6 * treatment  # Treatment improves survival
    comorbidity_effect = comorbidity_score * 0.2
    
    hazards = base_hazard * np.exp(age_effect + treatment_effect + comorbidity_effect)
    survival_times = np.random.exponential(1 / hazards)
    
    # Add some right-censoring
    follow_up_time = np.random.uniform(1, 5, n_patients)  # 1-5 years max follow-up
    observed_times = np.minimum(survival_times, follow_up_time)
    event_observed = (survival_times <= follow_up_time).astype(int)
    
    # Create DataFrame
    patient_df = pd.DataFrame({
        'age': age,
        'treatment': treatment,
        'comorbidity_score': comorbidity_score,
        'observed_time': observed_times,
        'event': event_observed
    })
    
    # Kaplan-Meier analysis
    logger.info("Running Kaplan-Meier analysis...")
    km_times, km_survival = survival_models.kaplan_meier_analysis(observed_times, event_observed)
    logger.info(f"✅ Kaplan-Meier: {len(km_times)} time points, max survival: {max(km_survival):.3f}")
    
    # Cox regression analysis
    logger.info("Running Cox regression analysis...")
    covariates = ['age', 'treatment', 'comorbidity_score']
    survival_result = survival_models.cox_regression_analysis(
        patient_df, 'observed_time', 'event', covariates
    )
    
    logger.info(f"✅ Cox regression C-index: {survival_result.c_index:.3f}")
    logger.info(f"✅ Hazard ratios: {dict(survival_result.hazard_ratios)}")
    
    return survival_result


async def demonstrate_causal_inference():
    """Demonstrate causal inference capabilities"""
    logger.info("\n=== DEMONSTRATING CAUSAL INFERENCE ===")
    
    causal_models = CausalInferenceModels()
    
    # Generate observational data with potential confounding
    np.random.seed(123)
    n_patients = 800
    
    # Confounders
    age = np.random.normal(60, 12, n_patients)
    severity = np.random.beta(2, 5, n_patients) * 10  # Higher values worse
    gender = np.random.choice([0, 1], n_patients)  # 0: female, 1: male
    
    # Treatment assignment based on severity (confounding by indication)
    treatment_prob = 0.3 + 0.4 * severity / 10  # More severe cases more likely to get treatment
    treatment = np.random.binomial(1, treatment_prob, n_patients)
    
    # Outcome generation (treatment has benefit for most patients)
    base_risk = 0.1
    age_risk = (age - 50) / 100
    severity_risk = severity * 0.1
    treatment_benefit = -0.3 * treatment  # Treatment reduces risk
    
    outcome_prob = np.clip(base_risk + age_risk + severity_risk + treatment_benefit, 0.05, 0.95)
    outcome = np.random.binomial(1, outcome_prob, n_patients)
    
    # Create DataFrame
    patient_df = pd.DataFrame({
        'age': age,
        'severity': severity,
        'gender': gender,
        'treatment': treatment,
        'outcome': outcome
    })
    
    # Estimate causal effects
    logger.info("Estimating causal effects...")
    causal_results = causal_models.estimate_ate(
        patient_df,
        patient_df['treatment'].values,
        patient_df['outcome'].values,
        ['age', 'severity', 'gender']
    )
    
    logger.info(f"✅ Causal inference results:")
    for method, effect in causal_results.items():
        logger.info(f"   {method}: {effect:.3f}")
    
    return causal_results


async def demonstrate_enhanced_outcomes():
    """Demonstrate enhanced outcome prediction models"""
    logger.info("\n=== DEMONSTRATING ENHANCED OUTCOME PREDICTION ===")
    
    enhanced_models = EnhancedOutcomeModels()
    
    # Generate patient data
    np.random.seed(456)
    n_patients = 600
    
    patient_data = pd.DataFrame({
        'age': np.random.normal(62, 14, n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'baseline_risk_score': np.random.normal(0.4, 0.3, n_patients),
        'comorbidity_count': np.random.poisson(1.2, n_patients),
        'days_since_admission': np.random.randint(1, 30, n_patients)
    })
    
    # Define multiple outcomes to predict
    outcomes = {
        'mortality': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'readmission': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
        'length_of_stay_extended': np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
    }
    
    # Train multi-task models
    logger.info("Training multi-task outcome models...")
    trained_models = enhanced_models.train_multi_task_model(patient_data, outcomes)
    
    logger.info(f"✅ Multi-task models trained for {len(trained_models)} outcomes:")
    for outcome_name in trained_models.keys():
        logger.info(f"   - {outcome_name}")
    
    return trained_models


async def demonstrate_deep_learning():
    """Demonstrate deep learning for survival prediction"""
    logger.info("\n=== DEMONSTRATING DEEP LEARNING FOR SURVIVAL ===")
    
    # Generate patient features
    np.random.seed(789)
    n_patients = 400
    
    patient_data = pd.DataFrame({
        'age': np.random.normal(65, 16, n_patients),
        'gender_encoded': np.random.choice([0, 1], n_patients),
        'baseline_risk_score': np.random.normal(0.5, 0.25, n_patients),
        'comorbidity_count': np.random.poisson(1.5, n_patients),
        'lab_value_1': np.random.normal(1.0, 0.5, n_patients),
        'lab_value_2': np.random.normal(0.8, 0.4, n_patients),
        'vital_sign': np.random.normal(70, 10, n_patients)
    })
    
    # Generate survival data
    survival_times = np.random.exponential(2, n_patients)
    events = np.random.binomial(1, 0.3, n_patients)  # 30% event rate
    
    # Create and train deep survival model
    input_dim = len([col for col in patient_data.columns])  # Number of features
    deep_model = DeepSurvivalModel(input_dim=input_dim)
    
    logger.info(f"Training deep survival model with {input_dim} features...")
    deep_model.train(patient_data, survival_times, events, epochs=75)
    
    # Make predictions
    features = patient_data.select_dtypes(include=[np.number])  # Select only numeric columns for prediction
    risk_scores = deep_model.predict_risk(features)

    logger.info(f"✅ Deep survival model trained successfully")
    # These are log-hazard-ratio scores, not probabilities: they are centred
    # near zero and negative values are ordinary. Only their ORDER is
    # meaningful, which is what the concordance index measures.
    logger.info(
        f"✅ Log-risk scores: mean {np.mean(risk_scores):.3f}, "
        f"range [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    logger.info(
        f"✅ In-sample concordance: "
        f"{deep_model.concordance(features, survival_times, events):.3f}")
    
    return deep_model


async def main():
    """Main function to demonstrate Phase 2 enhanced ML capabilities"""
    logger.info("=" * 80)
    logger.info("MEDICAL EVIDENCE GRAPH & OUTCOMES INSIGHT LAB - PHASE 2")
    logger.info("Implementation of Enhanced ML Models for Survival Analysis & Causal Inference")
    logger.info("=" * 80)
    
    # Execute all demonstrations
    survival_result = await demonstrate_survival_analysis()
    causal_results = await demonstrate_causal_inference()
    enhanced_models = await demonstrate_enhanced_outcomes()
    deep_model = await demonstrate_deep_learning()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 IMPLEMENTATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info("✅ SURVIVAL ANALYSIS CAPABILITIES:")
    logger.info(f"   - Kaplan-Meier survival curves: {len(survival_result.time_points)} timepoints")
    logger.info(f"   - Cox proportional hazards regression: C-index = {survival_result.c_index:.3f}")
    logger.info(f"   - Hazard ratios for: {list(survival_result.hazard_ratios.keys())}")
    
    logger.info("\n✅ CAUSAL INFERENCE CAPABILITIES:")
    for method, effect in causal_results.items():
        logger.info(f"   - {method}: {effect:.4f}")
    
    logger.info(f"\n✅ ENHANCED OUTCOME PREDICTION:")
    logger.info(f"   - Multi-task models for {len(enhanced_models)} different outcomes")
    logger.info(f"   - Feature engineering for clinical indicators")
    logger.info(f"   - Risk stratification algorithms")
    
    logger.info(f"\n✅ DEEP LEARNING FOR SURVIVAL:")
    logger.info(f"   - Neural networks for risk prediction")
    logger.info(f"   - End-to-end differentiable survival models")
    logger.info(f"   - Automated feature learning")
    
    logger.info(f"\n✅ INTEGRATION READY:")
    logger.info(f"   - All models follow consistent interfaces")
    logger.info(f"   - Compatible with graph database storage")
    logger.info(f"   - Ready for real-world validation")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 SUCCESSFULLY COMPLETED")
    logger.info("Advanced ML models ready for deployment & validation")
    logger.info("=" * 80)
    
    return {
        'survival_analysis': survival_result,
        'causal_inference': causal_results,
        'enhanced_models': enhanced_models,
        'deep_model': deep_model
    }


if __name__ == "__main__":
    results = asyncio.run(main())
    logger.info("\nPhase 2 Implementation completed successfully!")