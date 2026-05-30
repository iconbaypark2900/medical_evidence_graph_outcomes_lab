"""
Debug script to check data format for Cox regression
"""
import pandas as pd
import numpy as np

# Create the same synthetic data as in enhanced_ml_models.py
np.random.seed(42)
n_patients = 1000

# Generate synthetic features
age = np.random.normal(65, 15, n_patients)
age = np.clip(age, 18, 95)  # Realistic age range
gender = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
treatment = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])  # 40% receive treatment

# Generate survival times (exponential-like with treatment effect)
base_hazard = 0.001
age_effect = (age - 50) / 100
treatment_effect = -0.5 if treatment.sum() > 0 else 0  # Treatment reduces risk

hazards = base_hazard * np.exp(age_effect + treatment_effect * treatment)
survival_times = np.random.exponential(1 / hazards)

# Add some right-censoring (patients lost to follow-up)
follow_up_time = np.random.uniform(0.5, 5, n_patients)  # 6 months to 5 years max follow-up
observed_times = np.minimum(survival_times, follow_up_time)
event_observed = (survival_times <= follow_up_time).astype(int)

# Create DataFrame
patient_data = pd.DataFrame({
    'age': age,
    'gender': gender,
    'treatment': treatment,
    'observed_time': observed_times,
    'event': event_observed,
    'baseline_risk_score': np.random.normal(0.5, 0.2, n_patients),
    'comorbidity_count': np.random.poisson(1.5, n_patients)
})

print("Data types:")
print(patient_data.dtypes)
print("\nSample data:")
print(patient_data.head())
print(f"\nCovariates: ['age', 'treatment', 'baseline_risk_score']")

# Check for gender column - this is likely the issue
print(f"\nGender column type: {patient_data['gender'].dtype}")
print(f"Gender unique values: {patient_data['gender'].unique()}")

# The lifelines Cox model requires numeric inputs
# We need to encode categorical variables
from sklearn.preprocessing import LabelEncoder

patient_data_encoded = patient_data.copy()
le = LabelEncoder()
patient_data_encoded['gender_encoded'] = le.fit_transform(patient_data['gender'])

print(f"\nGender encoded: {patient_data['gender'].unique()} -> {patient_data_encoded['gender_encoded'].unique()}")

# Show what we'd pass to Cox model
covariates = ['age', 'treatment', 'baseline_risk_score', 'gender_encoded']
print(f"\nCovariates with encoded gender: {covariates}")
print(f"Encoded data types:")
print(patient_data_encoded[covariates].dtypes)