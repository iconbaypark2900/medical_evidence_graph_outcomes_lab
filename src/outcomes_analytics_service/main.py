"""
Main entry point for the Outcomes Analytics Service
"""
import asyncio
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import io
import base64


@dataclass
class CohortDefinition:
    """Defines a patient cohort for analysis"""
    id: str
    name: str
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    follow_up_period: int  # in days
    outcome_definition: Dict[str, Any]


@dataclass
class SurvivalResult:
    """Represents survival analysis results"""
    time_points: List[float]
    survival_probabilities: List[float]
    confidence_intervals: List[tuple]
    p_value: float


class OutcomesAnalyticsService:
    def __init__(self):
        """
        Initialize the outcomes analytics service
        """
        self.cohorts = {}
        self.population_data = pd.DataFrame()
    
    def create_cohort(self, definition: CohortDefinition):
        """
        Create a cohort based on definition
        """
        print(f"Creating cohort: {definition.name}")
        
        # In a real implementation, this would query a database to extract
        # patient data based on inclusion/exclusion criteria
        # For demo purposes, generating synthetic data
        
        # Create synthetic patient data
        n_patients = 1000
        np.random.seed(42)  # For reproducible results
        
        patient_ids = [f"pt_{i}" for i in range(n_patients)]
        ages = np.random.normal(55, 15, n_patients)  # Mean age 55, std 15
        ages = np.clip(ages, 18, 90)  # Clip to reasonable range
        genders = np.random.choice(['M', 'F'], n_patients, p=[0.45, 0.55])
        
        # Generate time-to-event data (survival times)
        # For demonstration, using exponential distribution
        # Base hazard rate depends on age and gender
        base_hazard = 0.001
        age_factor = (ages - 50) / 100  # Higher hazard for older patients
        gender_factor = np.where(genders == 'M', 0.1, -0.1)  # Males have slightly higher risk
        
        hazards = base_hazard * np.exp(age_factor + gender_factor)
        survival_times = np.random.exponential(1 / hazards)
        
        # Apply follow-up period constraint
        censored = survival_times > definition.follow_up_period
        observed_times = np.where(censored, definition.follow_up_period, survival_times)
        
        # Create outcome status (1 for event, 0 for censored)
        event_status = np.where(censored, 0, 1)
        
        # Create the dataset
        self.population_data = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'survival_time': observed_times,
            'event_status': event_status,
            'cohort_id': definition.id
        })
        
        self.cohorts[definition.id] = definition
        print(f"Created cohort with {len(self.population_data)} patients")
    
    def load_cohort_definition(self, cohort_id: str) -> CohortDefinition:
        """
        Load cohort definition by ID
        """
        if cohort_id not in self.cohorts:
            raise ValueError(f"Cohort {cohort_id} not found")
        return self.cohorts[cohort_id]
    
    def extract_population_data(self, cohort_definition: CohortDefinition) -> pd.DataFrame:
        """
        Extract population data based on cohort definition
        """
        print(f"Extracting population data for cohort: {cohort_definition.name}")
        
        # In a real implementation, this would apply inclusion/exclusion criteria
        # to a real patient database
        # For demo, returning the synthetic data we generated
        cohort_data = self.population_data[self.population_data['cohort_id'] == cohort_definition.id].copy()
        
        print(f"Extracted {len(cohort_data)} patients for analysis")
        return cohort_data
    
    def run_survival_analysis(self, population_data: pd.DataFrame) -> SurvivalResult:
        """
        Run survival analysis (Kaplan-Meier estimator)
        """
        print("Running survival analysis using Kaplan-Meier estimator...")
        
        # Sort by survival time
        sorted_data = population_data.sort_values('survival_time')
        
        # Get unique time points where events occur
        event_times = sorted_data[sorted_data['event_status'] == 1]['survival_time'].unique()
        
        # Calculate survival probabilities using Kaplan-Meier
        n_at_risk = len(sorted_data)
        survival_probs = []
        time_points = []
        conf_intervals = []
        
        for t in event_times:
            # Calculate number of events at time t
            events_at_t = len(sorted_data[(sorted_data['survival_time'] == t) & 
                                         (sorted_data['event_status'] == 1)])
            
            # Calculate number at risk just before time t
            n_at_risk = len(sorted_data[sorted_data['survival_time'] >= t])
            
            # Calculate survival probability
            if n_at_risk > 0:
                prob = events_at_t / n_at_risk
                survival_prob = (1 - prob) if survival_probs else (1 - prob)
                if survival_probs:
                    survival_prob = survival_probs[-1] * (1 - prob)
                
                survival_probs.append(survival_prob)
                time_points.append(t)
                
                # Calculate confidence interval (simplified)
                se = np.sqrt((survival_prob * (1 - survival_prob)) / n_at_risk)
                ci_lower = max(0, survival_prob - 1.96 * se)
                ci_upper = min(1, survival_prob + 1.96 * se)
                conf_intervals.append((ci_lower, ci_upper))
        
        # Generate a p-value for demonstration (in reality, this would come from 
        # statistical tests comparing groups)
        p_value = np.random.uniform(0.001, 0.1)  # Random p-value for demo
        
        result = SurvivalResult(
            time_points=time_points,
            survival_probabilities=survival_probs,
            confidence_intervals=conf_intervals,
            p_value=p_value
        )
        
        print(f"Survival analysis completed with {len(time_points)} time points")
        return result
    
    def run_comparative_effectiveness_analysis(self, 
                                             population_data: pd.DataFrame,
                                             group_col: str = 'treatment_group') -> Dict[str, Any]:
        """
        Run comparative effectiveness analysis between groups
        """
        print(f"Running comparative effectiveness analysis by {group_col}...")
        
        # In a real implementation, this would perform statistical tests
        # comparing outcomes between different groups
        # For demo, creating synthetic groups and comparing
        
        # Create synthetic treatment groups if not already present
        if group_col not in population_data.columns:
            # Simulate treatment assignment
            np.random.seed(123)
            population_data[group_col] = np.random.choice(
                ['treatment_A', 'treatment_B', 'usual_care'], 
                len(population_data), 
                p=[0.33, 0.33, 0.34]
            )
        
        # Calculate outcomes by group
        group_outcomes = {}
        for group in population_data[group_col].unique():
            group_data = population_data[population_data[group_col] == group]
            event_rate = group_data['event_status'].mean()
            median_survival = group_data['survival_time'].median()
            
            group_outcomes[group] = {
                'event_rate': event_rate,
                'median_survival': median_survival,
                'n_patients': len(group_data)
            }
        
        # Generate hazard ratio (simplified)
        # Comparing first group to others
        if len(group_outcomes) > 1:
            groups = list(group_outcomes.keys())
            baseline = group_outcomes[groups[0]]
            comparison = group_outcomes[groups[1]]
            
            hazard_ratio = comparison['event_rate'] / baseline['event_rate'] if baseline['event_rate'] > 0 else 1.0
        else:
            hazard_ratio = 1.0
        
        results = {
            'group_outcomes': group_outcomes,
            'hazard_ratio': hazard_ratio,
            'p_value': np.random.uniform(0.001, 0.05),  # Random p-value for demo
            'number_needed_to_treat': int(1 / abs(group_outcomes[groups[0]]['event_rate'] - 
                                                 group_outcomes[groups[1]]['event_rate']) if 
                                        len(groups) > 1 and 
                                        abs(group_outcomes[groups[0]]['event_rate'] - 
                                            group_outcomes[groups[1]]['event_rate']) > 0 else 100)
        }
        
        print(f"Comparative analysis completed for {len(group_outcomes)} groups")
        return results
    
    def run_subgroup_analysis(self, 
                            population_data: pd.DataFrame, 
                            subgroup_col: str = 'gender') -> Dict[str, Any]:
        """
        Run subgroup analysis
        """
        print(f"Running subgroup analysis by {subgroup_col}...")
        
        # Calculate outcomes by subgroup
        subgroup_results = {}
        for subgroup in population_data[subgroup_col].unique():
            subgroup_data = population_data[population_data[subgroup_col] == subgroup]
            event_rate = subgroup_data['event_status'].mean()
            
            subgroup_results[subgroup] = {
                'event_rate': event_rate,
                'n_patients': len(subgroup_data)
            }
        
        return subgroup_results
    
    def store_and_return_metrics(self, 
                               survival_result: SurvivalResult, 
                               comparative_results: Dict[str, Any],
                               cohort_definition: CohortDefinition) -> Dict[str, Any]:
        """
        Store results and return formatted metrics
        """
        print("Storing and formatting results...")
        
        # Create metrics summary
        metrics = {
            "cohort_id": cohort_definition.id,
            "cohort_name": cohort_definition.name,
            "total_patients": len(self.population_data),
            "survival_analysis": {
                "time_points": survival_result.time_points,
                "survival_probabilities": survival_result.survival_probabilities,
                "p_value": survival_result.p_value,
                "n_time_points": len(survival_result.time_points)
            },
            "comparative_effectiveness": comparative_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        print("Results stored and formatted successfully")
        return metrics


async def main():
    """
    Main function to run the outcomes analytics service
    """
    print("Starting Outcomes Analytics Service...")
    
    service = OutcomesAnalyticsService()
    
    # Define a sample cohort
    cohort_def = CohortDefinition(
        id="diabetes_cohort_1",
        name="Type 2 Diabetes Patients",
        inclusion_criteria={
            "diagnosis": "type_2_diabetes",
            "age_min": 18,
            "age_max": 80
        },
        exclusion_criteria={
            "end_stage_renal_disease": True,
            "pregnancy": True
        },
        follow_up_period=1825,  # 5 years in days
        outcome_definition={
            "primary": "cardiovascular_event",
            "secondary": ["all_cause_mortality", "diabetes_complications"]
        }
    )
    
    # Create the cohort
    service.create_cohort(cohort_def)
    
    # Load cohort definition
    loaded_def = service.load_cohort_definition("diabetes_cohort_1")
    
    # Extract population data
    pop_data = service.extract_population_data(loaded_def)
    
    # Run survival analysis
    survival_result = service.run_survival_analysis(pop_data)
    
    # Run comparative effectiveness analysis
    comp_effect_result = service.run_comparative_effectiveness_analysis(pop_data)
    
    # Run subgroup analysis
    subgroup_result = service.run_subgroup_analysis(pop_data)
    
    # Store and return metrics
    metrics = service.store_and_return_metrics(survival_result, comp_effect_result, loaded_def)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OUTCOMES ANALYTICS SUMMARY")
    print(f"{'='*60}")
    print(f"Cohort: {metrics['cohort_name']}")
    print(f"Total patients: {metrics['total_patients']}")
    print(f"Survival analysis time points: {metrics['survival_analysis']['n_time_points']}")
    print(f"Survival analysis p-value: {metrics['survival_analysis']['p_value']:.4f}")
    print(f"Comparative effectiveness - Hazard ratio: {metrics['comparative_effectiveness']['hazard_ratio']:.3f}")
    print(f"Number needed to treat: {metrics['comparative_effectiveness']['number_needed_to_treat']}")
    
    print(f"\nGroup outcomes:")
    for group, outcome in metrics['comparative_effectiveness']['group_outcomes'].items():
        print(f"  {group}: Event rate: {outcome['event_rate']:.3f}, "
              f"Median survival: {outcome['median_survival']:.1f} days, "
              f"Patients: {outcome['n_patients']}")
    
    print(f"\nSubgroup analysis:")
    for sub, result in subgroup_result.items():
        print(f"  {sub}: Event rate: {result['event_rate']:.3f}, Patients: {result['n_patients']}")
    
    print(f"{'='*60}")
    print("Outcomes Analytics Service completed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())