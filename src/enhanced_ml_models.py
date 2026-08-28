"""
Enhanced ML models for survival analysis and causal inference
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.stats import chi2
import lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SurvivalAnalysisResult:
    """Result from survival analysis"""
    hazard_ratios: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    c_index: float
    log_likelihood: float
    baseline_survival: List[float]
    time_points: List[float]
    survival_probabilities: List[float]


class SurvivalAnalysisModels:
    """Advanced survival analysis models using lifelines"""
    
    def __init__(self):
        self.kaplan_meier = KaplanMeierFitter()
        self.cox_model = CoxPHFitter()
        self.fitted = False
    
    def kaplan_meier_analysis(self, duration: np.ndarray, event_observed: np.ndarray) -> Tuple[List[float], List[float]]:
        """Perform Kaplan-Meier survival analysis"""
        try:
            self.kaplan_meier.fit(duration, event_observed)
            
            # Extract survival probabilities and time points
            survival_probs = self.kaplan_meier.survival_function_['KM_estimate'].values
            time_points = self.kaplan_meier.survival_function_.index.values
            
            return time_points.tolist(), survival_probs.tolist()
        except Exception as e:
            # Do NOT return ([], []) here. A caller cannot distinguish "the fit
            # failed" from "this cohort has no events", and the second is a
            # clinical finding while the first is a bug. An empty survival curve
            # renders as a blank chart, which reads as a result rather than an
            # error.
            logger.error(f"Error in Kaplan-Meier analysis: {e}")
            raise RuntimeError(
                f"Kaplan-Meier fit failed on {len(duration)} observations: {e}"
            ) from e
    
    def cox_regression_analysis(self, data: pd.DataFrame, duration_col: str, event_col: str, 
                               covariate_cols: List[str]) -> SurvivalAnalysisResult:
        """Perform Cox proportional hazards regression.

        Raises:
            KeyError: a requested column is not in `data` (caller error).
            ValueError: no usable rows remain after cleaning (data error).
            RuntimeError: the model itself failed to fit (estimator error).

        Note that `c_index` is computed on the same rows the model was fitted
        on. It is an in-sample fit statistic, not an estimate of how the model
        will discriminate on unseen patients, and must not be reported as one.
        """
        # Validation runs OUTSIDE the try below. A missing column is a mistake
        # at the call site, not a failure of the estimator, and the caller can
        # only act on that distinction if the exception type survives.
        required_cols = [duration_col, event_col] + covariate_cols
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            # Do NOT return an empty SurvivalAnalysisResult. A result object
            # with no hazard ratios and c_index=0.0 is indistinguishable from
            # a model that genuinely found no predictive signal — and 0.5 is
            # the c-index of a coin flip, so 0.0 is not even a neutral value
            # to fabricate.
            logger.error(f"Missing columns for Cox regression: {missing_cols}")
            raise KeyError(
                f"Cox regression requires columns {missing_cols}, which are "
                f"not in the supplied data (has: {list(data.columns)})")

        model_data = data[required_cols].copy().dropna()
        if len(model_data) == 0:
            raise ValueError(
                f"No rows remain after dropping missing values across "
                f"{required_cols}; {len(data)} rows were supplied")

        # Encode any non-numeric covariates; lifelines requires numeric input.
        for col in covariate_cols:
            if model_data[col].dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                model_data[col] = le.fit_transform(model_data[col].astype(str))
            model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

        model_data = model_data.dropna()
        if len(model_data) == 0:
            raise ValueError(
                f"No rows remain after coercing {covariate_cols} to numeric; "
                f"check for non-numeric values in the covariates")

        try:
            self.cox_model.fit(model_data, duration_col=duration_col, event_col=event_col)
        except Exception as e:
            # A SurvivalAnalysisResult with no hazard ratios and c_index=0.0 is
            # not an empty result — it is a claim: "no covariate predicts
            # survival, and the model discriminates worse than a coin flip"
            # (a c-index of 0.5 is chance; 0.0 is perfectly inverted). Returning
            # that because the fit crashed turns a bug into a clinical finding.
            logger.error(f"Error in Cox regression analysis: {e}")
            raise RuntimeError(
                f"Cox regression failed on {len(model_data)} rows with covariates "
                f"{covariate_cols}: {e}") from e

        summary = self.cox_model.summary
        # Effect estimates and their intervals both come from `summary`.
        # lifelines' `confidence_intervals_` holds intervals on the log-hazard
        # scale under the names '95% lower-bound'/'95% upper-bound'; reading it
        # with the exp(coef) names silently misses and used to yield (0, 0) for
        # every covariate — the claim "the hazard ratio is exactly zero, with
        # certainty". Fail loudly instead of substituting a number.
        required_summary_cols = ['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']
        missing_summary_cols = [c for c in required_summary_cols if c not in summary.columns]
        if missing_summary_cols:
            raise RuntimeError(
                f"lifelines {lifelines.__version__} summary is missing "
                f"{missing_summary_cols}; cannot report hazard ratios without "
                f"their confidence intervals")

        hazard_ratios = {}
        p_values = {}
        confidence_intervals = {}
        for param in self.cox_model.params_.index:
            hazard_ratios[param] = float(summary.loc[param, 'exp(coef)'])
            p_values[param] = float(summary.loc[param, 'p'])
            confidence_intervals[param] = (
                float(summary.loc[param, 'exp(coef) lower 95%']),
                float(summary.loc[param, 'exp(coef) upper 95%']),
            )

        c_index = self.cox_model.score(model_data, scoring_method="concordance_index")
        log_likelihood = self.cox_model.log_likelihood_

        # Single-column frame; index by position rather than by a literal name
        # so a rename upstream surfaces as a shape error rather than as an
        # empty curve. The previous code asked for 'baseline survival_' (the
        # column is 'baseline survival') behind a bare `except:`, so every
        # successful fit silently returned an empty baseline curve.
        baseline = self.cox_model.baseline_survival_
        if baseline.shape[1] != 1:
            raise RuntimeError(
                f"Expected a single baseline survival column, got "
                f"{list(baseline.columns)}")
        baseline_survival = baseline.iloc[:, 0].to_numpy().tolist()
        time_points = baseline.index.to_numpy().tolist()

        result = SurvivalAnalysisResult(
            hazard_ratios=hazard_ratios,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            c_index=c_index,
            log_likelihood=log_likelihood,
            baseline_survival=baseline_survival,
            time_points=time_points,
            survival_probabilities=baseline_survival,
        )

        self.fitted = True
        return result


class CausalInferenceModels:
    """Causal inference models for treatment effect estimation"""
    
    def __init__(self):
        self.propensity_model = None
        self.outcome_models = {}
        self.scaler = StandardScaler()
        self.fitted = False
    
    def propensity_score_matching(self, X: pd.DataFrame, treatment: np.ndarray, 
                                covariates: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate propensity scores and perform matching"""
        try:
            # Prepare features
            X_features = X[covariates].values
            X_scaled = self.scaler.fit_transform(X_features)
            
            # Simple logistic regression-like propensity model
            # In practice, you'd use more sophisticated models
            from sklearn.linear_model import LogisticRegression
            self.propensity_model = LogisticRegression(random_state=42)
            self.propensity_model.fit(X_scaled, treatment)
            
            # Get propensity scores
            propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
            
            # Perform 1:1 nearest neighbor matching (simplified)
            treated_idx = np.where(treatment == 1)[0]
            control_idx = np.where(treatment == 0)[0]
            
            matched_treated = []
            matched_control = []
            
            for t_idx in treated_idx:
                # Find closest control match
                t_score = propensity_scores[t_idx]
                control_scores = propensity_scores[control_idx]
                closest_control = control_idx[np.argmin(np.abs(control_scores - t_score))]
                
                if np.abs(propensity_scores[t_idx] - propensity_scores[closest_control]) < 0.25:  # Caliper
                    matched_treated.append(t_idx)
                    matched_control.append(closest_control)
            
            # Create balanced datasets
            all_matched = matched_treated + matched_control
            matched_treatment = treatment[all_matched]
            matched_outcomes = None  # This would come from the outcome data
            
            self.fitted = True
            return np.array(all_matched), matched_treatment
            
        except Exception as e:
            # Returning empty arrays reads downstream as "no units could be
            # matched", a legitimate result of poor covariate overlap, rather
            # than as "the matching code failed".
            logger.error(f"Error in propensity score matching: {e}")
            raise RuntimeError(f"Propensity score matching failed: {e}") from e
    
    def estimate_ate(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray, 
                    covariates: List[str]) -> Dict[str, float]:
        """Estimate Average Treatment Effect using various methods"""
        try:
            results = {}
            
            # Method 1: Simple difference (unadjusted)
            treated_outcome = outcome[treatment == 1].mean()
            control_outcome = outcome[treatment == 0].mean()
            ate_simple = treated_outcome - control_outcome
            results['ate_simple'] = ate_simple
            
            # Method 2: Propensity score weighted
            matched_idx, matched_treatment = self.propensity_score_matching(X, treatment, covariates)
            if len(matched_idx) > 0:
                matched_outcome = outcome[matched_idx]
                treated_matched = matched_outcome[matched_treatment == 1].mean()
                control_matched = matched_outcome[matched_treatment == 0].mean()
                ate_matched = treated_matched - control_matched
                results['ate_matched'] = ate_matched
            
            # Method 3: Regression adjustment
            X_features = X[covariates + ['treatment']].copy()
            X_features['treatment'] = treatment
            from sklearn.linear_model import LinearRegression
            outcome_model = LinearRegression()
            outcome_model.fit(X_features, outcome)
            
            # Predict outcomes for treated and control scenarios
            X_pred_treated = X_features.copy()
            X_pred_treated['treatment'] = 1
            pred_treated = outcome_model.predict(X_pred_treated).mean()
            
            X_pred_control = X_features.copy()
            X_pred_control['treatment'] = 0
            pred_control = outcome_model.predict(X_pred_control).mean()
            
            ate_regression = pred_treated - pred_control
            results['ate_regression'] = ate_regression
            
            return results
            
        except Exception as e:
            # The most dangerous return in this module. An average treatment
            # effect of 0.0 is not a neutral placeholder — it is the assertion
            # "this treatment does nothing", which is a substantive clinical
            # claim and one of the two most consequential answers this function
            # can give. Reporting it because the estimator crashed inverts the
            # meaning of a failure into a finding.
            logger.error(f"Error in ATE estimation: {e}")
            raise RuntimeError(f"ATE estimation failed: {e}") from e


class TorchRiskModel(nn.Module):
    """Feed-forward network producing a single scalar per patient.

    `final_activation` decides what that scalar means:
      "sigmoid" -- a probability in [0, 1], for binary event prediction.
      "none"    -- an unbounded log-risk score, for proportional-hazards
                   models where the output is a log hazard ratio.

    Choosing wrongly is not cosmetic: squashing a log-risk score through a
    sigmoid caps the hazard ratio and destroys the scale the Cox partial
    likelihood is defined on.
    """

    SUPPORTED_ACTIVATIONS = ("sigmoid", "none")

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        final_activation: str = "sigmoid",
    ):
        super(TorchRiskModel, self).__init__()

        if final_activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"final_activation must be one of {self.SUPPORTED_ACTIVATIONS}, "
                f"got {final_activation!r}")

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        if final_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.final_activation = final_activation
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EnhancedOutcomeModels:
    """Enhanced outcome models combining multiple ML approaches"""
    
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        self.models = {}
        self.preprocessors = {}
        self.is_fitted = False
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def build_patient_features(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """Build features from patient data for ML models"""
        features = pd.DataFrame()
        
        # Demographics
        if 'age' in patient_data.columns:
            features['age'] = patient_data['age']
            features['age_squared'] = patient_data['age'] ** 2
            features['is_elderly'] = (patient_data['age'] > 65).astype(int)
        
        if 'gender' in patient_data.columns:
            # Encode gender appropriately
            features['is_male'] = (patient_data['gender'] == 'M').astype(int)
            features['is_female'] = (patient_data['gender'] == 'F').astype(int)
        
        # Clinical indicators
        if 'baseline_risk_score' in patient_data.columns:
            features['baseline_risk_score'] = patient_data['baseline_risk_score']
        
        # Severity was previously accepted by the API schema and then dropped
        # here, so a caller supplying it had no way to tell it was ignored.
        if 'severity_score' in patient_data.columns:
            features['severity_score'] = patient_data['severity_score']
        
        # Time-based features
        if 'days_since_admission' in patient_data.columns:
            features['days_since_admission'] = patient_data['days_since_admission']
        
        # Comorbidity counts
        # Match on comorbidity/condition only. Including 'score' here used to
        # sweep in baseline_risk_score, so `comorbidity_count` was the count
        # plus the risk score -- and the risk score was separately kept as its
        # own feature, double-counting it.
        comorbidity_cols = [
            col for col in patient_data.columns
            if 'comorbidity' in col.lower() or 'condition' in col.lower()
        ]
        if comorbidity_cols:
            features['comorbidity_count'] = patient_data[comorbidity_cols].sum(axis=1)
        
        # Lab values (if available)
        lab_cols = [col for col in patient_data.columns if 'lab' in col.lower() or 'value' in col.lower()]
        for col in lab_cols:
            features[f'{col}_clean'] = patient_data[col]
        
        # Vital signs (if available)
        vital_cols = [col for col in patient_data.columns if 'vital' in col.lower() or 'sign' in col.lower()]
        for col in vital_cols:
            features[f'{col}_clean'] = patient_data[col]
        
        # Fill NaN values
        features = features.fillna(features.median())
        
        return features
    
    def train_risk_prediction_model(self, X: pd.DataFrame, y: np.ndarray, model_type: str = 'classifier') -> Any:
        """Train a risk prediction model"""
        if model_type == 'classifier':
            model = self.rf_classifier
        elif model_type == 'regressor':
            model = self.rf_regressor
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X, y)
        return model
    
    def predict_risk(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Make risk predictions using trained model"""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1] 
        elif hasattr(model, "predict"):
            return model.predict(X)
        else:
            raise ValueError("Model doesn't have predict or predict_proba method")
    
    def train_multi_task_model(self, patient_data: pd.DataFrame, outcomes: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train multi-task model for predicting multiple outcomes"""
        models = {}
        
        # Build features
        X = self.build_patient_features(patient_data)
        
        # Train a separate model for each outcome
        for outcome_name, outcome_data in outcomes.items():
            if len(outcome_data) != len(X):
                logger.warning(f"Length mismatch for outcome {outcome_name}: data={len(outcome_data)}, features={len(X)}")
                continue
                
            if len(np.unique(outcome_data)) > 1:  # Only train if outcome has variation
                # Determine if this is a classification or regression problem
                unique_vals = np.unique(outcome_data)
                if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):  # Binary classification
                    model = self.train_risk_prediction_model(X, outcome_data, 'classifier')
                    model_type = 'classification'
                elif len(unique_vals) <= 20 and all(v.is_integer() for v in unique_vals):  # Discrete classification
                    model = self.train_risk_prediction_model(X, outcome_data.astype(int), 'classifier')
                    model_type = 'classification' 
                else:  # Continuous or multi-class outcome - regression
                    model = self.train_risk_prediction_model(X, outcome_data, 'regressor')
                    model_type = 'regression'
                
                models[outcome_name] = {
                    'model': model,
                    'type': model_type
                }
                logger.info(f"✅ Trained {model_type} model for outcome: {outcome_name}")
        
        self.models = models
        self.is_fitted = True
        return models


class DeepSurvivalModel:
    """Neural proportional-hazards model (DeepSurv).

    The network outputs a log-risk score and is trained against the Cox
    negative partial log-likelihood, so both the event times and the
    censoring indicator take part in the fit.

    This replaces an earlier implementation that discarded `survival_times`
    and minimised binary cross-entropy on the event indicator alone. That
    is an event classifier, not a survival model: it has no notion of when
    an event happened, and it treats a patient censored at three months as
    a confirmed non-event rather than as someone who was simply not
    observed long enough.

    Ties in event time use the Breslow approximation.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, lr: float = 0.001):
        self.input_dim = input_dim
        # No output squashing: a proportional-hazards risk score is a
        # log-hazard-ratio on (-inf, inf), not a probability.
        self.model = TorchRiskModel(input_dim, hidden_dims, final_activation="none")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.is_fitted = False

    @staticmethod
    def cox_partial_log_likelihood(
        log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """Negative Cox partial log-likelihood (Breslow), averaged over events.

        For each patient i who had the event, the contribution is
        `risk_i - logsumexp(risk_j for all j still at risk at time T_i)`.
        Sorting by descending duration makes the risk set a prefix, so the
        inner term is a cumulative logsumexp.
        """
        order = torch.argsort(durations, descending=True)
        log_risk = log_risk[order]
        events = events[order]

        log_risk_set_sum = torch.logcumsumexp(log_risk, dim=0)
        observed = events > 0
        if not bool(observed.any()):
            raise ValueError(
                "Cox partial likelihood is undefined when no event is observed; "
                "every patient in this batch is censored")

        return -(log_risk - log_risk_set_sum)[observed].mean()

    def prepare_data(
        self, features: pd.DataFrame, survival_times: np.ndarray, events: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a cohort to tensors. All three inputs are used."""
        if len(features) != len(survival_times) or len(features) != len(events):
            raise ValueError(
                f"Length mismatch: {len(features)} feature rows, "
                f"{len(survival_times)} survival times, {len(events)} events")

        return (
            torch.FloatTensor(features.to_numpy(dtype=np.float32)),
            torch.FloatTensor(np.asarray(survival_times, dtype=np.float32)),
            torch.FloatTensor(np.asarray(events, dtype=np.float32)),
        )

    def train(
        self,
        features: pd.DataFrame,
        survival_times: np.ndarray,
        events: np.ndarray,
        epochs: int = 100,
    ) -> List[float]:
        """Fit the model, returning the loss at each epoch."""
        X, durations, event_flags = self.prepare_data(features, survival_times, events)

        self.model.train()
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            log_risk = self.model(X).squeeze(-1)
            loss = self.cox_partial_log_likelihood(log_risk, durations, event_flags)
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.item()))
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, partial log-likelihood loss: {loss.item():.4f}")

        self.is_fitted = True
        return losses

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        """Log-risk scores: higher means a higher hazard.

        These are relative, on the log-hazard-ratio scale. They are not
        probabilities and must not be presented to a clinician as one.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        X = torch.FloatTensor(features.to_numpy(dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            return self.model(X).squeeze(-1).numpy()

    def predict_partial_hazard(self, features: pd.DataFrame) -> np.ndarray:
        """exp(log-risk): the hazard ratio relative to the baseline."""
        return np.exp(self.predict_risk(features))

    def concordance(
        self, features: pd.DataFrame, survival_times: np.ndarray, events: np.ndarray
    ) -> float:
        """Concordance index of the fitted risk scores.

        Higher risk should mean shorter survival, hence the negation. 0.5 is
        chance. Computing this on the rows the model was fitted on gives an
        in-sample figure; pass held-out data for an honest estimate.
        """
        return float(
            concordance_index(survival_times, -self.predict_risk(features), events)
        )


async def main():
    """Main function to demonstrate enhanced ML models"""
    logger.info("Starting Enhanced ML Models for Medical Evidence Graph & Outcomes Insight Lab...")
    
    # Initialize models
    survival_models = SurvivalAnalysisModels()
    causal_models = CausalInferenceModels()
    enhanced_models = EnhancedOutcomeModels()
    deep_survival = DeepSurvivalModel(input_dim=5)  # Example input dimension
    
    # Create synthetic patient data for demonstration
    np.random.seed(42)
    n_patients = 1000
    
    logger.info(f"Creating synthetic patient data for {n_patients} patients...")
    
    # Generate synthetic features
    age = np.random.normal(65, 15, n_patients)
    age = np.clip(age, 18, 95)  # Realistic age range
    gender = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])
    treatment = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])  # 40% receive treatment
    
    # Generate survival times (exponential-like with treatment effect)
    # Calibrated against the follow-up window below. At the previous
    # value the per-patient event probability over 0.5-5 years was well
    # under 1%, so almost every patient was censored: the Cox fit rested
    # on a handful of events and the ATE was estimated from an outcome
    # that was zero for nearly everyone.
    base_hazard = 0.30
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
    
    # Demonstrate survival analysis
    logger.info("Running Kaplan-Meier survival analysis...")
    km_times, km_survival = survival_models.kaplan_meier_analysis(observed_times, event_observed)
    logger.info(f"Kaplan-Meier: {len(km_times)} time points calculated")
    
    # Prepare data for Cox regression by encoding categorical variables
    logger.info("Preparing data for Cox regression (encoding categorical variables)...")
    cox_data = patient_data.copy()
    
    # Encode categorical variables (gender) to numeric
    from sklearn.preprocessing import LabelEncoder
    le_gender = LabelEncoder()
    cox_data['gender_encoded'] = le_gender.fit_transform(cox_data['gender'])
    
    # Use encoded variables for Cox regression
    covariates = ['age', 'treatment', 'baseline_risk_score', 'gender_encoded']
    
    logger.info("Running Cox regression analysis...")
    survival_result = survival_models.cox_regression_analysis(
        cox_data, 'observed_time', 'event', covariates
    )
    logger.info(f"Cox regression C-index: {survival_result.c_index:.3f}")
    logger.info(f"Hazard ratios: {survival_result.hazard_ratios}")
    
    # Demonstrate causal inference
    logger.info("Running causal inference analysis...")
    causal_results = causal_models.estimate_ate(
        patient_data, 
        patient_data['treatment'].values, 
        patient_data['event'].values, 
        ['age', 'baseline_risk_score', 'comorbidity_count']
    )
    logger.info(f"Estimated treatment effects: {causal_results}")
    
    # Demonstrate enhanced outcome models
    logger.info("Training enhanced outcome models...")
    outcomes = {
        'mortality': patient_data['event'].values,
        'readmission': np.random.choice([0, 1], n_patients, p=[0.7, 0.3])
    }
    
    enhanced_models.train_multi_task_model(patient_data, outcomes)
    logger.info("Enhanced models trained successfully")
    
    # Demonstrate deep survival model
    logger.info("Training deep survival model...")
    features = enhanced_models.build_patient_features(patient_data)
    
    # Fix the neural network input dimensions based on actual feature count
    n_features = features.shape[1]
    deep_survival_fixed = DeepSurvivalModel(input_dim=n_features)
    
    # For deep model, we'll use a subset for faster demonstration
    subset_size = min(200, len(features))
    deep_survival_fixed.train(
        features.iloc[:subset_size], 
        observed_times[:subset_size], 
        event_observed[:subset_size],
        epochs=50  # Reduced epochs for demo
    )
    logger.info("Deep survival model trained successfully")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED ML MODELS - PHASE 2 IMPLEMENTATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Survival Analysis: Kaplan-Meier and Cox regression")
    logger.info(f"✅ Causal Inference: Propensity matching and ATE estimation") 
    logger.info(f"✅ Enhanced Models: Multi-task outcome prediction")
    logger.info(f"✅ Deep Learning: Neural network for survival prediction")
    logger.info(f"✅ All models successfully trained and tested")
    logger.info("="*60)
    
    return {
        'survival_analysis': survival_result,
        'causal_inference': causal_results,
        'enhanced_models_trained': len(enhanced_models.models),
        'deep_model_trained': deep_survival.is_fitted
    }


if __name__ == "__main__":
    result = asyncio.run(main())