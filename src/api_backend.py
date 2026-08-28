"""
FastAPI backend for Medical Evidence Graph & Outcomes Insight Lab
Provides RESTful APIs for the clinical decision support system.

Design rule for this module: an endpoint either analyses the data the
caller supplied, or it fails. It never substitutes generated data for
missing data.

Every analysis endpoint here used to accept a list of patients, discard
it, invent replacement outcomes with `np.random`, and return
`"status": "success"`. That is worse than an error, because the caller
has no way to tell a real finding from a sampled one. The schemas below
therefore require the observed outcome alongside the patient: a
time-to-event analysis needs follow-up time and an event indicator, a
causal analysis needs a treatment assignment and an outcome, and a risk
score needs a model that was fitted on labelled data beforehand. Where
that input is absent the request is rejected (422) or the service
reports that it is not ready (503).
"""
import asyncio
from dataclasses import dataclass, field

import os
import secrets

from fastapi import FastAPI, HTTPException, Depends, Query, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4
import json
from enum import Enum
from pathlib import Path
import logging

from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_ingestion import ClinicalTrialsFetcher, PubMedFetcher
from src.audit import AuditLog, actor_id
from src.phi import PHIDetected, scanner_from_config
from src.outcomes_analytics_service.main import (
    CohortDefinition,
    CohortError,
    OutcomesAnalyticsService,
)
from src.pathway_guideline_service.main import (
    ObservedPathway,
    PathwayGuidelineService,
)
from src.enhanced_ml_models import (
    SurvivalAnalysisModels,
    CausalInferenceModels,
    EnhancedOutcomeModels,
    DeepSurvivalModel
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Medical Evidence Graph & Outcomes Insight Lab API",
    description="API for clinical decision support and evidence-based medicine",
    version="2.0.0"
)

# ---------------------------------------------------------------------------
# Authentication
#
# There was none, and CORS was allow_origins=["*"] with credentials, which
# lets any page on the internet make authenticated requests on a viewer's
# behalf. This is patient-facing analysis; it should not be reachable by
# anything that can reach the port.
#
# Keys come from MEG_API_KEYS (comma-separated) or config's
# security.api_keys. When none are configured the API still runs -- the
# alternative is being unable to develop against it -- but it is loud
# about it: a warning at startup, and /api/health reports authentication
# as disabled rather than staying silent about it.
# ---------------------------------------------------------------------------

def load_api_keys(config_path: str = "config/settings.json") -> set:
    """Configured API keys, environment first."""
    from_env = os.environ.get("MEG_API_KEYS", "")
    keys = {k.strip() for k in from_env.split(",") if k.strip()}
    if keys:
        return keys

    try:
        config = json.loads(Path(config_path).read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {k for k in config.get("security", {}).get("api_keys", []) if k}


def load_allowed_origins(config_path: str = "config/settings.json") -> List[str]:
    from_env = os.environ.get("MEG_ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in from_env.split(",") if o.strip()]
    if origins:
        return origins

    try:
        config = json.loads(Path(config_path).read_text())
    except (OSError, json.JSONDecodeError):
        config = {}
    configured = config.get("security", {}).get("allowed_origins")
    # The Streamlit frontend, and nothing else, by default.
    return configured or ["http://localhost:8501", "http://127.0.0.1:8501"]


API_KEYS = load_api_keys()
ALLOWED_ORIGINS = load_allowed_origins()


def load_phi_scanner(config_path: str = "config/settings.json"):
    """Build the PHI scanner from config.

    A misconfiguration raises here, at import, rather than at the first
    request: an API that starts and quietly scans nothing while its config
    says otherwise is the failure this exists to prevent.
    """
    try:
        config = json.loads(Path(config_path).read_text())
    except (OSError, json.JSONDecodeError):
        config = {}
    return scanner_from_config(config)


phi_scanner = load_phi_scanner()
audit_log = AuditLog()

if phi_scanner.enabled:
    logger.info(f"PHI detection: {phi_scanner.backend}, "
                f"on detection {phi_scanner.on_detection}")
else:
    logger.info(
        "PHI detection is off. Send de-identified or synthetic data only.")

if not API_KEYS:
    logger.warning(
        "No API keys configured: every endpoint is open to anyone who can "
        "reach this port. Set MEG_API_KEYS or security.api_keys before "
        "exposing this beyond localhost.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> Optional[str]:
    """Reject a request that carries no valid key, when keys are configured."""
    if not API_KEYS:
        return None

    # Constant-time comparison against each key: a plain `in` on a set
    # leaks nothing useful here, but comparing the supplied value is where
    # timing differences would show.
    if api_key and any(secrets.compare_digest(api_key, k) for k in API_KEYS):
        return api_key

    raise HTTPException(
        status_code=401,
        detail="A valid X-API-Key header is required.",
        headers={"WWW-Authenticate": "APIKey"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

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
    """A patient's covariates. Carries no outcome, so it can be scored but
    not analysed."""
    patient_id: Optional[str] = Field(None, description="Unique patient identifier")
    demographics: PatientDemographics
    clinical_indicators: ClinicalIndicators
    treatment_history: Optional[TreatmentHistory] = Field(None)
    admission_date: Optional[str] = Field(None, description="Date of admission")


class FollowUp(BaseModel):
    """Observed follow-up for one patient.

    `event_observed=False` means the patient was censored at
    `observed_time_days` — they were event-free when last seen, not
    confirmed event-free forever.
    """
    observed_time_days: float = Field(
        ..., gt=0, description="Days from index date to event or last contact")
    event_observed: bool = Field(
        ..., description="True if the event occurred; False if censored")


class PatientRecord(PatientInput):
    """A patient plus their observed follow-up. Required by any
    time-to-event analysis — without it there is nothing to analyse."""
    follow_up: FollowUp


class TreatmentRecord(PatientInput):
    """A patient plus their treatment assignment and observed outcome."""
    treatment_assigned: bool = Field(..., description="True if the patient received the treatment")
    outcome_value: float = Field(..., description="Observed outcome for this patient")


class CohortMember(PatientInput):
    """A patient plus one or more observed outcomes, keyed by name."""
    outcomes: Dict[str, float] = Field(
        ..., min_length=1, description="Observed outcomes, e.g. {'mortality': 0}")


class SurvivalAnalysisRequest(BaseModel):
    patient_data: List[PatientRecord] = Field(..., min_length=2)
    time_horizon_days: int = Field(365, gt=0, description="Reporting horizon in days")


class CausalAnalysisRequest(BaseModel):
    patient_data: List[TreatmentRecord] = Field(..., min_length=4)
    treatment_variable: str = Field(..., description="Name of the treatment, for reporting")
    outcome_variable: str = Field(..., description="Name of the outcome, for reporting")
    confounders: List[str] = Field(
        [], description="Covariate names to adjust for; must exist on the patients")


class CohortAnalysisRequest(BaseModel):
    patient_cohort: List[CohortMember] = Field(..., min_length=2)
    comparator_cohort: Optional[List[CohortMember]] = Field(
        None, description="Comparator group")
    outcome_variables: List[str] = Field(
        [], description="Outcomes to compare; defaults to those present in both cohorts")
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level")


class CohortCriteria(BaseModel):
    """Inclusion / exclusion criteria, applied for real.

    Each value is matched exactly, by membership if a list, or as an
    inclusive range if a two-element [min, max].
    """
    inclusion: Dict[str, Any] = Field(default_factory=dict)
    exclusion: Dict[str, Any] = Field(default_factory=dict)


class CohortBuildRequest(BaseModel):
    cohort_id: str
    name: str = "Cohort"
    patients: List[PatientRecord] = Field(..., min_length=2)
    criteria: CohortCriteria = Field(default_factory=CohortCriteria)
    follow_up_period_days: int = Field(365, gt=0)


class ComparativeEffectivenessRequest(CohortBuildRequest):
    group_field: str = Field(
        "treatment_group",
        description="Field on each patient naming its arm; must be supplied")
    groups: List[str] = Field(
        ..., min_length=2,
        description="Arm label per patient, in the same order as `patients`")


class GuidelineStep(BaseModel):
    name: str
    description: str = ""
    type: str = "intervention"
    recommended: bool = True
    timing: str = "immediate"
    evidence_level: str = "unknown"


class GuidelineDecisionPoint(BaseModel):
    question: str
    description: str = ""
    options: List[str] = Field(default_factory=list)


class GuidelineRequest(BaseModel):
    id: str
    name: str
    condition: str
    version: str = "1.0"
    steps: List[GuidelineStep] = Field(..., min_length=1)
    decision_points: List[GuidelineDecisionPoint] = Field(default_factory=list)


class ObservedStep(BaseModel):
    name: str
    performed_at: Optional[str] = None


class AdherenceRequest(BaseModel):
    guideline_id: str
    patient_id: str
    condition: str
    steps: List[ObservedStep] = Field(default_factory=list)


class TrainRiskModelRequest(BaseModel):
    training_data: List[CohortMember] = Field(..., min_length=20)
    test_size: float = Field(0.25, gt=0.0, lt=1.0)
    random_state: int = Field(42)


# ---------------------------------------------------------------------------
# Shared conversion
# ---------------------------------------------------------------------------

# Fields that carry caller-supplied free text and so can carry an
# identifier. Everything else on PatientInput is numeric or an enum.
FREE_TEXT_FIELDS = (
    "patient_id", "race", "previous_treatments", "medication_list",
    "lab_value_names",
)


def free_text_of(patient: PatientInput) -> Dict[str, Any]:
    history = patient.treatment_history
    return {
        "patient_id": patient.patient_id,
        "race": patient.demographics.race,
        "previous_treatments": history.previous_treatments if history else [],
        "medication_list": history.medication_list if history else [],
        # Keys, not values: the values are floats, the keys are free-form.
        "lab_value_names": list((patient.clinical_indicators.lab_values or {})),
    }


def screen_for_phi(patients: List[PatientInput]) -> None:
    """Reject a payload carrying direct identifiers.

    Rejecting rather than redacting by default: silently altering the
    caller's data would change what is analysed without saying so, and
    this service's stated scope is de-identified or synthetic data.
    """
    if not phi_scanner.enabled:
        return
    for patient in patients:
        try:
            phi_scanner.enforce(free_text_of(patient))
        except PHIDetected as e:
            # The message names the field and the kind, never the matched
            # text -- echoing it back would copy the identifier into logs.
            raise HTTPException(status_code=422, detail=str(e)) from e


def patients_to_frame(patients: List[PatientInput]) -> pd.DataFrame:
    """Canonical patient -> DataFrame conversion.

    There is exactly one of these on purpose. The previous code built a
    frame with a 'sex' column at scoring time while models were fitted on
    `build_patient_features` output (which reads 'gender'), so every
    prediction raised a feature-name mismatch that was caught and turned
    into a risk of 0.0.
    """
    rows = []
    for patient in patients:
        row = {
            'age': patient.demographics.age,
            'gender': 'M' if patient.demographics.sex == SexEnum.male else 'F',
            'baseline_risk_score': patient.clinical_indicators.baseline_risk_score,
            'comorbidity_count': patient.clinical_indicators.comorbidity_count,
        }
        if patient.clinical_indicators.severity_score is not None:
            row['severity_score'] = patient.clinical_indicators.severity_score
        for lab_name, lab_value in (patient.clinical_indicators.lab_values or {}).items():
            row[f'lab_{lab_name}'] = lab_value
        rows.append(row)
    return pd.DataFrame(rows)


def median_survival_time(time_points: List[float], survival_probs: List[float]) -> Optional[float]:
    """First time at which survival drops to 0.5 or below.

    Returns None when the curve never reaches 0.5 — median survival is then
    "not reached", which is a real and reportable answer. The previous code
    returned `np.median(time_points)`, the midpoint of the time AXIS, which
    is unrelated to survival.
    """
    for time_point, survival in zip(time_points, survival_probs):
        # Tolerance: the product-limit estimate is a running product of
        # floats, so a curve that reaches exactly one half can land on
        # 0.5000000000000001 and a bare <= 0.5 then reports the next event
        # time instead.
        if survival <= 0.5 + 1e-9:
            return float(time_point)
    return None


# ---------------------------------------------------------------------------
# Risk model registry
# ---------------------------------------------------------------------------

# Where trained models and registered guidelines are kept between runs.
#
# Both registries were plain in-memory dicts, so a restart silently
# discarded every trained model and every registered guideline: risk
# assessment reverted to 503 "no risk model has been trained" and
# guidelines vanished, with nothing anywhere saying they had ever existed.
# For a score that is reported alongside held-out performance as evidence
# you can trust it, losing it on a deploy is not a small thing.
MODEL_STORE = Path(os.environ.get("MEG_MODEL_STORE", ".model_store"))


class RiskModel:
    """A fitted risk model together with what is needed to reproduce and
    judge it: the feature columns it saw, the medians used for imputation,
    and its held-out performance."""

    def __init__(self, outcome: str, model: Any, task: str, feature_columns: List[str],
                 feature_medians: Dict[str, float], metric_name: str,
                 metric_value: Optional[float], n_train: int, n_test: int):
        self.outcome = outcome
        self.model = model
        self.task = task
        self.feature_columns = feature_columns
        self.feature_medians = feature_medians
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.n_train = n_train
        self.n_test = n_test

    def align(self, features: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Reindex incoming features onto the fitted columns.

        Columns the caller did not supply are filled with the training
        median and reported back, so an imputed prediction is never
        presented as if it came from complete data.
        """
        imputed = [c for c in self.feature_columns if c not in features.columns]
        aligned = features.reindex(columns=self.feature_columns)
        return aligned.fillna(pd.Series(self.feature_medians)), imputed

    def describe(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "task": self.task,
            "holdout_metric": self.metric_name,
            "holdout_value": self.metric_value,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }

    def to_state(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome, "model": self.model, "task": self.task,
            "feature_columns": self.feature_columns,
            "feature_medians": self.feature_medians,
            "metric_name": self.metric_name, "metric_value": self.metric_value,
            "n_train": self.n_train, "n_test": self.n_test,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "RiskModel":
        return cls(**state)


class RiskModelRegistry:
    """Holds fitted risk models for the process.

    Empty at startup by design. Risk assessment cannot be served until
    something has been trained on labelled data: fitting a model on
    generated labels inside the request that then consumes it is a closed
    loop that reports its own inputs back as findings.
    """

    def __init__(self):
        self.models: Dict[str, RiskModel] = {}
        self.version: Optional[str] = None
        self.trained_at: Optional[str] = None

    def __init_store__(self, store: Path = MODEL_STORE):
        self.store = Path(store)

    @property
    def _path(self) -> Path:
        return getattr(self, "store", MODEL_STORE) / "risk_models.joblib"

    def is_ready(self) -> bool:
        return bool(self.models)

    def replace(self, models: Dict[str, RiskModel], persist: bool = True) -> None:
        self.models = models
        self.version = uuid4().hex[:12]
        self.trained_at = datetime.now(timezone.utc).isoformat()
        if persist:
            self.save()

    def save(self) -> None:
        import joblib
        import sklearn

        self._path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            # Recorded so a load can refuse rather than guess. Unpickling
            # an estimator built by a different scikit-learn is undefined:
            # it usually works, and when it does not it produces wrong
            # predictions rather than an error.
            "sklearn_version": sklearn.__version__,
            "version": self.version,
            "trained_at": self.trained_at,
            "models": {name: m.to_state() for name, m in self.models.items()},
        }, self._path)
        logger.info(f"Persisted {len(self.models)} risk models to {self._path}")

    def load(self) -> bool:
        """Restore models from disk. Returns whether anything was loaded."""
        if not self._path.exists():
            return False

        import joblib
        import sklearn

        try:
            state = joblib.load(self._path)
        except Exception as e:
            logger.error(
                f"Could not read {self._path}: {e}. Not loading; train again "
                f"or remove the file deliberately.")
            return False

        stored = state.get("sklearn_version")
        if stored != sklearn.__version__:
            # Refuse rather than warn. A model that silently scores
            # differently after an upgrade is worse than no model, because
            # the response still carries the held-out AUC measured before.
            logger.error(
                f"{self._path} was written by scikit-learn {stored}, running "
                f"{sklearn.__version__}. Refusing to load: predictions across "
                f"versions are not guaranteed to match. Retrain.")
            return False

        self.models = {name: RiskModel.from_state(m)
                       for name, m in state["models"].items()}
        self.version = state["version"]
        self.trained_at = state["trained_at"]
        logger.info(
            f"Restored {len(self.models)} risk models (version {self.version}, "
            f"trained {self.trained_at})")
        return True


risk_model_registry = RiskModelRegistry()
risk_model_registry.__init_store__()
risk_model_registry.load()

outcomes_service = OutcomesAnalyticsService()
pathway_service = PathwayGuidelineService()


class GuidelineStore:
    """Registered guidelines, kept between runs.

    JSON rather than a pickle: a guideline is plain data, and a format a
    human can read and correct matters more here than convenience.
    """

    def __init__(self, store: Path = MODEL_STORE):
        self.path = Path(store) / "guidelines.json"

    def save(self, guidelines: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(guidelines, indent=2) + "\n")

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read {self.path}: {e}. Starting empty.")
            return {}


guideline_store = GuidelineStore()


def restore_guidelines() -> int:
    """Re-register persisted guidelines into the pathway service."""
    restored = guideline_store.load()
    for payload in restored.values():
        pathway_service.represent_guideline_as_pathway(payload)
    if restored:
        logger.info(f"Restored {len(restored)} guidelines")
    return len(restored)


restore_guidelines()
survival_models = SurvivalAnalysisModels()
causal_models = CausalInferenceModels()
feature_builder = EnhancedOutcomeModels()


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSearchResponse:
    """What a search returned, and where it came from.

    `mode` is part of the payload because "our indexed corpus does not
    cover this" and "the literature does not cover this" are different
    claims, and a caller cannot tell them apart from the results alone.
    """
    results: List[Dict[str, Any]]
    mode: str  # "index" | "live" | "index_then_live"
    graph_context: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""


class EvidenceSearcher:
    """Retrieves evidence. Overridden in tests via FastAPI dependencies."""

    async def search(self, query: str, limit: int) -> EvidenceSearchResponse:
        raise NotImplementedError


class LiteratureEvidenceSearcher(EvidenceSearcher):
    """Live retrieval from PubMed and ClinicalTrials.gov.

    Replaces a stub that returned `f"Medical Evidence Title {i}: {query}"`
    — the query echoed back with a relevance score attached.
    """

    async def search(self, query: str, limit: int) -> EvidenceSearchResponse:
        return EvidenceSearchResponse(
            results=await self.fetch(query, limit),
            mode="live",
            note=("Retrieved live from PubMed and ClinicalTrials.gov. These "
                  "results are not in the local index and carry no graph "
                  "context."),
        )

    async def fetch(self, query: str, limit: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        async with PubMedFetcher() as pubmed:
            pmids = await pubmed.search_pubmed(query, limit)
            for article in await pubmed.fetch_pubmed_articles(pmids[:limit]):
                results.append({
                    "id": f"pubmed_{article['pmid']}",
                    "title": article["title"],
                    "abstract": article["abstract"],
                    "source": "PubMed",
                    "pub_date": article["pub_date"],
                    "authors": article["authors"],
                    "mesh_terms": article["mesh_terms"],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/",
                })

        remaining = limit - len(results)
        if remaining > 0:
            async with ClinicalTrialsFetcher() as trials:
                for study in await trials.search_trials(query, remaining):
                    identification = study.get("protocolSection", {}).get(
                        "identificationModule", {})
                    nct_id = identification.get("nctId", "")
                    results.append({
                        "id": f"nct_{nct_id}",
                        "title": identification.get("briefTitle", ""),
                        "abstract": study.get("protocolSection", {}).get(
                            "descriptionModule", {}).get("briefSummary", ""),
                        "source": "ClinicalTrials.gov",
                        "pub_date": study.get("protocolSection", {}).get(
                            "statusModule", {}).get("startDateStruct", {}).get("date", ""),
                        "authors": [],
                        "mesh_terms": [],
                        "url": f"https://clinicaltrials.gov/study/{nct_id}",
                    })

        return results[:limit]


class GraphRAGEvidenceSearcher(EvidenceSearcher):
    """Hybrid retrieval over the indexed corpus, with a labelled fallback.

    Searches the local index first: BM25 + vector + graph traversal, fused,
    with the graph context behind each hit. That is the point of indexing
    — before this, every search was a live round trip to PubMed on the hot
    path, subject to the rate limiter, and the indexed corpus was never
    read.

    When the index has nothing, it falls back to live retrieval and says
    so in `mode`, rather than blending the two silently. It does not
    ingest what it finds: writing to the corpus as a side effect of a read
    would mean the index quietly reshapes itself around whatever people
    happen to search for.
    """

    def __init__(self, service: Any, fallback: EvidenceSearcher):
        self.service = service
        self.fallback = fallback

    async def search(self, query: str, limit: int) -> EvidenceSearchResponse:
        answer = await self.service.answer_query(query, limit=limit)

        if not answer.results:
            live = await self.fallback.fetch(query, limit)
            return EvidenceSearchResponse(
                results=live,
                mode="index_then_live",
                note=("The indexed corpus returned nothing for this query; "
                      "these results came from a live search instead. "
                      "Absence from the index is not absence from the "
                      "literature."),
            )

        return EvidenceSearchResponse(
            results=[{
                "id": result.id,
                "title": result.title,
                "abstract": result.content,
                "source": result.source,
                "citation": result.citation,
                "pub_date": result.metadata.get("pub_date", ""),
                "journal": result.metadata.get("journal", ""),
                "authors": [],
                "mesh_terms": [],
                "url": _url_for(result),
                # How this result was found, so the ranking is inspectable.
                "found_by": result.found_by,
                "fused_score": result.fused_score,
            } for result in answer.results],
            mode="index",
            graph_context=answer.graph_context,
            note=answer.coverage["note"],
        )


def _url_for(result: Any) -> str:
    pmid = result.metadata.get("pmid")
    nct_id = result.metadata.get("nct_id")
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if nct_id:
        return f"https://clinicaltrials.gov/study/{nct_id}"
    return ""


_evidence_searcher: Optional[EvidenceSearcher] = None
_searcher_lock = asyncio.Lock()


async def build_evidence_searcher() -> EvidenceSearcher:
    """Prefer the indexed corpus; fall back to live retrieval.

    The graph stack is an optional extra, so its import is deferred: the
    analysis API, the risk models and the frontend all work without
    neo4j/qdrant/sentence-transformers installed.
    """
    live = LiteratureEvidenceSearcher()
    try:
        from src.graph_rag_service.main import GraphRAGService

        service = GraphRAGService()
        await service.connect()
    except Exception as e:
        # Not an error: retrieval degrades to live search, and every
        # response says which mode produced it.
        logger.warning(
            f"Graph-RAG unavailable ({e}); evidence search will use live "
            f"retrieval. Run `docker compose up -d` and index a corpus to "
            f"search locally.")
        return live

    logger.info("Evidence search backed by the indexed corpus (graph-RAG)")
    return GraphRAGEvidenceSearcher(service, live)


async def get_evidence_searcher() -> EvidenceSearcher:
    """Built once per process: connecting loads an embedding model."""
    global _evidence_searcher
    if _evidence_searcher is None:
        async with _searcher_lock:
            if _evidence_searcher is None:
                _evidence_searcher = await build_evidence_searcher()
    return _evidence_searcher


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {
        "message": "Medical Evidence Graph & Outcomes Insight Lab API",
        "status": "running",
        "version": "2.0.0",
        "risk_model_trained": risk_model_registry.is_ready(),
    }


@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Health check. Reports readiness honestly: a risk model that has not
    been trained is not ready, whatever the process uptime says."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_ready": {
            "survival_analysis": True,
            "causal_inference": True,
            "risk_assessment": risk_model_registry.is_ready(),
        },
        "risk_model_version": risk_model_registry.version,
        "risk_model_trained_at": risk_model_registry.trained_at,
        "risk_models_persisted": risk_model_registry._path.exists(),
        "guidelines_registered": len(pathway_service.guidelines),
        # Stated, not assumed. An unauthenticated deployment should be
        # visible to whoever is looking at it.
        "authentication": "api_key" if API_KEYS else "disabled",
        # What is actually done, not what a config flag asserts.
        "phi_detection": phi_scanner.describe,
    }


# ---------------------------------------------------------------------------
# Risk models
# ---------------------------------------------------------------------------

@app.post("/api/models/risk/train")
async def train_risk_models(request: TrainRiskModelRequest, _key: Optional[str] = Depends(require_api_key)):
    """Fit risk models on labelled patients and report held-out performance.

    Splitting matters here: a random forest scored on its own training rows
    reports near-perfect accuracy on any labels at all, including noise.
    """
    screen_for_phi(request.training_data)
    logger.info(f"Training risk models on {len(request.training_data)} patients")
    audit_log.record(
        "risk_model.train", actor=actor_id(_key),
        n_patients=len(request.training_data), test_size=request.test_size)

    frame = patients_to_frame(request.training_data)
    features = feature_builder.build_patient_features(frame)

    outcome_names = sorted({name for m in request.training_data for name in m.outcomes})
    incomplete = [
        name for name in outcome_names
        if any(name not in m.outcomes for m in request.training_data)
    ]
    if incomplete:
        raise HTTPException(
            status_code=422,
            detail=f"Outcomes {incomplete} are missing for some patients; every "
                   f"outcome must be recorded for every patient in the training set")

    trained: Dict[str, RiskModel] = {}
    skipped: Dict[str, str] = {}

    for outcome in outcome_names:
        values = np.array([m.outcomes[outcome] for m in request.training_data], dtype=float)
        unique = np.unique(values)

        if len(unique) < 2:
            skipped[outcome] = (
                f"constant outcome (every patient has {unique[0]}); nothing to learn")
            continue

        is_binary = set(unique) <= {0.0, 1.0}
        stratify = values if is_binary else None
        X_train, X_test, y_train, y_test = train_test_split(
            features, values, test_size=request.test_size,
            random_state=request.random_state, stratify=stratify)

        if is_binary:
            model = RandomForestClassifier(n_estimators=100, random_state=request.random_state)
            model.fit(X_train, y_train)
            metric_name = "roc_auc"
            if len(np.unique(y_test)) < 2:
                # AUC is undefined against a single-class holdout. Report that
                # rather than a number that looks like a score.
                metric_value = None
            else:
                metric_value = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            task = "classification"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=request.random_state)
            model.fit(X_train, y_train)
            metric_name = "r2"
            metric_value = float(r2_score(y_test, model.predict(X_test)))
            task = "regression"

        trained[outcome] = RiskModel(
            outcome=outcome,
            model=model,
            task=task,
            feature_columns=list(features.columns),
            feature_medians={c: float(features[c].median()) for c in features.columns},
            metric_name=metric_name,
            metric_value=metric_value,
            n_train=len(X_train),
            n_test=len(X_test),
        )

    if not trained:
        raise HTTPException(
            status_code=422,
            detail=f"No model could be trained. {skipped}")

    risk_model_registry.replace(trained)
    logger.info(f"Trained {len(trained)} risk models, version {risk_model_registry.version}")

    audit_log.record(
        "risk_model.trained", actor=actor_id(_key),
        model_version=risk_model_registry.version,
        outcomes=sorted(trained), skipped=sorted(skipped))

    return {
        "status": "success",
        "model_version": risk_model_registry.version,
        "trained_at": risk_model_registry.trained_at,
        "models": {name: model.describe() for name, model in trained.items()},
        "skipped_outcomes": skipped,
    }


@app.post("/api/patients/risk-assessment")
async def assess_patient_risk(patients: List[PatientInput], _key: Optional[str] = Depends(require_api_key)):
    """Score patients with the currently registered risk models."""
    if not patients:
        raise HTTPException(status_code=422, detail="No patients supplied")

    screen_for_phi(patients)

    if not risk_model_registry.is_ready():
        # The alternative — training on generated labels inside this request
        # — produced a mortality risk of 0.0 for every patient alongside
        # "status": "success".
        raise HTTPException(
            status_code=503,
            detail="No risk model has been trained. POST labelled patients to "
                   "/api/models/risk/train first.")

    logger.info(f"Assessing risk for {len(patients)} patients")
    audit_log.record(
        "risk.assess", actor=actor_id(_key), n_patients=len(patients),
        model_version=risk_model_registry.version)

    frame = patients_to_frame(patients)
    features = feature_builder.build_patient_features(frame)

    risk_assessments = []
    for position, patient in enumerate(patients):
        risks = {}
        for outcome, risk_model in risk_model_registry.models.items():
            aligned, imputed = risk_model.align(features.iloc[[position]])
            if risk_model.task == "classification":
                score = float(risk_model.model.predict_proba(aligned)[0, 1])
            else:
                score = float(risk_model.model.predict(aligned)[0])

            # Every score travels with the evidence for trusting it.
            risks[outcome] = {
                "score": score,
                "holdout_metric": risk_model.metric_name,
                "holdout_value": risk_model.metric_value,
                "imputed_features": imputed,
            }

        risk_assessments.append({
            "patient_id": patient.patient_id or f"patient_{position}",
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "risks": risks,
        })

    return {
        "status": "success",
        "model_version": risk_model_registry.version,
        "risk_assessments": risk_assessments,
        "total_patients": len(risk_assessments),
    }


# ---------------------------------------------------------------------------
# Survival analysis
# ---------------------------------------------------------------------------

@app.post("/api/survival-analysis/kaplan-meier")
async def kaplan_meier_analysis(request: SurvivalAnalysisRequest, _key: Optional[str] = Depends(require_api_key)):
    """Kaplan-Meier estimate from the supplied follow-up.

    Both inputs come from `follow_up` on each patient. The endpoint
    previously ignored `patient_data` entirely and drew its curve from
    np.random.exponential, making the response a function of the cohort
    SIZE and the RNG state.
    """
    screen_for_phi(request.patient_data)
    logger.info(f"Running Kaplan-Meier analysis for {len(request.patient_data)} patients")
    audit_log.record(
        "survival.kaplan_meier", actor=actor_id(_key),
        n_patients=len(request.patient_data),
        horizon_days=request.time_horizon_days)

    duration = np.array([p.follow_up.observed_time_days for p in request.patient_data])
    event = np.array([int(p.follow_up.event_observed) for p in request.patient_data])

    if event.sum() == 0:
        raise HTTPException(
            status_code=422,
            detail="No events observed in this cohort; a Kaplan-Meier curve "
                   "would be flat at 1.0 and carries no information")

    try:
        time_points, survival_probs = survival_models.kaplan_meier_analysis(duration, event)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    at_horizon = [
        survival for time_point, survival in zip(time_points, survival_probs)
        if time_point <= request.time_horizon_days
    ]

    return {
        "status": "success",
        "time_horizon_days": request.time_horizon_days,
        "survival_curve": {
            "time_points": time_points,
            "survival_probability": survival_probs,
        },
        "stats": {
            "median_survival_days": median_survival_time(time_points, survival_probs),
            "survival_at_horizon": at_horizon[-1] if at_horizon else None,
            "total_patients": len(request.patient_data),
            "events_occurred": int(event.sum()),
            "censored": int(len(event) - event.sum()),
            "max_follow_up_days": float(duration.max()),
        },
    }


@app.post("/api/survival-analysis/cox-regression")
async def cox_regression_analysis(request: SurvivalAnalysisRequest, _key: Optional[str] = Depends(require_api_key)):
    """Cox proportional hazards on the supplied covariates and follow-up.

    The covariates were always real; the outcome was not. `observed_time`
    and `event` were drawn per row from np.random, so the reported hazard
    ratios described the association between a real covariate and noise.
    """
    screen_for_phi(request.patient_data)
    logger.info(f"Running Cox regression for {len(request.patient_data)} patients")
    audit_log.record(
        "survival.cox", actor=actor_id(_key),
        n_patients=len(request.patient_data))

    frame = patients_to_frame(request.patient_data)
    frame['gender_encoded'] = (frame['gender'] == 'M').astype(int)
    frame['observed_time'] = [p.follow_up.observed_time_days for p in request.patient_data]
    frame['event'] = [int(p.follow_up.event_observed) for p in request.patient_data]

    covariates = ['age', 'gender_encoded', 'baseline_risk_score', 'comorbidity_count']
    # Drop covariates with no variation: they cannot be estimated and would
    # abort the whole fit.
    usable = [c for c in covariates if frame[c].nunique() > 1]
    dropped = [c for c in covariates if c not in usable]
    if not usable:
        raise HTTPException(
            status_code=422,
            detail="Every covariate is constant across this cohort; there is "
                   "nothing for a Cox model to estimate")

    try:
        result = survival_models.cox_regression_analysis(
            frame, 'observed_time', 'event', usable)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "status": "success",
        "c_index": result.c_index,
        "c_index_is_in_sample": True,
        "hazard_ratios": result.hazard_ratios,
        "p_values": result.p_values,
        "confidence_intervals": {k: list(v) for k, v in result.confidence_intervals.items()},
        "covariates_dropped_as_constant": dropped,
        "model_stats": {
            "log_likelihood": result.log_likelihood,
            "baseline_survival_points": len(result.baseline_survival),
            "total_patients": len(request.patient_data),
            "events_observed": int(frame['event'].sum()),
        },
    }


# ---------------------------------------------------------------------------
# Causal inference
# ---------------------------------------------------------------------------

@app.post("/api/causal-inference/ate-estimation")
async def estimate_ate(request: CausalAnalysisRequest, _key: Optional[str] = Depends(require_api_key)):
    """Average treatment effect from observed assignments and outcomes.

    Treatment and outcome now come from the request. They were previously
    synthesised — treatment from a threshold on the risk score, outcome
    from np.random.binomial(1, 0.1 + 0.1 * treatment) — so the returned
    "effect" was the 0.1 constant written into that line, recovered.
    """
    screen_for_phi(request.patient_data)
    logger.info(f"Estimating ATE for {len(request.patient_data)} patients")
    audit_log.record(
        "causal.ate", actor=actor_id(_key),
        n_patients=len(request.patient_data),
        treatment=request.treatment_variable, outcome=request.outcome_variable)

    frame = patients_to_frame(request.patient_data)
    frame['gender_encoded'] = (frame['gender'] == 'M').astype(int)
    treatment = np.array([int(p.treatment_assigned) for p in request.patient_data])
    outcome = np.array([p.outcome_value for p in request.patient_data])

    if len(np.unique(treatment)) < 2:
        raise HTTPException(
            status_code=422,
            detail="All patients are in the same treatment arm; a treatment "
                   "effect cannot be estimated without a comparison group")

    default_covariates = ['age', 'gender_encoded', 'baseline_risk_score', 'comorbidity_count']
    covariates = request.confounders or default_covariates
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Confounders {missing} are not available on these patients. "
                   f"Available: {sorted(frame.columns)}")

    usable = [c for c in covariates if frame[c].nunique() > 1]
    if not usable:
        raise HTTPException(
            status_code=422,
            detail=f"Confounders {covariates} are constant across this cohort")

    # estimate_ate reads a 'treatment' column off the frame for its
    # regression-adjustment step, then overwrites it with the array.
    frame['treatment'] = treatment

    try:
        causal_results = causal_models.estimate_ate(frame, treatment, outcome, usable)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "status": "success",
        "treatment_variable": request.treatment_variable,
        "outcome_variable": request.outcome_variable,
        "ate_estimates": {k: float(v) for k, v in causal_results.items()},
        "cohort": {
            "total_patients": len(request.patient_data),
            "treated": int(treatment.sum()),
            "control": int(len(treatment) - treatment.sum()),
        },
        "confounders_adjusted_for": usable,
        "confounders_dropped_as_constant": [c for c in covariates if c not in usable],
        "caveat": (
            "Observational estimate. Adjustment covers the listed confounders "
            "only; unmeasured confounding is not addressed."
        ),
    }


# ---------------------------------------------------------------------------
# Cohort comparison
# ---------------------------------------------------------------------------

def compare_one_outcome(values_1: np.ndarray, values_2: np.ndarray, alpha: float) -> Dict[str, Any]:
    """Run the appropriate two-sample test and report what was run.

    Binary outcomes go to a chi-square test of a 2x2 table, or Fisher's
    exact test when an expected cell is small. Continuous outcomes go to
    Welch's t-test. The previous implementation ran no test at all: it
    returned `cohorts_have_difference: True` and `p = 0.05` as literals.
    """
    is_binary = set(np.unique(np.concatenate([values_1, values_2]))) <= {0.0, 1.0}

    if is_binary:
        table = np.array([
            [int(values_1.sum()), int(len(values_1) - values_1.sum())],
            [int(values_2.sum()), int(len(values_2) - values_2.sum())],
        ])
        if table.sum(axis=0).min() == 0 or table.sum(axis=1).min() == 0:
            return {
                "test": None,
                "p_value": None,
                "significant": None,
                "reason": "a cohort has no variation in this outcome; no test is applicable",
                "cohort_1_rate": float(values_1.mean()),
                "cohort_2_rate": float(values_2.mean()),
            }

        expected = stats.contingency.expected_freq(table)
        if expected.min() < 5:
            _, p_value = stats.fisher_exact(table)
            test_name = "fisher_exact"
        else:
            _, p_value, _, _ = stats.chi2_contingency(table, correction=False)
            test_name = "chi2_contingency"

        rate_1, rate_2 = float(values_1.mean()), float(values_2.mean())
        return {
            "test": test_name,
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "cohort_1_rate": rate_1,
            "cohort_2_rate": rate_2,
            "risk_difference": rate_2 - rate_1,
            "risk_ratio": (rate_2 / rate_1) if rate_1 > 0 else None,
        }

    statistic, p_value = stats.ttest_ind(values_1, values_2, equal_var=False)
    return {
        "test": "welch_t",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "cohort_1_mean": float(values_1.mean()),
        "cohort_2_mean": float(values_2.mean()),
        "mean_difference": float(values_2.mean() - values_1.mean()),
    }


def summarise_cohort(members: List[CohortMember]) -> Dict[str, Any]:
    frame = patients_to_frame(members)
    gender_counts = frame['gender'].value_counts().to_dict()
    return {
        "size": len(members),
        "characteristics": {
            "mean_age": float(frame['age'].mean()),
            "mean_baseline_risk_score": float(frame['baseline_risk_score'].mean()),
            "mean_comorbidity_count": float(frame['comorbidity_count'].mean()),
            "gender_distribution": {k: int(v) for k, v in gender_counts.items()},
        },
    }


@app.post("/api/cohorts/compare")
async def compare_cohorts(request: CohortAnalysisRequest, _key: Optional[str] = Depends(require_api_key)):
    """Compare two cohorts on their observed outcomes, with real tests."""
    if not request.comparator_cohort:
        raise HTTPException(
            status_code=422,
            detail="A comparator cohort is required; there is nothing to "
                   "compare a single cohort against")

    screen_for_phi(request.patient_cohort + request.comparator_cohort)
    audit_log.record(
        "cohort.compare", actor=actor_id(_key),
        n_cohort=len(request.patient_cohort),
        n_comparator=len(request.comparator_cohort), alpha=request.alpha)
    logger.info(
        f"Comparing cohorts: {len(request.patient_cohort)} vs "
        f"{len(request.comparator_cohort)} patients")

    shared = sorted(
        {name for m in request.patient_cohort for name in m.outcomes}
        & {name for m in request.comparator_cohort for name in m.outcomes}
    )
    requested = request.outcome_variables or shared
    missing = [name for name in requested if name not in shared]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Outcomes {missing} are not recorded in both cohorts. "
                   f"Available in both: {shared}")
    if not requested:
        raise HTTPException(
            status_code=422,
            detail="The two cohorts share no outcome to compare")

    comparisons = {}
    for outcome in requested:
        values_1 = np.array(
            [m.outcomes[outcome] for m in request.patient_cohort if outcome in m.outcomes],
            dtype=float)
        values_2 = np.array(
            [m.outcomes[outcome] for m in request.comparator_cohort if outcome in m.outcomes],
            dtype=float)
        comparisons[outcome] = compare_one_outcome(values_1, values_2, request.alpha)

    tested = [c for c in comparisons.values() if c["significant"] is not None]

    return {
        "status": "success",
        "cohort_comparison": {
            "cohort_1": summarise_cohort(request.patient_cohort),
            "cohort_2": summarise_cohort(request.comparator_cohort),
            "outcomes": comparisons,
            "comparison_metrics": {
                "alpha": request.alpha,
                "outcomes_tested": len(tested),
                # Derived from the tests actually run, not asserted.
                "cohorts_have_difference": any(c["significant"] for c in tested) if tested else None,
                "note": (
                    "Unadjusted comparisons. With several outcomes, consider "
                    "correcting for multiple comparisons before acting on any "
                    "single p-value."
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Outcomes analytics
#
# OutcomesAnalyticsService and PathwayGuidelineService were both working
# and tested, and neither was reachable by any user of the product: the
# API exposed ten endpoints and none of them touched either one.
# ---------------------------------------------------------------------------

def records_to_cohort_frame(
    patients: List[PatientRecord], groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """Patient records in the shape the outcomes service analyses."""
    frame = patients_to_frame(patients)
    frame["patient_id"] = [p.patient_id or f"patient_{i}"
                           for i, p in enumerate(patients)]
    frame["survival_time"] = [p.follow_up.observed_time_days for p in patients]
    frame["event_status"] = [int(p.follow_up.event_observed) for p in patients]
    if groups is not None:
        frame["treatment_group"] = groups
    return frame


def build_cohort(request: CohortBuildRequest, groups: Optional[List[str]] = None):
    definition = CohortDefinition(
        id=request.cohort_id,
        name=request.name,
        # A two-element list means a range; the service applies it as one.
        inclusion_criteria={
            k: tuple(v) if isinstance(v, list) and len(v) == 2 and
            all(isinstance(x, (int, float)) for x in v) else v
            for k, v in request.criteria.inclusion.items()
        },
        exclusion_criteria=dict(request.criteria.exclusion),
        follow_up_period=request.follow_up_period_days,
        outcome_definition={},
    )
    frame = records_to_cohort_frame(request.patients, groups)
    try:
        cohort = outcomes_service.create_cohort(definition, frame)
    except CohortError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return definition, cohort


@app.post("/api/outcomes/cohort")
async def build_cohort_endpoint(request: CohortBuildRequest,
                                _key: Optional[str] = Depends(require_api_key)):
    """Apply inclusion/exclusion criteria and report the resulting cohort.

    Criteria are applied to the patients you supply. A criterion naming a
    field that is not present is refused rather than skipped: skipping it
    would silently produce a cohort broader than the one defined, and every
    rate computed from it would be wrong with no sign of it.
    """
    screen_for_phi(request.patients)
    definition, cohort = build_cohort(request)
    audit_log.record(
        "cohort.build", actor=actor_id(_key), cohort_id=request.cohort_id,
        supplied=len(request.patients), retained=int(len(cohort)))

    try:
        survival = outcomes_service.run_survival_analysis(cohort)
    except CohortError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return {
        "status": "success",
        "cohort_id": definition.id,
        "size": int(len(cohort)),
        "supplied": len(request.patients),
        "excluded_by_criteria": len(request.patients) - int(len(cohort)),
        "survival": {
            "time_points": survival.time_points,
            "survival_probabilities": survival.survival_probabilities,
            # Greenwood intervals: they widen as the risk set shrinks,
            # which the binomial standard error does not.
            "confidence_intervals": survival.confidence_intervals,
            "median_survival_days": survival.median_survival,
            "events": survival.n_events,
            "censored": survival.n_censored,
        },
    }


@app.post("/api/outcomes/comparative-effectiveness")
async def comparative_effectiveness(request: ComparativeEffectivenessRequest,
                                    _key: Optional[str] = Depends(require_api_key)):
    """Compare arms with a log-rank test.

    Arm assignment must be supplied. Inferring it here would make the
    comparison a comparison of whatever rule did the inferring.
    """
    if len(request.groups) != len(request.patients):
        raise HTTPException(
            status_code=422,
            detail=f"{len(request.groups)} arm labels for "
                   f"{len(request.patients)} patients")

    screen_for_phi(request.patients)
    definition, cohort = build_cohort(request, request.groups)
    audit_log.record(
        "cohort.comparative_effectiveness", actor=actor_id(_key),
        cohort_id=request.cohort_id, n_patients=len(request.patients),
        n_arms=len(set(request.groups)))

    try:
        result = outcomes_service.run_comparative_effectiveness_analysis(
            cohort, request.group_field)
        survival = outcomes_service.run_survival_analysis(cohort)
    except CohortError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return {
        "status": "success",
        "cohort_id": definition.id,
        "group_outcomes": result.group_outcomes,
        "test": result.comparison.test,
        "test_statistic": result.comparison.test_statistic,
        "p_value": result.comparison.p_value,
        "degrees_of_freedom": result.comparison.degrees_of_freedom,
        # A p-value with no denominators cannot be judged.
        "group_sizes": result.comparison.group_sizes,
        "group_events": result.comparison.group_events,
        "risk_difference": result.risk_difference,
        "number_needed_to_treat": result.number_needed_to_treat,
        "median_survival_days": survival.median_survival,
        "notes": result.notes,
    }


# ---------------------------------------------------------------------------
# Pathways and guidelines
# ---------------------------------------------------------------------------

@app.post("/api/pathways/guidelines")
async def register_guideline(request: GuidelineRequest,
                             _key: Optional[str] = Depends(require_api_key)):
    """Register a guideline as a machine-readable pathway."""
    payload = {
        "id": request.id,
        "name": request.name,
        "condition": request.condition,
        "version": request.version,
        "steps": [step.model_dump() for step in request.steps],
        "decision_points": [d.model_dump() for d in request.decision_points],
    }
    pathway = pathway_service.represent_guideline_as_pathway(payload)

    stored = guideline_store.load()
    stored[request.id] = payload
    guideline_store.save(stored)
    audit_log.record(
        "guideline.register", actor=actor_id(_key), guideline_id=request.id,
        n_steps=len(request.steps), version=request.version)

    return {
        "status": "success",
        "guideline_id": pathway.id,
        "steps": len(request.steps),
        "required_steps": sum(1 for s in request.steps if s.recommended),
        "decision_points": len(request.decision_points),
    }


@app.get("/api/pathways/guidelines")
async def list_guidelines(_key: Optional[str] = Depends(require_api_key)):
    return {
        "status": "success",
        "guidelines": [
            {"id": g.id, "name": g.name, "condition": g.condition,
             "version": g.version, "steps": len(g.nodes)}
            for g in pathway_service.guidelines.values()
        ],
    }


@app.post("/api/pathways/adherence")
async def evaluate_adherence(request: AdherenceRequest,
                             _key: Optional[str] = Depends(require_api_key)):
    """Score observed care against a registered guideline.

    Adherence is recomputed from the observed steps. It is never taken
    from the caller: a claimed adherence score reported back as a finding
    would be worth nothing.
    """
    observed = ObservedPathway(
        patient_id=request.patient_id,
        condition=request.condition,
        steps=[{"name": s.name} for s in request.steps],
        timestamps=[s.performed_at or "" for s in request.steps],
        outcomes=[],
        adherence_score=0.0,
    )

    try:
        comparison = pathway_service.compare_observed_to_recommended(
            observed, request.guideline_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    audit_log.record(
        "guideline.adherence", actor=actor_id(_key),
        guideline_id=request.guideline_id, patient_id=request.patient_id,
        adherence_score=comparison["adherence_score"],
        n_required=comparison["n_required"])

    return {
        "status": "success",
        "comparison": comparison,
        "opportunities": pathway_service.highlight_optimization_opportunities(
            comparison),
    }


@app.get("/api/pathways/guidelines/{guideline_id}/evidence")
async def guideline_evidence(
    guideline_id: str,
    per_step: int = Query(3, ge=1, le=10),
    searcher: EvidenceSearcher = Depends(get_evidence_searcher),
    _key: Optional[str] = Depends(require_api_key),
):
    """Retrieve current evidence for each step of a guideline.

    This is the join the project is named for and did not have: the
    evidence graph and the outcomes side ran entirely separately, so
    "the literature says X" and "this pathway recommends Y" could not be
    put next to each other. Each recommended step is used as a retrieval
    query, and what comes back is reported with the mode that produced it
    -- a step with no supporting evidence in the corpus is a finding worth
    seeing, and is not the same as a step the literature does not support.
    """
    if guideline_id not in pathway_service.guidelines:
        raise HTTPException(
            status_code=404,
            detail=f"Guideline {guideline_id!r} is not registered. Known: "
                   f"{sorted(pathway_service.guidelines)}")

    pathway = pathway_service.guidelines[guideline_id]
    steps = [n for n in pathway.nodes
             if n["type"] != "decision" and n.get("recommended", True)]

    findings = []
    for step in steps:
        query = f"{pathway.condition} {step['name']}"
        response = await searcher.search(query, per_step)
        findings.append({
            "step": step["name"],
            "query": query,
            "retrieval_mode": response.mode,
            "evidence": [
                {"citation": r.get("citation", r.get("id")),
                 "title": r.get("title", ""),
                 "url": r.get("url", "")}
                for r in response.results
            ],
            "supporting_records": len(response.results),
        })

    unsupported = [f["step"] for f in findings if not f["supporting_records"]]
    return {
        "status": "success",
        "guideline_id": guideline_id,
        "condition": pathway.condition,
        "steps_examined": len(findings),
        "findings": findings,
        "steps_with_no_evidence_in_corpus": unsupported,
        "note": (
            "Retrieval over the indexed corpus. A step with no supporting "
            "records means this corpus does not cover it, which is not a "
            "statement about the literature or about the recommendation."
        ),
    }


@app.get("/api/audit")
async def read_audit(
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = Query(None),
    _key: Optional[str] = Depends(require_api_key),
):
    """Recent audit events, newest first.

    Metadata only: who, when, which action, over how many patients. No
    patient content is recorded, so none can be read back out.
    """
    return {
        "status": "success",
        "events": audit_log.read(limit=limit, action=action),
        "note": ("Metadata only. Actors are a hash of the API key, so the "
                 "log is not a list of live credentials."),
    }


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------

@app.get("/api/evidence/search")
async def search_evidence(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
    filters: str = Query('{}', description="JSON filters for search"),
    searcher: EvidenceSearcher = Depends(get_evidence_searcher),
    _key: Optional[str] = Depends(require_api_key),
):
    """Search published evidence.

    GET, because it has no request body — this also matches what
    `src/frontend_interface.py` has always called, which the previous POST
    declaration answered with 405.
    """
    try:
        parsed_filters = json.loads(filters)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in filters parameter")

    try:
        response = await searcher.search(query, limit)
    except RuntimeError as e:
        # An upstream failure is reported as one. Returning [] here would be
        # read as "no evidence exists for this query".
        logger.error(f"Evidence search failed: {e}")
        raise HTTPException(status_code=502, detail=f"Evidence source unavailable: {e}") from e

    audit_log.record(
        "evidence.search", actor=actor_id(_key), query=query,
        limit=limit, mode=response.mode, n_results=len(response.results))

    return {
        "status": "success",
        "query": query,
        "filters": parsed_filters,
        "results": response.results,
        "total_results": len(response.results),
        "retrieval_mode": response.mode,
        "graph_context": response.graph_context,
        "note": response.note,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
