"""Shared fixtures.

Every synthetic dataset here is generated from a *known* data-generating
process with a seeded RNG. That is deliberate: a test that only asserts
"the estimator returned a number" cannot tell a working estimator from a
broken one. These fixtures let the tests assert that the estimators
recover the parameters they were given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEED = 20260828

# Ground-truth coefficients used to simulate the survival cohort.
# Tests assert the Cox model recovers these.
TRUE_BETA_TREATMENT = -0.70   # hazard ratio ~0.497 (treatment is protective)
TRUE_BETA_AGE_Z = 0.40        # hazard ratio ~1.492 per SD of age
BASELINE_HAZARD = 0.05

# Ground-truth average treatment effect used to simulate the causal cohort.
TRUE_ATE = 2.0


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def survival_cohort() -> pd.DataFrame:
    """Right-censored cohort simulated from an exponential PH model.

    With a constant baseline hazard, T | x ~ Exponential(rate=h0*exp(b'x)),
    so T = -log(U) / (h0 * exp(b'x)). Cox regression should recover b.
    """
    gen = np.random.default_rng(SEED)
    n = 4000

    age = np.clip(gen.normal(65, 12, n), 18, 95)
    age_z = (age - age.mean()) / age.std()
    treatment = gen.binomial(1, 0.45, n)
    gender = gen.choice(["M", "F"], n, p=[0.48, 0.52])

    linear_predictor = TRUE_BETA_TREATMENT * treatment + TRUE_BETA_AGE_Z * age_z
    rate = BASELINE_HAZARD * np.exp(linear_predictor)
    event_time = -np.log(gen.uniform(size=n)) / rate

    # Administrative censoring: uniform follow-up window.
    censor_time = gen.uniform(1.0, 40.0, n)
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "age_z": age_z,
            "gender": gender,
            "treatment": treatment,
            "observed_time": observed_time,
            "event": event,
            "baseline_risk_score": gen.normal(0.5, 0.2, n),
            "comorbidity_count": gen.poisson(1.5, n),
        }
    )


@pytest.fixture(scope="session")
def unconfounded_cohort() -> pd.DataFrame:
    """Randomised treatment: the unadjusted difference in means is unbiased."""
    gen = np.random.default_rng(SEED + 1)
    n = 3000

    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    treatment = gen.binomial(1, 0.5, n)
    outcome = 1.0 + TRUE_ATE * treatment + 0.5 * x1 - 0.3 * x2 + gen.normal(0, 1, n)

    return pd.DataFrame(
        {"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome}
    )


@pytest.fixture(scope="session")
def confounded_cohort() -> pd.DataFrame:
    """x1 drives both treatment assignment and outcome.

    The unadjusted difference in means is biased upward by roughly
    3.0 * (E[x1 | T=1] - E[x1 | T=0]); adjustment should remove most of it.
    """
    gen = np.random.default_rng(SEED + 2)
    n = 3000

    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    propensity = 1.0 / (1.0 + np.exp(-(1.2 * x1)))
    treatment = gen.binomial(1, propensity)
    outcome = 1.0 + TRUE_ATE * treatment + 3.0 * x1 - 0.3 * x2 + gen.normal(0, 1, n)

    return pd.DataFrame(
        {"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome}
    )


# --------------------------------------------------------------------------
# PubMed EFetch fixture.
#
# Tag names below are the real ones from the NLM PubMed DTD -- in particular
# <ForeName>, not <FirstName>, and a structured abstract carrying several
# <AbstractText> elements. Tests use this to check the parser against the
# format PubMed actually returns.
# --------------------------------------------------------------------------
PUBMED_EFETCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">31234567</PMID>
      <Article PubModel="Print">
        <Journal>
          <ISSN IssnType="Electronic">1533-4406</ISSN>
          <JournalIssue CitedMedium="Internet">
            <Volume>381</Volume>
            <PubDate>
              <Year>2019</Year>
              <Month>Nov</Month>
            </PubDate>
          </JournalIssue>
          <Title>The New England Journal of Medicine</Title>
        </Journal>
        <ArticleTitle>Dapagliflozin in Patients with Heart Failure and Reduced Ejection Fraction</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">In patients with type 2 diabetes, inhibitors of sodium-glucose cotransporter 2 reduce hospitalization for heart failure.</AbstractText>
          <AbstractText Label="METHODS">We randomly assigned 4744 patients with New York Heart Association class II, III, or IV heart failure.</AbstractText>
          <AbstractText Label="RESULTS">Over a median of 18.2 months, the primary outcome occurred in 386 of 2373 patients in the dapagliflozin group.</AbstractText>
          <AbstractText Label="CONCLUSIONS">Among patients with heart failure and a reduced ejection fraction, the risk of worsening heart failure was lower among those who received dapagliflozin.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y">
            <LastName>McMurray</LastName>
            <ForeName>John J V</ForeName>
            <Initials>JJV</Initials>
          </Author>
          <Author ValidYN="Y">
            <LastName>Solomon</LastName>
            <ForeName>Scott D</ForeName>
            <Initials>SD</Initials>
          </Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName UI="D006333" MajorTopicYN="N">Heart Failure</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName UI="D000077203" MajorTopicYN="Y">Sodium-Glucose Transporter 2 Inhibitors</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class _FakeResponse:
    """Stands in for an aiohttp response used as an async context manager."""

    def __init__(self, status: int = 200, text: str = "", payload=None):
        self.status = status
        self._text = text
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text

    async def json(self):
        return self._payload


class FakeSession:
    """Minimal aiohttp.ClientSession stand-in.

    `get` is intentionally NOT async: aiohttp's real `session.get(...)`
    returns an async context manager directly, and the code under test
    relies on that shape.
    """

    def __init__(self, response: _FakeResponse | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.requested_urls: list[str] = []

    def get(self, url, *args, **kwargs):
        self.requested_urls.append(url)
        if self._raises is not None:
            raise self._raises
        return self._response


@pytest.fixture
def fake_response():
    return _FakeResponse


@pytest.fixture
def fake_session():
    return FakeSession


# --------------------------------------------------------------------------
# Backing-stack detection
#
# Tests marked `requires_stack` need the docker-compose services. They skip
# rather than fail when those are down, so the default suite stays runnable
# with nothing but a venv -- but they do NOT skip silently once the stack is
# up, which is the point: the retrieval path has to be exercised against
# real Neo4j, OpenSearch and Qdrant, not against fakes only.
# --------------------------------------------------------------------------

def _probe(url: str, timeout: float = 1.5) -> bool:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _graph_extra_installed() -> bool:
    """Whether the optional `graph` dependencies are importable."""
    import importlib.util

    return all(
        importlib.util.find_spec(module) is not None
        for module in ("neo4j", "opensearchpy", "qdrant_client", "sentence_transformers")
    )


@pytest.fixture(scope="session")
def stack_available() -> bool:
    """Both halves are required, and both are genuinely separable.

    Reachable services are not enough: a machine can have the containers
    up while the venv lacks the optional `graph` extra, and the fixtures
    below then fail at import rather than skipping. Checking only the
    ports made that combination an error instead of a skip.
    """
    import json
    import pathlib as _pathlib

    if not _graph_extra_installed():
        return False

    config = json.loads(_pathlib.Path("config/settings.json").read_text())
    rag = config["services"]["graph_rag_service"]
    qdrant_port = rag["qdrant"].get("port", 6333)
    opensearch_port = rag["opensearch"].get("port", 9200)

    return (
        _probe(f"http://localhost:{opensearch_port}/_cluster/health")
        and _probe(f"http://localhost:{qdrant_port}/readyz")
        and _probe("http://localhost:7474/")
    )


@pytest.fixture
def require_stack(stack_available):
    if not stack_available:
        if not _graph_extra_installed():
            pytest.skip(
                "the optional graph extra is not installed; "
                "`pip install -e '.[graph]'`")
        pytest.skip(
            "Neo4j/OpenSearch/Qdrant not reachable; run `docker compose up -d`")
