# Medical Evidence Graph & Outcomes Insight Lab

A comprehensive platform for evidence-based medicine that combines medical knowledge graphs, advanced analytics, and AI-powered research assistance to improve clinical decision-making and patient outcomes.

## Overview

The Medical Evidence Graph & Outcomes Insight Lab is a sophisticated platform designed to:
- Ingest and process medical evidence from diverse sources (trials, publications, guidelines)
- Build and maintain knowledge graphs of medical entities and relationships
- Provide AI-powered research assistance with citations and context
- Analyze patient outcomes and compare observed care to recommended guidelines
- Support evidence-based clinical decision-making

## Architecture

### Tech Stack
- **Graph Database**: Neo4j for knowledge graph storage and traversal
- **Search**: OpenSearch for keyword search, Qdrant for semantic search
- **ML/AI**: PyTorch for outcome models, lifelines for survival analysis
- **Security**: HashiCorp Vault for secrets, Open Policy Agent for access control, Microsoft Presidio for PHI detection
- **Observability**: MLflow, Langfuse, Prometheus + Grafana

### Core Services

#### 1. Evidence Ingestion Service
- Ingests clinical trial registries, PubMed/Medline, guideline sites, payer/quality datasets
- Parses, normalizes, and enriches data with medical ontologies
- Maps entities: condition, intervention, outcome, cohort definition, setting
- Emits events for downstream processing

#### 2. Evidence Graph Service
- Maintains Neo4j graph with medical entities:
  - Nodes: `Condition`, `Intervention`, `Drug`, `Procedure`, `Outcome`, `Population`, `Trial`, `Guideline`, `Provider`, `Organization`
  - Edges: `TREATS`, `ASSOCIATED_WITH`, `CONTRAINDICATED_IN`, `RECOMMENDED_FOR`, `HAS_OUTCOME`, `DERIVED_FROM`, `SUPPORTED_BY`
- Uses KGE and GNN-based methods for link suggestions and conflict detection

#### 3. Graph-RAG Service
- Hybrid retrieval combining:
  - BM25 search via OpenSearch
  - Vector search via Qdrant for semantic queries
  - Graph traversal in Neo4j for contextual retrieval
- Returns answers with citations and graph context
- Ensures explainability by design

#### 4. Outcomes & Cohort Analytics Service
- Cohort builder with inclusion/exclusion criteria
- Survival analysis (Kaplan-Meier, Cox regression)
- Comparative effectiveness analysis
- Subgroup and disparity analysis
- Configurable outcome definitions

#### 5. Pathway & Guideline Service
- Represents guidelines as machine-readable graphs and rules
- Compares observed care vs. recommended pathways
- Identifies optimization opportunities
- Highlights outdated recommendations based on new evidence

## Status

| Area | State |
|---|---|
| Survival analysis (Kaplan-Meier, Cox) | working; tested against simulated cohorts with known hazard ratios |
| Causal inference (propensity matching, ATE) | working; tested against a known treatment effect under confounding |
| Risk models (train / score, held-out metrics) | working, via `/api/models/risk/train` |
| Evidence retrieval (PubMed, ClinicalTrials.gov) | working, live |
| Streamlit clinical frontend | working against the API above |
| Evidence ingestion service | working; entities come from curated NLM/registry metadata, not keyword matching |
| Outcomes analytics service | working; cohort criteria applied for real, log-rank tests for group comparison |
| Evidence graph service | link suggestion working (Adamic-Adar over the real graph); **KGE embeddings not implemented** — `recompute_kge_features` raises rather than returning random vectors |
| Evidence storage (Neo4j + OpenSearch + Qdrant) | working — `src/integration.py` indexes into all three |
| Graph-RAG hybrid retrieval | working — BM25 + vector + graph traversal, fused, with citations |
| Evidence search API + frontend | working — served from the index, with a labelled live fallback |
| Pathway & guideline service | working, and reachable — adherence, plus evidence per guideline step |
| API authentication | API key + restricted CORS; OIDC/OPA still not integrated |
| Vault / OPA / Presidio / MLflow / Langfuse | not integrated |

## Setup

### Development environment (this is the path that is tested)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,ui]'
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch  # CPU-only
.venv/bin/python -m pytest
```

`pyproject.toml` installs only the dependencies the exercised code paths
import. It is deliberately narrower than `requirements.txt`, which lists the
full intended stack (Neo4j, OpenSearch, Qdrant, transformers, spacy, MLflow,
DoWhy, EconML) — none of which any tested path imports yet.

See [`tests/README.md`](tests/README.md) for what the suite covers and why
several tests assert that the same request twice returns the same answer.

### Backing services (for graph-RAG)

```bash
.venv/bin/pip install -e '.[graph]'
docker compose up -d          # Neo4j, OpenSearch, Qdrant
```

Qdrant is published on **6343**, not its default 6333, which is commonly
already taken by another Qdrant on a developer machine. Neo4j credentials
are `neo4j` / `password123`, set in `docker-compose.yml` and matched in
`config/settings.json`.

Everything except retrieval runs without these — the analysis API, the
risk models and the frontend need no backing stores.

## Usage

### API

```bash
.venv/bin/python -m uvicorn src.api_backend:app --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Authentication

```bash
MEG_API_KEYS="a-long-random-key" .venv/bin/python -m uvicorn src.api_backend:app --port 8000
MEG_API_KEY="a-long-random-key"  .venv/bin/python -m streamlit run src/frontend_interface.py
```

Every analysis endpoint requires `X-API-Key`; `/`, `/health` and
`/api/health` stay open so a load balancer can probe them. Keys come from
`MEG_API_KEYS` (comma-separated) or `security.api_keys` in the config.

With no keys configured the API still runs — you have to be able to
develop against it — but it says so: a warning at startup, a banner in
the frontend, and `"authentication": "disabled"` in the health payload.
It is unauthenticated, not silently unauthenticated.

CORS is restricted to `security.allowed_origins` (or `MEG_ALLOWED_ORIGINS`),
defaulting to the local Streamlit frontend. It was previously `["*"]` with
credentials allowed, which lets any page on the internet make
authenticated requests on a viewer's behalf.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness, including whether a risk model has been trained |
| `POST /api/models/risk/train` | Fit risk models on labelled patients; returns held-out AUC / R² |
| `POST /api/patients/risk-assessment` | Score patients with the trained model (503 until one exists) |
| `POST /api/survival-analysis/kaplan-meier` | KM curve, median survival, survival at a horizon |
| `POST /api/survival-analysis/cox-regression` | Hazard ratios with 95% CIs and p-values |
| `POST /api/causal-inference/ate-estimation` | ATE: unadjusted, matched, regression-adjusted |
| `POST /api/cohorts/compare` | Two-cohort comparison with a named statistical test |
| `POST /api/outcomes/cohort` | Apply inclusion/exclusion criteria; KM curve with Greenwood intervals |
| `POST /api/outcomes/comparative-effectiveness` | Log-rank comparison between arms, with NNT |
| `POST /api/pathways/guidelines` | Register a guideline as a machine-readable pathway |
| `POST /api/pathways/adherence` | Score observed care against a guideline |
| `GET /api/pathways/guidelines/{id}/evidence` | **Current evidence for each step of a guideline** |
| `GET /api/evidence/search` | Retrieval over the indexed corpus, live fallback |

**Every analysis endpoint requires the observed outcome as part of the
request.** A survival request carries `follow_up` per patient, a causal
request carries `treatment_assigned` and `outcome_value`, a cohort
comparison carries named `outcomes`. Omitting them is a 422, not a result.
This is the central design rule of the API: it analyses the data you gave
it, or it fails. It does not generate substitutes.

Risk assessment additionally requires a model to have been trained, and
returns each score together with that model's held-out performance and a
list of any features that had to be imputed.

Trained models and registered guidelines persist to `MEG_MODEL_STORE`
(default `.model_store/`) and are restored at startup, so a restart does
not silently discard them. A model written by a different scikit-learn
version is refused rather than loaded: unpickling an estimator across
versions usually works, and when it does not it produces wrong
predictions rather than an error — while the response still carries the
held-out AUC measured before.

### Both together

```bash
.venv/bin/python start_system.py                    # loopback only
MEG_API_KEYS="a-long-random-key" \
  .venv/bin/python start_system.py --host 0.0.0.0   # needs keys, refuses without
```

Binding beyond `127.0.0.1` requires API keys and is refused without them.

### Frontend alone

```bash
.venv/bin/python -m streamlit run src/frontend_interface.py
```

Cohorts are supplied as CSV uploads. Required columns are listed on each
page; `src/cohort_io.py` converts them and reports the offending row and
column when a file cannot be used.

### Graph-RAG

Ingest from the live APIs and index into all three stores:

```bash
# First run for a term: fetches everything matching.
.venv/bin/python -m src.integration --term "metformin cardiovascular outcomes"

# Later runs: only what PubMed added or the registry revised since.
.venv/bin/python -m src.integration --term "metformin cardiovascular outcomes" --incremental
```

`--incremental` keeps a per-term watermark in `.ingest_state.json` and
narrows both sources to that window (PubMed by Entrez date, the registry
by last-update date). Re-indexing an unchanged record updates it in
place, so overlapping windows are harmless.

There is no in-process scheduler. `config/settings.json` declares a
`schedule` for the ingestion service; that value is the crontab line to
use, and this command is what it should run:

```cron
0 2 * * *  cd /path/to/repo && .venv/bin/python -m src.integration --term "..." --incremental
```

Without a refresh the corpus silently stops representing the literature
while retrieval keeps answering from it — which is why
`/api/evidence/search` reports `retrieval_mode` on every response.

Then query. Retrieval runs BM25 (OpenSearch), vector search (Qdrant,
`all-MiniLM-L6-v2`) and graph traversal (Neo4j) over the same corpus and
combines them with reciprocal rank fusion — rank-based, because a BM25
score and a cosine similarity are not on the same scale:

```bash
.venv/bin/python -m src.graph_rag_service.main
```

Each result carries its citation (PMID or NCT id), which retrievers found
it and at what rank, and the condition/intervention/outcome context from
the graph. The reported `coverage` measures how many retrievers agreed —
it is a statement about retrieval consensus, not about whether the
evidence answers the question.

### Analysis library directly

```bash
.venv/bin/python -m src.enhanced_ml_models   # survival + causal + deep survival demo
.venv/bin/python -m src.phase2_demo          # fuller Phase 2 walkthrough
```

### Other services

```bash
.venv/bin/python main.py evidence_ingestion_service   # live PubMed + ClinicalTrials.gov
.venv/bin/python main.py evidence_graph_service       # in-memory graph, link suggestion
.venv/bin/python main.py graph_rag_service            # needs docker compose + an index
.venv/bin/python main.py outcomes_analytics_service   # cohort analytics demo
.venv/bin/python main.py pathway_guideline_service    # not yet reviewed
```

### Metrics

`GET /metrics` serves Prometheus exposition — request counts and
durations by route, analyses and patients processed by kind, PHI
rejections, and the loaded risk model's held-out metric. Open like the
health endpoints, since a scraper cannot present a key and the series are
counts, never patient content.

Requests are labelled by route template rather than concrete path, so a
guideline id does not create a new time series.

MLflow experiment tracking is optional and off (`observability.mlflow`);
enabling it without the `tracking` extra raises at startup. Langfuse was
removed rather than implemented — it traces LLM and RAG generation, and
this system has no generation in it.

### Audit trail

Every analysis records who ran it, when, over how many patients, and with
which model or guideline — to `MEG_AUDIT_LOG` (default
`.audit/audit.jsonl`), append-only, one JSON object per line. Ingestion
runs record themselves too, since "ingested datasets" is the first thing
the promise names.

The active file rotates at 10 MB and keeps 5 generations
(`MEG_AUDIT_LOG` sets the path). That is a retention decision as much as
an operational one: past those generations the history is gone, so ship
the files somewhere durable if the record has to outlive the host.
Reading spans rotations, since a reader that only saw the active file
would lose history the moment it first rotated.

**Metadata only.** Passing anything shaped like a patient record is
refused, by shape rather than by field name: an audit log that accumulates
the data it audits becomes the largest copy of that data in the system,
in the file least likely to be access-controlled. Actors are a hash of the
API key, so the log is not a list of live credentials.

`GET /api/audit` reads it back, and the **Audit Trail** page shows it.

### PHI screening

Every endpoint that accepts patient data screens its free-text fields —
`patient_id`, `race`, `previous_treatments`, `medication_list`, and the
keys of `lab_values` — and rejects a payload carrying direct identifiers
with a 422 that names the field and the kind but never the matched text.

The default `patterns` backend matches national ids, emails, phone
numbers, dates of birth, and long digit runs. **It does not detect names
or addresses.** `security.phi.backend = "presidio"` uses Presidio's NLP
recognisers and needs the `phi` extra; selecting it without installing
that extra raises at startup rather than quietly scanning nothing.

`/api/health` reports what is actually being done, including that
limitation. The config previously declared `presidio.enabled: true` with
no Presidio code anywhere in the repository.

## Security, Privacy & Governance

- De-identified or synthetic data for patient-level pipelines
- PHI detection and masking via Presidio
- Access control via OPA with organization, role, jurisdiction, and dataset-level policies
- Full audit trails of ingested datasets, transformations, and queries
- Deployable in single-tenant, VPC, or on-prem setups

## Non-Functional Requirements

- **Explainability by Design**: No unexplained black-box recommendations
- **High Integrity**: Versioned datasets, schemas, and models
- **Scalability**: Graph and search infrastructure to support large corpora
- **Regulatory Compliance**: Configurable data residency and regulatory alignment (HIPAA, GDPR)

## Contributing

See the project documentation for details on how to contribute to the Medical Evidence Graph & Outcomes Insight Lab.

## License

[To be determined]