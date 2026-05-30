# Medical Evidence Graph & Outcomes Insight Lab
- **ML & Analytics:**
- PyTorch for outcome models (risk scores, survival, uplift).
- Lifelines / survival analysis libraries for time-to-event modeling.
- Causal inference/Uplift: DoWhy/EconML-style components (if included locally).
- **Security & Governance:**
- Vault for key and secret management.
- OPA for access policies (role, organization, geography, data sensitivity).
- Presidio for PHI detection/masking when ingesting semi-structured data.
- **Observability:**
- MLflow for experiment/model tracking.
- Langfuse for LLM/RAG traces.
- Prometheus + Grafana + OpenSearch for metrics, logs, traces.


## 5. Core Services & Components


### 5.1 Evidence Ingestion & Normalization Service


- Ingest sources:
- Clinical trial registries, PubMed/Medline, guideline sites, payer/quality datasets.
- De-identified EMR/claims or registry data (where provided).
- Functions:
- Parse, normalize, and enrich with medical ontologies.
- Map entities: condition, intervention, outcome, cohort definition, setting.
- Emit events: `evidence.ingested`, `dataset.updated`.


### 5.2 Evidence Graph Builder


- Construct and maintain Neo4j graph:
- Nodes: `Condition`, `Intervention`, `Drug`, `Procedure`, `Outcome`, `Population`, `Trial`, `Guideline`, `Provider`, `Organization`.
- Edges: `TREATS`, `ASSOCIATED_WITH`, `CONTRAINDICATED_IN`, `RECOMMENDED_FOR`, `HAS_OUTCOME`, `DERIVED_FROM`, `SUPPORTED_BY`.
- Use KGE and GNN-based methods to:
- Suggest related interventions and outcomes for given cohorts.
- Detect conflicting or weakly supported recommendations.


### 5.3 Graph-RAG & Evidence Query Service


- Hybrid retrieval combining:
- BM25 search over indexed documents (OpenSearch).
- Embedding search via Qdrant for semantic queries.
- Graph traversal in Neo4j to constrain/expand relevant evidence subgraphs.
- All answers include:
- Cited sources (trials, guidelines, studies).
- Graph context (e.g., condition → intervention → outcomes → populations).
- Confidence and coverage indicators.


### 5.4 Outcomes & Cohort Analytics Service


- Cohort builder supporting:
- Inclusion/exclusion criteria (diagnoses, procedures, demographics, comorbidities).
- Time windows, follow-up rules, and outcome definitions.
- Analytics:
- Survival curves, hazard ratios, incidence rates.
- Comparative effectiveness (A vs B vs usual care) with adjustment methods.
- Subgroup analysis and disparity analysis.
- Outputs can be exported, versioned, and attached to decisions or guidelines.


### 5.5 Pathway & Guideline Support Service


- Represent guidelines and pathways as machine-readable graphs and rules.
- Evaluate adherence and variance against observed data and outcomes.
- Highlight opportunities to:
- Update pathways when evidence accumulates.
- Identify low-value or harmful variations in care.


## 6. Security, Privacy & Governance


- De-identified or synthetic data for patient-level pipelines unless explicit agreements.
- PHI detection and masking via Presidio where needed.
- Access control via OPA with:
- Organization, role, jurisdiction, and dataset-level policies.
- Full audit trails of ingested datasets, transformations, and queries.
- Deployable in single-tenant or on-prem/VPC setups for sensitive partners.


## 7. Non-Functional Requirements


- Explainability by design: no unexplained black-box recommendations.
- High integrity: versioned datasets, schemas, and models.
- Scalable graph and search infrastructure to support large corpora.
- Configurable data residency and regulatory alignment (HIPAA, GDPR where applicable).