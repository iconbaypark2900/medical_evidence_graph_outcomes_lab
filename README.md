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

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up required services (Neo4j, OpenSearch, Qdrant)
4. Configure settings in `config/settings.json`
5. Run individual services as needed

## Usage

Each service can be run independently or as part of the integrated pipeline:

```bash
# Run evidence ingestion service
cd src/evidence_ingestion_service && python main.py

# Run evidence graph service
cd src/evidence_graph_service && python main.py

# Run graph-RAG service
cd src/graph_rag_service && python main.py

# Run outcomes analytics service
cd src/outcomes_analytics_service && python main.py

# Run pathway & guideline service
cd src/pathway_guideline_service && python main.py
```

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