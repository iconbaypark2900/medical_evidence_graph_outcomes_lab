# Medical Evidence Graph & Outcomes Lab Workflow

This document describes how the Medical Evidence Graph & Outcomes Lab works, the processes it follows, and the workflows it supports.

## Overview

The Medical Evidence Graph & Outcomes Lab implements a **Search → Extract → Synthesize → Graph → Report** workflow:

```
Search Literature → Extract Evidence → Assess Quality → Synthesize Findings → Build Knowledge Graph → Generate Report
```

## Core Workflows

### 1. Literature Search

#### Search PubMed

```bash
# Search PubMed for articles
curl -X POST http://localhost:8080/api/search/pubmed \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "drug X hypertension RCT",
    "date_from": "2020-01-01",
    "date_to": "2026-09-25",
    "max_results": 100
  }'
```

#### Search Clinical Trials

```bash
# Search ClinicalTrials.gov
curl -X POST http://localhost:8080/api/search/clinical_trials \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "drug X hypertension",
    "status": "completed",
    "max_results": 50
  }'
```

#### Search arXiv

```bash
# Search arXiv for preprints
curl -X POST http://localhost:8080/api/search/arxiv \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning drug discovery",
    "categories": ["q-bio.QM", "cs.AI"],
    "max_results": 50
  }'
```

### 2. Evidence Extraction

#### Extract PICO Elements

```bash
# Extract PICO elements from a paper
curl -X POST http://localhost:8080/api/extract/pico \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "text": "Full text of the paper"
  }'
```

#### Extract Entities

```bash
# Extract entities from a paper
curl -X POST http://localhost:8080/api/extract/entities \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "entity_types": ["drug", "disease", "gene", "protein"]
  }'
```

#### Extract Relationships

```bash
# Extract relationships from a paper
curl -X POST http://localhost:8080/api/extract/relationships \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "relationship_types": ["treats", "causes", "diagnoses"]
  }'
```

### 3. Quality Assessment

#### Assess Study Design

```bash
# Assess study design
curl -X POST http://localhost:8080/api/quality/study_design \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "text": "Full text of the paper"
  }'
```

#### Assess Risk of Bias

```bash
# Assess risk of bias
curl -X POST http://localhost:8080/api/quality/risk_of_bias \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "study_design": "RCT"
  }'
```

#### Assign GRADE

```bash
# Assign GRADE rating
curl -X POST http://localhost:8080/api/quality/grade \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "pubmed-123456",
    "study_design": "RCT",
    "risk_of_bias": "low",
    "inconsistency": "low",
    "indirectness": "low",
    "imprecision": "low",
    "publication_bias": "low"
  }'
```

### 4. Evidence Synthesis

#### Perform Meta-analysis

```bash
# Perform meta-analysis
curl -X POST http://localhost:8080/api/synthesis/meta_analysis \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_ids": ["pubmed-123456", "pubmed-123457", "pubmed-123458"],
    "outcome": "systolic_bp",
    "model": "random_effects"
  }'
```

#### Generate Systematic Review

```bash
# Generate systematic review
curl -X POST http://localhost:8080/api/synthesis/systematic_review \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "research_question": "Does drug X reduce blood pressure in adults with hypertension?",
    "search_strategy": "PubMed, Embase, Cochrane",
    "inclusion_criteria": "RCT, adults, hypertension, drug X",
    "exclusion_criteria": "Children, other interventions"
  }'
```

#### Analyze Heterogeneity

```bash
# Analyze heterogeneity
curl -X POST http://localhost:8080/api/synthesis/heterogeneity \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_ids": ["pubmed-123456", "pubmed-123457", "pubmed-123458"],
    "outcome": "systolic_bp"
  }'
```

### 5. Knowledge Graph

#### Build Graph

```bash
# Build knowledge graph from papers
curl -X POST http://localhost:8080/api/graph/build \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "paper_ids": ["pubmed-123456", "pubmed-123457", "pubmed-123458"],
    "entity_types": ["drug", "disease", "gene"],
    "relationship_types": ["treats", "causes", "expresses"]
  }'
```

#### Query Graph

```bash
# Query knowledge graph
curl -X POST http://localhost:8080/api/graph/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "MATCH (d:Drug {name: \"Drug X\"})-[:TREATS]->(disease:Disease) RETURN disease",
    "language": "cypher"
  }'
```

#### Visualize Graph

```bash
# Visualize knowledge graph
curl -X POST http://localhost:8080/api/graph/visualize \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": ["Drug X", "Hypertension", "ACE Inhibitors"],
    "edges": ["TREATS", "CLASS_OF"],
    "format": "svg"
  }'
```

### 6. Reporting & Visualization

#### Generate Report

```bash
# Generate evidence synthesis report
curl -X POST http://localhost:8080/api/report/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "synthesis_id": "synthesis-001",
    "format": "pdf",
    "sections": ["abstract", "introduction", "methods", "results", "discussion", "conclusions"]
  }'
```

#### Generate Dashboard

```bash
# Generate evidence dashboard
curl -X POST http://localhost:8080/api/dashboard/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Drug X for hypertension",
    "visualizations": ["forest_plot", "network_diagram", "timeline"]
  }'
```

#### Export Data

```bash
# Export evidence data
curl -X POST http://localhost:8080/api/export \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "evidence",
    "format": "csv",
    "filters": {
      "study_design": "RCT",
      "grade": "A"
    }
  }'
```

## Research Workflows

### Drug Discovery Workflow

1. **Define Question**: What is the effect of Drug X on blood pressure?
2. **Search Literature**: Search PubMed, ClinicalTrials.gov, arXiv
3. **Extract Evidence**: Extract PICO elements, entities, relationships
4. **Assess Quality**: Assess study design, risk of bias, GRADE
5. **Synthesize**: Perform meta-analysis, generate systematic review
6. **Build Graph**: Build knowledge graph of drugs, diseases, genes
7. **Report**: Generate report, dashboard, export data

### Disease Mechanism Workflow

1. **Define Question**: What are the mechanisms of Disease Y?
2. **Search Literature**: Search for disease mechanisms, genes, pathways
3. **Extract Evidence**: Extract genes, proteins, pathways, relationships
4. **Assess Quality**: Assess evidence quality, consistency
5. **Synthesize**: Generate narrative review, identify key mechanisms
6. **Build Graph**: Build knowledge graph of disease mechanisms
7. **Report**: Generate report with mechanism diagrams

### Treatment Comparison Workflow

1. **Define Question**: Which treatment is better for Condition Z?
2. **Search Literature**: Search for comparative studies
3. **Extract Evidence**: Extract treatment arms, outcomes, effect sizes
4. **Assess Quality**: Assess risk of bias, heterogeneity
5. **Synthesize**: Perform network meta-analysis, ranking
6. **Build Graph**: Build evidence network
7. **Report**: Generate report with treatment rankings

## Quality Control Workflows

### Data Quality Workflow

1. **Validate**: Validate extracted data against source papers
2. **Deduplicate**: Remove duplicate entries
3. **Resolve**: Resolve entity conflicts
4. **Enrich**: Add missing metadata
5. **Verify**: Verify relationships, evidence

### Analysis Quality Workflow

1. **Validate**: Validate statistical methods
2. **Reproduce**: Reproduce results from published studies
3. **Sensitivity**: Test sensitivity to assumptions
4. **Robustness**: Test robustness to outliers
5. **Peer Review**: Internal peer review of analyses

## Best Practices

1. **Pre-register protocols** — Pre-register review protocols
2. **Follow PRISMA** — Follow PRISMA guidelines for systematic reviews
3. **Use GRADE** — Use GRADE for evidence quality assessment
4. **Assess bias** — Assess risk of bias for all studies
5. **Analyze heterogeneity** — Analyze heterogeneity in meta-analyses
6. **Search comprehensively** — Search multiple databases, grey literature
7. **Extract independently** — Extract data independently, resolve conflicts
8. **Verify extraction** — Verify extracted data against source papers
9. **Reproduce analyses** — Reproduce analyses for verification
10. **Document everything** — Document all decisions, methods, results

## Related Documentation

- [README.md](README.md) — Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [MASTER.md](MASTER.md) — Project roadmap
- [PHASE_1_COMPLETE.md](PHASE_1_COMPLETE.md) — Phase 1 completion report
- [PHASE_2_SUMMARY.md](PHASE_2_SUMMARY.md) — Phase 2 summary
- [PHASE_3_COMPLETE_SUMMARY.md](PHASE_3_COMPLETE_SUMMARY.md) — Phase 3 completion summary
