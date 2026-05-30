"""
Evidence Ingestion Service

This service handles:
- Ingest sources: Clinical trial registries, PubMed/Medline, guideline sites, payer/quality datasets
- Functions: Parse, normalize, and enrich with medical ontologies
- Map entities: condition, intervention, outcome, cohort definition, setting
- Emit events: evidence.ingested, dataset.updated
"""