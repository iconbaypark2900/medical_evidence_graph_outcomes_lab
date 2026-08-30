# LIAISON PROJECT BRIEF — medical_evidence_graph_outcomes_lab

> Machine: DGX Spark | Org: dataScience | Phase: working system
> Path: `/home/iconbaypark2900/medical_evidence_graph_outcomes_lab`
> Last updated: 2026-08-29

---

## Problem statement

Evidence-based medicine platform: retrieval over indexed medical
literature, outcomes and cohort analytics, and guideline adherence, with
a knowledge graph joining them.

---

## Happy path

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -m pytest
```

576 tests. 13 of them need `docker compose up -d` (Neo4j, OpenSearch,
Qdrant) and skip with a message when it is not running.

```bash
.venv/bin/python start_system.py     # API + frontend, loopback only
```

---

## Non-goals

- Real patient EHR records. Free-text fields are screened for direct
  identifiers and a payload carrying them is rejected.
- HIPAA-certified deployment. Authentication is a shared API key; there
  is no per-role or per-dataset authorization.

---

## Validation profile

| Field | Value |
|-------|-------|
| Profile | `python` |
| Command | `cd /home/iconbaypark2900/medical_evidence_graph_outcomes_lab && .venv/bin/python -m pytest` |
| CI | GitHub Actions: unit matrix on 3.11/3.13, plus an integration job that brings up docker-compose |

---

## Current state

Working and tested: evidence ingestion (PubMed, ClinicalTrials.gov,
incremental), storage into Neo4j/OpenSearch/Qdrant, hybrid graph-RAG
retrieval, survival and causal analytics, risk models with held-out
metrics, guideline adherence, knowledge graph embeddings evaluated
against baselines, PHI screening, audit trail, Prometheus metrics.

Not implemented, and the config says so rather than implying otherwise:
OIDC, Open Policy Agent, Vault, Langfuse.

The corpus is 2874 documents and 26,984 triples (measured 2026-08-30) from
120 search terms across twelve clinical areas. It was 48 documents and 460
triples, which was the binding constraint on whether any retrieval or
embedding number meant anything: the held-out set was 32 triples and the
gate could not separate a model from the frequency baseline. It now can.

Reproduce or extend with:
`python -m src.integration --terms-file data/ingestion_terms.txt --max-per-source 12`

---

## Open risks

| Risk | Mitigation |
|------|------------|
| biomedical | Public literature only; no PHI accepted, and screening rejects payloads carrying direct identifiers |
| no-real-patient-data | Enforced at the API boundary, not just documented |
| evidence-provenance | Every retrieval result carries its citation and the retrieval mode that produced it |

---

## Next actions

- Real authentication if this is to be reachable by anyone else. The API now
  refuses rather than serving open when no keys are configured, but a valid
  key still grants every endpoint: no roles, no per-dataset scope, no tenancy.
- DistMult loses to the frequency baseline on mean rank (417.9 against 62.9)
  while beating it on every ranking metric. Worth understanding before the
  suggestions are relied on for anything that uses the whole ordering.

---

## Related

- [README.md](/home/iconbaypark2900/medical_evidence_graph_outcomes_lab/README.md) — setup, endpoints, and what is and is not implemented
- [tests/README.md](/home/iconbaypark2900/medical_evidence_graph_outcomes_lab/tests/README.md) — what the suite covers and why
