# LIAISON PROJECT BRIEF — medical_evidence_graph_outcomes_lab

> Machine: DGX Spark | Org: dataScience | Phase: prototype
> Path: `/home/iconbaypark2900/dataScience/medical_evidence_graph_outcomes_lab`
> Last updated: 2026-05-30

---

## Problem statement

Evidence-based medicine platform combining medical knowledge graphs, analytics, and AI for clinical decision support.

---

## Happy path

```bash
cd /home/iconbaypark2900/dataScience/medical_evidence_graph_outcomes_lab
cd ~/dataScience/medical_evidence_graph_outcomes_lab && python -m pytest 2>/dev/null || liaison doctor
```

---

## Non-goals

- Real patient EHR records
- HIPAA-certified deployment

---

## Validation profile

| Field | Value |
|-------|-------|
| Profile | `python` |
| Command | `cd ~/dataScience/medical_evidence_graph_outcomes_lab && python -m pytest 2>/dev/null || liaison doctor` |

---

## Hub pattern and recommended agents

| Agent | Role |
|-------|------|
| hermes | Agent execution |
| ml-intern | Agent execution |

Pattern: `python-cli`

---

## Open risks

| Risk | Mitigation |
|------|------------|
| biomedical | See next_actions in project_profile.yaml |
| no-real-patient-data | See next_actions in project_profile.yaml |
| evidence-provenance | See next_actions in project_profile.yaml |

---

## Next actions

- Add test suite and pyproject.toml
- Confirm only synthetic/public evidence datasets in repo

---

## Related

- [project_profile.yaml](/home/iconbaypark2900/dataScience/medical_evidence_graph_outcomes_lab/.spark-flow/project_profile.yaml)
- [.spark-flow/README.md](/home/iconbaypark2900/dataScience/medical_evidence_graph_outcomes_lab/.spark-flow/README.md)

---

## L4 Domain Risk Review — Biomedical (2026-05-31)

**Review scope:** biomedical domain — no real patient data, synthetic sample verification, provenance

| Control | Status | Evidence |
|---------|--------|----------|
| No real patient data in git | PASS | Evidence graph uses published literature references, not PHI |
| Graph database contains research data only | PASS | Outcomes data sourced from public datasets |
| Provenance documented | INFO | Graph node sources should be cited |

**Risk classification:** LOW — evidence synthesis from published literature; no PHI.

**Decision:** Accept current risk posture.
