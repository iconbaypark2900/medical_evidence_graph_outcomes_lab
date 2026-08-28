# Test suite

```bash
.venv/bin/python -m pytest              # everything, ~11s from cold
.venv/bin/python -m pytest --cov=src    # with coverage
.venv/bin/python -m pytest -m known_defect   # proven-but-unfixed bugs (currently none)
```

Most tests here touch nothing external. The exceptions are marked
`requires_stack` and need `docker compose up -d` (Neo4j, OpenSearch,
Qdrant); they **skip** with an explanatory message when those are not
reachable, so the default suite runs with nothing but a venv:

```bash
.venv/bin/python -m pytest -m requires_stack   # the 13 integration tests
```

CI runs both halves: one job without the stack (where these skip by
design) and one that brings up `docker-compose.yml` and runs them for
real. That second job fails if any of them *skips* — an integration test
that quietly skips reports green while proving nothing.

Those ten are the ones that prove retrieval works against the real
stores rather than against fakes — including that vector search matches a
paraphrase sharing no words with the document, which the previous
`np.random.rand(32)` "embeddings" could not do by construction.

Otherwise, no test touches the network, Docker, or a database. `tests/conftest.py`
drives the PubMed fetchers with a fake `aiohttp` session, and
`test_api_backend.py` swaps the live evidence searcher out through a FastAPI
dependency override.

## What the tests are for

Where an estimator is involved, tests check it against a **known** answer
rather than against itself. `conftest.py` simulates cohorts from a
data-generating process with hazard ratios and an average treatment effect
we chose, so "did Cox work?" has a checkable answer:
`test_cox_recovers_the_simulated_hazard_ratios` asserts the fit returns the
coefficients the data was built from. A test that only checks "a number came
back" cannot tell a working estimator from a broken one — and every bug this
suite has caught so far returned a perfectly plausible number.

| File | Covers |
|---|---|
| `test_survival_analysis.py` | Kaplan-Meier and Cox: recovery of known hazard ratios, curve shape, confidence intervals, failure modes |
| `test_causal_inference.py` | ATE recovery under randomisation and under confounding; propensity matching balance |
| `test_outcome_models.py` | Feature building, multi-task training, the Cox partial likelihood, DeepSurv |
| `test_data_ingestion.py` | PubMed/ClinicalTrials URL building, XML parsing, HTTP status handling |
| `test_cohort_io.py` | Cohort CSV → API payload conversion and its error messages |
| `test_api_backend.py` | Every endpoint: real analysis, validation, and the determinism guards |
| `test_outcomes_analytics.py` | Cohort criteria, Kaplan-Meier, log-rank comparison, NNT |
| `test_evidence_graph.py` | Stable entity ids, structural link suggestion, refusal to fake embeddings |
| `test_evidence_ingestion.py` | The ingestion pipeline over an injected retrieval function |
| `test_evidence_store.py` | Point identity, document shape, write-failure handling |
| `test_graph_rag.py` | Rank fusion, each retriever, and hybrid answers end to end |

## The determinism guards

Several tests post the same request twice and assert the two responses are
identical. They look trivial. They are the regression guard for the defect
this suite was written to catch: the analysis endpoints used to discard the
submitted patients, generate replacement outcomes with `np.random`, and
return `"status": "success"`. A deterministic analysis of a fixed cohort
must return a fixed answer, and these fail loudly if that stops being true.

The same idea covers the library layer, where the tests assert that a
failure *raises* rather than returning a plausible value — an empty survival
curve, an ATE of 0.0, a confidence interval of (0, 0), an empty PubMed
result list. Each of those is a substantive clinical claim when it is
returned by a bug.

## The `known_defect` marker

A bug that is understood but not yet fixed gets a test asserting the
behaviour we *want*, marked `@pytest.mark.known_defect` and
`@pytest.mark.xfail(strict=True)`, with the diagnosis in `reason=`. Because
the xfail is strict, fixing the bug turns the test from XFAIL into a hard
failure (`XPASS(strict)`) — the signal to delete the marker and let the test
stand as an ordinary regression guard.

The suite opened with 18 of these and now has none: all 18 were fixed, and
each marker was removed as its test flipped. Nothing here is skipped or
silently tolerated.

## Everything under src/ is now real

`pyproject.toml` sets `testpaths = ["tests"]`. It used to also have to
work around five modules under `src/` with test-shaped names that were
not tests — mock database implementations, debug scratch, and one
production class called `db_connection_test.py`. All five are gone, along
with `verify_phase1.py`, which existed only to drive the mocks. The real
stores replaced what they stood in for.

## Adding a test

Put shared synthetic data in `conftest.py` with a fixed seed and a
documented data-generating process, so the expected answer is derivable
rather than recorded from a previous run.
