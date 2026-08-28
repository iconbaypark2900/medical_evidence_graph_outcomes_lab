# Test suite

```bash
.venv/bin/python -m pytest              # everything, ~11s from cold
.venv/bin/python -m pytest --cov=src    # with coverage
.venv/bin/python -m pytest -m known_defect   # proven-but-unfixed bugs (currently none)
```

No test here touches the network, Docker, or a database. `tests/conftest.py`
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

## What is deliberately not collected

`pyproject.toml` sets `testpaths = ["tests"]`. Four modules under `src/`
have test-shaped names but are not tests, and would execute on import:

| File | What it actually is |
|---|---|
| `src/test_cox_simple.py` | Debug script; fits a Cox model and prints. |
| `src/debug_cox_data.py` | Debug script; prints dtypes of synthetic data. |
| `src/mock_databases_test.py` | In-memory Neo4j/OpenSearch/Qdrant fakes. |
| `src/mock_integration_test.py` | Phase-1 demo; re-declares those fakes inline. |
| `src/db_connection_test.py` | The `DatabaseManager` class — production code. |

The ground the first two explored is now covered by
`test_survival_analysis.py`, against known ground truth rather than printed
output.

## Adding a test

Put shared synthetic data in `conftest.py` with a fixed seed and a
documented data-generating process, so the expected answer is derivable
rather than recorded from a previous run.
