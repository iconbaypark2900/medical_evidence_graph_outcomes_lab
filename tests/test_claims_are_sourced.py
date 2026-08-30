"""The README's numbers must come from a committed artifact.

The published embedding results once reported an MRR table on a
"607-triple corpus". Nothing in the repository substantiated the 607: the
measured graph had 460 edges, docs/LIAISON_PROJECT_BRIEF.md said ~460, and
the audit log's only recorded count was 559. The figures themselves came
from a real evaluation, but on a graph state that no longer existed and had
never been written down, so no reader and no test could tell.

No unit test could have caught that, because nothing was wrong with the
code. What was wrong was that a number reached the README from a place
nobody could go and check. These tests close that: `python -m src.kge`
writes data/kge_evaluation.json, that file is committed, and the README has
to agree with it.

The tests need no database. They compare two files in the repository, which
is the point -- a claim that can only be checked by rebuilding the world is
a claim that will not be checked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
EVALUATION = REPO / "data" / "kge_evaluation.json"

# The README rounds; the artifact keeps full precision. Comparisons round
# the artifact to whatever precision the README printed, rather than
# allowing a tolerance -- a tolerance has to be picked, and 0.313 against
# 0.3125 lands exactly on any sensible one.

# README label -> key in the evaluation artifact. The published table names
# models the way a reader would; the artifact names them the way the code does.
MODEL_LABELS = {
    "DistMult": ("distmult", "model"),
    "TransE": ("transe", "model"),
    "frequency baseline": ("distmult", "frequency"),
    "Adamic-Adar": ("distmult", "adamic_adar"),
}


@pytest.fixture(scope="module")
def evaluation():
    assert EVALUATION.exists(), (
        f"{EVALUATION.relative_to(REPO)} is missing. The README quotes these "
        f"numbers; regenerate it with `python -m src.kge` against the graph "
        f"and commit it.")
    return json.loads(EVALUATION.read_text())


def _metrics(evaluation, model_key, which):
    """The metric dict for a model or for one of its baselines."""
    report = evaluation["models"][model_key]
    if which == "model":
        return report["model"]
    return next(b for b in report["baselines"] if b["model"] == which)


def _table_after(heading_pattern):
    """Rows of the first markdown table following a matching line."""
    lines = README.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if re.search(heading_pattern, line))
    rows, seen_header = [], False
    for line in lines[start:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                seen_header = True
                continue
            if seen_header:
                rows.append(cells)
        elif rows:
            break
    return rows


def _clean(cell):
    """A table cell without its bold markers or typographic minus."""
    return cell.replace("*", "").replace("−", "-").strip()


def _agrees(published, actual):
    """Is the published figure a correct rounding of the actual one?

    Stated as a half-unit-in-the-last-place bound rather than
    `round(actual, n)`: Python rounds halves to even, so it turns 0.5625
    into 0.562, and a person writing the table writes 0.563. Both are
    correct roundings, and the check is about whether the number came from
    the artifact -- not about which rounding convention was used.
    """
    cleaned = _clean(published)
    decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
    half_ulp = 0.5 * 10 ** -decimals
    return abs(float(cleaned) - actual) <= half_ulp + 1e-9


def test_the_published_corpus_size_matches_the_artifact(evaluation):
    """"460 triples" in the README has to be the graph that was measured."""
    stated = re.search(r"\*\*(\d[\d,]*) triples", README.read_text())
    assert stated, "the README no longer states the corpus size it measured on"

    assert int(stated.group(1).replace(",", "")) == evaluation["graph"]["triples"]


def test_the_published_metrics_match_the_artifact(evaluation):
    """Every figure in the results table came from the committed evaluation."""
    rows = _table_after(r"^\| model \| MRR \| Hits@1")
    assert len(rows) == len(MODEL_LABELS), f"unexpected table shape: {rows}"

    for label, mrr, h1, h3, h10, mean_rank in rows:
        model_key, which = MODEL_LABELS[label]
        actual = _metrics(evaluation, model_key, which)

        for published, key in ((mrr, "mrr"), (h1, "hits_at_1"),
                               (h3, "hits_at_3"), (h10, "hits_at_10"),
                               (mean_rank, "mean_rank")):
            assert _agrees(published, actual[key]), (
                f"README says {label} {key} = {published}, "
                f"data/kge_evaluation.json says {actual[key]}")


def test_the_published_margins_match_the_artifact(evaluation):
    """The margins table is the evidence behind the serving verdict."""
    rows = _table_after(r"^\| model \| MRR margin")
    assert rows, "the README no longer publishes the paired margins"

    for label, margin, ci, decisive in rows:
        model_key, _ = MODEL_LABELS[label]
        against = next(m for m in evaluation["models"][model_key]["margins"]
                       if m["baseline"] == "frequency")

        assert _agrees(margin, against["mrr_margin"]), (
            f"README says {label} margin {margin}, artifact says "
            f"{against['mrr_margin']}")

        low, high = ci.strip("[]").split(",")
        assert _agrees(low, against["ci95"][0])
        assert _agrees(high, against["ci95"][1])
        assert against["decisive"] is ("no" not in decisive.lower())


def test_the_serving_claim_matches_the_artifact(evaluation):
    """"Neither model is served" has to still be what the gate decided."""
    served = [name for name, report in evaluation["models"].items()
              if report["beats_baselines"]]
    claims_none = "Neither model is served" in README.read_text()

    assert claims_none is (not served), (
        f"README says neither model is served; the artifact serves {served}. "
        f"Whichever is stale, they have to agree.")
