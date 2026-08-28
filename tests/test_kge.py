"""Knowledge graph embeddings.

`recompute_kge_features` returned `np.random.rand(128)` per node as
"embeddings" and `np.random.random()` as a link confidence. That was
replaced with an honest NotImplementedError, and this is the
implementation it pointed at.

The evaluation is what these tests are mostly about. A link predictor
that is never scored against a baseline produces the same unfalsifiable
suggestions the random confidences did, only in a more convincing
costume. Every model is measured with filtered MRR and Hits@K on held-out
triples against two baselines, and one that loses is reported as losing
and is not served.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.kge import (
    DistMult,
    Evaluation,
    TransE,
    TripleStore,
    adamic_adar_baseline,
    build_and_evaluate,
    evaluate_ranking,
    frequency_baseline,
    kge_scorer,
    train_kge,
)


def learnable_graph(n_docs: int = 60) -> list:
    """A graph with structure an embedding model can actually learn.

    Documents fall into three topics; each topic has its own conditions and
    interventions. A model that learns topics should rank within-topic
    tails above the rest.
    """
    topics = {
        "cardio": (["heart failure", "hypertension"], ["dapagliflozin", "ramipril"]),
        "onco": (["breast cancer", "melanoma"], ["trastuzumab", "pembrolizumab"]),
        "endo": (["type 2 diabetes", "obesity"], ["metformin", "semaglutide"]),
    }
    triples = []
    for i in range(n_docs):
        topic = list(topics)[i % 3]
        conditions, interventions = topics[topic]
        doc = f"doc_{i}"
        for condition in conditions:
            triples.append((doc, "HAS_CONDITION", condition))
        for intervention in interventions:
            triples.append((doc, "HAS_INTERVENTION", intervention))
    return triples


# --------------------------------------------------------------------------
# Triple store and splitting
# --------------------------------------------------------------------------

def test_the_store_builds_a_vocabulary():
    store = TripleStore([("a", "R", "b"), ("a", "S", "c")])

    assert store.entities == ["a", "b", "c"]
    assert store.relations == ["R", "S"]


def test_encoding_maps_to_indices():
    store = TripleStore([("a", "R", "b")])

    assert store.encode([("a", "R", "b")]).tolist() == [[0, 0, 1]]


def test_an_empty_graph_is_refused():
    with pytest.raises(ValueError, match="nothing to train on"):
        TripleStore([])


def test_the_split_is_reproducible():
    store = TripleStore(learnable_graph())

    first = store.split(seed=7)
    second = store.split(seed=7)

    assert first == second


def test_held_out_triples_with_unseen_entities_go_back_to_training():
    """No embedding model can rank an entity it has no vector for, so
    leaving those in the test set depresses a score for a question the
    model was never given a chance to answer."""
    triples = learnable_graph() + [("lonely_doc", "HAS_CONDITION", "lonely_condition")]
    store = TripleStore(triples)

    train, valid, test = store.split(seed=0)

    seen = {e for h, _, t in train for e in (h, t)}
    for split in (valid, test):
        for head, _, tail in split:
            assert head in seen and tail in seen


def test_the_split_covers_every_triple():
    store = TripleStore(learnable_graph())

    train, valid, test = store.split(seed=1)

    assert len(train) + len(valid) + len(test) == len(store.triples)


# --------------------------------------------------------------------------
# The models
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model_class", [TransE, DistMult])
def test_scoring_returns_one_value_per_triple(model_class):
    torch.manual_seed(0)
    model = model_class(10, 3, dim=8)

    scores = model.score(torch.tensor([0, 1]), torch.tensor([0, 1]),
                         torch.tensor([2, 3]))

    assert scores.shape == (2,)


def test_transe_scores_a_satisfied_translation_highest():
    """h + r ~ t is the whole model; a score that ignored it would be a
    random number generator with extra steps."""
    model = TransE(3, 1, dim=4)
    with torch.no_grad():
        model.entity.weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.entity.weight[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        model.entity.weight[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        model.relation.weight[0] = torch.tensor([-1.0, 1.0, 0.0, 0.0])

    good = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    bad = model.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]))

    assert good.item() > bad.item()


def test_normalisation_bounds_entity_vectors():
    """Without it the loss falls by growing norms rather than by learning."""
    model = TransE(5, 2, dim=8)
    with torch.no_grad():
        model.entity.weight *= 100

    model.normalise()

    norms = model.entity.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_an_unknown_model_is_refused():
    store = TripleStore(learnable_graph())
    with pytest.raises(ValueError, match="Unknown model"):
        train_kge(store, store.triples, model_name="magic")


def test_training_is_deterministic_for_a_seed():
    store = TripleStore(learnable_graph())
    train, _, _ = store.split(seed=0)

    first = train_kge(store, train, epochs=20, seed=3)
    second = train_kge(store, train, epochs=20, seed=3)

    assert torch.allclose(first.entity.weight, second.entity.weight)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def test_a_perfect_scorer_scores_one():
    store = TripleStore([("a", "R", "b"), ("a", "R", "c"), ("d", "R", "b")])

    def perfect(head, relation, candidates):
        # Rank the correct tail top for every query.
        return np.array([10.0 if c == "b" else 0.0 for c in candidates])

    result = evaluate_ranking(perfect, store, [("d", "R", "b")], store.triples, "perfect")

    assert result.mrr == 1.0
    assert result.hits_at_1 == 1.0


def test_filtering_does_not_penalise_a_correct_alternative():
    """Ranking another true tail above the held-out one is being right, and
    an unfiltered metric would score it as being wrong."""
    all_triples = [("a", "R", "b"), ("a", "R", "c"), ("a", "R", "d")]
    store = TripleStore(all_triples)

    def prefers_c(head, relation, candidates):
        return np.array([5.0 if c == "c" else 1.0 for c in candidates])

    # 'c' is also true for (a, R, ?), so it must be filtered out.
    result = evaluate_ranking(prefers_c, store, [("a", "R", "b")], all_triples, "x")

    assert result.hits_at_1 == 1.0


def test_evaluating_nothing_is_an_error():
    store = TripleStore(learnable_graph())

    with pytest.raises(ValueError, match="No test triples"):
        evaluate_ranking(lambda h, r, c: np.zeros(len(c)), store, [], store.triples, "x")


def test_metrics_are_ordered_as_they_must_be():
    store = TripleStore(learnable_graph())
    train, _, test = store.split(seed=0)

    result = evaluate_ranking(frequency_baseline(train), store, test,
                              store.triples, "frequency")

    assert result.hits_at_1 <= result.hits_at_3 <= result.hits_at_10
    assert 0.0 <= result.mrr <= 1.0


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def test_the_frequency_baseline_ranks_by_how_often_a_tail_appears():
    train = [("d1", "R", "common"), ("d2", "R", "common"), ("d3", "R", "rare")]

    scores = frequency_baseline(train)("d4", "R", ["common", "rare", "unseen"])

    assert scores[0] > scores[1] > scores[2]


def test_the_frequency_baseline_is_relation_specific():
    train = [("d1", "R", "x"), ("d2", "S", "y")]

    assert frequency_baseline(train)("d3", "R", ["y"])[0] == 0.0


def test_adamic_adar_rewards_shared_neighbours():
    """It scores two-hop overlap, so head and candidate must share a direct
    neighbour. d1 and d2 both cite `a`; d3 shares nothing.

    This is also why its number in the real evaluation is not a verdict on
    the service's use of it: there it ranks entities against entities, and
    the evaluation asks it for evidence-to-entity tail prediction, which is
    three hops away and not what it measures.
    """
    train = [("d1", "R", "a"), ("d2", "R", "a"), ("d3", "R", "z")]

    scores = adamic_adar_baseline(train)("d2", "R", ["d1", "d3"])

    assert scores[0] > 0
    assert scores[1] == 0


# --------------------------------------------------------------------------
# The gate: a model that loses is not served
# --------------------------------------------------------------------------

def test_a_model_that_beats_the_baselines_is_returned():
    model, store, report = build_and_evaluate(
        learnable_graph(), model_name="distmult", dim=32, epochs=200, seed=0)

    assert report.beats_baselines is (model is not None)
    assert report.evaluation.n_test > 0


def test_a_model_that_loses_is_withheld(monkeypatch):
    """The whole point. Serving a predictor that loses to "suggest whatever
    usually appears" is the random-confidence problem in a better costume.
    """
    import src.kge as kge

    losing = Evaluation("distmult", mrr=0.01, hits_at_1=0.0, hits_at_3=0.0,
                        hits_at_10=0.0, n_test=10, mean_rank=99.0)
    monkeypatch.setattr(
        kge, "evaluate_ranking",
        lambda score_fn, store, test, all_triples, name:
            losing if name == "distmult" else
            Evaluation(name, 0.5, 0.4, 0.5, 0.6, 10, 2.0))

    model, _, report = build_and_evaluate(
        learnable_graph(), model_name="distmult", dim=8, epochs=5)

    assert model is None
    assert report.beats_baselines is False


def test_the_report_names_every_metric_it_lost_on():
    """A model can win on MRR and lose on Hits@10; one boolean hides that
    the baseline finds the answer in its top ten more often."""
    report = build_and_evaluate(
        learnable_graph(), model_name="distmult", dim=32, epochs=100, seed=0)[2]

    described = report.describe()
    assert set(described["comparison"]) == {"frequency", "adamic_adar"}
    for metric in described["loses_to_a_baseline_on"]:
        assert any(not per_metric[metric]
                   for per_metric in described["comparison"].values())


def test_the_report_disclaims_the_adamic_adar_comparison():
    """It is scored here on a task it is not used for in the service."""
    report = build_and_evaluate(
        learnable_graph(), model_name="distmult", dim=16, epochs=20)[2]

    assert "not the entity-to-entity suggestion" in report.describe()["note"]


def test_a_graph_too_small_to_evaluate_is_refused():
    """An unevaluated link predictor is not worth serving, and silently
    skipping the evaluation is how one gets served anyway."""
    with pytest.raises(ValueError, match="too small or too sparse"):
        build_and_evaluate([("a", "R", "b"), ("a", "R", "c")], epochs=2)


def test_the_scorer_returns_one_score_per_candidate():
    store = TripleStore(learnable_graph())
    train, _, _ = store.split(seed=0)
    model = train_kge(store, train, epochs=10, seed=0)

    scores = kge_scorer(model, store)("doc_0", "HAS_CONDITION", store.entities)

    assert scores.shape == (len(store.entities),)
