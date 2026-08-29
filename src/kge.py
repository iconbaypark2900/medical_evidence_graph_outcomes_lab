"""Knowledge graph embeddings over the evidence graph.

`recompute_kge_features` used to return `np.random.rand(128)` per node as
"embeddings" and `np.random.random()` as a link confidence. That was
replaced by an honest NotImplementedError, and this is the implementation
it pointed at.

The model is TransE or DistMult in PyTorch rather than a KGE library:
torch is already a core dependency, the models are a few dozen lines
each, and owning the training loop is what makes an honest evaluation
possible.

The evaluation is the point of this module, not the training. A link
predictor that is never scored against a baseline produces exactly the
kind of unfalsifiable "suggestion" the random confidences did -- the
numbers merely look more sophisticated. Every model here is measured with
filtered MRR and Hits@K on held-out triples, against two baselines, and a
model that does not beat them is reported as not beating them.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]  # (head, relation, tail)


@dataclass
class TripleStore:
    """Triples with an entity/relation vocabulary and a reproducible split."""
    triples: List[Triple]
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.triples:
            raise ValueError("No triples supplied; there is nothing to train on")
        self.entities = sorted({h for h, _, _ in self.triples}
                               | {t for _, _, t in self.triples})
        self.relations = sorted({r for _, r, _ in self.triples})
        self.entity_index = {e: i for i, e in enumerate(self.entities)}
        self.relation_index = {r: i for i, r in enumerate(self.relations)}

    def encode(self, triples: Sequence[Triple]) -> torch.Tensor:
        return torch.tensor(
            [[self.entity_index[h], self.relation_index[r], self.entity_index[t]]
             for h, r, t in triples], dtype=torch.long)

    def split(self, valid_fraction: float = 0.1, test_fraction: float = 0.1,
              seed: int = 0) -> Tuple[List[Triple], List[Triple], List[Triple]]:
        """Random split.

        Held-out triples whose entities never appear in training cannot be
        predicted by any embedding model -- there is no vector for them --
        so they are moved back into training rather than left to depress
        the score of a model that was never given a chance at them.
        """
        generator = np.random.default_rng(seed)
        order = generator.permutation(len(self.triples))
        n_test = int(len(order) * test_fraction)
        n_valid = int(len(order) * valid_fraction)

        test = [self.triples[i] for i in order[:n_test]]
        valid = [self.triples[i] for i in order[n_test:n_test + n_valid]]
        train = [self.triples[i] for i in order[n_test + n_valid:]]

        seen = {e for h, _, t in train for e in (h, t)}
        recovered = []
        for split in (valid, test):
            keep = []
            for triple in split:
                if triple[0] in seen and triple[2] in seen:
                    keep.append(triple)
                else:
                    recovered.append(triple)
            split[:] = keep
        train.extend(recovered)

        if recovered:
            logger.info(
                f"Moved {len(recovered)} held-out triples back to training: "
                f"their entities appear nowhere else, so no embedding model "
                f"could rank them")
        return train, valid, test


class TransE(nn.Module):
    """Translational embeddings: a valid triple satisfies h + r ~ t."""

    def __init__(self, n_entities: int, n_relations: int, dim: int = 64,
                 p_norm: int = 1):
        super().__init__()
        self.dim = dim
        self.p_norm = p_norm
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def normalise(self) -> None:
        """Entity vectors are renormalised each step.

        Without it the loss is trivially reduced by growing every norm
        rather than by learning anything about the graph.
        """
        with torch.no_grad():
            self.entity.weight.data = nn.functional.normalize(
                self.entity.weight.data, p=2, dim=1)

    def score(self, heads, relations, tails) -> torch.Tensor:
        """Higher is more plausible."""
        translated = self.entity(heads) + self.relation(relations)
        return -torch.norm(translated - self.entity(tails), p=self.p_norm, dim=-1)


class DistMult(nn.Module):
    """Bilinear diagonal model: score is <h, r, t>."""

    def __init__(self, n_entities: int, n_relations: int, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def normalise(self) -> None:
        with torch.no_grad():
            self.entity.weight.data = nn.functional.normalize(
                self.entity.weight.data, p=2, dim=1)

    def score(self, heads, relations, tails) -> torch.Tensor:
        return (self.entity(heads) * self.relation(relations)
                * self.entity(tails)).sum(dim=-1)


MODELS = {"transe": TransE, "distmult": DistMult}


@dataclass
class Evaluation:
    """Filtered ranking metrics on held-out triples."""
    model: str
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    n_test: int
    mean_rank: float

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model, "mrr": round(self.mrr, 4),
            "hits_at_1": round(self.hits_at_1, 4),
            "hits_at_3": round(self.hits_at_3, 4),
            "hits_at_10": round(self.hits_at_10, 4),
            "mean_rank": round(self.mean_rank, 1),
            "n_test_triples": self.n_test,
        }


def _known_tails(triples: Iterable[Triple]) -> Dict[Tuple[str, str], set]:
    known = defaultdict(set)
    for head, relation, tail in triples:
        known[(head, relation)].add(tail)
    return known


def evaluate_ranking(score_fn, store: TripleStore, test: Sequence[Triple],
                     all_triples: Sequence[Triple], name: str) -> Evaluation:
    """Filtered tail-prediction ranking.

    Filtered: when ranking candidates for (h, r, ?), other tails known to
    be true are removed before ranking. Without that, a model is penalised
    for ranking a genuinely correct alternative above the held-out one,
    and the score understates it for being right.
    """
    known = _known_tails(all_triples)
    ranks = []

    for head, relation, tail in test:
        scores = score_fn(head, relation, store.entities)
        target = scores[store.entity_index[tail]]

        competitors = known[(head, relation)] - {tail}
        better = 0
        for candidate, candidate_score in zip(store.entities, scores):
            if candidate == tail or candidate in competitors:
                continue
            if candidate_score > target:
                better += 1
        ranks.append(better + 1)

    if not ranks:
        raise ValueError("No test triples to evaluate")

    ranks_array = np.array(ranks, dtype=float)
    return Evaluation(
        model=name,
        mrr=float(np.mean(1.0 / ranks_array)),
        hits_at_1=float(np.mean(ranks_array <= 1)),
        hits_at_3=float(np.mean(ranks_array <= 3)),
        hits_at_10=float(np.mean(ranks_array <= 10)),
        mean_rank=float(np.mean(ranks_array)),
        n_test=len(ranks),
    )


def frequency_baseline(train: Sequence[Triple]):
    """Rank tails by how often they appear with that relation.

    The baseline any link predictor must beat to be worth its complexity:
    "suggest the thing that usually appears". On a graph where a handful of
    entities dominate, it is a surprisingly strong opponent.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    for _, relation, tail in train:
        counts[relation][tail] += 1

    def score(head: str, relation: str, candidates: Sequence[str]) -> np.ndarray:
        table = counts[relation]
        return np.array([float(table.get(c, 0)) for c in candidates])

    return score


def adamic_adar_baseline(train: Sequence[Triple]):
    """Rank by shared-neighbour similarity, down-weighting hubs.

    The structural baseline already serving suggestions in
    evidence_graph_service. If embeddings cannot beat this, they are
    complexity without benefit.
    """
    neighbours: Dict[str, set] = defaultdict(set)
    for head, _, tail in train:
        neighbours[head].add(tail)
        neighbours[tail].add(head)

    def score(head: str, relation: str, candidates: Sequence[str]) -> np.ndarray:
        head_neighbours = neighbours.get(head, set())
        out = np.zeros(len(candidates))
        for i, candidate in enumerate(candidates):
            shared = head_neighbours & neighbours.get(candidate, set())
            out[i] = sum(1.0 / math.log(len(neighbours[w])) for w in shared
                         if len(neighbours[w]) > 1)
        return out

    return score


def train_kge(store: TripleStore, train: Sequence[Triple],
              model_name: str = "transe", dim: int = 64, epochs: int = 200,
              margin: float = 1.0, learning_rate: float = 0.01,
              negatives: int = 4, seed: int = 0) -> nn.Module:
    """Train with margin ranking loss over corrupted triples."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; expected {sorted(MODELS)}")

    torch.manual_seed(seed)
    model = MODELS[model_name](len(store.entities), len(store.relations), dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin)

    encoded = store.encode(train)
    n_entities = len(store.entities)
    target = torch.ones(len(encoded) * negatives)

    for epoch in range(epochs):
        model.normalise()
        optimiser.zero_grad()

        heads, relations, tails = encoded[:, 0], encoded[:, 1], encoded[:, 2]
        positive = model.score(heads, relations, tails).repeat(negatives)

        # Corrupt head or tail, half each, uniformly over entities.
        corrupt_heads = heads.repeat(negatives).clone()
        corrupt_tails = tails.repeat(negatives).clone()
        replacement = torch.randint(0, n_entities, (len(corrupt_heads),))
        corrupt_head_side = torch.rand(len(corrupt_heads)) < 0.5
        corrupt_heads[corrupt_head_side] = replacement[corrupt_head_side]
        corrupt_tails[~corrupt_head_side] = replacement[~corrupt_head_side]

        negative = model.score(
            corrupt_heads, relations.repeat(negatives), corrupt_tails)

        loss = criterion(positive, negative, target)
        loss.backward()
        optimiser.step()

        if epoch % 50 == 0:
            logger.info(f"epoch {epoch}: loss {loss.item():.4f}")

    model.eval()
    return model


def kge_scorer(model: nn.Module, store: TripleStore):
    """Scoring function over candidate tails, for evaluate_ranking."""
    def score(head: str, relation: str, candidates: Sequence[str]) -> np.ndarray:
        with torch.no_grad():
            head_index = torch.tensor([store.entity_index[head]] * len(candidates))
            relation_index = torch.tensor(
                [store.relation_index[relation]] * len(candidates))
            tail_index = torch.tensor([store.entity_index[c] for c in candidates])
            return model.score(head_index, relation_index, tail_index).numpy()
    return score


def save_model(path, model: nn.Module, store: TripleStore, report: "KGEReport",
               edge_count: int) -> None:
    """Persist a trained model with everything needed to trust it later.

    The vocabulary travels with the weights because entity indices are
    positional: a state dict alone is a matrix of numbers whose rows mean
    whatever the graph meant at training time. The edge count travels with
    it so a graph that has since changed can be detected rather than
    silently scored against stale indices.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_class": type(model).__name__.lower(),
        "dim": model.dim,
        "state_dict": model.state_dict(),
        "entities": store.entities,
        "relations": store.relations,
        "edge_count": edge_count,
        "report": report.describe(),
        "mrr": report.evaluation.mrr,
    }, path)
    logger.info(f"Saved embeddings to {path}")


def load_model(path, edge_count: Optional[int] = None):
    """Restore a model, refusing one trained on a different graph.

    Returns (model, store, saved_report) or None. A model whose training
    graph had a different number of edges is refused rather than loaded:
    its entity indices no longer point at the same entities, so it would
    score confidently about the wrong things.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    if not path.exists():
        return None

    try:
        state = torch.load(path, weights_only=False)
    except Exception as e:
        logger.error(f"Could not read embeddings from {path}: {e}")
        return None

    if edge_count is not None and state["edge_count"] != edge_count:
        logger.warning(
            f"Saved embeddings were trained on a graph with "
            f"{state['edge_count']} edges; this graph has {edge_count}. "
            f"Refusing to load: entity indices no longer line up. Retrain.")
        return None

    model_class = MODELS[state["model_class"]]
    model = model_class(len(state["entities"]), len(state["relations"]),
                        state["dim"])
    model.load_state_dict(state["state_dict"])
    model.eval()

    store = TripleStore.__new__(TripleStore)
    store.triples = []
    store.entities = state["entities"]
    store.relations = state["relations"]
    store.entity_index = {e: i for i, e in enumerate(store.entities)}
    store.relation_index = {r: i for i, r in enumerate(store.relations)}

    logger.info(
        f"Restored embeddings from {path} "
        f"({state['model_class']}, MRR {state['mrr']:.4f})")
    return model, store, state["report"]


async def load_triples_from_neo4j(config_path: str = "config/settings.json") -> List[Triple]:
    """Read the evidence graph as triples."""
    import neo4j

    from src.integration import load_database_config

    config = load_database_config(config_path)["neo4j"]
    driver = neo4j.AsyncGraphDatabase.driver(
        config.get("uri", "bolt://localhost:7687"),
        auth=(config.get("username", "neo4j"), config.get("password", "")))
    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Evidence)-[r]->(n)
                WHERE n.name IS NOT NULL
                RETURN e.id AS head, type(r) AS relation, n.name AS tail
                """)
            return [(row["head"], row["relation"], row["tail"])
                    for row in await result.data()]
    finally:
        await driver.close()


@dataclass
class KGEReport:
    """A trained model together with the evidence for using it.

    `beats_baselines` is what decides whether the model is served. A link
    predictor that loses to "suggest whatever usually appears" is
    complexity without benefit, and serving its output anyway would be the
    random-confidence problem again in a more convincing costume.
    """
    evaluation: Evaluation
    baselines: List[Evaluation]
    beats_baselines: bool
    n_triples: int
    n_entities: int
    parameters: Dict[str, Any]

    def compare(self) -> Dict[str, Dict[str, bool]]:
        """Per-metric comparison against each baseline.

        Reported metric by metric rather than as one verdict: a model can
        win on MRR and lose on Hits@10, and collapsing that into a single
        boolean hides the case where the baseline finds the right answer
        in its top ten more often.
        """
        metrics = ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")
        return {
            baseline.model: {
                metric: getattr(self.evaluation, metric) > getattr(baseline, metric)
                for metric in metrics
            }
            for baseline in self.baselines
        }

    def describe(self) -> Dict[str, Any]:
        comparison = self.compare()
        lost_on = sorted({
            metric for per_metric in comparison.values()
            for metric, won in per_metric.items() if not won
        })
        return {
            "model": self.evaluation.describe(),
            "baselines": [b.describe() for b in self.baselines],
            "beats_baselines": self.beats_baselines,
            "comparison": comparison,
            "loses_to_a_baseline_on": lost_on,
            "graph": {"triples": self.n_triples, "entities": self.n_entities},
            "parameters": self.parameters,
            "note": (
                "Filtered tail-prediction on held-out triples. Serving is "
                "gated on MRR against every baseline; metrics the model "
                "loses on are listed rather than averaged away. Note that "
                "adamic_adar is scored here on evidence-to-entity tail "
                "prediction, which is not the entity-to-entity suggestion "
                "task it performs in evidence_graph_service -- its number "
                "here is not a verdict on that use."
            ),
        }


def build_and_evaluate(triples: Sequence[Triple], model_name: str = "transe",
                       dim: int = 64, epochs: int = 200, seed: int = 0
                       ) -> Tuple[Optional[nn.Module], TripleStore, KGEReport]:
    """Train, evaluate against baselines, and report whether to use it."""
    store = TripleStore(list(triples))
    train, _valid, test = store.split(seed=seed)

    if not test:
        raise ValueError(
            f"No held-out triples survived the split of {len(triples)} "
            f"triples; the graph is too small or too sparse to evaluate on, "
            f"and an unevaluated link predictor is not worth serving")

    model = train_kge(store, train, model_name=model_name, dim=dim,
                      epochs=epochs, seed=seed)

    evaluation = evaluate_ranking(
        kge_scorer(model, store), store, test, store.triples, model_name)
    baselines = [
        evaluate_ranking(frequency_baseline(train), store, test,
                         store.triples, "frequency"),
        evaluate_ranking(adamic_adar_baseline(train), store, test,
                         store.triples, "adamic_adar"),
    ]

    beats = all(evaluation.mrr > b.mrr for b in baselines)
    report = KGEReport(
        evaluation=evaluation, baselines=baselines, beats_baselines=beats,
        n_triples=len(triples), n_entities=len(store.entities),
        parameters={"model": model_name, "dim": dim, "epochs": epochs,
                    "train": len(train), "test": len(test)},
    )

    if not beats:
        logger.warning(
            f"{model_name} MRR {evaluation.mrr:.4f} does not beat every "
            f"baseline ({[f'{b.model} {b.mrr:.4f}' for b in baselines]}); "
            f"not recommending it for serving")

    return (model if beats else None), store, report


async def main():
    """Train and evaluate over the populated Neo4j graph."""
    import json

    logging.basicConfig(level=logging.INFO)
    triples = await load_triples_from_neo4j()
    logger.info(f"Loaded {len(triples)} triples")

    for model_name in ("transe", "distmult"):
        _model, _store, report = build_and_evaluate(
            triples, model_name=model_name, epochs=300)
        print(json.dumps(report.describe(), indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
