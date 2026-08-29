"""MeSH descriptor classification.

Curated extraction filed every non-chemical MeSH descriptor as a
condition. Measured against the indexed corpus, roughly half the
highest-degree Condition nodes were not conditions -- Primary Prevention,
Stroke Volume, Treatment Outcome, Drug Therapy Combination, Double-Blind
Method, Glomerular Filtration Rate, Kidney, Follow-Up Studies, Quality of
Life. Those are interventions, measurements, outcomes, study designs and
anatomy.

MeSH already classifies every descriptor by tree number, so the answer
did not have to be guessed. These tests run offline against a seeded
cache; `warm_cache_from_neo4j` is what populates it.
"""
from __future__ import annotations

import json

import pytest

from src.mesh import GRAPH_LABELS, TREE_AXES, MeshClassifier

# Real tree numbers, as returned by NCBI.
SEED = {
    "heart failure": ["C14.280.434"],
    "diabetes mellitus, type 2": ["C18.452.394.750.149", "C19.246.300"],
    "treatment outcome": ["E01.789.800", "N04.761.559.590.800"],
    "kidney": ["A05.810.453"],
    "double-blind method": ["E05.318.370.300", "N05.715.360.325.320"],
    "primary prevention": ["N02.421.726"],
    "metformin": ["D02.078.370.141.450", "D02.886.108.457"],
    "aged": ["M01.060.116"],
    "stroke volume": ["G09.330.553.700"],
    "not a real descriptor": [],
}


@pytest.fixture
def classifier(tmp_path) -> MeshClassifier:
    cache = tmp_path / "mesh.json"
    cache.write_text(json.dumps(SEED))
    # offline: a test that silently reaches NCBI is a test of the network.
    return MeshClassifier(cache_path=cache, offline=True)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("descriptor,axis", [
    ("Heart Failure", "condition"),
    ("Diabetes Mellitus, Type 2", "condition"),
    ("Metformin", "intervention"),
    ("Aged", "population"),
    ("Kidney", "anatomy"),
    ("Stroke Volume", "process"),
    ("Double-Blind Method", "study_method"),
    ("Primary Prevention", "health_care"),
    ("Treatment Outcome", "diagnostic"),
])
def test_descriptors_land_on_the_axis_their_tree_names(classifier, descriptor, axis):
    assert classifier.classify(descriptor) == axis


@pytest.mark.parametrize("descriptor", [
    "Treatment Outcome", "Kidney", "Double-Blind Method",
    "Primary Prevention", "Stroke Volume",
])
def test_the_misfiled_descriptors_no_longer_reach_the_graph(classifier, descriptor):
    """Each of these was a high-degree Condition node."""
    assert classifier.graph_label(descriptor) is None


@pytest.mark.parametrize("descriptor,label", [
    ("Heart Failure", "Condition"),
    ("Metformin", "Intervention"),
    ("Aged", "Population"),
])
def test_real_entities_still_reach_the_graph(classifier, descriptor, label):
    assert classifier.graph_label(descriptor) == label


def test_an_unknown_descriptor_is_not_guessed_onto_an_axis(classifier):
    """Defaulting is the defect being fixed."""
    assert classifier.classify("Not a real descriptor") == "unknown"
    assert classifier.graph_label("Not a real descriptor") is None


def test_classification_is_case_and_space_insensitive(classifier):
    assert classifier.classify("  HEART FAILURE  ") == "condition"


def test_the_longest_tree_prefix_wins():
    """E02 is Therapeutics; E05 is Investigative Techniques. A bare E would
    make a study method an intervention."""
    assert MeshClassifier.axis_for_tree("E02.319") == "intervention"
    assert MeshClassifier.axis_for_tree("E05.318") == "study_method"
    assert MeshClassifier.axis_for_tree("C14.280") == "condition"


def test_a_descriptor_in_several_trees_prefers_the_clinical_axis(tmp_path):
    """Clinical concepts are cross-listed into the health-care trees; the
    disease reading is the useful one."""
    cache = tmp_path / "mesh.json"
    cache.write_text(json.dumps({"cross listed": ["N04.761", "C14.280"]}))

    assert MeshClassifier(cache, offline=True).classify("Cross listed") == "condition"


def test_an_unmapped_tree_letter_is_not_an_error():
    assert MeshClassifier.axis_for_tree("Q99.999") is None


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_descriptors_are_split_across_the_axes(classifier):
    routed = classifier.split_descriptors([
        "Heart Failure", "Metformin", "Aged", "Treatment Outcome", "Kidney"])

    assert routed["conditions"] == ["Heart Failure"]
    assert routed["interventions"] == ["Metformin"]
    assert routed["populations"] == ["Aged"]
    assert set(routed["other"]) == {"Treatment Outcome", "Kidney"}


def test_nothing_is_dropped_by_the_split(classifier):
    """What is not an entity stays in `other`, so it remains searchable in
    OpenSearch without claiming to be a clinical entity."""
    descriptors = list(SEED)
    routed = classifier.split_descriptors(descriptors)

    assert sum(len(v) for v in routed.values()) == len(descriptors)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_a_cached_descriptor_is_not_fetched(classifier):
    classifier.classify("Heart Failure")

    assert classifier.lookups == 0
    assert classifier.cache_hits >= 1


def test_offline_mode_never_reaches_the_network(tmp_path):
    """An ingest that cannot reach NCBI still runs; the unknown descriptor
    is marked unknown rather than guessed."""
    offline = MeshClassifier(tmp_path / "empty.json", offline=True)

    assert offline.classify("Heart Failure") == "unknown"
    assert offline.lookups == 0


def test_the_cache_round_trips(tmp_path):
    path = tmp_path / "mesh.json"
    writer = MeshClassifier(path, offline=True)
    writer._cache["heart failure"] = ["C14.280.434"]
    writer.save_cache()

    assert MeshClassifier(path, offline=True).classify("Heart Failure") == "condition"


def test_a_corrupt_cache_does_not_stop_ingestion(tmp_path):
    path = tmp_path / "mesh.json"
    path.write_text("{not json")

    assert MeshClassifier(path, offline=True).classify("anything") == "unknown"


def test_only_three_axes_reach_the_graph():
    """Everything else stays on the evidence record."""
    assert set(GRAPH_LABELS) == {"condition", "intervention", "population"}
    assert set(GRAPH_LABELS.values()) == {"Condition", "Intervention", "Population"}


def test_the_shipped_cache_covers_the_common_descriptors():
    """It ships with the repository so ingestion classifies offline."""
    from pathlib import Path

    from src.mesh import DEFAULT_CACHE

    if not Path(DEFAULT_CACHE).exists():
        pytest.skip("cache not built; run python -m src.mesh")

    cached = MeshClassifier(DEFAULT_CACHE, offline=True)
    assert cached.classify("Heart Failure") == "condition"
    assert cached.graph_label("Treatment Outcome") is None
