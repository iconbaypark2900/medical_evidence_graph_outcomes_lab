"""Storage layer.

Covers the write half of graph-RAG. The regression guards here are for
three bugs that made a partially-written index look identical to a
complete one:

- Qdrant points were keyed by the loop index, so run two overwrote run
  one point for point.
- No schema was created, so the Qdrant collection did not exist and every
  upsert failed.
- Every failure returned False, which callers read as success.
"""
from __future__ import annotations

import pytest

from src.data_ingestion import MedicalEvidence
from src.integration import (
    ENTITY_SCHEMA,
    EvidenceStore,
    StorageError,
    evidence_to_document,
    load_database_config,
    point_id_for,
)


ARTICLE = MedicalEvidence(
    id="pubmed_31234567",
    title="Dapagliflozin in Heart Failure",
    abstract="Worsening heart failure was less common with dapagliflozin.",
    pub_date="2019",
    authors=["John J V McMurray"],
    journal="NEJM",
    source="PubMed",
    pmid="31234567",
    mesh_terms=["Heart Failure"],
    entities={"conditions": ["Heart Failure"], "interventions": ["dapagliflozin"],
              "outcomes": ["mortality"], "populations": []},
)


# --------------------------------------------------------------------------
# Point identity
# --------------------------------------------------------------------------

def test_point_ids_are_stable_for_the_same_evidence():
    """Regression guard. Ids were the enumerate() index, so every ingest
    wrote points 0..n-1 and silently replaced the previous run's."""
    assert point_id_for("pubmed_31234567") == point_id_for("pubmed_31234567")


def test_point_ids_differ_between_documents():
    assert point_id_for("pubmed_1") != point_id_for("pubmed_2")


def test_point_ids_do_not_depend_on_position_in_a_batch():
    batch_one = [point_id_for(i) for i in ["a", "b", "c"]]
    batch_two = [point_id_for(i) for i in ["c", "b", "a"]]

    assert batch_one == list(reversed(batch_two))


def test_point_ids_are_valid_qdrant_uuids():
    import uuid

    assert uuid.UUID(point_id_for("pubmed_1"))


# --------------------------------------------------------------------------
# Document shape
# --------------------------------------------------------------------------

def test_entities_are_flattened_to_top_level_keyword_fields():
    """Nested free-form entities under dynamic mapping grow the index
    mapping by one field per distinct key, without bound."""
    document = evidence_to_document(ARTICLE)

    assert document["conditions"] == ["Heart Failure"]
    assert document["interventions"] == ["dapagliflozin"]
    assert document["outcomes"] == ["mortality"]
    assert "entities" not in document


def test_the_document_carries_its_identifiers():
    document = evidence_to_document(ARTICLE)

    assert document["id"] == "pubmed_31234567"
    assert document["pmid"] == "31234567"
    assert document["mesh_terms"] == ["Heart Failure"]


def test_a_record_with_no_entities_still_produces_a_valid_document():
    bare = MedicalEvidence(
        id="pubmed_2", title="T", abstract="A", pub_date="2020",
        authors=[], journal="J", source="PubMed")

    document = evidence_to_document(bare)

    assert document["conditions"] == []
    assert document["authors"] == []


def test_the_entity_schema_covers_every_flattened_field():
    document = evidence_to_document(ARTICLE)

    for kind in ENTITY_SCHEMA:
        assert kind in document, f"{kind} has a graph label but no indexed field"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_database_config_exposes_all_three_stores():
    config = load_database_config()

    assert set(config) == {"neo4j", "opensearch", "qdrant"}
    assert config["neo4j"]["uri"].startswith("bolt://")


def test_a_missing_config_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_database_config("config/does_not_exist.json")


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------

class ExplodingOpenSearch:
    def bulk(self, body, refresh):
        raise ConnectionError("cluster unreachable")


class RejectingOpenSearch:
    def bulk(self, body, refresh):
        return {"errors": True, "items": [
            {"index": {"error": {"type": "strict_dynamic_mapping_exception"}}}]}


def test_a_write_failure_raises_rather_than_returning_false():
    """Regression guard: callers treated the False return as success, so a
    half-populated index was indistinguishable from a complete one."""
    store = EvidenceStore()
    store.opensearch_client = ExplodingOpenSearch()

    with pytest.raises(StorageError, match="OpenSearch write failed"):
        store.store_in_opensearch([ARTICLE])


def test_documents_rejected_by_the_index_are_reported():
    """A bulk request can return 200 with per-document errors inside."""
    store = EvidenceStore()
    store.opensearch_client = RejectingOpenSearch()

    with pytest.raises(StorageError, match="rejected 1 of 1"):
        store.store_in_opensearch([ARTICLE])


def test_storing_nothing_writes_nothing():
    store = EvidenceStore()

    assert store.store_in_opensearch([]) == 0
    assert store.store_in_qdrant([]) == 0


class OldNameEmbedder:
    """sentence-transformers renamed the dimension getter."""

    def get_sentence_embedding_dimension(self):
        return 384


class NewNameEmbedder:
    def get_embedding_dimension(self):
        return 768


class NamelessEmbedder:
    pass


def test_the_vector_dimension_is_read_under_either_api_name():
    """Guessing it would create the collection at the wrong width, and
    every later upsert would fail on dimension mismatch."""
    old = EvidenceStore(embedding_model=OldNameEmbedder())
    new = EvidenceStore(embedding_model=NewNameEmbedder())

    assert old.vector_size == 384
    assert new.vector_size == 768


def test_an_unrecognisable_embedder_is_an_error_not_a_guess():
    store = EvidenceStore(embedding_model=NamelessEmbedder())

    with pytest.raises(StorageError, match="Cannot determine embedding dimension"):
        _ = store.vector_size


# --------------------------------------------------------------------------
# Against the real stack
# --------------------------------------------------------------------------

@pytest.mark.requires_stack
async def test_schema_creation_is_idempotent(require_stack):
    from sentence_transformers import SentenceTransformer

    from src.integration import EMBEDDING_MODEL

    store = EvidenceStore(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL),
        index_name="test_idempotent_index",
        collection_name="test_idempotent_collection",
    )
    await store.connect()
    try:
        await store.ensure_schema()
        await store.ensure_schema()  # must not raise on the second call

        assert store.opensearch_client.indices.exists(index="test_idempotent_index")
        names = {c.name for c in store.qdrant_client.get_collections().collections}
        assert "test_idempotent_collection" in names
    finally:
        store.opensearch_client.indices.delete(
            index="test_idempotent_index", ignore=[404])
        store.qdrant_client.delete_collection("test_idempotent_collection")
        await store.close()


@pytest.mark.requires_stack
async def test_connecting_to_a_dead_host_raises(require_stack, tmp_path):
    """Partial connectivity is not a usable state: retrieval cannot tell a
    store that was never written from a document that does not exist."""
    import json

    config = json.loads(open("config/settings.json").read())
    config["services"]["graph_rag_service"]["opensearch"]["port"] = 9
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config))

    store = EvidenceStore(str(path), embedding_model=OldNameEmbedder())

    with pytest.raises(StorageError, match="Cannot reach OpenSearch"):
        await store.connect()
    await store.close()
