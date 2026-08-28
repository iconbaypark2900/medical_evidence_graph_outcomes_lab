"""Graph-RAG hybrid retrieval.

The module this covers was mock top to bottom. `MockQdrant` built its
"embeddings" as

    np.random.seed(hash(content) % 2**32)
    embedding = np.random.rand(32)

-- a random vector keyed by exact content, so two papers stating the same
finding in different words got orthogonal vectors. Semantic search could
not work by construction. `MockNeo4j` returned hardcoded nodes and
`np.random.random()` link confidences.

Fusion and ranking are tested directly. The retrievers are tested against
stub clients, and then against the real stores in the `requires_stack`
tests at the bottom, which skip when docker-compose is not up.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.graph_rag_service.main import (
    BM25_MINIMUM_SHOULD_MATCH,
    MIN_VECTOR_SIMILARITY,
    RRF_K,
    FusedResult,
    GraphRAGService,
    RetrievalError,
    SearchResult,
    reciprocal_rank_fusion,
)


def result(doc_id: str, retriever: str = "bm25", score: float = 1.0, **kwargs) -> SearchResult:
    return SearchResult(
        id=doc_id,
        title=kwargs.get("title", f"Title {doc_id}"),
        content=kwargs.get("content", f"Content of {doc_id}"),
        source=kwargs.get("source", "PubMed"),
        score=score,
        retriever=retriever,
        metadata=kwargs.get("metadata", {}),
    )


# --------------------------------------------------------------------------
# Reciprocal rank fusion
# --------------------------------------------------------------------------

def test_fusion_of_a_single_list_preserves_its_order():
    fused = reciprocal_rank_fusion({"bm25": [result("a"), result("b"), result("c")]})

    assert [r.id for r in fused] == ["a", "b", "c"]


def test_a_document_found_by_every_retriever_outranks_one_found_by_one():
    """The whole point of fusing: agreement across independent retrievers
    is evidence, a single retriever's top hit on its own is weaker."""
    fused = reciprocal_rank_fusion({
        "bm25": [result("solo"), result("shared", "bm25")],
        "vector": [result("shared", "vector")],
        "graph": [result("shared", "graph")],
    })

    assert fused[0].id == "shared"
    assert set(fused[0].found_by) == {"bm25", "vector", "graph"}


def test_fused_scores_use_reciprocal_rank():
    fused = reciprocal_rank_fusion({"bm25": [result("a"), result("b")]})

    assert fused[0].fused_score == pytest.approx(1 / (RRF_K + 1))
    assert fused[1].fused_score == pytest.approx(1 / (RRF_K + 2))


def test_fusion_ignores_raw_scores_entirely():
    """A BM25 score and a cosine similarity are different units. Fusing on
    rank avoids inventing a comparability between them."""
    modest = reciprocal_rank_fusion({
        "bm25": [result("a", score=0.01), result("b", score=0.001)]})
    enormous = reciprocal_rank_fusion({
        "bm25": [result("a", score=9999.0), result("b", score=5000.0)]})

    assert [r.id for r in modest] == [r.id for r in enormous]
    assert modest[0].fused_score == enormous[0].fused_score


def test_fusion_records_the_rank_each_retriever_gave():
    fused = reciprocal_rank_fusion({
        "bm25": [result("x"), result("target")],
        "vector": [result("target")],
    })

    target = next(r for r in fused if r.id == "target")
    assert target.found_by == {"bm25": 2, "vector": 1}


def test_fusion_keeps_the_richest_copy_of_a_document():
    """Qdrant payloads can hold a shorter abstract than the OpenSearch doc."""
    fused = reciprocal_rank_fusion({
        "vector": [result("a", content="short")],
        "bm25": [result("a", content="a considerably longer abstract body")],
    })

    assert fused[0].content == "a considerably longer abstract body"


def test_fusion_is_deterministic_for_tied_scores():
    """Ties break by id, so the same inputs always produce the same order.

    Not cosmetic: an unstable order means the same query returns different
    citations on different runs.
    """
    lists = {"bm25": [result("z")], "vector": [result("a")]}

    first = [r.id for r in reciprocal_rank_fusion(lists)]
    second = [r.id for r in reciprocal_rank_fusion(lists)]

    assert first == second == ["a", "z"]


def test_fusion_of_nothing_is_nothing():
    assert reciprocal_rank_fusion({"bm25": [], "vector": [], "graph": []}) == []


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------

def test_a_pubmed_result_cites_its_pmid():
    fused = reciprocal_rank_fusion({
        "bm25": [result("pubmed_31234567", metadata={"pmid": "31234567"})]})

    assert fused[0].citation == "PMID:31234567"


def test_a_trial_result_cites_its_nct_id():
    fused = reciprocal_rank_fusion({
        "bm25": [result("trial_NCT03036124", metadata={"nct_id": "NCT03036124"})]})

    assert fused[0].citation == "NCT03036124"


def test_a_result_with_no_identifier_falls_back_to_its_id():
    fused = reciprocal_rank_fusion({"bm25": [result("local_1")]})

    assert fused[0].citation == "local_1"


# --------------------------------------------------------------------------
# Retrievers against stub clients
# --------------------------------------------------------------------------

class StubOpenSearch:
    def __init__(self, hits=None, error=None):
        self.hits = hits or []
        self.error = error
        self.last_body = None

    def search(self, index, body):
        self.last_body = body
        if self.error:
            raise self.error
        return {"hits": {"hits": self.hits}}


def opensearch_hit(doc_id, score=1.0, **source):
    return {"_score": score, "_source": {"id": doc_id, "title": f"T {doc_id}",
                                         "content": "body", "source": "PubMed", **source}}


@pytest.fixture
def service() -> GraphRAGService:
    return GraphRAGService()


def test_bm25_search_maps_hits_to_results(service):
    service.opensearch_client = StubOpenSearch([
        opensearch_hit("a", 3.2, pmid="1"), opensearch_hit("b", 1.1, pmid="2")])

    results = service.run_bm25_search("heart failure", limit=5)

    assert [r.id for r in results] == ["a", "b"]
    assert results[0].score == 3.2
    assert results[0].retriever == "bm25"
    assert results[0].metadata["pmid"] == "1"


def test_bm25_search_weights_the_title_above_the_body(service):
    service.opensearch_client = StubOpenSearch([])

    service.run_bm25_search("heart failure", limit=5)

    fields = service.opensearch_client.last_body["query"]["multi_match"]["fields"]
    assert "title^2" in fields
    assert "content" in fields


def test_a_bm25_failure_is_raised_not_returned_as_no_results(service):
    """[] would read as "no evidence matches this query" -- a finding.
    A search failure is not a finding."""
    service.opensearch_client = StubOpenSearch(error=ConnectionError("index closed"))

    with pytest.raises(RetrievalError, match="BM25 search failed"):
        service.run_bm25_search("heart failure")


class StubQdrantPoint:
    def __init__(self, evidence_id, score, **payload):
        self.score = score
        self.payload = {"evidence_id": evidence_id, "title": f"T {evidence_id}",
                        "abstract": "body", "source": "PubMed", **payload}


class StubQdrantResponse:
    def __init__(self, points):
        self.points = points


class StubQdrant:
    def __init__(self, points=None, error=None):
        self.points = points or []
        self.error = error

    def query_points(self, collection_name, query, limit, with_payload,
                     score_threshold=None):
        if self.error:
            raise self.error
        self.score_threshold = score_threshold
        points = self.points
        if score_threshold is not None:
            points = [p for p in points if p.score >= score_threshold]
        return StubQdrantResponse(points[:limit])


class StubEmbedder:
    def encode(self, text):
        return np.zeros(384, dtype=np.float32)

    def get_embedding_dimension(self):
        return 384


def test_vector_search_maps_points_to_results(service):
    service.qdrant_client = StubQdrant([
        StubQdrantPoint("a", 0.91, pmid="1"), StubQdrantPoint("b", 0.72)])
    service.embedding_model = StubEmbedder()

    results = service.run_vector_search("heart failure", limit=5)

    assert [r.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(0.91)
    assert results[0].retriever == "vector"


def test_a_vector_search_failure_is_raised(service):
    service.qdrant_client = StubQdrant(error=ConnectionError("collection missing"))
    service.embedding_model = StubEmbedder()

    with pytest.raises(RetrievalError, match="Vector search failed"):
        service.run_vector_search("heart failure")


async def test_an_empty_query_is_refused(service):
    with pytest.raises(ValueError, match="Query is empty"):
        await service.answer_query("   ")


# --------------------------------------------------------------------------
# Against the real stack
# --------------------------------------------------------------------------

CORPUS = [
    ("t_hf_dapa", "Dapagliflozin in heart failure with reduced ejection fraction",
     "Dapagliflozin lowered the risk of worsening heart failure and cardiovascular death.",
     ["Heart Failure"], ["dapagliflozin"], ["mortality"]),
    ("t_hf_empa", "Empagliflozin outcome trial in chronic heart failure",
     "Empagliflozin reduced hospitalisation for heart failure in patients with and without diabetes.",
     ["Heart Failure"], ["empagliflozin"], ["mortality"]),
    ("t_onc_tras", "Trastuzumab in HER2-positive early breast cancer",
     "Adjuvant trastuzumab improved disease-free survival in HER2-positive breast cancer.",
     ["Breast Neoplasms"], ["trastuzumab"], ["survival"]),
]


@pytest.fixture
async def indexed_stack(require_stack):
    """Index a tiny known corpus into dedicated test targets.

    Its own index and collection, so the tests neither depend on nor
    disturb whatever the live corpus holds.
    """
    from sentence_transformers import SentenceTransformer

    from src.data_ingestion import MedicalEvidence
    from src.integration import EMBEDDING_MODEL, EvidenceStore

    store = EvidenceStore(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL),
        index_name="test_medical_evidence",
        collection_name="test_medical_evidence_embeddings",
    )
    await store.connect()
    await store.ensure_schema()

    evidence = [
        MedicalEvidence(
            id=doc_id, title=title, abstract=abstract, pub_date="2020",
            authors=["A Author"], journal="Test Journal", source="PubMed",
            pmid=doc_id.replace("t_", ""), mesh_terms=conditions,
            entities={"conditions": conditions, "interventions": interventions,
                      "outcomes": outcomes, "populations": []},
        )
        for doc_id, title, abstract, conditions, interventions, outcomes in CORPUS
    ]
    await store.store_all_evidence(evidence)

    service = GraphRAGService(
        embedding_model=store.embedding_model,
        index_name="test_medical_evidence",
        collection_name="test_medical_evidence_embeddings",
    )
    await service.connect()
    try:
        yield service
    finally:
        async with store.neo4j_driver.session() as session:
            await session.run(
                "MATCH (e:Evidence) WHERE e.id IN $ids DETACH DELETE e",
                ids=[c[0] for c in CORPUS])
        store.opensearch_client.indices.delete(
            index="test_medical_evidence", ignore=[404])
        store.qdrant_client.delete_collection("test_medical_evidence_embeddings")
        await service.close()
        await store.close()


@pytest.mark.requires_stack
async def test_bm25_finds_an_exact_drug_name(indexed_stack):
    results = indexed_stack.run_bm25_search("dapagliflozin", limit=5)

    assert results
    assert results[0].id == "t_hf_dapa"


@pytest.mark.requires_stack
async def test_vector_search_matches_a_paraphrase_bm25_would_miss(indexed_stack):
    """The capability the random-vector mock could not have.

    None of these words appear in the dapagliflozin abstract, but the
    meaning does.
    """
    query = "SGLT2 drug that prevents cardiac decompensation admissions"

    lexical = {r.id for r in indexed_stack.run_bm25_search(query, limit=2)}
    semantic = [r.id for r in indexed_stack.run_vector_search(query, limit=2)]

    heart_failure_papers = {"t_hf_dapa", "t_hf_empa"}
    assert set(semantic) & heart_failure_papers, semantic
    # The oncology paper must not outrank the heart-failure ones.
    assert semantic[0] != "t_onc_tras"
    assert lexical is not None  # BM25 ran; it simply has no term overlap here


@pytest.mark.requires_stack
async def test_semantically_unrelated_evidence_is_not_returned_first(indexed_stack):
    results = indexed_stack.run_vector_search("HER2 positive breast cancer", limit=3)

    assert results[0].id == "t_onc_tras"


@pytest.mark.requires_stack
async def test_graph_search_reaches_evidence_through_shared_entities(indexed_stack):
    results = await indexed_stack.run_graph_search("dapagliflozin", limit=5)

    assert {r.id for r in results} >= {"t_hf_dapa"}
    assert "dapagliflozin" in results[0].metadata["matched_entities"]


@pytest.mark.requires_stack
async def test_the_hybrid_answer_carries_citations_and_graph_context(indexed_stack):
    answer = await indexed_stack.answer_query("heart failure", limit=3)

    assert answer.results
    assert answer.retrievers_used == ["bm25", "graph", "vector"]
    assert all(c.startswith("PMID:") for c in answer.citations())
    assert answer.graph_context
    assert {row["entity"] for row in answer.graph_context} & {"Heart Failure"}


@pytest.mark.requires_stack
async def test_the_same_query_returns_the_same_answer(indexed_stack):
    """Determinism guard, as elsewhere: the previous implementation's
    scores came from np.random."""
    first = await indexed_stack.answer_query("heart failure", limit=3)
    second = await indexed_stack.answer_query("heart failure", limit=3)

    assert [r.id for r in first.results] == [r.id for r in second.results]
    assert [r.fused_score for r in first.results] == [
        r.fused_score for r in second.results]


@pytest.mark.requires_stack
async def test_coverage_reports_retriever_agreement_not_correctness(indexed_stack):
    answer = await indexed_stack.answer_query("heart failure", limit=3)

    coverage = answer.coverage
    assert coverage["results_returned"] == len(answer.results)
    assert 1 <= coverage["retrievers_agreeing_on_top_result"] <= 3
    assert "not whether the evidence answers the question" in coverage["note"]


@pytest.mark.requires_stack
async def test_reindexing_the_same_document_updates_rather_than_duplicates(indexed_stack):
    """Regression guard: Qdrant points were keyed by loop index, so a
    second run overwrote the first run's points position by position."""
    before = indexed_stack.qdrant_client.count(
        collection_name="test_medical_evidence_embeddings").count

    from sentence_transformers import SentenceTransformer  # noqa: F401
    from src.data_ingestion import MedicalEvidence
    from src.integration import EvidenceStore

    store = EvidenceStore(
        embedding_model=indexed_stack.embedding_model,
        index_name="test_medical_evidence",
        collection_name="test_medical_evidence_embeddings",
    )
    await store.connect()
    await store.store_all_evidence([
        MedicalEvidence(
            id="t_hf_dapa", title="Dapagliflozin in heart failure",
            abstract="Updated abstract.", pub_date="2021", authors=[],
            journal="Test Journal", source="PubMed", pmid="hf_dapa",
            mesh_terms=["Heart Failure"], entities={"conditions": ["Heart Failure"]},
        )])
    after = store.qdrant_client.count(
        collection_name="test_medical_evidence_embeddings").count
    await store.close()

    assert after == before


# --------------------------------------------------------------------------
# Relevance floors
#
# Neither retriever says "nothing matches" on its own. Vector search
# returns the k nearest neighbours however far away they are, and a
# multi_match ORs its terms so a stopword in common is a hit. Without a
# floor, a query the corpus does not cover comes back with a full page of
# confident-looking results.
# --------------------------------------------------------------------------

def test_weak_vector_hits_are_dropped(service):
    service.qdrant_client = StubQdrant([
        StubQdrantPoint("relevant", 0.72), StubQdrantPoint("noise", 0.11)])
    service.embedding_model = StubEmbedder()

    results = service.run_vector_search("heart failure", limit=5)

    assert [r.id for r in results] == ["relevant"]


def test_the_similarity_floor_is_passed_to_the_engine(service):
    """Filtered server-side, so the limit applies after the cut."""
    service.qdrant_client = StubQdrant([StubQdrantPoint("a", 0.9)])
    service.embedding_model = StubEmbedder()

    service.run_vector_search("heart failure", limit=5)

    assert service.qdrant_client.score_threshold == MIN_VECTOR_SIMILARITY


def test_the_similarity_floor_is_overridable(service):
    service.qdrant_client = StubQdrant([StubQdrantPoint("borderline", 0.20)])
    service.embedding_model = StubEmbedder()

    assert service.run_vector_search("q", 5) == []
    assert len(service.run_vector_search("q", 5, min_similarity=0.1)) == 1


def test_the_floor_sits_between_relevant_and_nonsense_scores():
    """Measured on this corpus: relevant queries score 0.56-0.79, nonsense
    tops out near 0.19. The floor has to separate those."""
    assert 0.19 < MIN_VECTOR_SIMILARITY < 0.56


def test_bm25_requires_a_meaningful_share_of_the_query_terms(service):
    service.opensearch_client = StubOpenSearch([])

    service.run_bm25_search("the quick brown fox", limit=5)

    query = service.opensearch_client.last_body["query"]["multi_match"]
    assert query["minimum_should_match"] == BM25_MINIMUM_SHOULD_MATCH


@pytest.mark.requires_stack
async def test_an_uncovered_query_returns_nothing_and_says_so(indexed_stack):
    """The honest failure. Returning the nearest three papers to a query
    about quantum chromodynamics would be indistinguishable from a match."""
    answer = await indexed_stack.answer_query(
        "quantum chromodynamics lattice gauge theory", limit=3)

    assert answer.results == []
    assert answer.coverage["matched"] is False
    assert "does not cover it" in answer.coverage["note"]
    assert "not a statement about the literature" in answer.coverage["note"]


@pytest.mark.requires_stack
async def test_a_covered_query_still_matches(indexed_stack):
    answer = await indexed_stack.answer_query("heart failure", limit=3)

    assert answer.results
    assert answer.coverage["matched"] is True


@pytest.mark.requires_stack
async def test_a_paraphrase_matches_semantically_with_no_lexical_overlap(indexed_stack):
    """The capability the np.random.rand(32) mock could not have had.

    None of these terms appear in the indexed abstracts, and BM25 finds
    nothing; vector search still retrieves the right papers.
    """
    query = "SGLT2 drug preventing cardiac decompensation admissions"

    assert indexed_stack.run_bm25_search(query, limit=5) == []
    semantic = [r.id for r in indexed_stack.run_vector_search(query, limit=3)]
    assert set(semantic) & {"t_hf_dapa", "t_hf_empa"}, semantic
