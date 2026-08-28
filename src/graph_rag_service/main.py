"""
Graph-RAG service: hybrid retrieval over the real evidence stores.

Read half of the pipeline; `src/integration.py` is the write half.

Retrieval combines three views of the same corpus:

  BM25 over OpenSearch     — exact terms, drug names, trial identifiers
  Vector search over Qdrant — paraphrase and synonym matching
  Traversal in Neo4j        — evidence reachable through shared entities

They are combined with reciprocal rank fusion, which needs no score
normalisation between retrievers whose scores are not comparable in the
first place (a BM25 score and a cosine similarity are different units).

This replaces a module that was mock top to bottom: `MockOpenSearch` did
substring matching, `MockNeo4j` returned hardcoded nodes, and
`MockQdrant` produced "embeddings" with

    np.random.seed(hash(content) % 2**32)
    embedding = np.random.rand(32)

which is a random vector keyed by exact content. Two papers saying the
same thing in different words got orthogonal vectors, so semantic search
could not work by construction — it was an expensive exact-match. Link
confidence was `np.random.random()`.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import neo4j
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient

from src.integration import (
    EMBEDDING_MODEL,
    EVIDENCE_COLLECTION,
    EVIDENCE_INDEX,
    load_database_config,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard RRF damping. Keeps any single retriever's top hit from
# dominating the fused ranking on its own.
RRF_K = 60

# Cosine similarity below which a vector hit is not treated as a match.
#
# Vector search has no natural cutoff: it returns the k nearest neighbours
# however far away they are, so an unrelated query still gets a full page
# of confident-looking results. Measured against this corpus with
# all-MiniLM-L6-v2, genuinely relevant queries score 0.56-0.79 while
# nonsense queries top out around 0.19, so anything below this is noise.
MIN_VECTOR_SIMILARITY = 0.35

# A query only matches lexically if enough of its terms hit. Without this,
# multi_match ORs the terms and "the quick brown fox" matches every
# document containing "the".
BM25_MINIMUM_SHOULD_MATCH = "2<70%"


class RetrievalError(RuntimeError):
    """A retriever failed. Never reported as an empty result set."""


@dataclass
class SearchResult:
    """One retrieved document, with the reason it was retrieved."""
    id: str
    title: str
    content: str
    source: str
    score: float
    retriever: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    """A document after fusion, carrying its per-retriever provenance."""
    id: str
    title: str
    content: str
    source: str
    fused_score: float
    found_by: Dict[str, int]  # retriever -> 1-based rank in that retriever
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        pmid = self.metadata.get("pmid")
        nct_id = self.metadata.get("nct_id")
        if pmid:
            return f"PMID:{pmid}"
        if nct_id:
            return nct_id
        return self.id


@dataclass
class EvidenceAnswer:
    """A retrieval response.

    `coverage` is deliberately not a confidence in the answer's
    correctness. It reports how many of the retrievers independently
    surfaced the top results, which is a statement about retrieval
    agreement and nothing more.
    """
    query: str
    results: List[FusedResult]
    graph_context: List[Dict[str, Any]]
    retrievers_used: List[str]
    coverage: Dict[str, Any]

    def citations(self) -> List[str]:
        return [result.citation for result in self.results]


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, Sequence[SearchResult]], k: int = RRF_K
) -> List[FusedResult]:
    """Combine ranked lists by reciprocal rank fusion.

    Each list contributes `1 / (k + rank)` per document. Rank-based rather
    than score-based on purpose: a BM25 score and a cosine similarity are
    not on the same scale, and normalising them against each other invents
    a comparability that does not exist.
    """
    scores: Dict[str, float] = {}
    ranks: Dict[str, Dict[str, int]] = {}
    documents: Dict[str, SearchResult] = {}

    for retriever, results in ranked_lists.items():
        for position, result in enumerate(results, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(result.id, {})[retriever] = position
            # Keep the richest copy of the document we have seen.
            if result.id not in documents or len(result.content or "") > len(
                    documents[result.id].content or ""):
                documents[result.id] = result

    fused = [
        FusedResult(
            id=doc_id,
            title=documents[doc_id].title,
            content=documents[doc_id].content,
            source=documents[doc_id].source,
            fused_score=score,
            found_by=ranks[doc_id],
            metadata=documents[doc_id].metadata,
        )
        for doc_id, score in scores.items()
    ]
    # Ties broken by id so the ordering is stable across runs.
    fused.sort(key=lambda r: (-r.fused_score, r.id))
    return fused


class GraphRAGService:
    """Hybrid retrieval over OpenSearch, Qdrant and Neo4j."""

    def __init__(self, config_path: str = "config/settings.json",
                 embedding_model: Any = None,
                 index_name: str = EVIDENCE_INDEX,
                 collection_name: str = EVIDENCE_COLLECTION):
        self.db_config = load_database_config(config_path)
        self.index_name = index_name
        self.collection_name = collection_name
        self.opensearch_client: Optional[OpenSearch] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.neo4j_driver: Optional[neo4j.AsyncDriver] = None
        self.embedding_model = embedding_model

    async def connect(self) -> None:
        opensearch_config = self.db_config["opensearch"]
        host = opensearch_config.get("host", "localhost")
        port = opensearch_config.get("port", 9200)
        try:
            self.opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                use_ssl=False, verify_certs=False, ssl_show_warn=False)
            self.opensearch_client.info()
        except Exception as e:
            raise RetrievalError(f"Cannot reach OpenSearch at {host}:{port}: {e}") from e

        qdrant_config = self.db_config["qdrant"]
        q_host = qdrant_config.get("host", "localhost")
        q_port = qdrant_config.get("port", 6333)
        try:
            self.qdrant_client = QdrantClient(host=q_host, port=q_port)
            self.qdrant_client.get_collections()
        except Exception as e:
            raise RetrievalError(f"Cannot reach Qdrant at {q_host}:{q_port}: {e}") from e

        neo4j_config = self.db_config["neo4j"]
        uri = neo4j_config.get("uri", "bolt://localhost:7687")
        try:
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
                uri,
                auth=(neo4j_config.get("username", "neo4j"),
                      neo4j_config.get("password", "")))
            async with self.neo4j_driver.session() as session:
                await (await session.run("RETURN 1 AS ok")).single()
        except Exception as e:
            raise RetrievalError(f"Cannot reach Neo4j at {uri}: {e}") from e

        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Graph-RAG service connected")

    async def close(self) -> None:
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            self.neo4j_driver = None
        if self.qdrant_client:
            self.qdrant_client.close()
            self.qdrant_client = None

    # -----------------------------------------------------------------
    # Retrievers
    # -----------------------------------------------------------------

    def run_bm25_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Lexical retrieval. Finds exact drug names and trial identifiers."""
        try:
            response = self.opensearch_client.search(
                index=self.index_name,
                body={
                    "size": limit,
                    "query": {
                        "multi_match": {
                            "query": query,
                            # Title matches weigh more than body matches.
                            "fields": ["title^2", "content", "mesh_terms",
                                       "conditions", "interventions", "outcomes"],
                            # Otherwise a stopword in common is a match.
                            "minimum_should_match": BM25_MINIMUM_SHOULD_MATCH,
                        }
                    },
                },
            )
        except Exception as e:
            # Returning [] here would read as "no evidence matches this
            # query", which is a finding rather than a failure.
            raise RetrievalError(f"BM25 search failed for {query!r}: {e}") from e

        return [
            SearchResult(
                id=hit["_source"]["id"],
                title=hit["_source"].get("title", ""),
                content=hit["_source"].get("content", ""),
                source=hit["_source"].get("source", ""),
                score=float(hit["_score"]),
                retriever="bm25",
                metadata={
                    "pmid": hit["_source"].get("pmid"),
                    "nct_id": hit["_source"].get("nct_id"),
                    "journal": hit["_source"].get("journal"),
                    "pub_date": hit["_source"].get("pub_date"),
                },
            )
            for hit in response["hits"]["hits"]
        ]

    def run_vector_search(self, query: str, limit: int = 10,
                          min_similarity: float = MIN_VECTOR_SIMILARITY) -> List[SearchResult]:
        """Semantic retrieval. Finds paraphrases BM25 misses.

        Hits below `min_similarity` are dropped. Without a floor, a query
        the corpus does not cover still returns its k nearest neighbours,
        and nothing in the response distinguishes those from real matches.
        """
        try:
            vector = self.embedding_model.encode(query)
            hits = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=[float(v) for v in vector],
                limit=limit,
                with_payload=True,
                score_threshold=min_similarity,
            ).points
        except Exception as e:
            raise RetrievalError(f"Vector search failed for {query!r}: {e}") from e

        return [
            SearchResult(
                id=hit.payload["evidence_id"],
                title=hit.payload.get("title", ""),
                content=hit.payload.get("abstract", ""),
                source=hit.payload.get("source", ""),
                score=float(hit.score),
                retriever="vector",
                metadata={
                    "pmid": hit.payload.get("pmid"),
                    "nct_id": hit.payload.get("nct_id"),
                    "journal": hit.payload.get("journal"),
                    "pub_date": hit.payload.get("pub_date"),
                },
            )
            for hit in hits
        ]

    async def run_graph_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Retrieval through the graph.

        Finds evidence connected to entities whose names appear in the
        query, and ranks by how many of those entities each document
        shares. This is the view BM25 and vector search cannot provide:
        it reaches documents that never mention the query terms but are
        linked to the same conditions or interventions.
        """
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    """
                    MATCH (n)
                    WHERE (n:Condition OR n:Intervention OR n:Outcome
                           OR n:Population)
                      AND toLower($search_text) CONTAINS toLower(n.name)
                    MATCH (e:Evidence)-->(n)
                    WITH e, collect(DISTINCT n.name) AS matched
                    RETURN e.id AS id, e.title AS title, e.abstract AS abstract,
                           e.source AS source, e.pmid AS pmid,
                           e.nct_id AS nct_id, e.journal AS journal,
                           e.pub_date AS pub_date, matched,
                           size(matched) AS shared
                    ORDER BY shared DESC, e.id
                    LIMIT $limit
                    """, search_text=query, limit=limit)
                records = await result.data()
        except Exception as e:
            raise RetrievalError(f"Graph search failed for {query!r}: {e}") from e

        return [
            SearchResult(
                id=record["id"],
                title=record["title"] or "",
                content=record["abstract"] or "",
                source=record["source"] or "",
                score=float(record["shared"]),
                retriever="graph",
                metadata={
                    "pmid": record["pmid"],
                    "nct_id": record["nct_id"],
                    "journal": record["journal"],
                    "pub_date": record["pub_date"],
                    "matched_entities": record["matched"],
                },
            )
            for record in records
        ]

    async def graph_context_for(
        self, evidence_ids: List[str], limit: int = 25
    ) -> List[Dict[str, Any]]:
        """The condition -> intervention -> outcome context for a result set.

        This is what makes an answer inspectable: the reader can see which
        entities the cited evidence actually connects.
        """
        if not evidence_ids:
            return []

        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Evidence)-[r]->(n)
                    WHERE e.id IN $ids
                    RETURN e.id AS evidence_id, type(r) AS relationship,
                           labels(n)[0] AS entity_type, n.name AS entity
                    ORDER BY evidence_id, entity_type, entity
                    LIMIT $limit
                    """, ids=evidence_ids, limit=limit)
                return await result.data()
        except Exception as e:
            raise RetrievalError(f"Graph context lookup failed: {e}") from e

    # -----------------------------------------------------------------
    # Hybrid query
    # -----------------------------------------------------------------

    async def answer_query(
        self, query: str, limit: int = 5, per_retriever: int = 10
    ) -> EvidenceAnswer:
        """Run all three retrievers, fuse, and attach graph context."""
        if not query or not query.strip():
            raise ValueError("Query is empty; there is nothing to retrieve")

        ranked_lists = {
            "bm25": self.run_bm25_search(query, per_retriever),
            "vector": self.run_vector_search(query, per_retriever),
            "graph": await self.run_graph_search(query, per_retriever),
        }

        fused = reciprocal_rank_fusion(ranked_lists)[:limit]
        graph_context = await self.graph_context_for([r.id for r in fused])
        matched = any(ranked_lists.values())

        # How many retrievers found each returned document. Agreement
        # across independent retrievers is worth surfacing; it is not a
        # claim about whether the evidence answers the question.
        agreement = [len(result.found_by) for result in fused]

        return EvidenceAnswer(
            query=query,
            results=fused,
            graph_context=graph_context,
            retrievers_used=sorted(ranked_lists),
            coverage={
                "results_returned": len(fused),
                "candidates_per_retriever": {
                    name: len(results) for name, results in ranked_lists.items()
                },
                "retrievers_agreeing_on_top_result": agreement[0] if agreement else 0,
                "mean_retriever_agreement": (
                    sum(agreement) / len(agreement) if agreement else 0.0),
                "matched": matched,
                "note": (
                    "Agreement counts how many retrievers independently "
                    "surfaced each result. It measures retrieval consensus, "
                    "not whether the evidence answers the question."
                    if matched else
                    "No retriever matched this query above its relevance "
                    "floor. The indexed corpus does not cover it; that is "
                    "not a statement about the literature."
                ),
            },
        )


async def main():
    """Query the live stores. Requires `docker compose up -d` and an index."""
    service = GraphRAGService()
    await service.connect()
    try:
        for query in ["metformin cardiovascular outcomes",
                      "does empagliflozin reduce hospitalisation for heart failure"]:
            print(f"\n{'='*70}\nQuery: {query}\n{'='*70}")
            answer = await service.answer_query(query, limit=3)

            for rank, result in enumerate(answer.results, start=1):
                found = ", ".join(
                    f"{name}#{position}" for name, position in sorted(result.found_by.items()))
                print(f"\n{rank}. [{result.citation}] {result.title[:78]}")
                print(f"   fused score {result.fused_score:.4f}  found by: {found}")

            print(f"\n   coverage: {answer.coverage['candidates_per_retriever']}, "
                  f"mean agreement "
                  f"{answer.coverage['mean_retriever_agreement']:.2f}/3")
            if answer.graph_context:
                entities = sorted({row["entity"] for row in answer.graph_context})
                print(f"   graph context: {entities[:8]}")
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
