"""
Storage layer: index medical evidence into Neo4j, OpenSearch and Qdrant.

This module holds the real database clients. It is the write half of the
graph-RAG pipeline; `src/graph_rag_service/main.py` is the read half.

Three bugs made the previous version unsafe to rely on:

- Qdrant points were keyed by the loop index, so every ingest run wrote
  points 0..n-1 and silently overwrote whatever the previous run had put
  there. Ids are now derived from the evidence id and are stable across
  runs.
- No schema was ever created. The Qdrant collection did not exist (so
  upserts failed), Neo4j had no uniqueness constraint on Evidence.id (so
  MERGE could duplicate), and the OpenSearch index was left to dynamic
  mapping, where the free-form `entities` object grows a new field per
  distinct key.
- Every failure path logged and returned False. Callers treated a failed
  store as a completed one, so a half-populated index looked identical to
  a full one — and the retrieval built on top would then report "no
  evidence found" for evidence that was simply never written.
"""
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import neo4j
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.data_ingestion import MedicalEvidence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EVIDENCE_INDEX = "medical_evidence"
EVIDENCE_COLLECTION = "medical_evidence_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Entity kind -> (Neo4j node label, relationship type)
ENTITY_SCHEMA = {
    "conditions": ("Condition", "HAS_CONDITION"),
    "interventions": ("Intervention", "HAS_INTERVENTION"),
    "outcomes": ("Outcome", "HAS_OUTCOME"),
    "populations": ("Population", "HAS_POPULATION"),
}

# Qdrant ids must be an unsigned integer or a UUID. Deriving a UUID5 from
# the evidence id makes re-indexing the same document update it in place
# rather than appending a duplicate under a fresh id.
POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class StorageError(RuntimeError):
    """A write did not complete. Never swallowed into a False return."""


def point_id_for(evidence_id: str) -> str:
    """Stable Qdrant point id for an evidence id."""
    return str(uuid.uuid5(POINT_NAMESPACE, evidence_id))


def load_database_config(config_path: str = "config/settings.json") -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = json.loads(config_file.read_text())
    services = config.get("services", {})
    return {
        "neo4j": services.get("graph_rag_service", {}).get("neo4j", {}),
        "opensearch": services.get("graph_rag_service", {}).get("opensearch", {}),
        "qdrant": services.get("graph_rag_service", {}).get("qdrant", {}),
    }


def evidence_to_document(evidence: MedicalEvidence) -> Dict[str, Any]:
    """Flatten one record into the OpenSearch document shape.

    Entities are flattened to top-level keyword arrays rather than left as
    a nested free-form object: under dynamic mapping every new entity key
    would add a field to the index mapping, which grows without bound.
    """
    entities = evidence.entities or {}
    return {
        "id": evidence.id,
        "title": evidence.title,
        "content": evidence.abstract,
        "source": evidence.source,
        "pub_date": evidence.pub_date,
        "authors": evidence.authors or [],
        "journal": evidence.journal,
        "pmid": evidence.pmid,
        "nct_id": evidence.nct_id,
        "mesh_terms": evidence.mesh_terms or [],
        "conditions": entities.get("conditions", []),
        "interventions": entities.get("interventions", []),
        "outcomes": entities.get("outcomes", []),
        "populations": entities.get("populations", []),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


class EvidenceStore:
    """Writes evidence into all three backing stores."""

    def __init__(self, config_path: str = "config/settings.json",
                 embedding_model: Any = None,
                 index_name: str = EVIDENCE_INDEX,
                 collection_name: str = EVIDENCE_COLLECTION):
        self.db_config = load_database_config(config_path)
        self.neo4j_driver: Optional[neo4j.AsyncDriver] = None
        self.opensearch_client: Optional[OpenSearch] = None
        self.qdrant_client: Optional[QdrantClient] = None
        # Injectable so tests can supply a stub instead of downloading a model.
        self.embedding_model = embedding_model
        # Overridable so tests can target their own index and collection.
        self.index_name = index_name
        self.collection_name = collection_name

    # -----------------------------------------------------------------
    # Connections
    # -----------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to every store. Raises if any one is unreachable.

        Partial connectivity is not a usable state: it produces an index
        that is populated in some stores and not others, and retrieval
        cannot tell that apart from a document that does not exist.
        """
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
            raise StorageError(f"Cannot reach Neo4j at {uri}: {e}") from e

        opensearch_config = self.db_config["opensearch"]
        host = opensearch_config.get("host", "localhost")
        port = opensearch_config.get("port", 9200)
        try:
            self.opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                use_ssl=False, verify_certs=False, ssl_show_warn=False)
            self.opensearch_client.info()
        except Exception as e:
            raise StorageError(f"Cannot reach OpenSearch at {host}:{port}: {e}") from e

        qdrant_config = self.db_config["qdrant"]
        q_host = qdrant_config.get("host", "localhost")
        q_port = qdrant_config.get("port", 6333)
        try:
            self.qdrant_client = QdrantClient(host=q_host, port=q_port)
            self.qdrant_client.get_collections()
        except Exception as e:
            raise StorageError(f"Cannot reach Qdrant at {q_host}:{q_port}: {e}") from e

        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Connected to Neo4j, OpenSearch and Qdrant")

    async def close(self) -> None:
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            self.neo4j_driver = None
        if self.qdrant_client:
            self.qdrant_client.close()
            self.qdrant_client = None
        logger.info("Closed database connections")

    @property
    def vector_size(self) -> int:
        # sentence-transformers renamed this; support both spellings so the
        # collection is never created with a guessed dimension.
        for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            getter = getattr(self.embedding_model, attribute, None)
            if getter is not None:
                return int(getter())
        raise StorageError(
            f"Cannot determine embedding dimension from "
            f"{type(self.embedding_model).__name__}")

    # -----------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create constraints, index mapping and collection if absent.

        Idempotent, and required before the first write: without it the
        Qdrant collection does not exist and every upsert fails.
        """
        async with self.neo4j_driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT evidence_id IF NOT EXISTS "
                "FOR (e:Evidence) REQUIRE e.id IS UNIQUE")
            for label, _ in ENTITY_SCHEMA.values():
                await session.run(
                    f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE")

        if not self.opensearch_client.indices.exists(index=self.index_name):
            self.opensearch_client.indices.create(
                index=self.index_name,
                body={
                    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                    "mappings": {
                        # Explicit, so a new entity key cannot add a field.
                        "dynamic": "strict",
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "content": {"type": "text"},
                            "source": {"type": "keyword"},
                            "pub_date": {"type": "keyword"},
                            "authors": {"type": "keyword"},
                            "journal": {"type": "text"},
                            "pmid": {"type": "keyword"},
                            "nct_id": {"type": "keyword"},
                            "mesh_terms": {"type": "keyword"},
                            "conditions": {"type": "keyword"},
                            "interventions": {"type": "keyword"},
                            "outcomes": {"type": "keyword"},
                            "populations": {"type": "keyword"},
                            "indexed_at": {"type": "date"},
                        },
                    },
                })
            logger.info(f"Created OpenSearch index {self.index_name}")

        existing = {c.name for c in self.qdrant_client.get_collections().collections}
        if self.collection_name not in existing:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(
                f"Created Qdrant collection {self.collection_name} "
                f"({self.vector_size}d, cosine)")

    # -----------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------

    async def store_in_neo4j(self, evidence: List[MedicalEvidence]) -> int:
        """Upsert evidence nodes and their entity relationships."""
        if not evidence:
            return 0

        rows = [{
            "id": e.id,
            "title": e.title,
            "abstract": e.abstract,
            "pub_date": e.pub_date,
            "authors": e.authors or [],
            "journal": e.journal,
            "source": e.source,
            "pmid": e.pmid,
            "nct_id": e.nct_id,
            "mesh_terms": e.mesh_terms or [],
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        } for e in evidence]

        try:
            async with self.neo4j_driver.session() as session:
                # One round trip for all evidence nodes, rather than one per
                # document and one per entity as before.
                await session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Evidence {id: row.id})
                    SET e += row
                    """, rows=rows)

                for kind, (label, relationship) in ENTITY_SCHEMA.items():
                    links = [
                        {"evidence_id": e.id, "name": name}
                        for e in evidence
                        for name in (e.entities or {}).get(kind, [])
                        if name
                    ]
                    if not links:
                        continue
                    await session.run(
                        f"""
                        UNWIND $links AS link
                        MATCH (e:Evidence {{id: link.evidence_id}})
                        MERGE (n:{label} {{name: link.name}})
                        MERGE (e)-[:{relationship}]->(n)
                        """, links=links)
        except Exception as e:
            raise StorageError(
                f"Neo4j write failed for {len(evidence)} records: {e}") from e

        logger.info(f"Stored {len(evidence)} evidence nodes in Neo4j")
        return len(evidence)

    def store_in_opensearch(self, evidence: List[MedicalEvidence]) -> int:
        """Index documents for BM25 retrieval."""
        if not evidence:
            return 0

        operations = []
        for item in evidence:
            operations.append({"index": {"_index": self.index_name, "_id": item.id}})
            operations.append(evidence_to_document(item))

        try:
            response = self.opensearch_client.bulk(body=operations, refresh=True)
        except Exception as e:
            raise StorageError(
                f"OpenSearch write failed for {len(evidence)} records: {e}") from e

        if response.get("errors"):
            failures = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise StorageError(
                f"OpenSearch rejected {len(failures)} of {len(evidence)} "
                f"documents: {failures[:3]}")

        logger.info(f"Indexed {len(evidence)} documents in OpenSearch")
        return len(evidence)

    def store_in_qdrant(self, evidence: List[MedicalEvidence]) -> int:
        """Embed and upsert vectors for semantic retrieval."""
        if not evidence:
            return 0

        texts = [f"{e.title}\n\n{e.abstract}".strip() for e in evidence]
        vectors = self.embedding_model.encode(texts)

        points = [
            PointStruct(
                # Stable id: re-indexing updates in place. Using the loop
                # index meant run two overwrote run one, point for point.
                id=point_id_for(item.id),
                vector=[float(v) for v in vector],
                payload={
                    "evidence_id": item.id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "source": item.source,
                    "pub_date": item.pub_date,
                    "journal": item.journal,
                    "pmid": item.pmid,
                    "nct_id": item.nct_id,
                },
            )
            for item, vector in zip(evidence, vectors)
        ]

        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points, wait=True)
        except Exception as e:
            raise StorageError(
                f"Qdrant write failed for {len(evidence)} records: {e}") from e

        logger.info(f"Stored {len(points)} embeddings in Qdrant")
        return len(points)

    async def store_all_evidence(self, evidence: List[MedicalEvidence]) -> Dict[str, int]:
        """Write to all three stores. Any failure raises.

        The counts are what was actually written. The previous version
        returned {"neo4j": True, ...} where False meant a logged exception,
        which callers read as success.
        """
        logger.info(f"Storing {len(evidence)} evidence items")
        return {
            "neo4j": await self.store_in_neo4j(evidence),
            "opensearch": self.store_in_opensearch(evidence),
            "qdrant": self.store_in_qdrant(evidence),
        }


# Backwards-compatible alias: the class was called EvidenceStorage.
EvidenceStorage = EvidenceStore


async def integrate_ingestion_and_storage(
    search_terms: Optional[List[str]] = None, max_per_source: int = 3
) -> Dict[str, int]:
    """Ingest from the live APIs and index the result."""
    from src.evidence_ingestion_service.main import EvidenceIngestionService

    search_terms = search_terms or ["metformin cardiovascular outcomes"]

    store = EvidenceStore()
    await store.connect()
    try:
        await store.ensure_schema()

        async with EvidenceIngestionService() as ingestion:
            evidence = await ingestion.fetch_sources(search_terms, max_per_source)
            evidence = ingestion.enrich_entities(evidence)

        if not evidence:
            # A genuinely empty result, because fetch raises on failure.
            logger.warning(f"No evidence matched {search_terms}")
            return {"neo4j": 0, "opensearch": 0, "qdrant": 0}

        return await store.store_all_evidence(evidence)
    finally:
        await store.close()


async def main():
    results = await integrate_ingestion_and_storage()
    logger.info(f"Indexed: {results}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
