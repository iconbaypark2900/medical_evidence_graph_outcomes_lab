"""
Integration module to store ingested medical evidence into databases
"""
import asyncio
import neo4j
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
from dataclasses import dataclass
import json
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MedicalEvidence:
    """Represents a piece of medical evidence"""
    id: str
    title: str
    abstract: str
    pub_date: str
    authors: List[str]
    journal: str
    source: str
    pmid: str = None
    nct_id: str = None
    mesh_terms: List[str] = None
    entities: Dict[str, List[str]] = None


class EvidenceStorage:
    """Handles storage of medical evidence in all three databases"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        # Load config
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract database config
        self.db_config = {
            "neo4j": config.get("services", {}).get("evidence_graph_service", {}).get("graph_database", {}),
            "opensearch": config.get("services", {}).get("graph_rag_service", {}).get("opensearch", {}),
            "qdrant": config.get("services", {}).get("graph_rag_service", {}).get("qdrant", {})
        }
        
        # Initialize connection objects
        self.neo4j_driver = None
        self.opensearch_client = None
        self.qdrant_client = None
        self.embedding_model = None
    
    async def connect_all_databases(self):
        """Connect to all databases"""
        logger.info("Connecting to all databases...")
        
        # Connect to Neo4j
        try:
            uri = self.db_config["neo4j"].get("uri", "bolt://localhost:7687")
            username = self.db_config["neo4j"].get("username", "neo4j")
            password = self.db_config["neo4j"].get("password", "password")
            
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if record and record["test"] == 1:
                    logger.info("Successfully connected to Neo4j")
                else:
                    logger.error("Failed to test Neo4j connection")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            return False
        
        # Connect to OpenSearch
        try:
            host = self.db_config["opensearch"].get("host", "localhost")
            port = self.db_config["opensearch"].get("port", 9200)
            
            self.opensearch_client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
            
            # Test connection
            info = self.opensearch_client.info()
            logger.info(f"Successfully connected to OpenSearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Error connecting to OpenSearch: {e}")
            return False
        
        # Connect to Qdrant
        try:
            host = self.db_config["qdrant"].get("host", "localhost")
            port = self.db_config["qdrant"].get("port", 6333)
            
            self.qdrant_client = QdrantClient(host=host, port=port)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Successfully connected to Qdrant, collections: {len(collections.collections)}")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return False
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
        
        return True
    
    async def store_in_neo4j(self, evidence: List[MedicalEvidence]):
        """Store evidence in Neo4j graph database"""
        if not self.neo4j_driver:
            logger.error("No Neo4j driver available")
            return False
        
        try:
            async with self.neo4j_driver.session() as session:
                for ev in evidence:
                    # Create the evidence node
                    await session.run(
                        """
                        MERGE (e:Evidence {id: $id})
                        SET e.title = $title,
                            e.abstract = $abstract,
                            e.pub_date = $pub_date,
                            e.authors = $authors,
                            e.journal = $journal,
                            e.source = $source,
                            e.pmid = $pmid,
                            e.nct_id = $nct_id,
                            e.mesh_terms = $mesh_terms,
                            e.created_at = $created_at
                        """,
                        id=ev.id,
                        title=ev.title,
                        abstract=ev.abstract,
                        pub_date=ev.pub_date,
                        authors=ev.authors,
                        journal=ev.journal,
                        source=ev.source,
                        pmid=ev.pmid,
                        nct_id=ev.nct_id,
                        mesh_terms=ev.mesh_terms,
                        created_at=datetime.now().isoformat()
                    )
                    
                    # Create entity nodes and relationships based on extracted entities
                    if ev.entities:
                        # Create condition nodes and relationships
                        for condition in ev.entities.get("conditions", []):
                            await session.run(
                                """
                                MERGE (c:Condition {name: $condition})
                                WITH c
                                MATCH (e:Evidence {id: $evidence_id})
                                MERGE (e)-[:HAS_CONDITION]->(c)
                                """,
                                condition=condition,
                                evidence_id=ev.id
                            )
                        
                        # Create intervention nodes and relationships
                        for intervention in ev.entities.get("interventions", []):
                            await session.run(
                                """
                                MERGE (i:Intervention {name: $intervention})
                                WITH i
                                MATCH (e:Evidence {id: $evidence_id})
                                MERGE (e)-[:HAS_INTERVENTION]->(i)
                                """,
                                intervention=intervention,
                                evidence_id=ev.id
                            )
                        
                        # Create outcome nodes and relationships
                        for outcome in ev.entities.get("outcomes", []):
                            await session.run(
                                """
                                MERGE (o:Outcome {name: $outcome})
                                WITH o
                                MATCH (e:Evidence {id: $evidence_id})
                                MERGE (e)-[:HAS_OUTCOME]->(o)
                                """,
                                outcome=outcome,
                                evidence_id=ev.id
                            )
                
                logger.info(f"Stored {len(evidence)} evidence items in Neo4j")
                return True
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
            return False
    
    def store_in_opensearch(self, evidence: List[MedicalEvidence]):
        """Store evidence in OpenSearch for BM25 search"""
        if not self.opensearch_client:
            logger.error("No OpenSearch client available")
            return False
        
        try:
            index_name = "medical_evidence"
            success_count = 0
            
            for ev in evidence:
                doc = {
                    "id": ev.id,
                    "title": ev.title,
                    "content": ev.abstract,
                    "source": ev.source,
                    "pub_date": ev.pub_date,
                    "authors": ev.authors,
                    "journal": ev.journal,
                    "pmid": ev.pmid,
                    "nct_id": ev.nct_id,
                    "entities": ev.entities or {},
                    "mesh_terms": ev.mesh_terms or [],
                    "metadata": {
                        "created_at": datetime.now().isoformat()
                    }
                }
                
                response = self.opensearch_client.index(
                    index=index_name,
                    body=doc,
                    id=ev.id
                )
                
                if response.get('result') in ['created', 'updated']:
                    success_count += 1
            
            logger.info(f"Successfully stored {success_count}/{len(evidence)} evidence items in OpenSearch")
            return True
        except Exception as e:
            logger.error(f"Error storing in OpenSearch: {e}")
            return False
    
    def store_in_qdrant(self, evidence: List[MedicalEvidence]):
        """Store evidence embeddings in Qdrant for vector search"""
        if not self.qdrant_client or not self.embedding_model:
            logger.error("No Qdrant client or embedding model available")
            return False
        
        try:
            collection_name = "medical_evidence_embeddings"
            points = []
            
            for i, ev in enumerate(evidence):
                # Create embedding from title and abstract
                text_to_embed = f"{ev.title} {ev.abstract}"
                embedding = self.embedding_model.encode(text_to_embed).tolist()
                
                # Create a payload with metadata
                payload = {
                    "id": ev.id,
                    "title": ev.title,
                    "abstract": ev.abstract,
                    "source": ev.source,
                    "pub_date": ev.pub_date,
                    "authors": ev.authors,
                    "journal": ev.journal,
                    "entities": ev.entities or {},
                    "created_at": datetime.now().isoformat()
                }
                
                points.append({
                    "id": i,  # Using index as ID for simplicity
                    "vector": embedding,
                    "payload": payload
                })
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Successfully stored {len(points)} evidence embeddings in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            return False
    
    async def store_all_evidence(self, evidence: List[MedicalEvidence]):
        """Store evidence in all three databases"""
        logger.info(f"Storing {len(evidence)} evidence items in all databases...")
        
        # Store in Neo4j (graph)
        neo4j_ok = await self.store_in_neo4j(evidence)
        
        # Store in OpenSearch (BM25)
        opensearch_ok = self.store_in_opensearch(evidence)
        
        # Store in Qdrant (vectors)
        qdrant_ok = self.store_in_qdrant(evidence)
        
        results = {
            "neo4j": neo4j_ok,
            "opensearch": opensearch_ok,
            "qdrant": qdrant_ok
        }
        
        logger.info(f"Storage results: {results}")
        return results
    
    async def close_connections(self):
        """Close all database connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        logger.info("All database connections closed")


async def integrate_ingestion_and_storage():
    """Integrate the ingestion and storage modules"""
    logger.info("Starting ingestion and storage integration...")
    
    # Initialize storage
    storage = EvidenceStorage()
    
    # Connect to databases
    if not await storage.connect_all_databases():
        logger.error("Failed to connect to databases")
        return False
    
    # Import ingestion module to get evidence
    from data_ingestion import ingest_medical_evidence
    
    # Define search terms for testing
    search_terms = [
        "diabetes treatment",
        "cancer therapy"
    ]
    
    # Ingest evidence
    logger.info("Starting evidence ingestion...")
    evidence = await ingest_medical_evidence(search_terms, max_per_source=2)
    
    if not evidence:
        logger.warning("No evidence was ingested, skipping storage")
        await storage.close_connections()
        return True
    
    # Store in all databases
    logger.info("Starting evidence storage in databases...")
    results = await storage.store_all_evidence(evidence)
    
    # Close connections
    await storage.close_connections()
    
    logger.info(f"Ingestion and storage completed. Results: {results}")
    return all(results.values())


async def main():
    """Main function to run the integration"""
    logger.info("Starting medical evidence ingestion and storage integration test...")
    
    success = await integrate_ingestion_and_storage()
    
    if success:
        logger.info("Integration completed successfully!")
    else:
        logger.error("Integration failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())