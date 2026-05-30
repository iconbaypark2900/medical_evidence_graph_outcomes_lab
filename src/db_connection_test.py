"""
Database connection utilities and initial schema setup for Medical Evidence Graph & Outcomes Insight Lab
"""
import asyncio
import neo4j
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
import logging
from typing import Optional


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages connections to all databases used in the system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.neo4j_driver: Optional[neo4j.AsyncGraphDatabase] = None
        self.opensearch_client: Optional[OpenSearch] = None
        self.qdrant_client: Optional[QdrantClient] = None
    
    async def connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            uri = self.config.get("neo4j", {}).get("uri", "bolt://localhost:7687")
            username = self.config.get("neo4j", {}).get("username", "neo4j")
            password = self.config.get("neo4j", {}).get("password", "password")
            
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if record and record["test"] == 1:
                    logger.info("Successfully connected to Neo4j")
                    return True
                else:
                    logger.error("Failed to test Neo4j connection")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            return False
    
    def connect_opensearch(self):
        """Connect to OpenSearch database"""
        try:
            host = self.config.get("opensearch", {}).get("host", "localhost")
            port = self.config.get("opensearch", {}).get("port", 9200)
            
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
            return True
        except Exception as e:
            logger.error(f"Error connecting to OpenSearch: {e}")
            return False
    
    def connect_qdrant(self):
        """Connect to Qdrant database"""
        try:
            host = self.config.get("qdrant", {}).get("host", "localhost")
            port = self.config.get("qdrant", {}).get("port", 6333)
            
            self.qdrant_client = QdrantClient(host=host, port=port)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Successfully connected to Qdrant, collections: {len(collections.collections)}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return False
    
    async def setup_neo4j_schema(self):
        """Set up initial Neo4j schema with constraints and indexes"""
        if not self.neo4j_driver:
            logger.error("No Neo4j driver available")
            return False
        
        try:
            # Create constraints for important node properties
            constraints = [
                "CREATE CONSTRAINT unique_condition IF NOT EXISTS FOR (c:Condition) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT unique_intervention IF NOT EXISTS FOR (i:Intervention) REQUIRE i.id IS UNIQUE", 
                "CREATE CONSTRAINT unique_outcome IF NOT EXISTS FOR (o:Outcome) REQUIRE o.id IS UNIQUE",
                "CREATE CONSTRAINT unique_trial IF NOT EXISTS FOR (t:Trial) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT unique_guideline IF NOT EXISTS FOR (g:Guideline) REQUIRE g.id IS UNIQUE"
            ]
            
            async with self.neo4j_driver.session() as session:
                for constraint in constraints:
                    await session.run(constraint)
                    logger.info(f"Applied constraint: {constraint}")
            
            logger.info("Neo4j schema setup completed")
            return True
        except Exception as e:
            logger.error(f"Error setting up Neo4j schema: {e}")
            return False
    
    def setup_opensearch_indices(self):
        """Set up initial OpenSearch indices"""
        if not self.opensearch_client:
            logger.error("No OpenSearch client available")
            return False
        
        try:
            # Define the mapping for evidence documents
            evidence_mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "pub_date": {"type": "date"},
                        "authors": {"type": "keyword"},
                        "journal": {"type": "keyword"},
                        "entities": {
                            "properties": {
                                "conditions": {"type": "keyword"},
                                "interventions": {"type": "keyword"},
                                "outcomes": {"type": "keyword"},
                                "populations": {"type": "keyword"}
                            }
                        },
                        "metadata": {"type": "object", "enabled": False}
                    }
                }
            }
            
            # Create the evidence index
            index_name = "medical_evidence"
            if not self.opensearch_client.indices.exists(index=index_name):
                self.opensearch_client.indices.create(index=index_name, body=evidence_mapping)
                logger.info(f"Created OpenSearch index: {index_name}")
            else:
                logger.info(f"OpenSearch index {index_name} already exists")
                
            return True
        except Exception as e:
            logger.error(f"Error setting up OpenSearch indices: {e}")
            return False
    
    def setup_qdrant_collections(self):
        """Set up initial Qdrant collections"""
        if not self.qdrant_client:
            logger.error("No Qdrant client available")
            return False
        
        try:
            # Define vector size for medical text embeddings
            vector_size = 384  # Using sentence-transformers all-MiniLM-L6-v2 size
            
            # Create collection for document embeddings
            from qdrant_client.http import models
            
            collection_name = "medical_evidence_embeddings"
            
            if collection_name not in [c.name for c in self.qdrant_client.get_collections().collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Qdrant collection {collection_name} already exists")
                
            return True
        except Exception as e:
            logger.error(f"Error setting up Qdrant collections: {e}")
            return False
    
    async def setup_all_schemas(self):
        """Set up schemas for all databases"""
        logger.info("Setting up schemas for all databases...")
        
        neo4j_ok = await self.setup_neo4j_schema() if self.neo4j_driver else False
        opensearch_ok = self.setup_opensearch_indices() if self.opensearch_client else False
        qdrant_ok = self.setup_qdrant_collections() if self.qdrant_client else False
        
        return neo4j_ok and opensearch_ok and qdrant_ok
    
    async def close_connections(self):
        """Close all database connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        
        logger.info("All database connections closed")


async def test_database_connections(config_path: str = "config/settings.json"):
    """Test connections to all databases"""
    import json
    from pathlib import Path
    
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract database config
    db_config = {
        "neo4j": config.get("services", {}).get("evidence_graph_service", {}).get("graph_database", {}),
        "opensearch": config.get("services", {}).get("graph_rag_service", {}).get("opensearch", {}),
        "qdrant": config.get("services", {}).get("graph_rag_service", {}).get("qdrant", {})
    }
    
    # Create database manager
    db_manager = DatabaseManager(db_config)
    
    # Test connections
    logger.info("Testing database connections...")
    
    neo4j_connected = await db_manager.connect_neo4j()
    opensearch_connected = db_manager.connect_opensearch()
    qdrant_connected = db_manager.connect_qdrant()
    
    if neo4j_connected and opensearch_connected and qdrant_connected:
        logger.info("All database connections successful!")
        
        # Set up schemas
        schemas_ok = await db_manager.setup_all_schemas()
        if schemas_ok:
            logger.info("All database schemas set up successfully!")
        else:
            logger.error("Error setting up database schemas")
    
    else:
        logger.error("One or more database connections failed")
    
    # Close connections
    await db_manager.close_connections()
    
    return neo4j_connected and opensearch_connected and qdrant_connected


async def main():
    """Main function to test database connections"""
    logger.info("Starting database connection test for Medical Evidence Graph & Outcomes Insight Lab...")
    
    success = await test_database_connections()
    
    if success:
        logger.info("Database setup completed successfully!")
    else:
        logger.error("Database setup failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())