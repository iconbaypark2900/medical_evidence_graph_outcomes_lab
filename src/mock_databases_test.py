"""
Mock database implementations for testing the medical evidence system
This simulates Neo4j, OpenSearch, and Qdrant for development without Docker
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class MockNode:
    """Mock Neo4j node"""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class MockRelationship:
    """Mock Neo4j relationship"""
    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any]


class MockNeo4jDriver:
    """Mock Neo4j driver for testing"""
    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        self.constraints = set()
    
    async def close(self):
        """Close the mock driver"""
        pass
    
    class MockSession:
        def __init__(self, driver):
            self.driver = driver
        
        async def run(self, query: str, **params):
            """Mock query execution"""
            print(f"Neo4j mock executing query: {query[:50]}... with params {list(params.keys())}")
            
            # Handle simple mock queries
            if "CREATE CONSTRAINT" in query.upper():
                constraint_name = params.get('constraint_name', 'unknown')
                self.driver.constraints.add(constraint_name)
                print(f"Created constraint: {constraint_name}")
                return MockResult([{"test": 1}])
            elif "MERGE" in query.upper() or "CREATE" in query.upper():
                # Mock MERGE/CREATE operations
                if "Evidence" in query:
                    evidence_id = params.get('id', 'unknown')
                    print(f"Created/merged evidence: {evidence_id}")
                    # Store in mock DB
                    self.driver.nodes[evidence_id] = {
                        'id': evidence_id,
                        'type': 'Evidence',
                        'properties': params
                    }
                elif "Condition" in query:
                    condition_name = params.get('condition', 'unknown')
                    print(f"Created/merged condition: {condition_name}")
                    # Store in mock DB
                    condition_id = f"cond_{hash(condition_name)}"
                    self.driver.nodes[condition_id] = {
                        'id': condition_id,
                        'type': 'Condition',
                        'properties': {'name': condition_name}
                    }
                    # Create relationship
                    if 'evidence_id' in params:
                        rel_id = f"rel_{hash(condition_name+params['evidence_id'])}"
                        self.driver.relationships[rel_id] = {
                            'start_node': params['evidence_id'],
                            'end_node': condition_id,
                            'type': 'HAS_CONDITION'
                        }
                elif "Intervention" in query:
                    intervention_name = params.get('intervention', 'unknown')
                    print(f"Created/merged intervention: {intervention_name}")
                    # Store in mock DB
                    intervention_id = f"int_{hash(intervention_name)}"
                    self.driver.nodes[intervention_id] = {
                        'id': intervention_id,
                        'type': 'Intervention',
                        'properties': {'name': intervention_name}
                    }
                    # Create relationship
                    if 'evidence_id' in params:
                        rel_id = f"rel_{hash(intervention_name+params['evidence_id'])}"
                        self.driver.relationships[rel_id] = {
                            'start_node': params['evidence_id'],
                            'end_node': intervention_id,
                            'type': 'HAS_INTERVENTION'
                        }
                elif "Outcome" in query:
                    outcome_name = params.get('outcome', 'unknown')
                    print(f"Created/merged outcome: {outcome_name}")
                    # Store in mock DB
                    outcome_id = f"out_{hash(outcome_name)}"
                    self.driver.nodes[outcome_id] = {
                        'id': outcome_id,
                        'type': 'Outcome',
                        'properties': {'name': outcome_name}
                    }
                    # Create relationship
                    if 'evidence_id' in params:
                        rel_id = f"rel_{hash(outcome_name+params['evidence_id'])}"
                        self.driver.relationships[rel_id] = {
                            'start_node': params['evidence_id'],
                            'end_node': outcome_id,
                            'type': 'HAS_OUTCOME'
                        }
                
                return MockResult([{"result": "success"}])
            elif "RETURN 1" in query:
                return MockResult([{"test": 1}])
            else:
                # Handle other queries
                return MockResult([])
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def session(self):
        """Return a mock session"""
        return self.MockSession(self)


class MockResult:
    """Mock result for Neo4j queries"""
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    async def single(self):
        """Get single result"""
        if self.data:
            return self.data[0]
        return None


class MockOpenSearch:
    """Mock OpenSearch client for testing"""
    def __init__(self):
        self.indices = {}
        self.documents = {}
    
    def info(self):
        """Mock info response"""
        return {"version": {"number": "2.9.0"}}
    
    def indices_exists(self, index: str):
        """Check if index exists"""
        return index in self.indices
    
    def indices_create(self, index: str, body: Dict[str, Any]):
        """Create an index"""
        self.indices[index] = body
        print(f"Created OpenSearch mock index: {index}")
    
    def index(self, index: str, body: Dict[str, Any], id: str):
        """Index a document"""
        if index not in self.documents:
            self.documents[index] = {}
        self.documents[index][id] = body
        print(f"Indexed document in {index}: {id}")
        return {"result": "created"}


class MockQdrantClient:
    """Mock Qdrant client for testing"""
    def __init__(self):
        self.collections = {}
        self.points = {}
    
    def get_collections(self):
        """Get collections"""
        class Collections:
            def __init__(self, names):
                self.collections = [type('Collection', (), {'name': name})() for name in names]
        
        return Collections(list(self.collections.keys()))
    
    def create_collection(self, collection_name: str, vectors_config: Dict[str, Any]):
        """Create a collection"""
        self.collections[collection_name] = vectors_config
        self.points[collection_name] = []
        print(f"Created Qdrant mock collection: {collection_name}")
    
    def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        """Upsert points"""
        if collection_name in self.points:
            self.points[collection_name].extend(points)
            print(f"Upserted {len(points)} points to {collection_name}")
        else:
            print(f"Collection {collection_name} not found!")


# Replace the real database libraries with mocks for testing purposes
class MockDatabaseManager:
    """Manages mock connections to databases for testing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.neo4j_driver = None
        self.opensearch_client = None
        self.qdrant_client = None
        self.embedding_model = None
    
    async def connect_neo4j(self):
        """Connect to mock Neo4j"""
        try:
            self.neo4j_driver = MockNeo4jDriver()
            print("Successfully connected to mock Neo4j")
            return True
        except Exception as e:
            print(f"Error connecting to mock Neo4j: {e}")
            return False
    
    def connect_opensearch(self):
        """Connect to mock OpenSearch"""
        try:
            self.opensearch_client = MockOpenSearch()
            print("Successfully connected to mock OpenSearch")
            return True
        except Exception as e:
            print(f"Error connecting to mock OpenSearch: {e}")
            return False
    
    def connect_qdrant(self):
        """Connect to mock Qdrant"""
        try:
            self.qdrant_client = MockQdrantClient()
            print("Successfully connected to mock Qdrant")
            return True
        except Exception as e:
            print(f"Error connecting to mock Qdrant: {e}")
            return False
    
    async def setup_neo4j_schema(self):
        """Set up mock Neo4j schema"""
        if not self.neo4j_driver:
            print("No Neo4j driver available")
            return False
        
        try:
            constraints = [
                "unique_condition",
                "unique_intervention", 
                "unique_outcome",
                "unique_trial",
                "unique_guideline"
            ]
            
            async with self.neo4j_driver.session() as session:
                for constraint in constraints:
                    await session.run(f"CREATE CONSTRAINT {constraint} IF NOT EXISTS FOR (c:Label) REQUIRE c.id IS UNIQUE", 
                                    constraint_name=constraint)
            
            print("Mock Neo4j schema setup completed")
            return True
        except Exception as e:
            print(f"Error setting up mock Neo4j schema: {e}")
            return False
    
    def setup_opensearch_indices(self):
        """Set up mock OpenSearch indices"""
        if not self.opensearch_client:
            print("No OpenSearch client available")
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
            if not self.opensearch_client.indices_exists(index=index_name):
                self.opensearch_client.indices_create(index=index_name, body=evidence_mapping)
                print(f"Created mock OpenSearch index: {index_name}")
            else:
                print(f"Mock OpenSearch index {index_name} already exists")
                
            return True
        except Exception as e:
            print(f"Error setting up mock OpenSearch indices: {e}")
            return False
    
    def setup_qdrant_collections(self):
        """Set up mock Qdrant collections"""
        if not self.qdrant_client:
            print("No Qdrant client available")
            return False
        
        try:
            # Define vector size for medical text embeddings
            vector_size = 384  # Using sentence-transformers all-MiniLM-L6-v2 size
            
            # Create collection for document embeddings
            collection_name = "medical_evidence_embeddings"
            
            if collection_name not in [c.name for c in self.qdrant_client.get_collections().collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"size": vector_size, "distance": "cosine"}
                )
                print(f"Created mock Qdrant collection: {collection_name}")
            else:
                print(f"Mock Qdrant collection {collection_name} already exists")
                
            return True
        except Exception as e:
            print(f"Error setting up mock Qdrant collections: {e}")
            return False
    
    async def setup_all_schemas(self):
        """Set up schemas for all mock databases"""
        print("Setting up schemas for all mock databases...")
        
        neo4j_ok = await self.setup_neo4j_schema() if self.neo4j_driver else False
        opensearch_ok = self.setup_opensearch_indices() if self.opensearch_client else False
        qdrant_ok = self.setup_qdrant_collections() if self.qdrant_client else False
        
        return neo4j_ok and opensearch_ok and qdrant_ok
    
    async def close_connections(self):
        """Close mock database connections"""
        print("Closing mock database connections")


async def test_mock_databases():
    """Test the mock database setup"""
    print("Testing mock database connections...")
    
    # Create a mock config similar to what would be in the real config
    config = {
        "services": {
            "evidence_graph_service": {
                "graph_database": {
                    "uri": "bolt://mock:7687",
                    "username": "mock",
                    "password": "mock"
                }
            },
            "graph_rag_service": {
                "opensearch": {
                    "host": "mock",
                    "port": 9200
                },
                "qdrant": {
                    "host": "mock",
                    "port": 6333
                }
            }
        }
    }
    
    # Create database manager with mock databases
    db_manager = MockDatabaseManager(config["services"])
    
    # Test connections
    print("Testing mock database connections...")
    
    neo4j_connected = await db_manager.connect_neo4j()
    opensearch_connected = db_manager.connect_opensearch()
    qdrant_connected = db_manager.connect_qdrant()
    
    print(f"Mock connections - Neo4j: {neo4j_connected}, OpenSearch: {opensearch_connected}, Qdrant: {qdrant_connected}")
    
    if neo4j_connected and opensearch_connected and qdrant_connected:
        print("All mock database connections successful!")
        
        # Set up schemas
        schemas_ok = await db_manager.setup_all_schemas()
        if schemas_ok:
            print("All mock database schemas set up successfully!")
        else:
            print("Error setting up mock database schemas")
    
    else:
        print("One or more mock database connections failed")
    
    # Close connections
    await db_manager.close_connections()
    
    return neo4j_connected and opensearch_connected and qdrant_connected


if __name__ == "__main__":
    asyncio.run(test_mock_databases())