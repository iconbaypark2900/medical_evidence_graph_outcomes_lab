"""
Main entry point for the Graph-RAG Service
"""
import asyncio
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class Document:
    """Represents a document in the system"""
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result"""
    id: str
    score: float
    content: str
    source: str
    metadata: Dict[str, Any]


class MockOpenSearch:
    """Mock implementation of OpenSearch for BM25 search"""
    
    def __init__(self):
        self.documents = {}
    
    def index_document(self, doc: Document):
        """Index a document"""
        self.documents[doc.id] = doc
        print(f"Indexed document in OpenSearch: {doc.id} - {doc.title}")
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform BM25 search"""
        # Simple keyword matching for demonstration
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in self.documents.items():
            # Check if query terms appear in title or content
            score = 0
            if query_lower in doc.title.lower():
                score += 2  # Higher weight for title matches
            if query_lower in doc.content.lower():
                score += doc.content.lower().count(query_lower)  # Count occurrences in content
            
            if score > 0:
                results.append(SearchResult(
                    id=doc_id,
                    score=score,
                    content=doc.content[:200] + "...",  # Truncate for display
                    source=doc.source,
                    metadata=doc.metadata
                ))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]  # Return top_k results


class MockQdrant:
    """Mock implementation of Qdrant for vector search"""
    
    def __init__(self):
        self.vectors = {}
        self.documents = {}
    
    def add_document(self, doc_id: str, content: str, embedding: np.ndarray = None):
        """Add document with vector representation"""
        # Create a simple embedding by taking hash of content if not provided
        if embedding is None:
            # This is a very simplified embedding for demo purposes
            embedding = np.random.rand(32)  # 32-dimensional vector
            # Make embedding deterministic based on content hash
            content_hash = hash(content) % 1000000
            np.random.seed(content_hash)
            embedding = np.random.rand(32)
        
        self.vectors[doc_id] = embedding
        self.documents[doc_id] = content
        print(f"Added document to Qdrant: {doc_id}")
    
    def search_vectors(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents using vector similarity"""
        # Create embedding for query
        query_hash = hash(query) % 1000000
        np.random.seed(query_hash)
        query_embedding = np.random.rand(32)
        
        similarities = []
        for doc_id, doc_embedding in self.vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Get corresponding document
            content = self.documents[doc_id]
            
            similarities.append(SearchResult(
                id=doc_id,
                score=similarity,
                content=content[:200] + "...",
                source="vector_search",
                metadata={"similarity": similarity}
            ))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x.score, reverse=True)
        return similarities[:top_k]


class MockNeo4j:
    """Mock implementation of Neo4j for graph traversal"""
    
    def __init__(self):
        self.graph_structure = {}
        self.node_properties = {}
    
    def add_node(self, node_id: str, label: str, properties: Dict[str, Any]):
        """Add a node to the graph"""
        self.node_properties[node_id] = {
            "label": label,
            "properties": properties
        }
        if label not in self.graph_structure:
            self.graph_structure[label] = []
        self.graph_structure[label].append(node_id)
        print(f"Added node to Neo4j: {node_id} ({label})")
    
    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        """Add a relationship to the graph"""
        if properties is None:
            properties = {}
        print(f"Added relationship to Neo4j: {source_id} -[{rel_type}]-> {target_id}")
    
    def graph_traversal(self, start_node_label: str, relationship_type: str = None, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Perform graph traversal to find related nodes"""
        print(f"Performing graph traversal from {start_node_label}")
        
        # In a real implementation, this would perform actual graph traversal
        # For demo purposes, we'll return simulated results
        results = []
        
        # Find all nodes of the starting label
        if start_node_label in self.graph_structure:
            for node_id in self.graph_structure[start_node_label][:2]:  # Just take first 2 for demo
                results.append({
                    "node_id": node_id,
                    "label": start_node_label,
                    "properties": self.node_properties[node_id]["properties"],
                    "context": f"Related to {start_node_label} via graph relationships",
                    "confidence": np.random.random()
                })
        
        return results


class GraphRAGService:
    def __init__(self):
        """
        Initialize the graph-RAG service
        """
        self.opensearch = MockOpenSearch()
        self.qdrant = MockQdrant() 
        self.neo4j = MockNeo4j()
        self.documents = []
        self.graph_initialized = False
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents into OpenSearch and Qdrant
        """
        print(f"Indexing {len(documents)} documents into OpenSearch and Qdrant...")
        
        for doc in documents:
            # Index in OpenSearch for BM25 search
            self.opensearch.index_document(doc)
            
            # Add to Qdrant for vector search
            self.qdrant.add_document(doc.id, doc.content)
            
            self.documents.append(doc)
    
    def build_graph_from_documents(self):
        """
        Build graph structure from indexed documents
        """
        print("Building graph from indexed documents...")
        
        # In a real implementation, this would extract entities and relationships
        # from documents and create corresponding graph nodes and edges
        # For demo purposes, creating some sample entities
        
        sample_nodes = [
            ("condition_1", "Condition", {"name": "diabetes mellitus", "code": "E11"}),
            ("intervention_1", "Intervention", {"name": "metformin", "type": "drug"}),
            ("outcome_1", "Outcome", {"name": "HbA1c reduction", "type": "laboratory"}),
            ("trial_1", "Trial", {"name": "Sample Clinical Trial", "nct_id": "NCT00000000"})
        ]
        
        sample_edges = [
            ("trial_1", "condition_1", "HAS_CONDITION", {}),
            ("trial_1", "intervention_1", "HAS_INTERVENTION", {}),
            ("trial_1", "outcome_1", "HAS_OUTCOME", {})
        ]
        
        for node_id, label, properties in sample_nodes:
            self.neo4j.add_node(node_id, label, properties)
        
        for source_id, target_id, rel_type, properties in sample_edges:
            self.neo4j.add_relationship(source_id, target_id, rel_type, properties)
        
        self.graph_initialized = True
        print("Graph building completed")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and entities
        """
        print(f"Analyzing query: '{query}'")
        
        # Simple entity extraction for demonstration
        entities = []
        if "diabetes" in query.lower():
            entities.append({"text": "diabetes", "type": "condition", "confidence": 0.9})
        if "metformin" in query.lower():
            entities.append({"text": "metformin", "type": "intervention", "confidence": 0.85})
        if "trial" in query.lower():
            entities.append({"text": "trial", "type": "study_type", "confidence": 0.7})
        
        analysis = {
            "original_query": query,
            "detected_entities": entities,
            "query_type": "evidence_lookup"  # This would be determined by more complex NLP
        }
        
        return analysis
    
    def run_bm25_search(self, query: str) -> List[SearchResult]:
        """
        Run BM25 search over indexed documents
        """
        print(f"Running BM25 search for query: '{query}'")
        return self.opensearch.bm25_search(query)
    
    def run_vector_search(self, query: str) -> List[SearchResult]:
        """
        Run vector/semantic search over indexed documents
        """
        print(f"Running vector search for query: '{query}'")
        return self.qdrant.search_vectors(query)
    
    def run_graph_traversal(self, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run graph traversal based on query analysis
        """
        if not self.graph_initialized:
            print("Graph not initialized, skipping graph traversal")
            return []
        
        print(f"Running graph traversal for query analysis")
        
        # Determine starting point for traversal based on detected entities
        start_label = "Condition"  # Default
        if query_analysis["detected_entities"]:
            for entity in query_analysis["detected_entities"]:
                if entity["type"] in ["condition", "intervention", "outcome", "trial"]:
                    start_label = entity["type"].capitalize()
                    break
        
        return self.neo4j.graph_traversal(start_label)
    
    def merge_rerank_results(self, 
                           bm25_results: List[SearchResult], 
                           vector_results: List[SearchResult],
                           graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and rerank results from different sources
        """
        print(f"Merging {len(bm25_results)} BM25, {len(vector_results)} vector, and {len(graph_results)} graph results")
        
        merged_results = []
        
        # Add BM25 results
        for result in bm25_results:
            merged_results.append({
                "id": result.id,
                "content": result.content,
                "source": f"BM25 from {result.source}",
                "score": result.score,
                "type": "document"
            })
        
        # Add vector results
        for result in vector_results:
            merged_results.append({
                "id": result.id,
                "content": result.content,
                "source": f"Vector search from {result.source}",
                "score": result.score,
                "type": "document"
            })
        
        # Add graph results
        for result in graph_results:
            merged_results.append({
                "id": result["node_id"],
                "content": f"Graph entity: {result['label']} - {result['properties']}",
                "source": f"Graph traversal ({result['label']})",
                "score": result["confidence"],
                "type": "graph"
            })
        
        # Simple reranking by score
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        return merged_results[:10]  # Return top 10 results
    
    def attach_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attach citations and context to results
        """
        print(f"Attaching citations to {len(results)} results")
        
        for result in results:
            # Add citation information
            result["citations"] = [
                {
                    "source": result["source"],
                    "confidence": result["score"],
                    "relevance": "high" if result["score"] > 0.8 else "medium"
                }
            ]
            
            # Add graph context if it's a graph result
            if result["type"] == "graph":
                result["graph_context"] = {
                    "node_label": result["content"].split(" - ")[0].split(": ")[1] if " - " in result["content"] else "Unknown",
                    "related_nodes": ["sample_related_node_1", "sample_related_node_2"]
                }
        
        return results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline
        """
        print(f"\nProcessing query: '{query}'")
        
        # Analyze query
        query_analysis = self.analyze_query(query)
        
        # Run different search methods
        bm25_results = self.run_bm25_search(query)
        vector_results = self.run_vector_search(query)
        graph_results = self.run_graph_traversal(query_analysis)
        
        # Merge and rerank results
        merged_results = self.merge_rerank_results(bm25_results, vector_results, graph_results)
        
        # Attach citations and context
        final_results = self.attach_citations(merged_results)
        
        # Format final response
        response = {
            "query": query,
            "query_analysis": query_analysis,
            "results": final_results,
            "total_results": len(final_results),
            "processing_time": "0.5s"  # This would be calculated in a real implementation
        }
        
        return response


async def main():
    """
    Main function to run the graph-RAG service
    """
    print("Starting Graph-RAG Service...")
    
    service = GraphRAGService()
    
    # Create sample documents to index
    sample_documents = [
        Document(
            id="doc_1",
            title="Metformin for Type 2 Diabetes: A Systematic Review",
            content="Metformin remains the first-line therapy for type 2 diabetes mellitus. This systematic review analyzes 47 randomized controlled trials showing significant HbA1c reduction of 1.0-2.0% compared to placebo. The drug is associated with weight loss and reduced cardiovascular events.",
            source="PubMed",
            metadata={"journal": "Diabetes Care", "year": 2023, "pmid": "12345678"}
        ),
        Document(
            id="doc_2", 
            title="Clinical Trial of Metformin vs Insulin in Pregnancy",
            content="The randomized controlled trial compared metformin to insulin in 100 pregnant women with gestational diabetes. Results showed non-inferiority of metformin for glycemic control with fewer side effects. Primary outcome was neonatal hypoglycemia rate.",
            source="ClinicalTrials.gov",
            metadata={"nct_id": "NCT12345678", "phase": "Phase 3", "status": "Completed"}
        ),
        Document(
            id="doc_3",
            title="ACE Inhibitors in Diabetic Nephropathy",
            content="Meta-analysis of 23 studies examining ACE inhibitors for diabetic nephropathy prevention. Significant reduction in albuminuria and progression to end-stage renal disease. Number needed to treat was 15 over 5 years.",
            source="PubMed", 
            metadata={"journal": "NEJM", "year": 2022, "pmid": "87654321"}
        )
    ]
    
    # Index documents
    service.index_documents(sample_documents)
    
    # Build graph from documents
    service.build_graph_from_documents()
    
    # Process sample queries
    queries = [
        "metformin for diabetes",
        "clinical trials on diabetes treatments",
        "ACE inhibitors for diabetic complications"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        result = service.process_query(query)
        
        print(f"\nTop 3 results:")
        for i, res in enumerate(result["results"][:3], 1):
            print(f"  {i}. {res['content'][:100]}...")
            print(f"     Source: {res['source']}, Score: {res['score']:.3f}")
    
    print("\nGraph-RAG Service completed successfully")


if __name__ == "__main__":
    asyncio.run(main())