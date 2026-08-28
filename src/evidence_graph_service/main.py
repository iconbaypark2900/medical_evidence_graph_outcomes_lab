"""
Main entry point for the Evidence Graph Service
"""
import asyncio
import hashlib
import logging
import math
from typing import Any, Dict, List, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


def entity_id(label: str, name: str) -> str:
    """Stable identifier for a named entity.

    Uses a content hash rather than Python's built-in hash(), which is
    salted per process: the same condition received a different node id on
    every run, so an upsert created a duplicate node instead of merging
    into the existing one, and the graph accumulated a fresh disconnected
    copy of every entity each time the service started.
    """
    digest = hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()
    return f"{label}_{digest[:12]}"


@dataclass
class GraphNode:
    """Represents a node in the medical evidence graph"""
    id: str
    label: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge in the medical evidence graph"""
    id: str
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]


class EvidenceGraphService:
    def __init__(self):
        """
        Initialize the evidence graph service
        """
        self.nodes = {}
        self.edges = {}
        self.graph_db = None  # This would connect to Neo4j in a real implementation
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        print(f"Added node: {node.id} ({node.label})")
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.edges[edge.id] = edge
        print(f"Added edge: {edge.id} ({edge.source} -> {edge.target}, {edge.relationship})")
    
    def extract_entities_relations(self, evidence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract entities and relations from evidence data
        """
        nodes = []
        edges = []
        
        for item in evidence_data:
            # Extract entities from the evidence item
            source_id = item["id"]
            
            # Create a trial/study node
            study_node = GraphNode(
                id=source_id,
                label="Trial" if item["type"] == "clinical_trial" else "Publication",
                properties={
                    "source": item["source"],
                    "title": f"Sample {item['type']} from {item['source']}",
                    "timestamp": item["timestamp"]
                }
            )
            nodes.append(study_node)
            
            # Extract condition nodes
            for condition in item["entities"].get("conditions", []):
                condition_id = entity_id("condition", condition)
                condition_node = GraphNode(
                    id=condition_id,
                    label="Condition",
                    properties={"name": condition}
                )
                nodes.append(condition_node)
                
                # Create edge from study to condition
                edge = GraphEdge(
                    id=f"edge_{source_id}_{condition_id}",
                    source=source_id,
                    target=condition_id,
                    relationship="HAS_CONDITION",
                    properties={"evidence_level": "high"}
                )
                edges.append(edge)
            
            # Extract intervention nodes
            for intervention in item["entities"].get("interventions", []):
                intervention_id = entity_id("intervention", intervention)
                intervention_node = GraphNode(
                    id=intervention_id,
                    label="Intervention", 
                    properties={"name": intervention}
                )
                nodes.append(intervention_node)
                
                # Create edge from study to intervention
                edge = GraphEdge(
                    id=f"edge_{source_id}_{intervention_id}",
                    source=source_id,
                    target=intervention_id,
                    relationship="HAS_INTERVENTION",
                    properties={"evidence_level": "high"}
                )
                edges.append(edge)
            
            # Extract outcome nodes
            for outcome in item["entities"].get("outcomes", []):
                outcome_id = entity_id("outcome", outcome)
                outcome_node = GraphNode(
                    id=outcome_id,
                    label="Outcome",
                    properties={"name": outcome}
                )
                nodes.append(outcome_node)
                
                # Create edge from study to outcome
                edge = GraphEdge(
                    id=f"edge_{source_id}_{outcome_id}",
                    source=source_id,
                    target=outcome_id,
                    relationship="HAS_OUTCOME",
                    properties={"evidence_level": "high"}
                )
                edges.append(edge)
        
        return {"nodes": nodes, "edges": edges}
    
    def upsert_graph_nodes_edges(self, graph_elements: Dict[str, Any]):
        """
        Upsert (update or insert) nodes and edges into the graph database
        """
        print(f"Upserting {len(graph_elements['nodes'])} nodes and {len(graph_elements['edges'])} edges")
        
        # Add nodes
        for node in graph_elements["nodes"]:
            self.add_node(node)
        
        # Add edges
        for edge in graph_elements["edges"]:
            self.add_edge(edge)
    
    def _adjacency(self) -> Dict[str, Set[str]]:
        """Undirected neighbour map built from the edges actually present."""
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for edge in self.edges.values():
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        return adjacency

    def recompute_kge_features(self, model_name: str = "distmult",
                               dim: int = 64, epochs: int = 300,
                               triples=None, seed: int = 0):
        """Train knowledge graph embeddings and report whether to use them.

        This used to return `np.random.rand(128)` per node as "embeddings",
        then an honest NotImplementedError pointing at the work this now
        does.

        Returns the evaluation report whether or not the model is worth
        serving. A link predictor that loses to "suggest whatever usually
        appears with this relation" is complexity without benefit, and
        serving it anyway would be the random-confidence problem again in a
        more convincing costume -- so `beats_baselines` in the report is
        what decides, and a losing model is reported as losing.
        """
        from src.kge import build_and_evaluate

        if triples is None:
            triples = [
                (edge.source, edge.relationship, edge.target)
                for edge in self.edges.values()
            ]
        if not triples:
            raise ValueError(
                "The graph holds no edges; there is nothing to embed. Index a "
                "corpus first (python -m src.integration).")

        model, store, report = build_and_evaluate(
            triples, model_name=model_name, dim=dim, epochs=epochs, seed=seed)

        self.kge_model = model
        self.kge_store = store
        self.kge_report = report

        logger.info(
            f"{model_name}: MRR {report.evaluation.mrr:.4f}, "
            f"beats baselines {report.beats_baselines}")
        return report

    def kge_suggestions(self, head: str, relation: str, limit: int = 5):
        """Rank candidate tails with the trained embeddings.

        Refuses when no model has been trained, and when the trained model
        did not beat its baselines -- in which case the honest answer is to
        use `suggest_related_entities`, which is the predictor that
        actually won.
        """
        model = getattr(self, "kge_model", None)
        report = getattr(self, "kge_report", None)

        if report is None:
            raise RuntimeError(
                "No embeddings have been trained. Call recompute_kge_features.")

        # Arguments are validated before the serving gate. An unknown
        # entity is a caller error whether or not the model was good
        # enough to serve, and reporting it as "the model lost" would send
        # the reader to fix the wrong thing.
        store = self.kge_store
        if head not in store.entity_index:
            raise KeyError(
                f"{head!r} has no embedding: it was not in the training graph")
        if relation not in store.relation_index:
            raise KeyError(
                f"Unknown relation {relation!r}; known: {store.relations}")

        if model is None:
            raise RuntimeError(
                f"The trained model did not beat its baselines "
                f"(MRR {report.evaluation.mrr:.4f} against "
                f"{[f'{b.model} {b.mrr:.4f}' for b in report.baselines]}), so "
                f"its suggestions are not served. Use "
                f"suggest_related_entities.")

        from src.kge import kge_scorer

        scores = kge_scorer(model, store)(head, relation, store.entities)
        known = {edge.target for edge in self.edges.values()
                 if edge.source == head and edge.relationship == relation}

        ranked = sorted(zip(store.entities, scores), key=lambda pair: -pair[1])
        suggestions = []
        for entity, score in ranked:
            if entity == head or entity in known:
                continue
            suggestions.append({
                "target": entity,
                "score": float(score),
                "scoring_method": f"kge:{report.parameters['model']}",
                "relationship_type": relation,
                # The score is a model output, not a probability. Shipping
                # the evaluation with it is what makes it checkable.
                "model_mrr": round(report.evaluation.mrr, 4),
                "model_hits_at_10": round(report.evaluation.hits_at_10, 4),
            })
            if len(suggestions) >= limit:
                break
        return suggestions

    def suggest_related_entities(
        self,
        target_node_id: str,
        candidate_label: str = "Intervention",
        relationship_type: str = "TREATS",
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Suggest links by Adamic-Adar similarity over shared neighbours.

        A standard, untrained link-prediction baseline: two entities are
        related to the degree that they co-occur with the same studies,
        and a study that mentions few entities is stronger evidence of a
        connection than one that mentions many. The score is
        `sum(1 / log(degree(w)))` over shared neighbours w.

        The returned `score` is unbounded and deliberately not called a
        confidence — it is a ranking signal, not a calibrated probability.
        The previous version returned `np.random.random()` as "confidence"
        with the explanation "Similar embedding space to {target}", which
        attached a fabricated justification to a random number.

        `shared_neighbours` lists exactly which nodes produced the score,
        so every suggestion can be traced back to the evidence behind it.
        """
        if target_node_id not in self.nodes:
            raise KeyError(
                f"Node {target_node_id!r} is not in the graph "
                f"({len(self.nodes)} nodes known)")

        adjacency = self._adjacency()
        target_neighbours = adjacency.get(target_node_id, set())

        suggestions = []
        for node_id, node in self.nodes.items():
            if node_id == target_node_id or node.label != candidate_label:
                continue
            if node_id in target_neighbours:
                continue  # already linked; nothing to suggest

            shared = target_neighbours & adjacency.get(node_id, set())
            if not shared:
                continue

            # A shared neighbour of degree 1 cannot connect two nodes, and
            # log(1) = 0 would divide by zero.
            score = sum(
                1.0 / math.log(len(adjacency[w])) for w in shared
                if len(adjacency[w]) > 1)
            if score <= min_score:
                continue

            suggestions.append({
                "target_node_id": node_id,
                "name": node.properties.get("name", node_id),
                "score": score,
                "scoring_method": "adamic_adar",
                "relationship_type": relationship_type,
                "shared_neighbours": sorted(shared),
            })

        suggestions.sort(key=lambda s: s["score"], reverse=True)
        print(f"Found {len(suggestions)} suggestions for {target_node_id}")
        return suggestions


async def main():
    """
    Main function to run the evidence graph service
    """
    print("Starting Evidence Graph Service...")
    
    service = EvidenceGraphService()
    
    # Simulate evidence data that would come from the ingestion service
    sample_evidence_data = [
        {
            "id": "evidence_1",
            "source": "PubMed",
            "type": "publication", 
            "entities": {
                "conditions": ["diabetes mellitus", "hypertension"],
                "interventions": ["metformin", "ACE inhibitors"],
                "outcomes": ["HbA1c reduction", "cardiovascular events"],
                "populations": ["adults", "elderly"]
            },
            "timestamp": "2023-10-01T00:00:00Z"
        },
        {
            "id": "evidence_2", 
            "source": "ClinicalTrials.gov",
            "type": "clinical_trial",
            "entities": {
                "conditions": ["breast cancer"],
                "interventions": ["trastuzumab", "chemotherapy"],
                "outcomes": ["overall survival", "progression-free survival"],
                "populations": ["HER2-positive patients"]
            },
            "timestamp": "2023-10-01T00:00:00Z"
        }
    ]
    
    # Extract entities and relations
    graph_elements = service.extract_entities_relations(sample_evidence_data)
    
    # Upsert nodes and edges
    service.upsert_graph_nodes_edges(graph_elements)
    
    # Suggest links from the graph structure. KGE embeddings are not
    # available -- see recompute_kge_features for why that is now an error
    # rather than a list of random vectors.
    conditions = [n for n in service.nodes.values() if n.label == "Condition"]
    if conditions:
        target = conditions[0]
        print(f"\nSuggestions for condition {target.properties.get('name')!r}:")
        for suggestion in service.suggest_related_entities(target.id):
            print(f"  {suggestion['name']} "
                  f"(score {suggestion['score']:.3f}, "
                  f"via {len(suggestion['shared_neighbours'])} shared study/studies)")
    
    print("Evidence Graph Service completed successfully")


if __name__ == "__main__":
    asyncio.run(main())