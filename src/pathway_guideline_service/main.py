"""
Main entry point for the Pathway & Guideline Service
"""
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO


@dataclass
class GuidelinePathway:
    """Represents a clinical guideline pathway"""
    id: str
    name: str
    condition: str
    nodes: List[Dict[str, Any]]  # Pathway steps
    edges: List[Dict[str, Any]]  # Connections between steps
    version: str
    last_updated: str
    metadata: Dict[str, Any]


@dataclass
class ObservedPathway:
    """Represents an observed care pathway from patient data"""
    patient_id: str
    condition: str
    steps: List[Dict[str, Any]]  # Observed care steps
    timestamps: List[str]
    outcomes: List[Dict[str, Any]]
    adherence_score: float


class PathwayGuidelineService:
    def __init__(self):
        """
        Initialize the pathway and guideline service
        """
        self.guidelines = {}
        self.observed_pathways = {}
        self.pathway_graphs = {}
    
    def represent_guideline_as_pathway(self, guideline_data: Dict[str, Any]) -> GuidelinePathway:
        """
        Represent a guideline as a machine-readable pathway
        """
        print(f"Representing guideline '{guideline_data['name']}' as pathway")
        
        # Create pathway nodes (steps in the guideline)
        nodes = []
        for i, step in enumerate(guideline_data.get('steps', [])):
            nodes.append({
                'id': f"step_{i}",
                'name': step['name'],
                'description': step.get('description', ''),
                'type': step.get('type', 'intervention'),  # intervention, test, assessment
                'recommended': step.get('recommended', True),
                'timing': step.get('timing', 'immediate'),  # immediate, delayed, conditional
                'evidence_level': step.get('evidence_level', 'unknown')
            })
        
        # Create pathway edges (relationships between steps)
        edges = []
        for i in range(len(nodes) - 1):
            edges.append({
                'source': nodes[i]['id'],
                'target': nodes[i+1]['id'],
                'type': 'follows',
                'condition': 'default'
            })
        
        # Add decision points if present
        for i, step in enumerate(guideline_data.get('decision_points', [])):
            decision_id = f"decision_{i}"
            nodes.append({
                'id': decision_id,
                'name': step['question'],
                'description': step['description'],
                'type': 'decision',
                'options': step.get('options', [])
            })
            
            # Add edges from previous step to decision
            if i > 0:
                edges.append({
                    'source': nodes[i-1]['id'],
                    'target': decision_id,
                    'type': 'leads_to',
                    'condition': 'needs_decision'
                })
        
        pathway = GuidelinePathway(
            id=guideline_data['id'],
            name=guideline_data['name'],
            condition=guideline_data['condition'],
            nodes=nodes,
            edges=edges,
            version=guideline_data.get('version', '1.0'),
            last_updated=guideline_data.get('last_updated', datetime.now().isoformat()),
            metadata=guideline_data.get('metadata', {})
        )
        
        self.guidelines[pathway.id] = pathway
        self._build_pathway_graph(pathway)
        
        print(f"Successfully created pathway with {len(nodes)} nodes and {len(edges)} edges")
        return pathway
    
    def _build_pathway_graph(self, pathway: GuidelinePathway):
        """Build a NetworkX graph representation of the pathway"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in pathway.nodes:
            G.add_node(
                node['id'], 
                name=node['name'], 
                type=node['type'],
                recommended=node.get('recommended', True)
            )
        
        # Add edges
        for edge in pathway.edges:
            G.add_edge(
                edge['source'], 
                edge['target'], 
                type=edge['type'],
                condition=edge.get('condition', 'default')
            )
        
        self.pathway_graphs[pathway.id] = G
    
    def compare_observed_to_recommended(self, 
                                      observed_pathway: ObservedPathway, 
                                      guideline_id: str) -> Dict[str, Any]:
        """
        Compare observed care pathway to recommended guideline pathway
        """
        print(f"Comparing observed pathway for patient {observed_pathway.patient_id} "
              f"to guideline {guideline_id}")
        
        if guideline_id not in self.guidelines:
            raise ValueError(f"Guideline {guideline_id} not found")
        
        guideline_pathway = self.guidelines[guideline_id]
        observed = observed_pathway.steps
        recommended = guideline_pathway.nodes
        
        # Calculate adherence metrics
        recommended_step_names = {node['name'] for node in recommended if node['type'] != 'decision'}
        observed_step_names = {step['name'] for step in observed}
        
        # Steps that should have been performed according to guideline
        required_steps = {node['name'] for node in recommended if node['recommended']}
        
        # Steps that were actually performed
        performed_steps = observed_step_names.intersection(recommended_step_names)
        
        # Calculate adherence score
        if required_steps:
            adherence_score = len(performed_steps.intersection(required_steps)) / len(required_steps)
        else:
            adherence_score = 1.0  # If no required steps, perfect adherence
        
        # Identify variances
        missing_steps = required_steps - performed_steps
        extra_steps = performed_steps - required_steps
        
        comparison_result = {
            "patient_id": observed_pathway.patient_id,
            "guideline_id": guideline_id,
            "condition": observed_pathway.condition,
            "adherence_score": adherence_score,
            "required_steps": list(required_steps),
            "performed_steps": list(performed_steps),
            "missing_steps": list(missing_steps),
            "extra_steps": list(extra_steps),
            "n_required": len(required_steps),
            "n_performed": len(performed_steps),
            "n_missing": len(missing_steps),
            "n_extra": len(extra_steps),
            "comparison_timestamp": datetime.now().isoformat()
        }
        
        print(f"Adherence score: {adherence_score:.3f}")
        print(f"Missing steps: {len(missing_steps)}, Extra steps: {len(extra_steps)}")
        
        return comparison_result
    
    def highlight_optimization_opportunities(self, 
                                           comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Highlight opportunities for pathway optimization
        """
        print("Highlighting optimization opportunities...")
        
        opportunities = []
        
        # Opportunity 1: Missing recommended steps
        if comparison_results["n_missing"] > 0:
            opportunities.append({
                "type": "missing_recommended",
                "description": f"Patient missed {comparison_results['n_missing']} recommended steps",
                "steps": comparison_results["missing_steps"],
                "priority": "high" if comparison_results["n_missing"] > 2 else "medium",
                "suggestion": "Implement protocols to ensure recommended steps are followed"
            })
        
        # Opportunity 2: Unnecessary steps
        if comparison_results["n_extra"] > 0:
            opportunities.append({
                "type": "unnecessary_steps",
                "description": f"Patient received {comparison_results['n_extra']} non-recommended steps",
                "steps": comparison_results["extra_steps"],
                "priority": "medium",
                "suggestion": "Review protocols to reduce unnecessary interventions"
            })
        
        # Opportunity 3: Timing issues (simplified)
        timing_variance = abs(comparison_results["n_performed"] - comparison_results["n_required"])
        if timing_variance > 2:
            opportunities.append({
                "type": "timing_optimization",
                "description": "Significant variance in care steps performed vs. recommended",
                "steps": [],
                "priority": "medium",
                "suggestion": "Review care coordination protocols"
            })
        
        print(f"Identified {len(opportunities)} optimization opportunities")
        return opportunities
    
    def generate_pathway_visualization(self, guideline_id: str) -> str:
        """
        Generate a visualization of the guideline pathway
        """
        if guideline_id not in self.pathway_graphs:
            return "Pathway not found"
        
        G = self.pathway_graphs[guideline_id]
        
        # Create a simple visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Separate nodes by type for different coloring
        intervention_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'intervention']
        decision_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'decision']
        test_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'test']
        
        # Draw nodes with different colors
        nx.draw_networkx_nodes(G, pos, nodelist=intervention_nodes, 
                              node_color='lightblue', node_shape='s', node_size=1000, label='Intervention')
        nx.draw_networkx_nodes(G, pos, nodelist=decision_nodes, 
                              node_color='orange', node_shape='d', node_size=800, label='Decision Point')
        nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, 
                              node_color='lightgreen', node_shape='o', node_size=800, label='Test')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'name')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Guideline Pathway: {self.guidelines[guideline_id].name}")
        plt.legend()
        plt.axis('off')
        
        # Save to a string buffer
        img_buffer = io.StringIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        
        return img_buffer.getvalue()


async def main():
    """
    Main function to run the pathway and guideline service
    """
    print("Starting Pathway & Guideline Service...")
    
    service = PathwayGuidelineService()
    
    # Define a sample guideline pathway
    sample_guideline = {
        "id": "dm2_management_2023",
        "name": "Type 2 Diabetes Management Guideline 2023",
        "condition": "type_2_diabetes",
        "version": "2023.1",
        "last_updated": "2023-01-15",
        "metadata": {
            "organization": "American Diabetes Association",
            "evidence_level": "A"
        },
        "steps": [
            {
                "name": "Initial Assessment",
                "description": "Comprehensive patient assessment",
                "type": "assessment",
                "recommended": True,
                "timing": "immediate"
            },
            {
                "name": "Lifestyle Modification",
                "description": "Diet and exercise counseling",
                "type": "intervention",
                "recommended": True,
                "timing": "immediate"
            },
            {
                "name": "Metformin Therapy",
                "description": "Start metformin unless contraindicated",
                "type": "intervention",
                "recommended": True,
                "timing": "immediate"
            },
            {
                "name": "HbA1c Monitoring",
                "description": "Check HbA1c every 3 months",
                "type": "test",
                "recommended": True,
                "timing": "every_3_months"
            },
            {
                "name": "Annual Eye Exam",
                "description": "Comprehensive eye examination",
                "type": "test",
                "recommended": True,
                "timing": "annual"
            }
        ],
        "decision_points": [
            {
                "question": "Is HbA1c >7% despite metformin?",
                "description": "Decision point for additional therapy",
                "options": [
                    {"name": "Yes", "next": "add_second_agent"},
                    {"name": "No", "next": "continue_current_therapy"}
                ]
            }
        ]
    }
    
    # Represent the guideline as a machine-readable pathway
    pathway = service.represent_guideline_as_pathway(sample_guideline)
    
    # Create a sample observed pathway (what actually happened to a patient)
    sample_observed = ObservedPathway(
        patient_id="pt_12345",
        condition="type_2_diabetes",
        steps=[
            {"name": "Initial Assessment", "timestamp": "2023-02-01"},
            {"name": "Lifestyle Modification", "timestamp": "2023-02-05"},
            {"name": "Metformin Therapy", "timestamp": "2023-02-10"},
            {"name": "HbA1c Monitoring", "timestamp": "2023-05-10"},
            {"name": "Unnecessary Lab Test", "timestamp": "2023-05-11"},  # Not in guideline
        ],
        timestamps=["2023-02-01", "2023-02-05", "2023-02-10", "2023-05-10", "2023-05-11"],
        outcomes=[{"outcome": "HbA1c_7.2", "timestamp": "2023-05-10"}],
        adherence_score=0.8  # 80% adherence
    )
    
    # Compare observed to recommended pathway
    comparison = service.compare_observed_to_recommended(sample_observed, "dm2_management_2023")
    
    # Highlight optimization opportunities
    opportunities = service.highlight_optimization_opportunities(comparison)
    
    # Print results
    print(f"\n{'='*70}")
    print("PATHWAY & GUIDELINE COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Patient: {comparison['patient_id']}")
    print(f"Condition: {comparison['condition']}")
    print(f"Adherence Score: {comparison['adherence_score']:.3f}")
    print(f"Required Steps: {comparison['n_required']}")
    print(f"Performed Steps: {comparison['n_performed']}")
    print(f"Missing Steps: {comparison['n_missing']}")
    print(f"Extra Steps: {comparison['n_extra']}")
    
    print(f"\nMissing Steps: {comparison['missing_steps']}")
    print(f"Extra Steps: {comparison['extra_steps']}")
    
    print(f"\nOPTIMIZATION OPPORTUNITIES:")
    for i, opp in enumerate(opportunities, 1):
        print(f"  {i}. {opp['type'].replace('_', ' ').title()}: {opp['description']}")
        print(f"     Priority: {opp['priority']}")
        print(f"     Suggestion: {opp['suggestion']}")
    
    if opportunities:
        print(f"\nBased on {len(opportunities)} opportunities identified, consider:")
        for opp in opportunities:
            if opp['priority'] in ['high', 'medium']:
                print(f"  - {opp['suggestion']}")
    
    print(f"{'='*70}")
    print("Pathway & Guideline Service completed successfully")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())