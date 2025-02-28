#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def generate_visualization():
    # Create visualization directory if it doesn't exist
    vis_dir = Path("results/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rankings
    rankings_path = Path("results/rankings.json")
    with open(rankings_path, 'r') as f:
        rankings_data = json.load(f)
    
    # Load evaluations data
    evals_path = Path("results/evaluations.csv")
    evals_df = pd.read_csv(evals_path)
    
    # Filter out failed evaluations
    evals_df = evals_df[evals_df["parse_failed"] == False]
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes from rankings
    for model_entry in rankings_data["rankings"]:
        model = model_entry[0]
        score = model_entry[1]
        G.add_node(model, pagerank=score)
    
    # Add edges from evaluations
    for _, row in evals_df.iterrows():
        judge = row["judge_model"]
        rated = row["rated_model"]
        score = float(row["score"])
        
        if G.has_edge(judge, rated):
            G[judge][rated]["weight"] += score
        else:
            G.add_edge(judge, rated, weight=score)
    
    # Normalize edge weights for visualization
    max_weight = max([G[u][v]["weight"] for u, v in G.edges()])
    for u, v in G.edges():
        G[u][v]["normalized_weight"] = G[u][v]["weight"] / max_weight
    
    # Create visualizations
    
    # 1. Static graph visualization
    plt.figure(figsize=(12, 10))
    
    # Calculate position using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get pagerank scores
    pagerank_scores = {node: G.nodes[node].get('pagerank', 0.1) for node in G.nodes()}
    
    # Draw nodes
    node_sizes = [pagerank_scores[node] * 5000 for node in G.nodes()]
    node_colors = list(pagerank_scores.values())
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.8
    )
    
    # Draw edges
    edge_widths = [G[u][v].get('normalized_weight', 0.1) * 5 for u, v in G.edges()]
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold'
    )
    
    # Add title
    plt.title("LLM Endorsement Graph (Node size = PageRank score, Edge width = Endorsement strength)")
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(vis_dir / "endorsement_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Save graph in GML format
    nx.write_gml(G, vis_dir / "endorsement_graph.gml")
    
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    generate_visualization()