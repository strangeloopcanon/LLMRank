"""
Graph visualization for SlopRank endorsement networks.
"""
import json
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..config import logger


def generate_visualization(
        rankings_path=None, 
        evaluations_path=None, 
        output_dir=None,
        vis_config=None
    ):
    """
    Generate visualizations of the SlopRank endorsement graph.
    
    Parameters:
    -----------
    rankings_path : Path or str
        Path to the rankings.json file
    evaluations_path : Path or str
        Path to the evaluations.csv file
    output_dir : Path or str
        Directory to save visualizations
    vis_config : VisualizationConfig
        Configuration for visualizations
    
    Returns:
    --------
    tuple
        Paths to generated visualization files
    """
    if rankings_path is None:
        rankings_path = Path("results/rankings.json")
    else:
        rankings_path = Path(rankings_path)
    
    if evaluations_path is None:
        evaluations_path = Path("results/evaluations.csv")
    else:
        evaluations_path = Path(evaluations_path)
    
    if output_dir is None:
        output_dir = Path("results/visualizations")
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rankings
    with open(rankings_path, 'r') as f:
        rankings_data = json.load(f)
    
    # Extract pagerank scores
    if isinstance(rankings_data['rankings'][0], list):
        # Old format with list of lists
        pagerank_scores = {model: score for model, score in rankings_data["rankings"]}
    else:
        # New format with list of dicts
        pagerank_scores = {item["model"]: item["score"] for item in rankings_data["rankings"]}
    
    # Load evaluations
    evals_df = pd.read_csv(evaluations_path)
    
    # Filter out failed evaluations
    evals_df = evals_df[evals_df["parse_failed"] == False]
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes from rankings
    for model, score in pagerank_scores.items():
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
    
    # Save graph in GML format
    gml_path = output_dir / "endorsement_graph.gml"
    nx.write_gml(G, gml_path)
    logger.info(f"Saved graph in GML format to {gml_path}")
    
    # Generate static visualization if matplotlib is available
    png_path = None
    if HAS_MATPLOTLIB:
        png_path = output_dir / "endorsement_graph.png"
        generate_static_visualization(G, pagerank_scores, png_path, vis_config)
        logger.info(f"Saved static visualization to {png_path}")
    
    # Generate interactive visualization if plotly is available
    html_path = None
    if HAS_PLOTLY and (vis_config is None or vis_config.interactive):
        html_path = output_dir / "endorsement_graph.html"
        generate_interactive_visualization(G, pagerank_scores, html_path, vis_config)
        logger.info(f"Saved interactive visualization to {html_path}")
    
    return gml_path, png_path, html_path


def generate_static_visualization(G, pagerank_scores, output_path, vis_config=None):
    """
    Generate a static visualization of the endorsement graph using matplotlib.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not found. Cannot generate static visualization.")
        return
    
    # Node size factor, edge width factor, color maps, etc.
    node_size_factor = 2000
    edge_width_factor = 2.0
    node_colormap = 'viridis'
    edge_colormap = 'plasma'
    
    if vis_config is not None:
        node_size_factor = vis_config.node_size_factor
        edge_width_factor = vis_config.edge_width_factor
        node_colormap = vis_config.node_colormap
        edge_colormap = vis_config.edge_colormap
    
    try:
        # Calculate position using spring layout
        layout_func = nx.spring_layout
        if vis_config is not None and hasattr(vis_config, 'layout'):
            if vis_config.layout == 'circular':
                layout_func = nx.circular_layout
            elif vis_config.layout == 'kamada_kawai':
                layout_func = nx.kamada_kawai_layout
            elif vis_config.layout == 'spectral':
                layout_func = nx.spectral_layout
        
        pos = layout_func(G, seed=42)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        node_sizes = [pagerank_scores.get(node, 0.01) * node_size_factor for node in G.nodes()]
        node_colors = [pagerank_scores.get(node, 0.0) for node in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.get_cmap(node_colormap),
            alpha=0.8
        )
        
        # Draw edges
        edge_widths = [G[u][v].get('normalized_weight', 0.1) * edge_width_factor for u, v in G.edges()]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.6,
            edge_color=range(len(G.edges())),
            edge_cmap=plt.cm.get_cmap(edge_colormap),
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>'
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating static visualization: {e}")


def generate_interactive_visualization(G, pagerank_scores, output_path, vis_config=None):
    """
    Generate an interactive visualization of the endorsement graph using Plotly.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not found. Cannot generate interactive visualization.")
        return
    
    # Node size factor, edge width factor, color maps, etc.
    node_size_factor = 2000
    edge_width_factor = 2.0
    node_colormap = 'Viridis'
    
    if vis_config is not None:
        node_size_factor = vis_config.node_size_factor
        edge_width_factor = vis_config.edge_width_factor
        node_colormap = vis_config.node_colormap
    
    try:
        # Calculate position using spring layout
        layout_func = nx.spring_layout
        if vis_config is not None and hasattr(vis_config, 'layout'):
            if vis_config.layout == 'circular':
                layout_func = nx.circular_layout
            elif vis_config.layout == 'kamada_kawai':
                layout_func = nx.kamada_kawai_layout
            elif vis_config.layout == 'spectral':
                layout_func = nx.spectral_layout
        
        pos = layout_func(G, seed=42)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            source, target = edge
            source_pos = pos[source]
            target_pos = pos[target]
            weight = G[source][target].get('weight', 1.0)
            
            # Calculate line transparency and width based on weight
            width = max(1, min(10, weight / 5))
            opacity = min(1.0, max(0.3, weight / 10.0))
            
            # Create edge line
            edge_trace = go.Scatter(
                x=[source_pos[0], target_pos[0]],
                y=[source_pos[1], target_pos[1]],
                line=dict(width=width, color=f'rgba(150, 150, 150, {opacity})'),
                hoverinfo='text',
                text=f"{source} → {target}<br>Weight: {weight:.2f}",
                mode='lines+markers',
                marker=dict(size=0),
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            # Create arrowhead
            # Simple approximation of arrow position (80% along the edge)
            arrow_x = source_pos[0] * 0.2 + target_pos[0] * 0.8
            arrow_y = source_pos[1] * 0.2 + target_pos[1] * 0.8
            
            arrow_trace = go.Scatter(
                x=[arrow_x],
                y=[arrow_y],
                mode='markers',
                marker=dict(
                    symbol='triangle-right',
                    size=10,
                    color=f'rgba(150, 150, 150, {opacity})',
                    angle=np.degrees(np.arctan2(
                        target_pos[1] - source_pos[1],
                        target_pos[0] - source_pos[0]
                    ))
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(arrow_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"{node}<br>PageRank: {pagerank_scores.get(node, 0):.4f}" for node in G.nodes()],
            marker=dict(
                showscale=True,
                colorscale=node_colormap,
                color=[pagerank_scores.get(node, 0) for node in G.nodes()],
                size=[pagerank_scores.get(node, 0.01) * node_size_factor / 10 for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text='PageRank Score',
                        side='right'
                    ),
                    xanchor='left'
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Interactive LLM Endorsement Graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                annotations=[
                    dict(
                        text="Node size = PageRank score<br>Edge width = Endorsement strength",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.01, y=-0.05
                    )
                ]
            )
        )
        
        # Save to HTML file
        fig.write_html(output_path)
        
    except Exception as e:
        logger.error(f"Error generating interactive visualization: {e}")


if __name__ == "__main__":
    # Run as a standalone script
    generate_visualization()