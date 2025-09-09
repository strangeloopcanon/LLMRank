import json
import random
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from .config import logger, EvalConfig
from . import __version__

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("Matplotlib not found. Graph visualization will be limited.")
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    logger.warning("Plotly not found. Interactive visualizations will be disabled.")
    HAS_PLOTLY = False


def build_endorsement_graph(evals_df: pd.DataFrame, config: EvalConfig, skip_failed: bool=True) -> nx.DiGraph:
    """
    Build a directed graph from numeric evaluations: judge -> rated, weight=score.
    """
    if skip_failed:
        evals_df = evals_df[evals_df["parse_failed"] == False]
    
    G = nx.DiGraph()
    G.add_nodes_from(config.model_names)

    # Add node metadata
    for model in config.model_names:
        G.nodes[model]["name"] = model
        G.nodes[model]["id"] = model
    
    # Process by category if categories exist
    if config.prompt_categories:
        # Initialize category scores in node metadata
        for model in config.model_names:
            for category in config.prompt_categories.keys():
                G.nodes[model][f"score_{category}"] = 0.0
                G.nodes[model][f"count_{category}"] = 0
    
    # Add edges with weights
    for _, row in evals_df.iterrows():
        judge = row["judge_model"]
        rated = row["rated_model"]
        score = float(row["score"])
        prompt = row["prompt"]
        
        # Check which category this prompt belongs to
        category = None
        if config.prompt_categories:
            for cat_name, prompts in config.prompt_categories.items():
                if prompt in prompts:
                    category = cat_name
                    break
        
        # Update edge weights
        if G.has_edge(judge, rated):
            G[judge][rated]["weight"] += score
            # Update category data if applicable
            if category:
                if f"weight_{category}" in G[judge][rated]:
                    G[judge][rated][f"weight_{category}"] += score
                    G[judge][rated][f"count_{category}"] += 1
                else:
                    G[judge][rated][f"weight_{category}"] = score
                    G[judge][rated][f"count_{category}"] = 1
        else:
            G.add_edge(judge, rated, weight=score)
            # Add category data if applicable
            if category:
                G[judge][rated][f"weight_{category}"] = score
                G[judge][rated][f"count_{category}"] = 1

    return G

def compute_pagerank(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute PageRank scores for the graph.
    """
    if len(G.edges) == 0:
        logger.warning("No edges in the endorsement graph—cannot compute PageRank.")
        return {}
    return nx.pagerank(G, weight="weight")

def compute_categorical_pageranks(G: nx.DiGraph, categories: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Compute PageRank scores for each category.
    """
    category_rankings = {}
    
    for category in categories.keys():
        # Create a subgraph with only the edges from this category
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        
        for u, v, data in G.edges(data=True):
            if f"weight_{category}" in data:
                H.add_edge(u, v, weight=data[f"weight_{category}"])
        
        # Compute PageRank on this subgraph
        if len(H.edges) > 0:
            category_rankings[category] = nx.pagerank(H, weight="weight")
        else:
            logger.warning(f"No edges for category {category}—skipping PageRank.")
            category_rankings[category] = {}
    
    return category_rankings

def compute_confidence_intervals(
    evaluations_df: pd.DataFrame, 
    config: EvalConfig,
    iterations: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Compute confidence intervals using bootstrap resampling.
    """
    if not config.confidence.enabled:
        return {}
    
    logger.info(f"Computing confidence intervals using {iterations} bootstrap iterations...")
    
    # Store results
    bootstrap_results = {model: [] for model in config.model_names}
    
    # Run bootstrap iterations
    for i in range(iterations):
        if i % 100 == 0:
            logger.info(f"Bootstrap iteration {i}/{iterations}...")
        
        # Resample evaluations with replacement
        sampled_evals = evaluations_df.sample(frac=1.0, replace=True)
        
        # Build graph from resampled data
        G = build_endorsement_graph(sampled_evals, config)
        
        # Compute PageRank
        scores = compute_pagerank(G)
        
        # Store scores
        for model, score in scores.items():
            bootstrap_results[model].append(score)
    
    # Calculate confidence intervals
    confidence_stats = {}
    alpha = 1.0 - config.confidence.confidence_level
    
    for model in config.model_names:
        if not bootstrap_results[model]:
            confidence_stats[model] = {
                "mean": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "std_dev": 0.0
            }
            continue
            
        sorted_scores = sorted(bootstrap_results[model])
        lower_idx = int(alpha/2 * len(sorted_scores))
        upper_idx = int((1-alpha/2) * len(sorted_scores))
        
        confidence_stats[model] = {
            "mean": np.mean(sorted_scores),
            "lower_bound": sorted_scores[lower_idx],
            "upper_bound": sorted_scores[upper_idx],
            "std_dev": np.std(sorted_scores)
        }
    
    return confidence_stats

def test_statistical_significance(
    confidence_stats: Dict[str, Dict[str, float]],
    config: EvalConfig
) -> Dict[Tuple[str, str], bool]:
    """
    Test whether differences between model rankings are statistically significant.
    """
    significance_results = {}
    
    # Create sorted list of models by mean score
    models_by_score = sorted(
        [(model, stats["mean"]) for model, stats in confidence_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Compare each adjacent pair in the ranking
    for i in range(len(models_by_score) - 1):
        model1, score1 = models_by_score[i]
        model2, score2 = models_by_score[i + 1]
        
        # Calculate t-statistic-like value
        mean_diff = confidence_stats[model1]["mean"] - confidence_stats[model2]["mean"]
        pooled_std = np.sqrt(
            confidence_stats[model1]["std_dev"]**2 + 
            confidence_stats[model2]["std_dev"]**2
        )
        
        # Determine if significant based on confidence intervals
        is_significant = (
            confidence_stats[model1]["lower_bound"] > confidence_stats[model2]["upper_bound"] or
            confidence_stats[model2]["lower_bound"] > confidence_stats[model1]["upper_bound"]
        )
        
        significance_results[(model1, model2)] = is_significant
    
    return significance_results

def visualize_graph(G: nx.DiGraph, config: EvalConfig, pagerank_scores: Dict[str, float]) -> None:
    """
    Create and save visualizations of the endorsement graph.
    """
    if not config.visualization.enabled:
        return
    
    logger.info("Generating graph visualizations...")
    
    # Create visualization directory
    vis_dir = config.output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Save graph in GML format if requested
    if "gml" in config.visualization.save_formats:
        gml_path = vis_dir / "endorsement_graph.gml"
        nx.write_gml(G, gml_path)
        logger.info(f"Saved graph in GML format to {gml_path}")
    
    # Save graph in GraphML format if requested
    if "graphml" in config.visualization.save_formats:
        graphml_path = vis_dir / "endorsement_graph.graphml"
        nx.write_graphml(G, graphml_path)
        logger.info(f"Saved graph in GraphML format to {graphml_path}")
    
    # Generate static visualization with matplotlib
    if HAS_MATPLOTLIB and "png" in config.visualization.save_formats:
        try:
            # Determine layout
            layout_func = getattr(nx, f"{config.visualization.layout}_layout", nx.spring_layout)
            pos = layout_func(G)
            
            # Normalize node sizes based on PageRank scores
            node_sizes = [pagerank_scores.get(node, 0.01) * config.visualization.node_size_factor 
                        for node in G.nodes()]
            
            # Normalize edge widths based on weights
            edge_widths = [G[u][v].get('weight', 1.0) * config.visualization.edge_width_factor / 10.0 
                         for u, v in G.edges()]
            
            # Create colormap for nodes
            node_colors = [pagerank_scores.get(node, 0.0) for node in G.nodes()]
            
            # Create the figure
            plt.figure(figsize=(12, 10))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                cmap=plt.cm.get_cmap(config.visualization.node_colormap),
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.5,
                edge_color=range(len(G.edges())),
                edge_cmap=plt.cm.get_cmap(config.visualization.edge_colormap),
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
            
            # Add a title
            plt.title("LLM Endorsement Graph (Node size = PageRank score, Edge width = Endorsement strength)")
            plt.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(vis_dir / "endorsement_graph.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved static visualization to {vis_dir / 'endorsement_graph.png'}")
            
        except Exception as e:
            logger.error(f"Error generating static visualization: {e}")
    
    # Generate interactive visualization with Plotly
    if HAS_PLOTLY and "html" in config.visualization.save_formats and config.visualization.interactive:
        try:
            # Determine layout
            layout_func = getattr(nx, f"{config.visualization.layout}_layout", nx.spring_layout)
            pos = layout_func(G)
            
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
                    colorscale=config.visualization.node_colormap,
                    color=[pagerank_scores.get(node, 0) for node in G.nodes()],
                    size=[pagerank_scores.get(node, 0.01) * config.visualization.node_size_factor / 10 
                          for node in G.nodes()],
                    colorbar=dict(
                        thickness=15,
                        title='PageRank Score',
                        xanchor='left',
                        titleside='right'
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
            html_path = vis_dir / "endorsement_graph.html"
            fig.write_html(html_path)
            logger.info(f"Saved interactive visualization to {html_path}")
            
        except Exception as e:
            logger.error(f"Error generating interactive visualization: {e}")

def finalize_rankings(
    rankings: dict, 
    config: EvalConfig, 
    G: Optional[nx.DiGraph] = None,
    evaluations_df: Optional[pd.DataFrame] = None,
    category_rankings: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Process final rankings, save results, and generate visualizations.
    """
    # Sort the rankings
    ranked_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    
    # Display rankings in console
    logger.info("=== PageRank Rankings ===")
    for model, score in ranked_items:
        logger.info(f"{model}\t{score:.6f}")
    
    # Prepare results object
    results = {
        "rankings": [{"model": m, "score": s} for m, s in ranked_items],
        "metadata": {
            "evaluation_method": config.evaluation_method,
            "timestamp": datetime.now().isoformat(),
            "version": __version__
        }
    }
    
    # Add category rankings if available
    if category_rankings:
        results["category_rankings"] = {}
        for category, scores in category_rankings.items():
            sorted_category_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results["category_rankings"][category] = [
                {"model": m, "score": s} for m, s in sorted_category_items
            ]
            
            # Display category rankings in console
            logger.info(f"\n=== Category: {category} Rankings ===")
            for model, score in sorted_category_items:
                logger.info(f"{model}\t{score:.6f}")
    
    # Add confidence intervals if enabled and provided
    if config.confidence.enabled and evaluations_df is not None:
        confidence_stats = compute_confidence_intervals(
            evaluations_df, 
            config,
            iterations=config.confidence.bootstrap_iterations
        )
        
        # Add confidence data to results
        results["confidence_intervals"] = {
            model: {
                "mean": stats["mean"],
                "lower_bound": stats["lower_bound"],
                "upper_bound": stats["upper_bound"],
                "std_dev": stats["std_dev"],
            }
            for model, stats in confidence_stats.items()
        }
        
        # Test statistical significance between adjacent ranks
        significance_results = test_statistical_significance(confidence_stats, config)
        results["significance"] = {
            f"{model1}_vs_{model2}": is_significant
            for (model1, model2), is_significant in significance_results.items()
        }
        
        # Display confidence intervals in console
        logger.info("\n=== Confidence Intervals (95%) ===")
        for model, stats in confidence_stats.items():
            logger.info(f"{model}: {stats['mean']:.6f} [{stats['lower_bound']:.6f}, {stats['upper_bound']:.6f}]")
        
        # Display significance information
        logger.info("\n=== Statistical Significance ===")
        for (model1, model2), is_significant in significance_results.items():
            significance_str = "Significant" if is_significant else "Not significant"
            logger.info(f"{model1} vs {model2}: {significance_str}")
    
    # Save the results to JSON
    outfile = config.output_dir / "rankings.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved final rankings to {outfile}")
    
    # Generate visualizations if graph is provided
    if G is not None and config.visualization.enabled:
        visualize_graph(G, config, rankings)
