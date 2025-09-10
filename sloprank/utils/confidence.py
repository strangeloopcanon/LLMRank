"""
Confidence interval calculation for SlopRank rankings.
"""
import json
import bodo.pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

from ..config import logger

def compute_confidence_intervals(
        evaluations_path=None,
        output_path=None,
        iterations=500,
        confidence_level=0.95
    ):
    """
    Compute confidence intervals for model rankings using bootstrap resampling.
    
    Parameters:
    -----------
    evaluations_path : Path or str
        Path to the evaluations CSV file
    output_path : Path or str
        Path for the output JSON file
    iterations : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level (0.0-1.0)
    
    Returns:
    --------
    dict
        Confidence statistics
    """
    if evaluations_path is None:
        evaluations_path = Path("results/evaluations.csv")
    else:
        evaluations_path = Path(evaluations_path)
    
    if output_path is None:
        output_path = Path("results/confidence_stats.json")
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Computing confidence intervals using {iterations} bootstrap iterations...")
    
    # Load evaluations
    evals_df = pd.read_csv(evaluations_path)
    
    # Filter out failed evaluations
    evals_df = evals_df[evals_df["parse_failed"] == False]
    
    # Get unique models
    models = list(set(evals_df["judge_model"].unique()) | set(evals_df["rated_model"].unique()))
    
    # Store bootstrap results
    bootstrap_results = {model: [] for model in models}
    
    # Run bootstrap iterations
    for i in range(iterations):
        if i % 100 == 0:
            logger.info(f"Bootstrap iteration {i}/{iterations}...")
        
        # Resample evaluations with replacement
        sampled_evals = evals_df.sample(frac=1.0, replace=True)
        
        # Build graph from resampled data
        G = nx.DiGraph()
        G.add_nodes_from(models)
        
        for _, row in sampled_evals.iterrows():
            judge = row["judge_model"]
            rated = row["rated_model"]
            score = float(row["score"])
            
            if G.has_edge(judge, rated):
                G[judge][rated]["weight"] += score
            else:
                G.add_edge(judge, rated, weight=score)
        
        # Compute PageRank
        if len(G.edges) > 0:
            scores = nx.pagerank(G, weight="weight")
            
            # Store scores
            for model, score in scores.items():
                bootstrap_results[model].append(score)
    
    # Calculate confidence intervals (95%)
    confidence_stats = {}
    alpha = 1.0 - confidence_level
    
    for model in models:
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
            "mean": float(np.mean(sorted_scores)),
            "lower_bound": float(sorted_scores[max(0, lower_idx)]),
            "upper_bound": float(sorted_scores[min(len(sorted_scores)-1, upper_idx)]),
            "std_dev": float(np.std(sorted_scores))
        }
    
    # Test statistical significance
    significance_results = {}
    
    # Create sorted list of models by mean score
    models_by_score = sorted(
        [(model, stats["mean"]) for model, stats in confidence_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Compare each adjacent pair in the ranking
    for i in range(len(models_by_score) - 1):
        model1, _ = models_by_score[i]
        model2, _ = models_by_score[i + 1]
        
        # Determine if significant based on confidence intervals
        is_significant = (
            confidence_stats[model1]["lower_bound"] > confidence_stats[model2]["upper_bound"] or
            confidence_stats[model2]["lower_bound"] > confidence_stats[model1]["upper_bound"]
        )
        
        significance_results[f"{model1}_vs_{model2}"] = is_significant
    
    # Save results
    results = {
        "confidence_intervals": confidence_stats,
        "significance": significance_results,
        "metadata": {
            "iterations": iterations,
            "confidence_level": confidence_level
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n=== Confidence Intervals ===")
    for model, stats in sorted(confidence_stats.items(), key=lambda x: x[1]["mean"], reverse=True):
        logger.info(f"{model}: {stats['mean']:.6f} [{stats['lower_bound']:.6f}, {stats['upper_bound']:.6f}]")
    
    logger.info("\n=== Statistical Significance ===")
    for pair, is_significant in significance_results.items():
        significance_str = "Significant" if is_significant else "Not significant"
        logger.info(f"{pair}: {significance_str}")
    
    logger.info(f"Confidence statistics saved to {output_path}")
    
    return confidence_stats


if __name__ == "__main__":
    # Run as a standalone script
    compute_confidence_intervals()