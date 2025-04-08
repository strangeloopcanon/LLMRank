#!/usr/bin/env python3
"""
Development script for running SlopRank directly from source without installing.
This allows testing changes immediately without reinstalling the package.

Usage: python run_dev.py --prompts prompts.csv --models "gpt-4o,claude-3-5-haiku-latest"
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to sys.path so Python can find the sloprank module
sys.path.insert(0, str(Path(__file__).parent))

# Now import from local sloprank package
from sloprank.config import EvalConfig, logger
from sloprank.collect import collect_responses, collect_raw_evaluations
from sloprank.parse import parse_evaluation_rows
from sloprank.rank import build_endorsement_graph, compute_pagerank, finalize_rankings
from sloprank.utils.categorization import analyze_categories
from sloprank.utils.confidence import calculate_confidence
from sloprank.utils.visualization import visualize_graph
from sloprank.utils.dashboard import generate_dashboard

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run SlopRank directly from source")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts CSV file")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of models")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--confidence", action="store_true", help="Calculate confidence intervals")
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = EvalConfig(
        model_names=args.models.split(","),
        evaluation_method=1,  # numeric ratings
        use_subset_evaluation=True,
        evaluators_subset_size=2,  # Rate 2 other models
        output_dir=Path(args.output_dir)
    )
    
    # Enable visualization if requested
    if args.visualize:
        config.visualization.enabled = True
    
    # Enable confidence calculations if requested
    if args.confidence:
        config.confidence.enabled = True
    
    # Print configuration
    logger.info(f"Using config: {config}")
    
    # Read prompts
    prompts_df = pd.read_csv(args.prompts)
    prompt_pairs = [(row["prompt"], row.get("answer_key", "")) 
                    for _, row in prompts_df.iterrows()]
    
    # Run SlopRank pipeline
    responses_df = collect_responses(prompt_pairs, config)
    raw_evals_df = collect_raw_evaluations(responses_df, config)
    
    # Extract prompt categories if available
    if "category" in prompts_df.columns:
        # Create mapping from prompt to category
        prompt_cats = {row["prompt"]: row["category"] for _, row in prompts_df.iterrows()}
        config.prompt_categories = {}
        
        # Group prompts by category
        for prompt, category in prompt_cats.items():
            if category:
                if category not in config.prompt_categories:
                    config.prompt_categories[category] = []
                config.prompt_categories[category].append(prompt)
    
    # Continue with evaluation parsing
    evals_df = parse_evaluation_rows(raw_evals_df, config)
    
    # Build graph and compute rankings
    G = build_endorsement_graph(evals_df)
    rankings = compute_pagerank(G)
    finalize_rankings(rankings, config)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_graph(G, rankings, config)
    
    # Calculate confidence intervals if requested
    if args.confidence:
        calculate_confidence(evals_df, config)
    
    # Analyze by category if categories exist
    if config.prompt_categories:
        analyze_categories(evals_df, config)
    
    logger.info("SlopRank pipeline completed successfully")

if __name__ == "__main__":
    main()