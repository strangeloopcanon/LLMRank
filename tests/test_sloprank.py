"""
Simple test script for SlopRank
"""
import pandas as pd
import json
from pathlib import Path
from sloprank.config import EvalConfig, VisualizationConfig
from sloprank.collect import collect_responses, collect_raw_evaluations
from sloprank.parse import parse_evaluation_rows
from sloprank.rank import build_endorsement_graph, compute_pagerank, finalize_rankings

# Define simpler prompts
prompts = [
    "What is the capital of France?",
    "Name three primary colors",
]

# Create a simple prompts file
test_df = pd.DataFrame({"Questions": prompts})
test_df.to_excel("tiny_prompts.xlsx", index=False)

# Define a simple test configuration
config = EvalConfig(
    model_names=["deepseek-reasoner", "claude-3.7-sonnet", "chatgpt-4o"],
    evaluation_method=1,  # numeric
    use_subset_evaluation=False,  # All models evaluate each other
    evaluators_subset_size=2,  # This will be ignored since subset_evaluation is False
    output_dir=Path("test_results"),
    request_delay=0.0
)

# Create output directory
config.output_dir.mkdir(exist_ok=True)

# Create prompt pairs (prompt, answer_key)
prompt_pairs = [(prompt, "") for prompt in prompts]

# Collect responses
print(f"Collecting responses from {len(config.model_names)} models for {len(prompts)} prompts...")
responses_df = collect_responses(prompt_pairs, config)
responses_df.to_csv(config.output_dir / "responses.csv", index=False)
print(f"Saved responses to {config.output_dir}/responses.csv")

# Collect evaluations
print("Collecting evaluations...")
raw_evaluations_df = collect_raw_evaluations(responses_df, config)
raw_evaluations_df.to_csv(config.output_dir / "raw_evaluations.csv", index=False)
print(f"Saved raw evaluations to {config.output_dir}/raw_evaluations.csv")

# Parse evaluations
print("Parsing evaluations...")
evaluations_df = parse_evaluation_rows(raw_evaluations_df)
evaluations_df.to_csv(config.output_dir / "evaluations.csv", index=False)
print(f"Saved parsed evaluations to {config.output_dir}/evaluations.csv")

# Build graph and compute rankings
print("Building graph and computing rankings...")
G = build_endorsement_graph(evaluations_df)
pagerank_scores = compute_pagerank(G)
rankings = finalize_rankings(config, pagerank_scores)

# Save rankings to file
rankings_file = config.output_dir / "rankings.json"
with open(rankings_file, "w") as f:
    json.dump({"rankings": rankings, "metadata": {"evaluation_method": config.evaluation_method}}, f, indent=4)
print(f"Saved rankings to {rankings_file}")

print("Test completed successfully!")