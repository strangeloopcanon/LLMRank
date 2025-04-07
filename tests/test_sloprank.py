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

# Use existing tiny_prompts.csv file
prompts_file = Path(__file__).parent / "tiny_prompts.csv"
test_df = pd.read_csv(prompts_file)
prompts = test_df["Questions"].tolist()

# Define a simple test configuration
config = EvalConfig(
    model_names=["deepseek-chat", "claude-3.5-haiku", "gpt-4o"],
    evaluation_method=1,  # numeric
    use_subset_evaluation=False,  # All models evaluate each other
    evaluators_subset_size=2,  # This will be ignored since subset_evaluation is False
    output_dir=Path(__file__).parent / "test_results",
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
evaluations_df = parse_evaluation_rows(raw_evaluations_df, config)
evaluations_df.to_csv(config.output_dir / "evaluations.csv", index=False)
print(f"Saved parsed evaluations to {config.output_dir}/evaluations.csv")

# Build graph and compute rankings
print("Building graph and computing rankings...")
G = build_endorsement_graph(evaluations_df, config)
pagerank_scores = compute_pagerank(G)
rankings = finalize_rankings(pagerank_scores, config, G, evaluations_df)

# Save rankings to file
rankings_file = config.output_dir / "rankings.json"
with open(rankings_file, "w") as f:
    json.dump(rankings, f, indent=4)
print(f"Saved rankings to {rankings_file}")

print("Test completed successfully!")