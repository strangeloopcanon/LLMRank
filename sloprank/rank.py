import json
import networkx as nx
import pandas as pd
from datetime import datetime
from .config import logger, EvalConfig

def build_endorsement_graph(evals_df: pd.DataFrame, config: EvalConfig, skip_failed: bool=True) -> nx.DiGraph:
    """
    Build a directed graph from numeric evaluations: judge -> rated, weight=score.
    """
    if skip_failed:
        evals_df = evals_df[evals_df["parse_failed"] == False]
    
    G = nx.DiGraph()
    G.add_nodes_from(config.model_names)

    for _, row in evals_df.iterrows():
        judge = row["judge_model"]
        rated = row["rated_model"]
        score = float(row["score"])
        if G.has_edge(judge, rated):
            G[judge][rated]["weight"] += score
        else:
            G.add_edge(judge, rated, weight=score)

    return G

def compute_pagerank(G: nx.DiGraph):
    if len(G.edges) == 0:
        logger.warning("No edges in the endorsement graph—cannot compute PageRank.")
        return {}
    return nx.pagerank(G, weight="weight")

def finalize_rankings(rankings: dict, config: EvalConfig):
    # Return a sorted list
    ranked_items = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    logger.info("=== PageRank Rankings ===")
    for model, score in ranked_items:
        logger.info(f"{model}\t{score:.6f}")

    results = {
        "rankings": [{"model": m, "score": s} for m, s in ranked_items],
        "metadata": {
            "evaluation_method": config.evaluation_method,
            "timestamp": datetime.now().isoformat()
        }
    }

    outfile = config.output_dir / "rankings.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved final rankings to {outfile}")
