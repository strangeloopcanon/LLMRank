"""
Smoke test that exercises the core pipeline without network calls.

It fabricates raw evaluations with valid JSON so parsing, graph
construction, PageRank, and finalization can run entirely offline.
"""
from pathlib import Path
import json as _json

import pandas as pd

from sloprank.config import EvalConfig, VisualizationConfig, ConfidenceConfig
from sloprank.parse import parse_evaluation_rows
from sloprank.rank import build_endorsement_graph, compute_pagerank, finalize_rankings
from sloprank import __version__ as PKG_VERSION


def _make_mock_raw_evaluations(models, prompts):
    """Create a DataFrame that mimics raw_evaluations.csv rows.

    Each judge rates the other two models for each prompt with JSON like
    {"Model_1": 9, "Model_2": 7} mapped consistently via model_mapping.
    """
    rows = []

    # Preferred ranking for determinism
    model_preference = {
        "gpt-5": 9.0,
        "claude-4-sonnet": 7.0,
        "gemini-2.5-pro": 5.0,
    }

    for prompt in prompts:
        for judge in models:
            others = [m for m in models if m != judge]
            # Keep stable order for mapping
            others_sorted = others  # already in order of models list passed in
            mapping = {m: f"Model_{i+1}" for i, m in enumerate(others_sorted)}
            # Build anon score dict using preferences
            anon_scores = {mapping[m]: float(model_preference.get(m, 6.0)) for m in others_sorted}
            raw_json = _json.dumps(anon_scores)

            rows.append({
                "prompt": prompt,
                "judge_model": judge,
                "raw_judgment": raw_json,
                "model_mapping": _json.dumps(mapping, sort_keys=True),
                "raw_judgment_token_count": len(raw_json.split())
            })

    return pd.DataFrame(rows)


def test_smoke_offline(tmp_path: Path = None):
    # Test config and output directory
    out_dir = tmp_path if tmp_path else Path(__file__).parent / "smoke_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["gpt-5", "claude-4-sonnet", "gemini-2.5-pro"]
    prompts = [
        "What is 2+2?",
        "Name a primary color.",
    ]

    # Build minimal config with heavy features disabled
    config = EvalConfig(
        model_names=models,
        evaluation_method=1,
        use_subset_evaluation=False,
        evaluators_subset_size=2,
        output_dir=out_dir,
        visualization=VisualizationConfig(enabled=False),
        confidence=ConfidenceConfig(enabled=False),
    )

    # Fabricate raw evaluations and parse them
    raw_eval_df = _make_mock_raw_evaluations(models, prompts)
    parsed_df = parse_evaluation_rows(raw_eval_df, config)

    # Ensure parsing yielded rows for all judged pairs
    assert not parsed_df.empty
    assert set(parsed_df.columns) >= {"prompt", "judge_model", "rated_model", "score", "parse_failed"}

    # Build graph and compute rankings
    G = build_endorsement_graph(parsed_df, config)
    pr = compute_pagerank(G)
    assert pr and all(m in pr for m in models)

    # Finalize (writes JSON file)
    finalize_rankings(pr, config, G=G, evaluations_df=parsed_df)

    rankings_path = out_dir / "rankings.json"
    assert rankings_path.exists()
    data = _json.loads(rankings_path.read_text())
    assert "rankings" in data and len(data["rankings"]) == len(models)
    assert data.get("metadata", {}).get("version") == PKG_VERSION


if __name__ == "__main__":
    # Run directly
    test_smoke_offline()
    print("Smoke test completed successfully.")

