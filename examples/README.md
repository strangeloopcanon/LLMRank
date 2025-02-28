# SlopRank Example Scripts

This directory contains standalone scripts that demonstrate each of the advanced features of SlopRank. These scripts can be run individually after running the main SlopRank tool.

## Available Scripts

### 1. Graph Visualization (`generate_visualization.py`)

Creates visual representations of the model endorsement network:

```bash
python examples/generate_visualization.py
```

**Outputs:**
- Static PNG visualization: `results/visualizations/endorsement_graph.png`
- GraphML file: `results/visualizations/endorsement_graph.gml`

### 2. Confidence Intervals (`compute_confidence.py`)

Uses bootstrap resampling to estimate statistical reliability:

```bash
python examples/compute_confidence.py
```

**Outputs:**
- `results/confidence_stats.json` containing:
  - Confidence intervals for each model's PageRank score
  - Statistical significance tests between adjacent ranks

### 3. Prompt Categorization (`prompt_categorization.py`)

Automatically categorizes prompts and provides per-category rankings:

```bash
python examples/prompt_categorization.py
```

**Outputs:**
- Categorized version of your prompts file
- Per-category rankings in `results/category_rankings.json`
- CSV analysis in `results/category_analysis.csv`

### 4. Interactive Dashboard

#### Dashboard Generation (`generate_dashboard.py`)
Creates an HTML dashboard from all the results:

```bash
python examples/generate_dashboard.py
```

#### Dashboard Server (`dashboard.py`)
Starts a local server to view the dashboard:

```bash
python examples/dashboard.py
```

## Recommended Workflow

For the best experience, run the tools in this order:

1. Run SlopRank: `sloprank --prompts prompts.xlsx --output-dir results`
2. Generate visualizations: `python examples/generate_visualization.py`
3. Compute confidence intervals: `python examples/compute_confidence.py`
4. Analyze categories: `python examples/prompt_categorization.py`
5. Generate dashboard: `python examples/generate_dashboard.py`
6. View the dashboard: `python examples/dashboard.py`

## Integrated CLI Alternative

All these features are now integrated into the main `sloprank` CLI tool:

```bash
sloprank run --prompts prompts.xlsx --output-dir results --visualize --confidence --categories --dashboard
```

These standalone example scripts are provided for educational purposes and for users who want to use each feature independently.