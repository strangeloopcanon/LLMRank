# SlopRank New Features Guide

This guide explains the new features added to SlopRank and how to use them.

## 1. Graph Visualization

The graph visualization feature creates visual representations of the model endorsement network:

```bash
# Generate graph visualization from existing evaluation data
python generate_visualization.py
```

This will create:
- A static PNG visualization in `results/visualizations/endorsement_graph.png`
- A GraphML file in `results/visualizations/endorsement_graph.gml`

## 2. Confidence Intervals and Statistical Significance

The confidence intervals feature uses bootstrap resampling to estimate statistical reliability:

```bash
# Compute confidence intervals for model rankings
python compute_confidence.py
```

This will create `results/confidence_stats.json` containing:
- Confidence intervals for each model's PageRank score
- Statistical significance tests between adjacent ranks

## 3. Prompt Categorization and Analysis

This feature automatically categorizes prompts and provides per-category rankings:

```bash
# Categorize prompts and analyze evaluations by category
python prompt_categorization.py
```

This will:
- Create a categorized version of your prompts file
- Generate per-category rankings in `results/category_rankings.json`
- Create a CSV analysis in `results/category_analysis.csv`

## 4. Interactive Dashboard

The dashboard provides a simple web interface to explore all the results:

```bash
# Generate dashboard HTML file
python generate_dashboard.py

# Start a local web server with the dashboard
python dashboard.py
```

The dashboard shows:
- Overall model rankings
- Confidence intervals (if available)
- Category-specific rankings (if available)
- Graph visualization (if available)

## Using All Features Together

For the best experience, run the tools in this order:

1. Run SlopRank: `sloprank --prompts prompts.xlsx --output-dir results`
2. Generate visualizations: `python generate_visualization.py`
3. Compute confidence intervals: `python compute_confidence.py`
4. Analyze categories: `python prompt_categorization.py`
5. Generate dashboard: `python generate_dashboard.py`
6. View the dashboard by opening `results/dashboard.html` in a browser