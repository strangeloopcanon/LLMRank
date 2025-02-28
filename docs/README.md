# SlopRank Dashboard

This directory contains the interactive dashboard for SlopRank LLM evaluation framework.

## Files

- `index.html` - The main dashboard file
- `visualizations/` - Directory containing graph visualizations and images

## How to Use

1. Open `index.html` in any modern web browser
2. Explore the model rankings, category performance, and graph visualizations

## Hosting on GitHub Pages

This directory is configured to be used with GitHub Pages. When GitHub Pages is enabled for this repo with the 'docs' folder as the source, the dashboard will be available at:

https://yourusername.github.io/llmrank/

## Updating the Dashboard

To update this dashboard with new evaluation results:

1. Run the SlopRank tool with the `--dashboard` option
2. Copy the resulting dashboard.html to this directory as index.html
3. Update the image paths if necessary
4. Commit and push the changes