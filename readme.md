# SlopRank

SlopRank is an evaluation framework for ranking LLMs using peer-based cross-evaluation and PageRank. It enables unbiased, dynamic, and scalable benchmarking of multiple models, fostering transparency and innovation in the development of AI systems.

You can use it with a large set of heterogeneous prompts to get overall rankings, or with smaller targeted sets to evaluate models for your specific use case.

## Interactive Dashboard

![Dashboard Preview](results/visualizations/endorsement_graph.png)

**[➡️ View Interactive Dashboard](https://htmlpreview.github.io/?https://github.com/strangeloopcanon/llmrank/blob/main/docs/index.html)**

### Example Ranking (OpenRouter run):
```
=== PageRank Rankings ===
   model                                   pagerank_score
0  openrouter/openai/gpt-5                 0.168470
1  openrouter/qwen/qwen3-max               0.155266
2  openrouter/google/gemini-2.5-pro        0.145787
3  openrouter/anthropic/claude-opus-4.1    0.135553
4  openrouter/x-ai/grok-4                  0.135202
5  openrouter/anthropic/claude-sonnet-4    0.133854
6  openrouter/nousresearch/hermes-4-405b   0.125868
```

Models in this run: gpt-5, claude opus 4.1, claude sonnet 4, grok 4, qwen 3 max, gemini 2.5 pro, nousresearch/hermes-4-405b. Results were computed using peer cross‑evaluation and PageRank over 37 prompts.

It supports pretty much all models, anything that can be run with the 'llm' library.

## Features
- **Peer-Based Evaluation**: Models evaluate each other's responses, mimicking a collaborative and competitive environment.
- **Customizable Scoring**:
  - **Numeric Ratings (1–10)** for granular evaluation.
  - **Upvote/Downvote** for simple binary scoring.
- **Subset Evaluation**: Reduce API costs by limiting the models each evaluator reviews.
- **Graph-Based Ranking**: Endorsements are represented in a graph, and PageRank is used to compute relative rankings.
- **Scalable Benchmarking**: Add more models or prompts with ease, maintaining flexibility and efficiency.
- **Graph Visualization**: Visualize model endorsements with interactive and static graph visualizations.
- **Category-Based Analysis**: Evaluate model performance across different prompt categories (reasoning, coding, etc.).
- **Statistical Confidence**: Calculate confidence intervals and significance tests for model rankings.
- **Interactive Dashboard**: Explore results through a web-based dashboard with interactive visualizations.

## How It Works
1. **Prompt Collection**: Define a set of questions or tasks to test the models.
2. **Model Responses**: Each model generates a response to the prompts.
3. **Cross-Evaluation**:
   - Each model evaluates the quality of other models' responses.
   - Evaluations are collected via predefined scoring methods.
4. **Graph Construction**: Build a directed graph where nodes are models, and edges represent endorsements.
5. **Ranking**: Apply the PageRank algorithm to rank models based on their relative endorsements.

## Installation

### Prerequisites
- Python 3.8+
- [SimonW's `llm` library](https://github.com/simonw/llm)
- `networkx` for graph computations
- `dotenv` for environment variable management

### Setup

SlopRank is on PyPI, so you can install it via:
```bash
pip install sloprank
```

From Source: If you prefer, clone this repo and install locally:
```bash
git clone https://github.com/strangeloopcanon/llmrank.git
cd sloprank
pip install .
```

### API Keys Setup

Set up API keys using Simon Willison's llm tool. Example:
```bash
llm keys set anthropic 
llm keys set openai
```

Or create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

After installing, you can run the entire SlopRank workflow via the `sloprank` command. By default, SlopRank uses the models defined in DEFAULT_CONFIG. You can override this by passing --models with a comma-separated list.

### Basic Usage

```bash
sloprank --prompts prompts.csv --output-dir results
```
- `--prompts prompts.csv` tells SlopRank where to find your list of prompts.
- `--output-dir results` puts all CSV and JSON outputs in the results/ folder.

If you want to override the default models:

```bash
sloprank --prompts prompts.csv --output-dir results --models "chatgpt-4o,o1,claude-3-7-sonnet-latest, deepseek-reasoner, gemini-2.0-pro-exp-02-05" --visualize --confidence
```

### Configuration
- **Models**: Update the `MODEL_NAMES` list to include the models you want to evaluate.
- **Prompts**: Define your prompts in the `raw_prompts` list.
- **Evaluation Method**: Choose between numeric ratings (`EVALUATION_METHOD = 1`) or upvotes/downvotes (`EVALUATION_METHOD = 2`).
- **Subset Evaluation**: Toggle `USE_SUBSET_EVALUATION` to reduce evaluation costs.

### Advanced Features

#### Visualization, Confidence Intervals, and Categories

Run SlopRank with all advanced features:

```bash
sloprank run --prompts prompts.csv --output-dir results --visualize --confidence --categories
```

#### Interactive Dashboard

Add the `--dashboard` flag to launch an interactive web dashboard:

```bash
sloprank run --prompts prompts.csv --output-dir results --dashboard
```

Launch the dashboard for existing results:

```bash
sloprank dashboard --output-dir results
```

#### Using Individual Tools

The `examples/` directory contains standalone scripts for each advanced feature:

1. Graph Visualization:
   ```bash
   python examples/generate_visualization.py
   ```

2. Confidence Intervals:
   ```bash
   python examples/compute_confidence.py
   ```

3. Prompt Categorization:
   ```bash
   python examples/prompt_categorization.py
   ```

4. Dashboard Generation:
   ```bash
   python examples/generate_dashboard.py
   python examples/dashboard.py
   ```

## Outputs
- **Ranked Models**: A list of models ordered by their PageRank scores.
- **Graph Representation**: A directed graph showing the flow of endorsements.
- **Processing Times**: Benchmark of evaluation times for each model.
- **Interactive Visualizations**: HTML-based interactive graphs with node and edge details.
- **Static Visualizations**: PNG images of the endorsement graph.
- **Confidence Intervals**: Statistical confidence bounds for model rankings.
- **Significance Tests**: Statistical significance indicators between adjacent ranks.
- **Category Rankings**: Model performance across different prompt categories.

#### Dashboard Details

The dashboard provides:
- Overall model rankings with confidence intervals
- Category-specific performance analysis
- Interactive graph visualizations
- Model comparison tools

#### Download Options

- **[⬇️ Download Dashboard HTML](https://raw.githubusercontent.com/strangeloopcanon/llmrank/main/docs/index.html)** - Save and open locally in any browser

## Applications
- **Benchmarking**: Evaluate and rank new or existing LLMs.
- **Specialization Analysis**: Test domain-specific capabilities (e.g., legal, medical).
- **Model Optimization**: Identify strengths and weaknesses for targeted fine-tuning.
- **Public Leaderboards**: Maintain transparency and foster healthy competition among models.

## Development

### Release Process

To build and release a new version of SlopRank to PyPI:

1. Update the version number in `pyproject.toml` following semantic versioning
2. Update the Changelog section below with all changes
3. Clean previous builds: `rm -rf build/ dist/ *.egg-info/`
4. Build the package: `python -m build`
5. Validate the package: `twine check dist/*`
6. Upload to PyPI: `twine upload dist/*`
7. Create a GitHub release with the changelog info

### Troubleshooting Releases

- If you get permission errors during upload, check your PyPI credentials
- If the build fails, ensure all dependencies are correctly listed in pyproject.toml
- If the package fails validation, fix the issues before attempting to upload again

## Version History

See the [CHANGELOG.md](CHANGELOG.md) file for a detailed version history and release notes.

## Ideas for Contributions

### Suggested Improvements
1. Improve visualization options and customization.
2. Add more statistical analysis methods.
3. Develop a public leaderboard to showcase rankings.
4. Enhance the web dashboard with more interactive features.
5. Add support for multi-language evaluation by introducing localized prompts.
6. Implement cost estimation and optimization features.

Contributions are welcome! If you have ideas for improving the framework, feel free to open an issue or submit a pull request.

## Acknowledgments
Special thanks to:
- [SimonW](https://github.com/simonw) for the `llm` library.
- The AI community
## Using parallm for More Efficient Response Collection

SlopRank uses the `parallm` library for more efficient parallel model querying:

```python
# Install with pip
pip install sloprank

# parallm is included as a dependency and automatically used
sloprank run --prompts prompts.csv --output-dir results --models "gpt-4o,claude-3.5-sonnet-latest"

# Or use parallm directly
from parallm import query_model_all

# Query multiple models with all prompts in a CSV file
df = query_model_all("prompts.csv", ["gpt-4", "claude-3-5-sonnet", "gemini-2.0-flash"])
print(df)
```

This integration significantly speeds up the response collection process by running queries in parallel.
