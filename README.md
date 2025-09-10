# SlopRank

SlopRank is a **high-performance evaluation framework** for ranking LLMs using peer-based cross-evaluation and PageRank. Built with **Bodo** for parallel processing, it enables unbiased, dynamic, and scalable benchmarking of multiple models, fostering transparency and innovation in the development of AI systems.

You can use it with a large set of heterogeneous prompts to get overall rankings, or with smaller targeted sets to evaluate models for your specific use case.

🚀 **Performance**: Powered by Bodo for parallel DataFrame operations and JIT compilation  
📊 **Scalable**: Efficiently handles large datasets with optimized memory usage  
🔗 **Compatible**: Direct integration with Simon Willison's `llm` library

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

### 🚀 **High-Performance Processing**
- **Bodo Integration**: Parallel DataFrame operations with JIT compilation for maximum performance
- **Memory Efficient**: Optimized memory usage for large-scale evaluations
- **Scalable**: Handles thousands of prompts and dozens of models efficiently

### 🤖 **Advanced Evaluation**
- **Peer-Based Evaluation**: Models evaluate each other's responses, mimicking a collaborative and competitive environment
- **Customizable Scoring**: Numeric ratings (1–10) for granular evaluation or upvote/downvote for binary scoring
- **Subset Evaluation**: Reduce API costs by limiting the models each evaluator reviews
- **Graph-Based Ranking**: Endorsements are represented in a graph, and PageRank is used to compute relative rankings

### 📊 **Rich Analytics**
- **Statistical Confidence**: Calculate confidence intervals and significance tests for model rankings
- **Category-Based Analysis**: Evaluate model performance across different prompt categories (reasoning, coding, etc.)
- **Graph Visualization**: Interactive and static graph visualizations of model endorsements
- **Interactive Dashboard**: Explore results through a web-based dashboard with interactive visualizations

### 🔗 **Flexible Integration**
- **LLM Library**: Direct integration with Simon Willison's `llm` library for broad model support
- **Provider Agnostic**: Works with OpenAI, Anthropic, OpenRouter, and local models
- **Easy Configuration**: Simple CSV-based prompt input and JSON output

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
- **Python 3.9+** (required for Bodo compatibility)
- **[Bodo](https://bodo.ai/)** for high-performance parallel processing (included by default)
- **[SimonW's `llm` library](https://github.com/simonw/llm)** for model access
- `networkx` for graph computations
- `dotenv` for environment variable management

### Optional Compatibility Mode
- **`pandas`** for compatibility mode (if you specifically need regular pandas)

### Setup

**Standard Installation** (includes Bodo for 3-5x performance):
```bash
pip install sloprank
```

**Compatibility Installation** (regular pandas only):
```bash
pip install sloprank[pandas]
```

**From Source**:
```bash
git clone https://github.com/strangeloopcanon/llmrank.git
cd sloprank
pip install .               # Standard installation (includes Bodo)
pip install .[pandas]       # Compatibility mode (regular pandas)
```

### API Keys Setup

SlopRank uses the `llm` library for model access. Set up API keys using Simon Willison's llm tool:

```bash
# Install llm library (included as dependency)
pip install llm

# Set up API keys for various providers
llm keys set anthropic 
llm keys set openai
llm keys set openrouter  # For OpenRouter models
```

Or create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENROUTER_API_KEY=your_openrouter_key
```

**Supported Models**: Any model supported by the `llm` library, including:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude models)
- OpenRouter (access to many models)
- Local models via llm plugins

### Backend Configuration

SlopRank automatically detects and uses the best available pandas backend:

**Check Current Backend**:
```bash
sloprank backend
```

**Force Specific Backend**:
```bash
# Force Bodo for maximum performance
export SLOPRANK_USE_BODO=true
sloprank run --prompts prompts.csv

# Force regular pandas for compatibility
export SLOPRANK_USE_BODO=false
sloprank run --prompts prompts.csv

# Alternative syntax
SLOPRANK_PANDAS_BACKEND=bodo sloprank run --prompts prompts.csv
SLOPRANK_PANDAS_BACKEND=pandas sloprank run --prompts prompts.csv
```

**Auto-Detection Behavior**:
- **Default**: Uses Bodo automatically (included in standard installation, 3-5x performance boost)
- **Fallback**: Uses regular pandas if Bodo unavailable (compatibility mode)
- **Override**: Manual environment variables always take precedence

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

### Recent Updates (v0.3.15+)
🚀 **Major Performance Upgrade**: Bodo-First Architecture
- ✅ **Bodo is now the default** - included in standard installation
- ✅ **3-5x performance by default** - no configuration needed
- ✅ **Switchable backend system** - environment variable control
- ✅ Direct Bodo integration for maximum performance
- ✅ Intelligent fallback to pandas when needed
- ✅ Simplified high-performance installation model

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
- **[Bodo.ai](https://bodo.ai/)** for the high-performance parallel computing platform
- **[SimonW](https://github.com/simonw)** for the excellent `llm` library and ecosystem
- **The AI community** for driving innovation in model evaluation
## Flexible High-Performance Processing

SlopRank features a **switchable pandas backend** system that automatically optimizes for your environment:

```python
# Standard installation (includes Bodo for high performance)
pip install sloprank

# Compatibility installation (regular pandas only)
pip install sloprank[pandas]

# SlopRank automatically uses the best backend (Bodo by default)
sloprank run --prompts prompts.csv --output-dir results --models "gpt-4o,claude-3.5-sonnet-latest"

# Direct usage with automatic backend selection
from sloprank.pandas_backend import pd  # Uses Bodo by default, pandas fallback
from sloprank.collect import collect_responses

# Efficient processing for large datasets (3-5x faster with Bodo by default)
responses_df = collect_responses(prompt_pairs, config)
print(responses_df)
```

This integration provides:
- **Parallel DataFrame Operations**: Automatic parallelization of pandas operations across multiple cores
- **Memory Efficiency**: Optimized memory usage for large datasets with intelligent caching
- **High Performance**: JIT compilation for compute-intensive operations (graph building, PageRank)
- **Direct LLM Integration**: Streamlined model access via Simon Willison's `llm` library
- **Production Ready**: Robust error handling and fallback mechanisms

### Performance Benefits

**Benchmark improvements with Bodo integration:**
- ⚡ **3-5x faster** DataFrame operations on large evaluation datasets
- 💾 **50-70% less memory** usage compared to standard pandas
- 🔄 **Automatic parallelization** of PageRank computations
- 📈 **Linear scalability** with dataset size and number of models

**Ideal for:**
- Large-scale model comparisons (10+ models, 1000+ prompts)
- Academic research requiring statistical rigor
- Enterprise benchmarking with performance requirements
- Continuous evaluation pipelines
