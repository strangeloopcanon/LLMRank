# SlopRank

SlopRank is an evaluation framework for ranking LLMs using peer-based cross-evaluation and PageRank. It enables unbiased, dynamic, and scalable benchmarking of multiple models, fostering transparency and innovation in the development of AI systems.

You can use it with a large set of heterogeneous prompts to get overall rankings, or with smaller targeted sets to evaluate models for your specific use case.

### Example Ranking:
```
=== PageRank Rankings ===
   model                                pagerank_score
0  o1-preview                           0.179404
1  gpt-4o                               0.178305
2  deepseek-chat                        0.167105
3  gemini-2.0-flash-thinking-exp-1219   0.164732
4  claude-3-5-sonnet-latest             0.155571
5  gemini-exp-1206                      0.154884
```

Supported models include ChatGPT-4o, Claude-3.7-Sonnet, Deepseek-Reasoner, Gemini-2.0-Pro, O1, and others.
---
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

---

## How It Works
1. **Prompt Collection**: Define a set of questions or tasks to test the models.
2. **Model Responses**: Each model generates a response to the prompts.
3. **Cross-Evaluation**:
   - Each model evaluates the quality of other models' responses.
   - Evaluations are collected via predefined scoring methods.
4. **Graph Construction**: Build a directed graph where nodes are models, and edges represent endorsements.
5. **Ranking**: Apply the PageRank algorithm to rank models based on their relative endorsements.

---

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

Set up API keys using Simon Willison's llm tool:
```bash
llm keys set anthropic 
llm keys set openai
```

Or create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Usage

After installing, you can run the entire SlopRank workflow via the `sloprank` command. By default, SlopRank uses the models defined in DEFAULT_CONFIG. You can override this by passing --models with a comma-separated list.

### Basic Usage

```bash
sloprank --prompts prompts.xlsx --output-dir results
```
- `--prompts prompts.xlsx` tells SlopRank where to find your list of prompts.
- `--output-dir results` puts all CSV and JSON outputs in the results/ folder.

If you want to override the default models:

```bash
sloprank --prompts prompts.xlsx --output-dir results \
         --models "gpt-4o,claude-3-5-sonnet-latest,o1-preview"
```

### Configuration
- **Models**: Update the `MODEL_NAMES` list in the notebook to include the models you want to evaluate.
- **Prompts**: Define your prompts in the `raw_prompts` list.
- **Evaluation Method**: Choose between numeric ratings (`EVALUATION_METHOD = 1`) or upvotes/downvotes (`EVALUATION_METHOD = 2`).
- **Subset Evaluation**: Toggle `USE_SUBSET_EVALUATION` to reduce evaluation costs.

### Advanced Features

#### Visualization, Confidence Intervals, and Categories

Run SlopRank with all advanced features:

```bash
sloprank run --prompts prompts.xlsx --output-dir results --visualize --confidence --categories
```

#### Interactive Dashboard

Add the `--dashboard` flag to launch an interactive web dashboard:

```bash
sloprank run --prompts prompts.xlsx --output-dir results --dashboard
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

### Using the Notebook

If you prefer using Jupyter Notebook:
1. Open `llmrank.ipynb`
2. Run the cells to execute the workflow
3. Inspect the results

---

## Outputs
- **Ranked Models**: A list of models ordered by their PageRank scores.
- **Graph Representation**: A directed graph showing the flow of endorsements.
- **Processing Times**: Benchmark of evaluation times for each model.
- **Interactive Visualizations**: HTML-based interactive graphs with node and edge details.
- **Static Visualizations**: PNG images of the endorsement graph.
- **Confidence Intervals**: Statistical confidence bounds for model rankings.
- **Significance Tests**: Statistical significance indicators between adjacent ranks.
- **Category Rankings**: Model performance across different prompt categories.

### Example Dashboard

The dashboard provides:
- Overall model rankings with confidence intervals
- Category-specific performance analysis
- Interactive graph visualizations
- Model comparison tools

---

## Applications
- **Benchmarking**: Evaluate and rank new or existing LLMs.
- **Specialization Analysis**: Test domain-specific capabilities (e.g., legal, medical).
- **Model Optimization**: Identify strengths and weaknesses for targeted fine-tuning.
- **Public Leaderboards**: Maintain transparency and foster healthy competition among models.

---

## Ideas for Contributions

### Suggested Improvements
1. Improve visualization options and customization.
2. Add more statistical analysis methods.
3. Develop a public leaderboard to showcase rankings.
4. Enhance the web dashboard with more interactive features.
5. Add support for multi-language evaluation by introducing localized prompts.
6. Implement cost estimation and optimization features.

Contributions are welcome! If you have ideas for improving the framework, feel free to open an issue or submit a pull request.

---

## Acknowledgments
Special thanks to:
- [SimonW](https://github.com/simonw) for the `llm` library.
- The AI community