
# LLMRank

"SlopRank" is an eval framework for ranking LLMs using peer-based cross-evaluation and PageRank. It enables unbiased, dynamic, and scalable benchmarking of multiple models, fostering transparency and innovation in the development of AI systems.

You can use it for one large set of heterogenous prompts to get the overall ranking, or smaller sets to get rankings for your particular usecase.

Example output from one set:
   ```bash
   === PageRank Scores ===
   gemini-2.0-flash-thinking-exp-1219: 0.1466
   o1-preview: 0.1456
   gemini-exp-1206: 0.1455
   deepseek-chat: 0.1424
   gpt-4o: 0.1422
   claude-3-5-sonnet-latest: 0.1398
   claude-3-opus-latest: 0.1380
   ```
---
## Features
- **Peer-Based Evaluation**: Models evaluate each other's responses, mimicking a collaborative and competitive environment.
- **Customizable Scoring**:
  - **Numeric Ratings (1–10)** for granular evaluation.
  - **Upvote/Downvote** for simple binary scoring.
- **Subset Evaluation**: Reduce API costs by limiting the models each evaluator reviews.
- **Graph-Based Ranking**: Endorsements are represented in a graph, and PageRank is used to compute relative rankings.
- **Scalable Benchmarking**: Add more models or prompts with ease, maintaining flexibility and efficiency.

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
1. Clone the repository
2. Install dependencies
3. Set up API keys for your LLMs by creating a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```
   I set them using llm keys set [MODEL] 

---

## Usage

### Configuration
- **Models**: Update the `MODEL_NAMES` list in the notebook to include the models you want to evaluate.
- **Prompts**: Define your prompts in the `raw_prompts` list.
- **Evaluation Method**: Choose between numeric ratings (`EVALUATION_METHOD = 1`) or upvotes/downvotes (`EVALUATION_METHOD = 2`).
- **Subset Evaluation**: Toggle `USE_SUBSET_EVALUATION` to reduce evaluation costs.

### Running the Framework
1. Open and run the notebook.
2. Inspect the results:
   - Ranked models based on PageRank.
   - Visualization of the endorsement graph (optional).

---

## Outputs
- **Ranked Models**: A list of models ordered by their PageRank scores.
- **Graph Representation**: A directed graph showing the flow of endorsements.
- **Processing Times**: Benchmark of evaluation times for each model.

---

## Applications
- **Benchmarking**: Evaluate and rank new or existing LLMs.
- **Specialization Analysis**: Test domain-specific capabilities (e.g., legal, medical).
- **Model Optimization**: Identify strengths and weaknesses for targeted fine-tuning.
- **Public Leaderboards**: Maintain transparency and foster healthy competition among models.

---

## Ideas for Contributions

### Suggested Improvements
1. Add graph visualization to show endorsement flow between models.
2. Create a heatmap of model scores across different prompts.
3. Develop a public leaderboard to showcase rankings.
4. Build a simple GUI for easier usage of the framework.
5. Add support for multi-language evaluation by introducing localized prompts.

Contributions are welcome! If you have ideas for improving the framework, feel free to open an issue or submit a pull request.

---

## Acknowledgments
Special thanks to:
- [SimonW](https://github.com/simonw) for the `llm` library.
- The AI community