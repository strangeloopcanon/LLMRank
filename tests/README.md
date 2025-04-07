# SlopRank Tests

This directory contains test files for the SlopRank library.

## Test Files

| File | Description |
|------|-------------|
| `test_sloprank.py` | Simple end-to-end test for the SlopRank library |
| `tiny_prompts.csv` | Minimal test prompts with just 2 simple questions |
| `mini_prompts.csv` | Small test prompts with 3 more comprehensive questions |

## Running Tests

To run the basic test:

```bash
python test_sloprank.py
```

### Test Process

The test will automatically:
1. Create a test output directory (`test_results/`)
2. Collect responses from configured models
3. Collect evaluations between models
4. Parse evaluations
5. Build the endorsement graph
6. Compute the PageRank scores
7. Output the final rankings

> **Note:** The full test may take several minutes to complete due to the time required for API calls to language models.

## Test Configuration

The test script uses a simple configuration with:
- 3 models: deepseek-reasoner, claude-3.7-sonnet, and chatgpt-4o
- Simple factual questions to ensure fast responses
- Full evaluation (all models evaluate each other)

You can modify the test script to use different models, prompts, or evaluation settings.