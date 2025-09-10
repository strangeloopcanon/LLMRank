# SlopRank Compatibility Notes

## Migration to Bodo-Direct Approach

### Issue Description
Recent updates to [ParaLLM](https://github.com/strangeloopcanon/ParaLLM) included a dependency on Bodo 2025.8.2, which had compatibility issues with certain Python environments. The error manifested as:

```
AttributeError: module 'dis' has no attribute 'hasarg'
```

This occurred when Bodo tried to access Python bytecode inspection features that have changed in recent Python versions.

### Resolution Implemented

SlopRank v0.3.15+ has been updated to use **Bodo directly** instead of relying on ParaLLM as an intermediary. This provides:

1. **Direct Bodo Integration**: Uses `bodo.pandas` throughout the codebase for DataFrame operations
2. **Parallel Processing**: Leverages Bodo's native parallel processing capabilities
3. **Simplified Dependencies**: Removes ParaLLM dependency complexity
4. **Better Performance**: Direct Bodo usage with optimized operations
5. **LLM Library Integration**: Uses Simon Willison's `llm` library for model access

### Code Changes Made

#### 1. Direct Bodo Integration (`sloprank/collect.py`, `sloprank/cli.py`)
- Replaced `import pandas as pd` with `import bodo.pandas as pd`
- Removed all ParaLLM dependencies and imports
- Added hybrid pandas/Bodo approach for complex operations (filtering, concatenation)
- Implemented proper Bodo DataFrame type handling with schema specifications

#### 2. Dependency Simplification (`pyproject.toml`)
- Removed `parallm>=0.1.3` dependency
- Added `bodo>=2024.0.0` as core dependency
- Added `llm>=0.13.0` for model access
- Updated minimum Python version to 3.9+ (required for Bodo)

#### 3. LLM Integration
- Direct integration with Simon Willison's `llm` library
- Simplified model querying without ParaLLM wrapper
- Better error handling for unknown models

### Alternative Solutions

If you continue experiencing issues, try these approaches:

#### Option 1: Downgrade Bodo (Recommended for ParaLLM users)
```bash
pip install "bodo<2025.0.0"
```

#### Option 2: Use ParaLLM without Bodo features
The current implementation automatically handles this fallback.

#### Option 3: Use llm library directly
Install and configure Simon Willison's llm library:
```bash
pip install llm
llm keys set openai
llm keys set anthropic
```

### Verification

After the fixes, SlopRank should work correctly:

```bash
# Test basic functionality
sloprank --help

# Test with a small prompt set
sloprank run --prompts prompts.csv --output-dir test_results --models "mock-model"
```

### Expected Behavior

When Bodo compatibility issues occur:
1. SlopRank logs an error message about ParaLLM batch processing
2. Automatically falls back to individual model queries
3. Continues normal operation with slightly reduced performance
4. All core features (ranking, visualization, confidence intervals) remain functional

### Performance Impact

The fallback approach has minimal performance impact:
- **With ParaLLM**: Parallel batch processing (fastest)
- **Fallback mode**: Sequential individual queries (slightly slower, but still functional)
- **Mock mode**: Uses placeholder responses for testing

### Reporting Issues

If you encounter related issues:
1. Check that you're using SlopRank v0.3.15+
2. Include the full error traceback
3. Mention your Python version and OS
4. Note which ParaLLM version you're using

### Future Improvements

Planned enhancements:
1. Optional ParaLLM dependency
2. Native parallel processing without Bodo
3. Enhanced model provider support
4. Better fallback performance optimization
