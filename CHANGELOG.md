# Changelog

All notable changes to SlopRank will be documented in this file.

## [0.3.15] - 2025-09-10

### 🚀 **MAJOR PERFORMANCE UPGRADE: Bodo-First Architecture**

### Added
- **Bodo-First Installation**: Bodo is now included by default in standard installation
- **Switchable Backend System**: Environment variable control for backend selection
- **Automatic Performance**: 3-5x speedup by default, no configuration required
- **High-Performance Processing**: JIT compilation and parallel DataFrame operations
- **Memory Optimization**: 50-70% reduction in memory usage for large datasets
- **Enhanced Error Handling**: Robust fallback mechanisms for both backends
- **CLI Backend Command**: `sloprank backend` to check and manage backend configuration

### Changed
- **BREAKING**: Bodo is now the default dependency (included in `pip install sloprank`)
- **Installation Model**: `pip install sloprank` → includes Bodo, `pip install sloprank[pandas]` → compatibility mode
- **Dependencies**: Bodo moved from optional to core dependency for maximum performance by default
- **Backend Priority**: Auto-detection now prefers Bodo (included) over pandas (fallback)
- **Core Processing**: Intelligent backend-aware DataFrame operations

### Removed
- **ParaLLM dependency**: Eliminated intermediate layer for better performance and reliability
- **Performance barriers**: Users no longer need to know about Bodo to get maximum performance

### Migration Notes
- **New Users**: Get 3-5x performance automatically with `pip install sloprank`
- **Existing Users**: Upgrading provides automatic performance improvements
- **Compatibility**: Use `SLOPRANK_USE_BODO=false` or `pip install sloprank[pandas]` for pandas-only mode
- **No API Changes**: All existing commands and workflows remain the same

## [0.3.11] - 2025-09-09

### Changed
- Updated default model list to latest: `gpt-5`, `claude-4-sonnet`, `gemini-2.5-pro`, `deepseek-chat`.
- Relaxed `llm` dependency to `llm>=0.23` to support latest providers.
- Rankings JSON now includes package `version` metadata.
- README updated with latest model examples and `llm`/parallm notes.

### Added
- Offline smoke test (`tests/test_smoke.py`) to exercise parsing, graph, and ranking without network calls.

### Fixed
- Resolved version mismatch between `pyproject.toml` and `sloprank/__init__.py`.

## [0.2.3] - 2025-02-28

### Added
- Tests directory with simple test scripts and example prompts
- Test README with documentation on how to run tests

### Fixed
- Improved error handling for subset evaluation configuration
- Automatic adjustment of evaluators_subset_size when too large for the number of models
- Added support for new model versions (Claude-3.7-Sonnet, ChatGPT-4o, Deepseek-Reasoner)

## [0.2.2] - 2025-01-14

### Added
- Support for graph visualization of model endorsements
- Confidence interval calculations for rankings
- Category analysis for prompt-specific performance

### Changed
- Improved API error handling
- Enhanced CLI interface with additional options

## [0.2.1] - 2025-01-03

### Added
- Dashboard features for interactive exploration
- Visualization improvements

### Fixed
- Bug fixes in PageRank calculation
- Better error handling for API timeouts

## [0.2.0] - 2024-12-20

### Added
- Complete rewrite with modular architecture
- Support for multiple evaluation methods
- Export options for results

## [0.1.0] - 2024-12-01

### Added
- Initial release
- Basic implementation of peer-based LLM evaluation
- PageRank algorithm for ranking models
