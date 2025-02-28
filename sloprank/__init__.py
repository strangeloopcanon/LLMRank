"""
SlopRank Package
----------------

Peer-based cross-evaluation of LLMs with PageRank-based scoring.

Key features:
- Peer-based evaluation where models score each other
- Graph visualization of model endorsements
- Confidence intervals and statistical significance tests
- Category-based evaluation and ranking
- Web dashboard for interactive exploration
"""

from .config import (
    EvalConfig, 
    VisualizationConfig, 
    ConfidenceConfig, 
    WebDashboardConfig,
    DEFAULT_CONFIG
)

__version__ = "0.2.2"
__all__ = [
    "EvalConfig",
    "VisualizationConfig",
    "ConfidenceConfig", 
    "WebDashboardConfig",
    "DEFAULT_CONFIG"
]