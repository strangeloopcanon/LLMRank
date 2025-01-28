"""
SlopRank Package
----------------

Peer-based cross-evaluation of LLMs with PageRank-based scoring.
"""

from .config import EvalConfig, DEFAULT_CONFIG

__version__ = "0.1.0"
__all__ = [
    "EvalConfig",
    "DEFAULT_CONFIG"
]