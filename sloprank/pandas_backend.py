"""
Pandas Backend Configuration
============================

This module provides a configurable pandas backend that can switch between
regular pandas and bodo.pandas based on environment variables or configuration.

Environment Variables:
- SLOPRANK_USE_BODO: Set to "true" to use Bodo, "false" for regular pandas
- SLOPRANK_PANDAS_BACKEND: Set to "bodo" or "pandas" 

Default: Uses Bodo if available, falls back to regular pandas
"""

import os
import logging
from typing import Any

logger = logging.getLogger("SlopRank.PandasBackend")

# Configuration flags
_USE_BODO = None
_pandas_module = None

def _determine_backend():
    """Determine which pandas backend to use based on configuration and availability."""
    global _USE_BODO
    
    if _USE_BODO is not None:
        return _USE_BODO
    
    # Check environment variables
    env_use_bodo = os.getenv("SLOPRANK_USE_BODO", "").lower()
    env_backend = os.getenv("SLOPRANK_PANDAS_BACKEND", "").lower()
    
    # Explicit configuration via environment
    if env_use_bodo in ("true", "1", "yes"):
        _USE_BODO = True
        return True
    elif env_use_bodo in ("false", "0", "no"):
        _USE_BODO = False
        return False
    elif env_backend == "bodo":
        _USE_BODO = True
        return True
    elif env_backend == "pandas":
        _USE_BODO = False
        return False
    
    # Auto-detection: prefer Bodo if available (now the default!)
    try:
        import bodo.pandas
        _USE_BODO = True
        logger.info("Auto-detected: Using Bodo pandas backend (default high-performance mode)")
        return True
    except ImportError:
        # Fallback to regular pandas
        try:
            import pandas
            _USE_BODO = False
            logger.info("Auto-detected: Using regular pandas backend (Bodo not available)")
            return False
        except ImportError:
            raise ImportError("Neither Bodo nor pandas is available. Please install one: 'pip install sloprank' (includes Bodo) or 'pip install sloprank[pandas]' (compatibility mode)")

def _get_pandas_module():
    """Get the configured pandas module."""
    global _pandas_module
    
    if _pandas_module is not None:
        return _pandas_module
    
    use_bodo = _determine_backend()
    
    if use_bodo:
        try:
            import bodo.pandas as pd
            _pandas_module = pd
            logger.info("Loaded Bodo pandas backend")
            return pd
        except ImportError as e:
            logger.warning(f"Failed to import Bodo pandas: {e}")
            logger.info("Falling back to regular pandas")
            # Fall through to regular pandas
    
    # Use regular pandas
    try:
        import pandas as pd
        _pandas_module = pd
        logger.info("Loaded regular pandas backend")
        return pd
    except ImportError:
        raise ImportError("Neither Bodo nor pandas is available. Please install one: 'pip install sloprank' (includes Bodo) or 'pip install sloprank[pandas]' (compatibility mode)")

def get_pandas():
    """Get the configured pandas module."""
    return _get_pandas_module()

def is_using_bodo():
    """Check if we're using the Bodo backend."""
    return _USE_BODO is True

def force_backend(backend: str):
    """
    Force a specific backend for testing or configuration.
    
    Args:
        backend: "bodo" or "pandas"
    """
    global _USE_BODO, _pandas_module
    
    if backend.lower() == "bodo":
        _USE_BODO = True
    elif backend.lower() == "pandas":
        _USE_BODO = False
    else:
        raise ValueError("Backend must be 'bodo' or 'pandas'")
    
    # Reset module cache to force reload
    _pandas_module = None
    logger.info(f"Forced backend to: {backend}")

def get_backend_info():
    """Get information about the current backend configuration."""
    pd = get_pandas()
    backend_name = "bodo" if is_using_bodo() else "pandas"
    
    return {
        "backend": backend_name,
        "module": str(type(pd)),
        "using_bodo": is_using_bodo(),
        "env_use_bodo": os.getenv("SLOPRANK_USE_BODO"),
        "env_backend": os.getenv("SLOPRANK_PANDAS_BACKEND")
    }

# Export the pandas module for easy importing
pd = get_pandas()

# For backwards compatibility and explicit access
pandas = pd
