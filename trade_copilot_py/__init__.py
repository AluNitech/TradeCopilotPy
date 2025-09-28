"""
TradeCopilotPy package initialization.
Expose package metadata and public subpackages.
"""

__version__ = "0.1.0"

# Public API of the top-level package
__all__ = [
    "__version__",
    "data_handler",
    "strategy",
    "utils",
]

# Optionally, you can re-export frequently used symbols here to shorten import paths.
# from .data_handler import get_multi_timeframe_data  # noqa: F401