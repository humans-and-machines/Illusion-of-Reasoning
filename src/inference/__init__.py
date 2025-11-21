"""Inference package initialization.

This module intentionally avoids importing heavy submodules (like carpark_core)
at import time so that lightweight callers (e.g., math-only runners) do not
require optional dependencies such as torch/transformers.
"""

__all__: list[str] = []
