"""
Inference package initialization.

The package is organized into layered subpackages:

- :mod:`src.inference.domains` – task-specific inference loops (math/carpark/crossword/summarize).
- :mod:`src.inference.utils` – shared helpers, backends, and registries.
- :mod:`src.inference.runners` and :mod:`src.inference.cli` – reusable runner logic plus thin CLIs.
- :mod:`src.inference.gateways` – remote math runners (Azure, OpenRouter, Portkey).

Only lightweight, stable symbols are re-exported here to keep import-time
dependencies minimal.
"""

from __future__ import annotations

from src.inference.utils.common import GenerationLimits, SamplingConfigBase

__all__ = [
    "GenerationLimits",
    "SamplingConfigBase",
]
