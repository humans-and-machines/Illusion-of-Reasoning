"""Analysis package initialization.

This module marks ``src.analysis`` as a regular package and provides a small
import surface for the main RQ entrypoints and commonly used helpers. Imports
are resolved lazily to avoid side effects (for example, matplotlib style loads)
when only a subset of modules is needed.
"""

from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "rq1_analysis",
    "rq2_analysis",
    "rq3_analysis",
    "h2_analysis",
    "h3_analysis",
    "export_cue_variants",
    "cue_delta_accuracy",
    "cue_entropy_regression",
    "graph_3",
    "graph_3_impl",
    "graph_4",
    "temperature_effects",
    "plotting",
    "core",
    "common",
    "figures",
    "io",
    "labels",
    "metrics",
    "utils",
]

_SUBMODULES = {
    "rq1_analysis",
    "rq2_analysis",
    "rq3_analysis",
    "h2_analysis",
    "h3_analysis",
    "export_cue_variants",
    "cue_delta_accuracy",
    "cue_entropy_regression",
    "graph_3",
    "graph_3_impl",
    "graph_4",
    "temperature_effects",
    "plotting",
    "core",
    "common",
    "figures",
}


def __getattr__(name: str) -> Any:
    """Lazily import analysis submodules and common helpers on demand."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"src.analysis.{name}")
        globals()[name] = module
        return module
    if name in {"io", "labels", "metrics"}:
        common = importlib.import_module("src.analysis.common")
        value = getattr(common, name)
        globals()[name] = value
        return value
    if name == "utils":
        module = importlib.import_module("src.analysis.utils")
        globals()[name] = module
        return module
    raise AttributeError(name)
