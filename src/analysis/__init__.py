"""Analysis package initialization.

This module marks ``src.analysis`` as a regular package and provides a small
import surface for the main RQ entrypoints and commonly used helpers.
"""

from __future__ import annotations

from . import rq1_analysis, rq2_analysis, rq3_analysis  # noqa: F401
from . import core, common, figures  # noqa: F401

from .common import io, labels, metrics  # noqa: F401

__all__ = [
    "rq1_analysis",
    "rq2_analysis",
    "rq3_analysis",
    "core",
    "common",
    "figures",
    "io",
    "labels",
    "metrics",
]
