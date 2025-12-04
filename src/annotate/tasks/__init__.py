"""
Task-oriented annotation utilities.

This subpackage is reserved for active tooling (e.g., ``math_cue_variants``).
Legacy aliases such as :mod:`src.annotate.tasks.shifts` now live in
``src.annotate.backcompat.tasks`` but remain importable via compatibility
hooks defined here.
"""

from __future__ import annotations

import importlib
import sys

from . import math_cue_variants


__all__ = ["math_cue_variants"]

_SHIFTS_MODULE = importlib.import_module("src.annotate.backcompat.tasks.shifts")
sys.modules[f"{__name__}.shifts"] = _SHIFTS_MODULE
