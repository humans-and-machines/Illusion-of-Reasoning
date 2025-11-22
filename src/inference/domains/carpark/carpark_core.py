#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin compatibility/shim module for Rush Hour (car-park) inference.

The implementation for two-pass inference now lives in
``src.inference.domains.carpark.carpark_solver`` and the dataset loader in
``src.inference.domains.carpark.carpark_data``. This module keeps the original public
API surface so that existing imports and task-registry dotted paths continue
to work.
"""

from __future__ import annotations

from importlib import import_module
from typing import List, Optional

from .carpark_data import (
    _canon_rush_generic,
    _canon_rush_gold,
    load_rush_dataset,
)
from .carpark_solver import (
    SYSTEM_PROMPT,
    CarparkInferenceConfig,
    InferenceContext,
    run_inference_on_split,
)


__all__ = [
    "SYSTEM_PROMPT",
    "CarparkInferenceConfig",
    "InferenceContext",
    "run_inference_on_split",
    "load_rush_dataset",
    "_canon_rush_generic",
    "_canon_rush_gold",
    "main",
]


def main(argv: Optional[List[str]] = None) -> None:
    """
    Backwards-compatible CLI wrapper.

    Delegates to :mod:`src.inference.domains.carpark.carpark_cli` to run the unified
    carpark inference entrypoint.
    """
    carpark_cli = import_module("src.inference.domains.carpark.carpark_cli")
    carpark_cli.main(argv)
