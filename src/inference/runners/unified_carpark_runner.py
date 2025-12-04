"""
Library helpers for the Rush Hour (car-park) unified runner.

The CLI wiring now lives in :mod:`src.inference.cli.unified_carpark`.
"""

from __future__ import annotations

from importlib import import_module


__all__ = ["load_carpark_module"]


def load_carpark_module():
    """
    Resolve the Rush Hour inference module.

    :returns: Imported :mod:`src.inference.domains.carpark.carpark_core` module.
    """
    return import_module("src.inference.domains.carpark.carpark_core")
