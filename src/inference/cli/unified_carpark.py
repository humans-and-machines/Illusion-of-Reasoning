#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Rush Hour (car-park) inference CLI.

This module wires :class:`HFBackend` and the shared dataset loader into
``run_carpark_main`` from :mod:`src.inference.runners.unified_runner_base`.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.inference.backends import HFBackend
from src.inference.runners.unified_carpark_runner import load_carpark_module
from src.inference.runners.unified_runner_base import run_carpark_main


__all__ = [
    "HFBackend",
    "load_carpark_module",
    "run_carpark_main",
    "main",
]


def _load_carpark_module():
    """Wrapper kept so tests can monkeypatch this symbol."""
    return load_carpark_module()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for Rush Hour inference CLI."""
    kwargs = {"load_module": _load_carpark_module, "backend_cls": HFBackend}
    if argv is not None:
        kwargs["argv"] = argv
    run_carpark_main(**kwargs)


if __name__ == "__main__":
    main()
