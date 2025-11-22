#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entrypoint for carpark (Rush Hour) inference.

This module is responsible only for CLI wiring. It invokes the unified
carpark runner, which in turn calls into :mod:`src.inference.domains.carpark.carpark_core`
for dataset loading and inference logic.
"""

from __future__ import annotations

from importlib import import_module
from typing import Optional, Sequence

from src.inference.backends import HFBackend
from src.inference.runners.unified_runner_base import run_carpark_main


def _load_carpark_module():
    """
    Indirection so the unified runner can resolve the carpark module.

    The returned module must expose ``load_rush_dataset`` and
    ``run_inference_on_split`` with the expected signatures.
    """
    return import_module("src.inference.domains.carpark.carpark_core")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    CLI entrypoint used by ``python -m src.inference.domains.carpark.carpark_cli``.
    """
    run_carpark_main(load_module=_load_carpark_module, backend_cls=HFBackend, argv=argv)


if __name__ == "__main__":
    main()
