#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified launcher for the Rush Hour (car-park) inference loop using carpark_core.
Thin wrapper over src.inference.unified_runner_base.
"""
from __future__ import annotations

from src.inference import carpark_core
from src.inference.backends import HFBackend
from src.inference.unified_runner_base import run_carpark_main


def _load_carpark_module():
    """Indirection for tests to monkeypatch the carpark_core module."""
    return carpark_core


def main() -> None:
    """Entry point for unified carpark (Rush Hour) inference."""
    run_carpark_main(load_module=_load_carpark_module, backend_cls=HFBackend)


__all__ = ["main"]


if __name__ == "__main__":
    main()
