#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers shared by CLI scripts that need optional ``_script_init`` shims.
"""

from __future__ import annotations

import importlib


def ensure_script_context() -> None:
    """
    Attempt to import the optional ``_script_init`` module used when these
    scripts are executed via ``python path/to/script.py``.
    """
    try:  # pragma: no cover - best-effort shim for direct execution
        importlib.import_module("_script_init")
    except ModuleNotFoundError:
        pass


__all__ = ["ensure_script_context"]
