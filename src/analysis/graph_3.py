#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thin wrapper for the PASS-1 entropy bucket plots (graph_3).

Canonical usage remains:

    python -m src.analysis.graph_3 ...

The implementation lives in ``graph_3_impl.py`` to keep this entrypoint
small and to satisfy lint rules about module size.
"""

from __future__ import annotations

from .graph_3_impl import main as _impl_main


def main() -> None:
    """CLI entry point that delegates to the implementation module."""
    _impl_main()


if __name__ == "__main__":
    main()
