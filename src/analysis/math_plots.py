#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper module for the Math summary plots script.

Canonical usage is::

  python -m src.analysis.math_plots ...

The implementation lives in ``math_plots_impl.py`` to keep the public entry
point stable while allowing the underlying script to evolve.
"""

from __future__ import annotations

from .math_plots_impl import main as _impl_main


def main() -> None:
    """CLI entry point that delegates to the implementation module."""
    _impl_main()


if __name__ == "__main__":
    main()
