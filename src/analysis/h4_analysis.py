#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper module for the H4 artificial recheck figures.

Canonical usage is::

  python -m src.analysis.h4_analysis ...

The implementation lives in ``h4_analysis_impl.py`` to keep the public entry
point stable while allowing the underlying script to evolve.
"""

from __future__ import annotations

from .h4_analysis_impl import main as _impl_main


def main() -> None:
    """Entry point that delegates to the implementation module."""
    _impl_main()


if __name__ == "__main__":
    main()
