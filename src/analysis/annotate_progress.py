#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backwards-compatible CLI wrapper for ``src.annotate.core.progress``.

This module used to live under :mod:`src.analysis`; it now simply re-exports
the implementation from :mod:`src.annotate.core.progress` so existing scripts
can keep invoking ``python -m src.analysis.annotate_progress``.
"""

from __future__ import annotations

from src.annotate.core.progress import count_progress, main, parse_args

__all__ = ["count_progress", "main", "parse_args"]


if __name__ == "__main__":
    main()
