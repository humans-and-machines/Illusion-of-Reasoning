#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap helper so analysis scripts continue working when executed directly.

Importing this module injects the repository root into ``sys.path`` which allows
``from src...`` imports even when running ``python path/to/script.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_root_on_path() -> None:
    """Append the repository root to ``sys.path`` when missing."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.append(repo_str)


ensure_repo_root_on_path()
