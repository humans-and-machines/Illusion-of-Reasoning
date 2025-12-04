"""
Domain-specific inference modules.

Each subpackage (``math``, ``carpark``, ``crossword``) contains the
task-specific inference loops plus helpers that previously lived in the
legacy ``core`` package.
"""

from __future__ import annotations


__all__ = ["math", "carpark", "crossword", "summarize"]
