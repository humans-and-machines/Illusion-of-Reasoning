"""
Backwards-compatible task wrappers.

Historically ``src.annotate.tasks`` exposed thin aliases like
``src.annotate.tasks.shifts``.  The canonical implementations now live under
``src.annotate.core`` and ``src.annotate.cli``; these shims simply re-export
that API for legacy imports.
"""

__all__ = ["shifts"]
