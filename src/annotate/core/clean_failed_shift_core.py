"""
Backwards-compatible shim for core cleaning helpers.

Canonical implementation now lives in :mod:`src.annotate.core.clean_core`.
"""

from .clean_core import BAD_RATIONALE_PREFIXES, SHIFT_FIELDS, clean_file, clean_root  # noqa: F401


__all__ = ["BAD_RATIONALE_PREFIXES", "SHIFT_FIELDS", "clean_file", "clean_root"]
