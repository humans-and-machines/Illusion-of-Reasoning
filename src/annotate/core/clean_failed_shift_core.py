"""
Backwards-compatible shim for core cleaning helpers.

Canonical implementation now lives in :mod:`src.annotate.core.clean_core`.
"""

from .clean_core import (  # noqa: F401
    BAD_RATIONALE_PREFIXES,
    SHIFT_FIELDS,
    clean_file,
    clean_root,
)

__all__ = ["BAD_RATIONALE_PREFIXES", "SHIFT_FIELDS", "clean_file", "clean_root"]
