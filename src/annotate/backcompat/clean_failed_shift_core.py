"""
Backwards-compatible shim for cleaning fallback shift annotations.

Canonical implementation now lives in :mod:`src.annotate.core.clean_core`.
"""

from ..core.clean_core import (
    BAD_RATIONALE_PREFIXES,
    SHIFT_FIELDS,
    clean_file,
    clean_root,
)

__all__ = ["BAD_RATIONALE_PREFIXES", "SHIFT_FIELDS", "clean_file", "clean_root"]
