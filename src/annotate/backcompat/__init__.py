"""
Legacy compatibility shims kept under :mod:`src.annotate.backcompat`.

These modules preserve historical import paths such as
``src.annotate.shift_cli`` or ``src.annotate.llm_client`` while allowing
the top-level ``src.annotate`` package to focus on the canonical ``core``,
``cli``, and ``infra`` subpackages.
"""

from __future__ import annotations

from . import clean_failed_shift_core, clean_failed_shift_labels, config, llm_client, prompts, shift_cli


__all__ = [
    "clean_failed_shift_core",
    "clean_failed_shift_labels",
    "config",
    "llm_client",
    "prompts",
    "shift_cli",
]
