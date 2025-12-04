"""
Backwards-compatible shim for prompt templates.

Canonical implementation now lives in :mod:`src.annotate.core.prompts`.
New code should import from :mod:`src.annotate` or
``src.annotate.core.prompts`` instead of this module.
"""

from ..core.prompts import SHIFT_JUDGE_SYSTEM_PROMPT, SHIFT_JUDGE_USER_TEMPLATE, SHIFT_PROMPT


__all__ = [
    "SHIFT_JUDGE_SYSTEM_PROMPT",
    "SHIFT_JUDGE_USER_TEMPLATE",
    "SHIFT_PROMPT",
]
