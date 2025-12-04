#!/usr/bin/env python3
"""
Shared helpers for Aha cue detection and gating logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import both_get, coerce_bool


def any_keys_true(
    pass1_fields: Dict[str, Any],
    record: Dict[str, Any],
    keys: List[str],
) -> int:
    """
    Return 1 if any of the provided keys coerce to True, else 0.

    ``both_get`` is used so callers can provide fields that exist either in the
    PASS-1 block or at the top of the record.
    """
    for key in keys:
        value = both_get(pass1_fields, record, key, None)
        if value is None:
            continue
        out = coerce_bool(value)
        if out is not None and out == 1:
            return 1
    return 0


def aha_native(
    pass1_fields: Dict[str, Any],
    record: Dict[str, Any],
) -> Optional[int]:
    """
    A native reconsider cue (``has_reconsider_cue``) that ignores injected cues.
    """
    aha_raw = coerce_bool(both_get(pass1_fields, record, "has_reconsider_cue"))
    markers = both_get(pass1_fields, record, "reconsider_markers") or []
    injected = isinstance(markers, list) and ("injected_cue" in markers)
    if injected:
        return 0
    return 0 if aha_raw is None else int(aha_raw)


def cue_gate_for_llm(
    pass1_fields: Dict[str, Any],
    record: Dict[str, Any],
    domain: Optional[str] = None,
    allow_judge_non_xword: bool = False,
) -> int:
    """
    Native cue gate used when gating GPT-detected shifts.
    """
    has_reconsider = (
        coerce_bool(
            both_get(pass1_fields, record, "has_reconsider_cue"),
        )
        == 1
    )
    rec_marks = both_get(pass1_fields, record, "reconsider_markers") or []
    injected = isinstance(rec_marks, list) and ("injected_cue" in rec_marks)
    reconsider_ok = has_reconsider and not injected

    prefilter_cues = both_get(pass1_fields, record, "_shift_prefilter_markers") or []
    judge_cues = both_get(pass1_fields, record, "shift_markers_v1") or []

    domain_lower = str(domain).lower() if domain is not None else ""
    if domain_lower == "crossword" or allow_judge_non_xword:
        return int(reconsider_ok or bool(prefilter_cues) or bool(judge_cues))
    return int(reconsider_ok or bool(prefilter_cues))
