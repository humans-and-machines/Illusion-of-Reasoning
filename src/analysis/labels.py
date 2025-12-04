#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha / shift labeling utilities.

Centralizes several slightly different "Aha!" definitions that appear across
analysis scripts (native cue-based, GPT-labeled, gated variants).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import coerce_bool


# ---------------------------------------------------------------------------
# Shared key sets for Aha labels
# ---------------------------------------------------------------------------

AHA_KEYS_CANONICAL: List[str] = [
    "change_way_of_thinking",
    "shift_in_reasoning_v1",
]

AHA_KEYS_BROAD: List[str] = AHA_KEYS_CANONICAL + [
    "shift_llm",
    "shift_gpt",
    "pivot_llm",
    "rechecked",
]


# ---------------------------------------------------------------------------
# Native (words-based) Aha
# ---------------------------------------------------------------------------


def aha_words(pass1: Dict[str, Any]) -> int:
    """
    Words-based Aha flag from pass1.

    Definition:
      - has_reconsider_cue == 1
      - EXCLUDING injected cues (marker "injected_cue").
    """
    reconsider_flag = coerce_bool(pass1.get("has_reconsider_cue"))
    markers = pass1.get("reconsider_markers") or []
    if isinstance(markers, list) and "injected_cue" in markers:
        return 0
    return 0 if reconsider_flag is None else int(reconsider_flag)


# ---------------------------------------------------------------------------
# GPT / LLM-based shift labels (sample-level)
# ---------------------------------------------------------------------------


def aha_gpt_canonical(pass1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    """
    Canonical GPT/LLM-labeled shift:
      change_way_of_thinking OR shift_in_reasoning_v1
    """
    for k in AHA_KEYS_CANONICAL:
        value = pass1.get(k, rec.get(k, None))
        if value is not None and coerce_bool(value) == 1:
            return 1
    return 0


def aha_gpt_broad(pass1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    """
    Broader GPT/LLM-labeled shift:
      any of AHA_KEYS_BROAD.
    """
    for k in AHA_KEYS_BROAD:
        value = pass1.get(k, rec.get(k, None))
        if value is not None and coerce_bool(value) == 1:
            return 1
    return 0


def _any_keys_true(pass1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    """
    Helper used by gating-aware variants: 1 if any of the keys are truthy.
    """
    for k in keys:
        value = pass1.get(k, rec.get(k, None))
        if value is None:
            continue
        out = coerce_bool(value)
        if out == 1:
            return 1
    return 0


def _cue_gate_for_llm(pass1: Dict[str, Any], domain: Optional[str]) -> int:
    """
    Gate GPT shift labels by native reconsideration cues and domain-specific
    rules.

    For Crossword, we allow gate activation via:
      - non-injected reconsider cue OR
      - prefilter markers OR
      - shift_markers_v1

    For other domains, we ignore judge markers.
    """
    has_reconsider = coerce_bool(pass1.get("has_reconsider_cue")) == 1
    rec_marks = pass1.get("reconsider_markers") or []
    injected = "injected_cue" in rec_marks
    reconsider_ok = has_reconsider and not injected
    prefilter = pass1.get("_shift_prefilter_markers") or []
    judge = pass1.get("shift_markers_v1") or []
    if str(domain).lower() == "crossword":
        return int(reconsider_ok or bool(prefilter) or bool(judge))
    return int(reconsider_ok or bool(prefilter))


def aha_gpt_for_rec(
    pass1: Dict[str, Any],
    rec: Dict[str, Any],
    gpt_subset_native: bool,
    gpt_keys: List[str],
    domain: Optional[str],
) -> int:
    """
    Reusable implementation of the gating logic used in the temperature
    analysis scripts:

      - gpt_raw: any(gpt_keys)
      - if gpt_subset_native is True, gate by native reconsider cues.
    """
    gpt_raw = _any_keys_true(pass1, rec, gpt_keys)
    if not gpt_subset_native:
        return int(gpt_raw)
    gate = _cue_gate_for_llm(pass1, domain)
    return int(gpt_raw & gate)


def aha_gpt(
    pass1: Dict[str, Any],
    rec: Dict[str, Any],
    mode: str = "canonical",
    gate_by_words: bool = True,
    domain: Optional[str] = None,
) -> int:
    """
    High-level convenience wrapper:
      - mode: "canonical" or "broad"
      - optionally enforce aha_gpt <= aha_words via gate_by_words
      - optionally apply domain-aware gating when gate_by_words is True
    """
    if mode not in {"canonical", "broad"}:
        raise ValueError(f"unsupported mode: {mode}")

    base = aha_gpt_canonical if mode == "canonical" else aha_gpt_broad
    gpt_raw = base(pass1, rec)

    if not gate_by_words:
        return int(gpt_raw)

    words = aha_words(pass1)
    # When gating by words, also respect the cue gate used in temp plots
    gate = _cue_gate_for_llm(pass1, domain)
    return int(bool(gpt_raw and words and gate))
