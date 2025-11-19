#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math-specific result packaging, canonicalization, and tag/entropy helpers.

These utilities were originally defined in ``inference.common`` and are split
out to keep that module smaller and more focused on torch/transformers glue
code. Public functions are still re-exported from ``inference.common`` for
backwards compatibility.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.inference.text_utils import find_markers_and_context


# ---------------------------------------------------------------------------
# Text + tag helpers
# ---------------------------------------------------------------------------
RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")


def extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (<think>..., <answer>...) contents (whitespace stripped)."""
    think = ans = None
    think_match = RE_THINK.search(txt)
    if think_match:
        think = think_match.group(1).strip()
    answer_match = RE_ANSWER.search(txt)
    if answer_match:
        ans = answer_match.group(1).strip()
    return think, ans


def valid_tag_structure(full_text: str) -> bool:
    """Require exactly one <think>…</think> before <answer>…</answer>."""
    opens_think = len(re.findall(r"(?i)<think>", full_text))
    closes_think = len(re.findall(r"(?i)</think>", full_text))
    opens_ans = len(re.findall(r"(?i)<answer>", full_text))
    closes_ans = len(re.findall(r"(?i)</answer>", full_text))
    if not (opens_think == closes_think == 1 and opens_ans == closes_ans == 1):
        return False
    think_open_match = re.search(r"(?i)<think>", full_text)
    if think_open_match is None:
        return False
    think_close_match = re.search(r"(?i)</think>", full_text)
    if think_close_match is None:
        return False
    answer_open_match = re.search(r"(?i)<answer>", full_text)
    if answer_open_match is None:
        return False
    answer_close_match = re.search(r"(?i)</answer>", full_text)
    if answer_close_match is None:
        return False

    think_open_pos = think_open_match.start()
    think_close_pos = think_close_match.start()
    answer_open_pos = answer_open_match.start()
    answer_close_pos = answer_close_match.start()
    return think_open_pos < think_close_pos < answer_open_pos < answer_close_pos


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
RE_LATEX_FRAC = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", re.I)
RE_LATEX_CMDS = re.compile(r"\\(left|right|,|;|!|:)", re.I)
RE_SPACES = re.compile(r"\s+")
RE_BRACES = re.compile(r"[{}]")
RE_PARENS_COMMAs = re.compile(r"[()\[\],]")


def canon_math(value: Optional[str]) -> Optional[str]:
    """
    Permissive canonicalizer for math answers. Lowercases, removes spacing/punctuation,
    simplifies common LaTeX forms, and normalizes pi.
    """
    if value is None:
        return None
    canonical = value.strip()
    canonical = (
        canonical.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("π", "pi")
        .replace("\\pi", "pi")
    )
    canonical = RE_LATEX_CMDS.sub("", canonical)
    canonical = RE_LATEX_FRAC.sub(r"\1/\2", canonical)
    canonical = RE_BRACES.sub("", canonical)
    canonical = RE_SPACES.sub("", canonical)
    canonical = RE_PARENS_COMMAs.sub("", canonical)
    canonical = canonical.replace("\\boxed", "").replace("$", "")
    canonical = canonical.lower().rstrip(".")
    canonical = re.sub(r"/{2,}", "/", canonical)
    canonical = re.sub(r"\+{2,}", "+", canonical)
    canonical = re.sub(r"-{2,}", "-", canonical)
    if canonical.startswith("+"):
        canonical = canonical[1:]
    return canonical


def contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    """Substring check after both sides are canonicalized."""
    return bool(hay and needle and (needle in hay))


# ---------------------------------------------------------------------------
# Torch-agnostic utilities
# ---------------------------------------------------------------------------
def finite_mean(values: Iterable[float]) -> Optional[float]:
    """Return the mean of finite values, ignoring NaNs."""
    finite_values = [
        float(value)
        for value in values
        if not math.isnan(float(value)) and math.isfinite(float(value))
    ]
    return (sum(finite_values) / len(finite_values)) if finite_values else None


def build_entropy_pass_base(
    *,
    prev_output: Optional[str],
    full_text: str,
    pred_answer_text: str,
    pred_canon: Optional[str],
    entropy_overall: Optional[float],
    entropy_think: Optional[float],
    entropy_answer: Optional[float],
) -> Dict[str, Any]:
    """
    Common core for per-pass result dicts with entropy stats and prediction fields.
    """
    return {
        "prev_output": prev_output,
        "output": full_text,
        "pred_answer": pred_answer_text,
        "pred_answer_canon": pred_canon,
        "entropy": entropy_overall,
        "entropy_think": entropy_think,
        "entropy_answer": entropy_answer,
    }


def add_token_and_tag_fields(
    base: Dict[str, Any],
    *,
    tokens_total: int,
    tokens_think: int,
    tokens_answer: int,
    full_text: str,
) -> Dict[str, Any]:
    """
    Add shared token-count and tag-structure fields to a per-pass result dict.
    """
    base.update(
        {
            "tokens_total": tokens_total,
            "tokens_end_think": tokens_think,
            "tokens_think": tokens_think,
            "tokens_answer": tokens_answer,
            "valid_tag_structure": valid_tag_structure(full_text),
        },
    )
    return base


# ---------------------------------------------------------------------------
# Reconsideration patterns
# ---------------------------------------------------------------------------
RECONSIDER_PATTERNS: Sequence[Tuple[str, re.Pattern]] = [
    ("wait_line", re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider", re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    (
        "step_by_step",
        re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I),
    ),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck", re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]


# ---------------------------------------------------------------------------
# Math pass metadata and packing
# ---------------------------------------------------------------------------
@dataclass
class MathPassMeta:
    """Metadata required to pack a single math pass result."""

    problem: str
    canon_gold: Optional[str]
    injected_cue: bool
    prev_output: Optional[str]
    cue_prefix_str: str
    stop_reason_think: Optional[str]
    stop_reason_answer: Optional[str]


@dataclass
class MathTokenStats:
    """Token-count summary for a single math pass."""

    tokens_think: int
    tokens_answer: int
    tokens_total: int


@dataclass
class MathEntropySummary:
    """Entropy statistics for a single math pass."""

    overall: Optional[float]
    think: Optional[float]
    answer: Optional[float]
    pre_cue: Optional[float]
    reconsider_think: Optional[float]
    reconsider_full: Optional[float]


@dataclass
class MathReconsiderationInfo:
    """Reconsideration markers and positions for a math pass."""

    markers: List[str]
    pos_in_think: Optional[int]
    context: Optional[str]
    excerpt: Optional[str]
    t_cue: Optional[int]


def _compute_math_token_stats(
    ent_think: List[float],
    ent_answer: List[float],
) -> Tuple[List[float], MathTokenStats]:
    """Return combined entropy series and basic token counts."""
    tok_ents_all = (ent_think or []) + (ent_answer or [])
    tokens_think = len(ent_think or [])
    tokens_answer = len(ent_answer or [])
    tokens_total = len(tok_ents_all)
    return tok_ents_all, MathTokenStats(
        tokens_think=tokens_think,
        tokens_answer=tokens_answer,
        tokens_total=tokens_total,
    )


def _compute_math_reconsideration_info(
    think_text: str,
    meta: MathPassMeta,
    tokens_think: int,
) -> MathReconsiderationInfo:
    """Derive reconsideration markers, context, and cue index from metadata."""
    skip_chars = len(meta.cue_prefix_str) if meta.injected_cue else 0
    markers, pos_in_think, reconsider_context, reconsider_excerpt = find_markers_and_context(
        think_text,
        f"Problem: {meta.problem}",
        RECONSIDER_PATTERNS,
        skip_prefix_chars=skip_chars,
    )
    if meta.injected_cue:
        markers = ["injected_cue"] + (markers or [])

    t_cue = 0 if meta.injected_cue else None
    if (not meta.injected_cue) and (pos_in_think is not None):
        t_cue = max(0, min(pos_in_think, tokens_think))

    return MathReconsiderationInfo(
        markers=markers or [],
        pos_in_think=pos_in_think,
        context=reconsider_context,
        excerpt=reconsider_excerpt,
        t_cue=t_cue,
    )


def _summarize_math_entropies(
    tok_ents_all: List[float],
    ent_think: List[float],
    ent_answer: List[float],
    stats: MathTokenStats,
    t_cue: Optional[int],
) -> MathEntropySummary:
    """Compute overall and segment-wise entropy summaries for a math pass."""
    tokens_think = stats.tokens_think
    tokens_total = stats.tokens_total
    entropy_overall = finite_mean(tok_ents_all) if tok_ents_all else None
    entropy_think = finite_mean(ent_think) if ent_think else None
    entropy_answer = finite_mean(ent_answer) if ent_answer else None
    entropy_pre_cue = None
    entropy_reconsider_think = None
    entropy_reconsider_full = None

    if t_cue is not None:
        if tokens_total > t_cue:
            entropy_reconsider_full = finite_mean(tok_ents_all[t_cue:])
        if tokens_think > t_cue:
            entropy_reconsider_think = finite_mean(tok_ents_all[t_cue:tokens_think])

    return MathEntropySummary(
        overall=entropy_overall,
        think=entropy_think,
        answer=entropy_answer,
        pre_cue=entropy_pre_cue,
        reconsider_think=entropy_reconsider_think,
        reconsider_full=entropy_reconsider_full,
    )


def build_math_pass_meta(
    *,
    problem: str,
    canon_gold: Optional[str],
    injected_cue: bool,
    prev_output: Optional[str],
    cue_prefix_str: str,
    stop_reason_think: Optional[str],
    stop_reason_answer: Optional[str],
) -> MathPassMeta:
    """Helper to construct MathPassMeta consistently across math inference cores."""
    return MathPassMeta(
        problem=problem,
        canon_gold=canon_gold,
        injected_cue=injected_cue,
        prev_output=prev_output,
        cue_prefix_str=cue_prefix_str,
        stop_reason_think=stop_reason_think,
        stop_reason_answer=stop_reason_answer,
    )


def pack_math_pass_result(
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    meta: MathPassMeta,
) -> Dict[str, Any]:
    """
    Assemble per-pass math result dict with entropy, reconsideration markers,
    and token/tag statistics.
    """
    tok_ents_all, token_stats = _compute_math_token_stats(ent_think, ent_answer)
    think, answer = extract_blocks(full_text)
    think_text = think or ""
    pred_answer_text = answer or ""

    reconsider_info = _compute_math_reconsideration_info(
        think_text,
        meta,
        token_stats.tokens_think,
    )
    entropy_summary = _summarize_math_entropies(
        tok_ents_all=tok_ents_all,
        ent_think=ent_think,
        ent_answer=ent_answer,
        stats=token_stats,
        t_cue=reconsider_info.t_cue,
    )

    pred_canon = canon_math(pred_answer_text)
    is_correct_pred = contains_canon(pred_canon, meta.canon_gold)

    base = build_entropy_pass_base(
        prev_output=meta.prev_output,
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        entropy_overall=entropy_summary.overall,
        entropy_think=entropy_summary.think,
        entropy_answer=entropy_summary.answer,
    )
    base.update(
        {
            "entropy_pre_cue": entropy_summary.pre_cue,
            "entropy_reconsider_think": entropy_summary.reconsider_think,
            "entropy_reconsider_full": entropy_summary.reconsider_full,
            "stop_reason_think": meta.stop_reason_think,
            "stop_reason_answer": meta.stop_reason_answer,
            "has_reconsider_cue": bool(reconsider_info.markers),
            "reconsider_markers": reconsider_info.markers,
            "reconsider_pos": reconsider_info.pos_in_think,
            "reconsider_context": reconsider_info.context,
            "reconsider_excerpt": reconsider_info.excerpt,
            "is_correct_pred": is_correct_pred,
            "is_correct_after_reconsideration": bool(reconsider_info.markers)
            and bool(is_correct_pred),
        },
    )
    return add_token_and_tag_fields(
        base,
        tokens_total=token_stats.tokens_total,
        tokens_think=token_stats.tokens_think,
        tokens_answer=token_stats.tokens_answer,
        full_text=full_text,
    )


__all__ = [
    "RE_THINK",
    "RE_ANSWER",
    "RECONSIDER_PATTERNS",
    "extract_blocks",
    "valid_tag_structure",
    "canon_math",
    "contains_canon",
    "finite_mean",
    "build_entropy_pass_base",
    "add_token_and_tag_fields",
    "MathPassMeta",
    "MathTokenStats",
    "MathEntropySummary",
    "MathReconsiderationInfo",
    "build_math_pass_meta",
    "pack_math_pass_result",
]
