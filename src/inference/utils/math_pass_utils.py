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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.inference.utils.text_utils import find_markers_and_context


# ---------------------------------------------------------------------------
# Text + tag helpers
# ---------------------------------------------------------------------------
RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")


def extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract ``<think>`` and ``<answer>`` contents from a model output.

    :param txt: Full text containing ``<think>…</think>`` and ``<answer>…</answer>`` blocks.
    :returns: Tuple ``(think_text, answer_text)`` with surrounding whitespace stripped,
        or ``(None, None)`` if blocks are not found.
    """
    think = ans = None
    think_match = RE_THINK.search(txt)
    if think_match:
        think = think_match.group(1).strip()
    answer_match = RE_ANSWER.search(txt)
    if answer_match:
        ans = answer_match.group(1).strip()
    return think, ans


def valid_tag_structure(full_text: str) -> bool:
    """
    Check that exactly one ``<think>…</think>`` appears before ``<answer>…</answer>``.

    :param full_text: Full text produced by the model.
    :returns: ``True`` if the tag structure is valid, otherwise ``False``.
    """
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

    :param value: Raw answer text to canonicalize.
    :returns: Canonicalized string, or ``None`` if ``value`` is ``None``.
    """
    if value is None:
        return None
    canonical = value.strip()
    canonical = (
        canonical.replace("–", "-").replace("—", "-").replace("−", "-").replace("π", "pi").replace("\\pi", "pi")
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
    """
    Substring check after both sides are canonicalized.

    :param hay: Canonicalized haystack string.
    :param needle: Canonicalized needle string.
    :returns: ``True`` if ``needle`` is a substring of ``hay``.
    """
    return bool(hay and needle and (needle in hay))


# ---------------------------------------------------------------------------
# Torch-agnostic utilities
# ---------------------------------------------------------------------------
def finite_mean(values: Iterable[float]) -> Optional[float]:
    """
    Compute the mean of finite values, ignoring NaNs and infinities.

    :param values: Iterable of numeric values.
    :returns: Mean of finite values, or ``None`` if no finite values are present.
    """
    finite_values = [float(value) for value in values if not math.isnan(float(value)) and math.isfinite(float(value))]
    return (sum(finite_values) / len(finite_values)) if finite_values else None


def build_entropy_pass_base(
    *,
    prev_output: Optional[str],
    full_text: str,
    pred_answer_text: str,
    pred_canon: Optional[str],
    entropy_summary: "MathEntropySummary",
) -> Dict[str, Any]:
    """
    Common core for per-pass result dicts with entropy stats and prediction fields.

    :param prev_output: Previous pass output text, if any.
    :param full_text: Full model output including tags.
    :param pred_answer_text: Extracted answer text for this pass.
    :param pred_canon: Canonicalized predicted answer.
    :param entropy_summary: Aggregated entropy statistics for this pass.
    :returns: Dictionary containing core prediction and entropy fields.
    """
    return {
        "prev_output": prev_output,
        "output": full_text,
        "pred_answer": pred_answer_text,
        "pred_answer_canon": pred_canon,
        "entropy": entropy_summary.overall,
        "entropy_think": entropy_summary.think,
        "entropy_answer": entropy_summary.answer,
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

    :param base: Base result dictionary to update.
    :param tokens_total: Total number of tokens across think and answer.
    :param tokens_think: Number of tokens in the think segment.
    :param tokens_answer: Number of tokens in the answer segment.
    :param full_text: Full model output including tags for structure checks.
    :returns: The updated result dictionary.
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


def finalize_pass_fields(
    base: Dict[str, Any],
    reconsideration_fields: Dict[str, Any],
    *,
    token_stats: MathTokenStats,
    full_text: str,
) -> Dict[str, Any]:
    """
    Merge reconsideration data and token/tag metadata into a pass result dict.
    """
    base.update(reconsideration_fields)
    return add_token_and_tag_fields(
        base,
        tokens_total=token_stats.tokens_total,
        tokens_think=token_stats.tokens_think,
        tokens_answer=token_stats.tokens_answer,
        full_text=full_text,
    )


# ---------------------------------------------------------------------------
# Reconsideration patterns / default cues
# ---------------------------------------------------------------------------
DEFAULT_SECOND_PASS_PHRASE = " ||| ".join(
    [
        ("Hold on, this reasoning might be wrong. Let's go back and check each step carefully."),
        (
            "Actually, this approach doesn't look correct. "
            "Let's restart and work through the solution more systematically."
        ),
        "Wait, we need to reconsider. Let's think this through step by step.",
    ],
)


def build_second_pass_cue_strings(phrase: Optional[str]) -> List[str]:
    """
    Parse a configurable second-pass phrase into a list of cue strings.

    Accepts a raw phrase where multiple cues are separated by ``\"|||\"``.
    Returns each cue with a trailing space, matching the behaviour used by the
    inference loops.

    :param phrase: Raw configuration string for second-pass cues.
    :returns: List of cue strings with trailing spaces.
    """
    raw_phrase = phrase or ""
    if "|||" in raw_phrase:
        cue_phrases = [part.strip() for part in raw_phrase.split("|||") if part.strip()]
    else:
        cue_phrases = [raw_phrase.strip()] if raw_phrase.strip() else []
    return [cue + " " for cue in cue_phrases]


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
    (
        "hold_on_reasoning_wrong",
        re.compile(r"\bhold on\b.*\breasoning\b.*\bwrong\b", re.I | re.S),
    ),
    (
        "approach_not_correct",
        re.compile(
            r"\bactually\b.*\bapproach\b.*(doesn['’]?t|does not)\s+look\s+correct",
            re.I | re.S,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Math pass metadata and packing
# ---------------------------------------------------------------------------
@dataclass
class MathPassMeta:
    """
    Metadata required to pack a single math pass result.

    :param problem: Normalized problem text.
    :param canon_gold: Canonicalized gold answer, if available.
    :param injected_cue: Whether a reconsideration cue was injected.
    :param prev_output: Previous pass output shown to the model.
    :param cue_prefix_str: Cue prefix string added before reconsideration text.
    :param stop_reason_think: Stop reason for the think phase.
    :param stop_reason_answer: Stop reason for the answer phase.
    """

    problem: str
    canon_gold: Optional[str]
    injected_cue: bool
    prev_output: Optional[str]
    cue_prefix_str: str
    stop_reason_think: Optional[str]
    stop_reason_answer: Optional[str]


@dataclass
class MathTokenStats:
    """
    Token-count summary for a single math pass.

    :param tokens_think: Number of tokens in the think segment.
    :param tokens_answer: Number of tokens in the answer segment.
    :param tokens_total: Total number of tokens across think and answer.
    """

    tokens_think: int
    tokens_answer: int
    tokens_total: int


@dataclass
class MathEntropySummary:
    """
    Entropy statistics for a single math pass.

    :param overall: Overall token-level entropy.
    :param think: Entropy restricted to the think segment.
    :param answer: Entropy restricted to the answer segment.
    :param pre_cue: Entropy before any reconsideration cue, if applicable.
    :param reconsider_think: Entropy over reconsideration tokens in think.
    :param reconsider_full: Entropy over reconsideration tokens in think+answer.
    """

    overall: Optional[float]
    think: Optional[float]
    answer: Optional[float]
    pre_cue: Optional[float]
    reconsider_think: Optional[float]
    reconsider_full: Optional[float]


@dataclass
class MathReconsiderationInfo:
    """
    Reconsideration markers and positions for a math pass.

    :param markers: List of reconsideration marker labels.
    :param pos_in_think: Character offset of the first reconsideration marker.
    :param context: Surrounding context string for the marker.
    :param excerpt: Short excerpt around the marker.
    :param t_cue: Token index at which reconsideration begins, if known.
    """

    markers: List[str]
    pos_in_think: Optional[int]
    context: Optional[str]
    excerpt: Optional[str]
    t_cue: Optional[int]


def _compute_math_token_stats(
    ent_think: List[float],
    ent_answer: List[float],
) -> Tuple[List[float], MathTokenStats]:
    """
    Compute combined entropy series and basic token counts.

    :param ent_think: Entropy values for tokens in the think segment.
    :param ent_answer: Entropy values for tokens in the answer segment.
    :returns: Tuple ``(tok_ents_all, stats)`` with concatenated entropies and
        a :class:`MathTokenStats` summary.
    """
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
    """
    Derive reconsideration markers, context, and cue index from metadata.

    :param think_text: Text inside the ``<think>`` block.
    :param meta: Pass metadata describing cues and problem text.
    :param tokens_think: Number of think tokens used for cue indexing.
    :returns: :class:`MathReconsiderationInfo` with markers and positions.
    """
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
    """
    Compute overall and segment-wise entropy summaries for a math pass.

    :param tok_ents_all: Entropy values for all tokens.
    :param ent_think: Entropy values for tokens in the think segment.
    :param ent_answer: Entropy values for tokens in the answer segment.
    :param stats: Token-count summary for the pass.
    :param t_cue: Token index from which reconsideration begins, if any.
    :returns: :class:`MathEntropySummary` with aggregated statistics.
    """
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


def build_reconsideration_fields(
    *,
    stop_reasons: Mapping[str, Optional[str]],
    reconsideration: Optional[Mapping[str, Any]],
    is_correct_pred: bool,
    entropy_summary: Optional[MathEntropySummary] = None,
    enumeration: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble reconsideration-related fields shared by math and crossword passes.
    """
    reconsideration = reconsideration or {}
    markers_list = list(reconsideration.get("markers", [])) if reconsideration else []
    reconsider_context = reconsideration.get(
        "reconsider_context",
        reconsideration.get("context"),
    )
    reconsider_excerpt = reconsideration.get(
        "reconsider_excerpt",
        reconsideration.get("excerpt"),
    )
    fields: Dict[str, Any] = {
        "stop_reason_think": stop_reasons.get("think"),
        "stop_reason_answer": stop_reasons.get("answer"),
        "has_reconsider_cue": bool(markers_list),
        "reconsider_markers": markers_list,
        "reconsider_pos": reconsideration.get("pos_in_think"),
        "reconsider_context": reconsider_context,
        "reconsider_excerpt": reconsider_excerpt,
        "is_correct_pred": is_correct_pred,
        "is_correct_after_reconsideration": bool(markers_list) and bool(is_correct_pred),
    }
    if enumeration is not None:
        fields["enumeration"] = enumeration
    if entropy_summary is not None:
        fields.update(
            {
                "entropy_pre_cue": entropy_summary.pre_cue,
                "entropy_reconsider_think": entropy_summary.reconsider_think,
                "entropy_reconsider_full": entropy_summary.reconsider_full,
            },
        )
    return fields


def build_reconsideration_base_kwargs(
    *,
    prev_output: Optional[str],
    full_text: str,
    pred_answer_text: str,
    pred_canon: str,
    entropy_summary: Optional[MathEntropySummary],
) -> Dict[str, Any]:
    """
    Common kwargs used when assembling reconsideration-aware pass results.
    """
    return {
        "prev_output": prev_output,
        "full_text": full_text,
        "pred_answer_text": pred_answer_text,
        "pred_canon": pred_canon,
        "entropy_summary": entropy_summary,
    }


def build_stop_reasons(
    stop_think: Optional[str],
    stop_answer: Optional[str],
) -> Dict[str, Optional[str]]:
    """Helper to normalize stop reasons into a consistent mapping."""
    return {
        "think": stop_think,
        "answer": stop_answer,
    }


@dataclass(frozen=True)
class ReconsiderationInputs:
    """
    Normalized inputs used to package reconsideration-aware pass results.
    """

    base_fields: Mapping[str, Any]
    stop_reasons: Mapping[str, Optional[str]]
    reconsideration: Optional[Mapping[str, Any]]
    is_correct_pred: bool
    token_stats: MathTokenStats
    enumeration: Optional[str] = None


def assemble_reconsideration_pass_result(inputs: ReconsiderationInputs) -> Dict[str, Any]:
    """
    Shared helper to package entropy/reconsideration fields into a pass result.
    """
    base = build_entropy_pass_base(
        prev_output=inputs.base_fields.get("prev_output"),
        full_text=inputs.base_fields["full_text"],
        pred_answer_text=inputs.base_fields["pred_answer_text"],
        pred_canon=inputs.base_fields["pred_canon"],
        entropy_summary=inputs.base_fields.get("entropy_summary"),
    )
    reconsideration_fields = build_reconsideration_fields(
        stop_reasons=inputs.stop_reasons,
        reconsideration=inputs.reconsideration,
        is_correct_pred=inputs.is_correct_pred,
        entropy_summary=inputs.base_fields.get("entropy_summary"),
        enumeration=inputs.enumeration,
    )
    return finalize_pass_fields(
        base,
        reconsideration_fields,
        token_stats=inputs.token_stats,
        full_text=inputs.base_fields["full_text"],
    )


def _build_math_pass_meta_kwonly(meta_kwargs: Dict[str, Any]) -> MathPassMeta:
    """
    Internal helper that constructs :class:`MathPassMeta` from keyword-only args.
    """
    required_keys = {
        "problem",
        "canon_gold",
        "injected_cue",
        "prev_output",
        "cue_prefix_str",
        "stop_reason_think",
        "stop_reason_answer",
    }
    missing = required_keys.difference(meta_kwargs)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise TypeError(f"build_math_pass_meta() missing required keyword arguments: {missing_list}")

    unexpected = set(meta_kwargs).difference(required_keys)
    if unexpected:
        unexpected_list = ", ".join(sorted(unexpected))
        raise TypeError(f"build_math_pass_meta() got unexpected keyword arguments: {unexpected_list}")

    return MathPassMeta(**meta_kwargs)


def build_math_pass_meta(*args, **kwargs) -> MathPassMeta:
    """
    Helper to construct :class:`MathPassMeta` consistently across math inference cores.

    This function supports both the newer keyword-only style::

        build_math_pass_meta(
            problem=problem,
            canon_gold=canon_gold,
            injected_cue=injected_cue,
            prev_output=prev_output,
            cue_prefix_str=cue_prefix_str,
            stop_reason_think=stop_reason_think,
            stop_reason_answer=stop_reason_answer,
        )

    and the older positional style used in legacy call sites::

        build_math_pass_meta(
            problem,
            canon_gold,
            injected_cue,
            prev_output,
            cue_prefix_str,
            stop_reason_think,
            stop_reason_answer,
        )

    :param args: Optional positional arguments following the legacy ordering.
    :param kwargs: Keyword arguments for the keyword-only interface.
    :returns: Constructed :class:`MathPassMeta` instance.
    :raises TypeError: If both positional and keyword arguments are mixed or an
        unexpected number of positional arguments is provided.
    """
    if args and kwargs:
        raise TypeError("build_math_pass_meta() cannot mix positional and keyword arguments")

    if args:
        if len(args) != 7:
            raise TypeError(
                f"build_math_pass_meta() expected 7 positional arguments, got {len(args)}",
            )
        (
            problem,
            canon_gold,
            injected_cue,
            prev_output,
            cue_prefix_str,
            stop_reason_think,
            stop_reason_answer,
        ) = args
        kwargs = {
            "problem": problem,
            "canon_gold": canon_gold,
            "injected_cue": injected_cue,
            "prev_output": prev_output,
            "cue_prefix_str": cue_prefix_str,
            "stop_reason_think": stop_reason_think,
            "stop_reason_answer": stop_reason_answer,
        }

    return _build_math_pass_meta_kwonly(kwargs)


def pack_math_pass_result(
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    meta: MathPassMeta,
) -> Dict[str, Any]:
    """
    Assemble per-pass math result dict with entropy, reconsideration markers,
    and token/tag statistics.

    :param full_text: Full model output including tags.
    :param ent_think: Entropy values for tokens in the think segment.
    :param ent_answer: Entropy values for tokens in the answer segment.
    :param meta: Pass metadata describing the problem and cue configuration.
    :returns: Dictionary representing a single math pass result.
    """
    tok_ents_all, token_stats = _compute_math_token_stats(ent_think, ent_answer)
    think_text, pred_answer_text = (segment or "" for segment in extract_blocks(full_text))

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

    base_fields = build_reconsideration_base_kwargs(
        prev_output=meta.prev_output,
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        entropy_summary=entropy_summary,
    )
    inputs = ReconsiderationInputs(
        base_fields=base_fields,
        stop_reasons=build_stop_reasons(
            meta.stop_reason_think,
            meta.stop_reason_answer,
        ),
        reconsideration=vars(reconsider_info),
        is_correct_pred=is_correct_pred,
        token_stats=token_stats,
    )
    return assemble_reconsideration_pass_result(inputs)
