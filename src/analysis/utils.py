#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines

"""
Shared low-level helpers for analysis scripts.

This module intentionally keeps dependencies light and behavior conservative so
it can be safely imported from many one-off scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# Allow references like ``src.analysis.utils.utils`` in environments that expect
# module-name aliases to be present on the module object.
utils = sys.modules[__name__]
_UNCERTAINTY_MODULE = None


def _load_uncertainty_module():
    """
    Lazy-load the uncertainty helpers to avoid mandatory heavy deps at import time.
    """
    global _UNCERTAINTY_MODULE  # pylint: disable=global-statement
    if _UNCERTAINTY_MODULE is None:
        _UNCERTAINTY_MODULE = import_module("src.analysis.common.uncertainty")
    return _UNCERTAINTY_MODULE


def standardize_uncertainty(
    data,
    source_col: str = "uncertainty",
    dest_col: str = "uncertainty_std",
):
    """
    Proxy to :func:`src.analysis.common.uncertainty.standardize_uncertainty`.

    Imported lazily to avoid mandatory pandas dependency at module import time.
    """
    return _load_uncertainty_module().standardize_uncertainty(
        data,
        source_col=source_col,
        dest_col=dest_col,
    )


def standardize_uncertainty_with_stats(
    data,
    source_col: str = "uncertainty",
    dest_col: str = "uncertainty_std",
):
    """
    Proxy to :func:`src.analysis.common.uncertainty.standardize_uncertainty_with_stats`.
    """
    return _load_uncertainty_module().standardize_uncertainty_with_stats(
        data,
        source_col=source_col,
        dest_col=dest_col,
    )


# ---------------------------------------------------------------------------
# Path / naming helpers
# ---------------------------------------------------------------------------

# Flexible step patterns used across scripts (dir names, file names, substrings)
STEP_PATS = [
    re.compile(r"step\s*[-_]?(?P<step>\d+)", re.I),
    re.compile(r"global[_-]?step\s*[-_]?(?P<step>\d+)", re.I),
    re.compile(r"checkpoint\s*[-_]?(?P<step>\d+)", re.I),
    # bare numeric subdirs at the end of the path (e.g., ".../01234/")
    re.compile(r"/(?P<step>\d{2,5})(?=/|$)"),
]

# Common temperature tokens in directory names
TEMP_PATS = [
    re.compile(r"(?:^|[-_])temp[-_](?P<t>low|[0-9]+(?:\.[0-9]+)?)$", re.I),
    re.compile(r"(?:^|[-_])(?P<t>low|[0-9]+(?:\.[0-9]+)?)[-_]temp$", re.I),
    re.compile(r"(?:^|[-_])low[-_]temp$", re.I),
]


def nat_step_from_path(path: str) -> Optional[int]:
    """
    Try to infer an integer training step from a path or filename.

    Uses a collection of regexes that cover the various conventions used in
    analysis scripts (step123, global_step-123, checkpoint_123, ...).
    """
    path_str = str(path)
    for pattern in STEP_PATS:
        match = pattern.search(path_str)
        if not match:
            continue
        step_str = match.groupdict().get("step") or match.group(1)
        try:
            return int(step_str)
        except (TypeError, ValueError):
            # Fall through to try other patterns
            continue
    return None


def step_from_rec_or_path(rec: Dict[str, Any], path: str) -> int:
    """
    Extract an integer training step from a record, falling back to the file
    path via :func:`nat_step_from_path` when needed.
    """
    step = rec.get("step") or rec.get("global_step") or rec.get("training_step")
    if step is None:
        step = nat_step_from_path(path)
    try:
        return int(step) if step is not None else 0
    except (TypeError, ValueError):
        return 0


def add_domain_root_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register the standard ``--roots_*`` arguments used by many analysis CLIs.

    The helper returns ``parser`` to support fluent-style setup.
    """
    parser.add_argument(
        "--roots_crossword",
        nargs="+",
        default=[],
        help="Paths or globs to Crossword results",
    )
    parser.add_argument(
        "--roots_math",
        nargs="+",
        default=[],
        help="Paths or globs to Math results",
    )
    parser.add_argument(
        "--roots_carpark",
        nargs="+",
        default=[],
        help="Paths or globs to Carpark results",
    )
    return parser


def parse_temp_from_dir(dirname: str, low_alias: float) -> Optional[float]:
    """
    Infer temperature from a directory name.

    Supports tokens like:
      temp-0.7, 0.7-temp, temp-low, low-temp
    """
    dirname_lower = dirname.lower()
    for pattern in TEMP_PATS:
        match = pattern.search(dirname_lower)
        if not match:
            continue
        token = match.groupdict().get("t", "low").lower()
        if token == "low":
            return float(low_alias)
        try:
            return float(token)
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Generic coercion / text helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """
    Convert an arbitrary string into a filesystem-friendly slug.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def coerce_bool(value: Any) -> Optional[int]:
    """
    Best-effort conversion to {0,1}.

    Returns:
      1/0 for recognized truthy/falsey values, or None if there is no
      reasonable interpretation (e.g., complex structures).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(bool(value))
    if isinstance(value, str):
        string_value = value.strip().lower()
        if string_value in {"1", "true", "t", "yes", "y"}:
            return 1
        if string_value in {"0", "false", "f", "no", "n"}:
            return 0

    return int(bool(value))


def truthy_flag(value: Any) -> bool:
    """
    Return True for common \"truthy\" encodings (booleans, ints, strings).
    """
    if value is True:
        return True
    if isinstance(value, (int, np.integer, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def coerce_float(value: Any) -> Optional[float]:
    """
    Best-effort conversion to float, returning None on failure instead of
    raising.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def both_get(
    pass1_data: Dict[str, Any],
    record: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Best-effort getter that prefers ``pass1_data[key]`` but falls back to the
    top-level record when missing.
    """
    value = pass1_data.get(key)
    return value if value is not None else record.get(key, default)


def lighten_hex(hex_color: str, factor: float = 0.65) -> str:
    """
    Lighten a hex color by interpolating towards white.
    """
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    red = 1 - (1 - red) * factor
    green = 1 - (1 - green) * factor
    blue = 1 - (1 - blue) * factor
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def combined_entropy_from_pass1(pass1: Dict[str, Any]) -> Optional[float]:
    """
    Prefer combined 'entropy' if present; else average of think/answer when
    both are available, else the first available of (think, answer).
    """
    entropy_overall = coerce_float(pass1.get("entropy"))
    entropy_think = coerce_float(pass1.get("entropy_think"))
    entropy_answer = coerce_float(pass1.get("entropy_answer"))
    if entropy_overall is not None:
        return entropy_overall
    if entropy_think is not None and entropy_answer is not None:
        return 0.5 * (entropy_think + entropy_answer)
    return entropy_think if entropy_think is not None else entropy_answer


def entropy_from_pass1(
    pass1: Dict[str, Any],
    mode: str = "sum",
) -> Optional[float]:
    """
    Compute a pass-1 entropy summary under a chosen mode.

    Modes:
      - ``\"sum\"``      : ``entropy_think + entropy_answer`` when available,
                           otherwise fall back to ``entropy``.
      - ``\"think\"``    : ``entropy_think`` with fallback to ``entropy``.
      - ``\"answer\"``   : ``entropy_answer`` with fallback to ``entropy``.
      - ``\"combined\"`` : use :func:`combined_entropy_from_pass1`.
      - anything else    : fall back to the raw ``entropy`` field.
    """
    ent_think = coerce_float(pass1.get("entropy_think"))
    ent_answer = coerce_float(pass1.get("entropy_answer"))
    ent_overall = coerce_float(pass1.get("entropy"))

    if mode == "sum":
        parts = [value for value in (ent_think, ent_answer) if value is not None]
        if parts:
            return float(sum(parts))
        return ent_overall
    if mode == "think":
        return ent_think if ent_think is not None else ent_overall
    if mode == "answer":
        return ent_answer if ent_answer is not None else ent_overall
    if mode == "combined":
        return combined_entropy_from_pass1(pass1)
    return ent_overall


def choose_uncertainty(
    pass1: Dict[str, Any],
    pref: str = "answer",
) -> Optional[float]:
    """
    Pick an entropy-style uncertainty value from a PASS-1 dict.

    The ``pref`` argument controls the priority order among
    ``entropy_answer``, ``entropy`` and ``entropy_think`` while treating
    valid zeros as meaningful (unlike ``or``-chaining).
    """
    if pref == "answer":
        value = pass1.get("entropy_answer")
        if value is None:
            value = pass1.get("entropy")
        if value is None:
            value = pass1.get("entropy_think")
        return float(value) if value is not None else None
    if pref == "overall":
        value = pass1.get("entropy")
        if value is None:
            value = pass1.get("entropy_answer")
        if value is None:
            value = pass1.get("entropy_think")
        return float(value) if value is not None else None
    if pref == "think":
        value = pass1.get("entropy_think")
        if value is None:
            value = pass1.get("entropy")
        if value is None:
            value = pass1.get("entropy_answer")
        return float(value) if value is not None else None
    return None


def extract_pass1_and_step(
    record: Dict[str, Any],
    step_from_name: Optional[int],
) -> Tuple[Dict[str, Any], Optional[int]]:
    """
    Extract a robust PASS-1 object and integer step from a record.

    Returns (pass1_dict, step_int); if either is unavailable or invalid,
    returns ({}, None).
    """
    pass1_data = record.get("pass1") or record.get("p1") or record.get("first_pass") or {}
    if not isinstance(pass1_data, dict):
        pass1_data = {}
    if not pass1_data:
        return {}, None

    step_raw = record.get(
        "step",
        step_from_name if step_from_name is not None else None,
    )
    if step_raw is None:
        return {}, None
    try:
        step_int = int(step_raw)
    except (TypeError, ValueError):
        return {}, None
    return pass1_data, step_int


@dataclass
class FormalThresholds:
    """
    Threshold configuration used by Formal Aha helpers.

    ``delta1`` and ``delta2`` control prior failure/stability; ``min_prior_steps``
    is the minimum number of previous steps required; ``delta3`` is an optional
    gain threshold used only by :func:`formal_flags_with_gain`.
    """

    delta1: float
    delta2: float
    min_prior_steps: int
    delta3: Optional[float] = None


def formal_prior_ok(
    freq: np.ndarray,
    rate: np.ndarray,
    index: int,
    thresholds: FormalThresholds,
) -> bool:
    """
    Shared check for Formal Aha marking:
      - require at least ``min_prior_steps`` previous steps; and
      - prior failure: max(freq[:index]) < delta1; and
      - prior stability: max(rate[:index]) < delta2.
    """
    if index < thresholds.min_prior_steps:
        return False
    prior_fail = float(np.max(freq[:index])) < thresholds.delta1
    prior_stable = float(np.max(rate[:index])) < thresholds.delta2
    return prior_fail and prior_stable


def formal_flags_from_series(
    freq: np.ndarray,
    rate: np.ndarray,
    shift: np.ndarray,
    thresholds: FormalThresholds,
) -> np.ndarray:
    """
    Compute Formal Aha flags for a single (problem, step) series.

    Returns an ``int`` array of the same length as ``freq`` / ``rate`` /
    ``shift`` with 1 where the Formal criteria are satisfied and 0 otherwise.
    """
    num_steps = len(freq)
    flags = np.zeros(num_steps, dtype=int)
    for index in range(num_steps):
        prior_ok = formal_prior_ok(freq, rate, index, thresholds)
        flags[index] = int(prior_ok and (shift[index] == 1))
    return flags


def formal_flags_with_gain(
    freq: np.ndarray,
    rate: np.ndarray,
    shift: np.ndarray,
    p_plus: np.ndarray,
    thresholds: FormalThresholds,
) -> np.ndarray:
    """
    Formal flags that additionally enforce a minimum gain over prior accuracy.

    When ``delta3`` is None, this reduces to the standard Formal Aha criteria;
    otherwise it also requires ``p_correct_given_shift - freq_correct > delta3``.
    """
    num_steps = len(freq)
    flags = np.zeros(num_steps, dtype=int)
    for index in range(num_steps):
        prior_ok = formal_prior_ok(freq, rate, index, thresholds)
        if not (prior_ok and (shift[index] == 1)):
            flags[index] = 0
        elif thresholds.delta3 is None:
            flags[index] = 1
        else:
            flags[index] = int(
                np.isfinite(p_plus[index]) and ((p_plus[index] - freq[index]) > thresholds.delta3),
            )
    return flags


def apply_formal_marking(
    problem_step_df: Any,
    *,
    delta1: float,
    delta2: float,
    min_prior_steps: int,
    mark_formal_fn: Callable[[Any, float, float, int], Any],
) -> Any:
    """
    Convenience wrapper to apply Formal Aha thresholds with a module-specific
    ``mark_formal`` implementation.
    """
    return mark_formal_fn(
        problem_step_df,
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
    )


def gpt_keys_for_mode(gpt_mode: str) -> list[str]:
    """
    Standard GPT shift label keys for a given mode.

    ``gpt_mode`` should be ``"canonical"`` or ``"broad"``.
    """
    base = ["change_way_of_thinking", "shift_in_reasoning_v1"]
    if gpt_mode == "canonical":
        return base
    return base + ["shift_llm", "shift_gpt", "pivot_llm", "rechecked"]


def get_aha_gpt_flag(
    pass1_dict: Optional[Dict[str, Any]],
    record: Optional[Dict[str, Any]],
) -> Optional[int]:
    """
    Return the first truthy GPT/LLM shift flag available on ``pass1`` or root.
    """
    pass1_dict = pass1_dict or {}
    record = record or {}
    candidates = [
        ("pass1", "shift_in_reasoning_v1"),
        ("pass1", "shift_llm"),
        ("pass1", "shift_gpt"),
        ("pass1", "pivot_llm"),
        ("pass1", "rechecked"),
        ("record", "rechecked"),
        ("pass1", "change_way_of_thinking"),
        ("record", "change_way_of_thinking"),
    ]
    for location, key in candidates:
        if location == "pass1":
            source = pass1_dict
        else:
            source = record
        value = source.get(key)
        if value is None:
            continue
        parsed = coerce_bool(value)
        if parsed is not None:
            return int(parsed)
    return None


def build_roots_by_temp_from_templates(
    temps: list[float],
    **template_kwargs: Optional[str],
) -> dict[float, dict[str, str]]:
    """
    Helper used by temp_graph / temperature_effects to turn a list of
    temperatures + directory templates into a mapping:

      { T_float: {domain_name -> existing_path, ...}, ... }

    Only keeps entries where the formatted path exists.
    """
    crossword_tpl = template_kwargs.get("crossword_tpl")
    math_tpl = template_kwargs.get("math_tpl")
    math2_tpl = template_kwargs.get("math2_tpl")
    math3_tpl = template_kwargs.get("math3_tpl")
    carpark_tpl = template_kwargs.get("carpark_tpl")

    roots_by_temp: dict[float, dict[str, str]] = {}
    for temp_value in temps:
        domain_paths: dict[str, Optional[str]] = {}
        if crossword_tpl:
            crossword_path = crossword_tpl.format(T=temp_value)
            domain_paths["Crossword"] = crossword_path if os.path.isdir(crossword_path) else None
        if math_tpl:
            math_path = math_tpl.format(T=temp_value)
            domain_paths["Math"] = math_path if os.path.isdir(math_path) else None
        if math2_tpl:
            math2_path = math2_tpl.format(T=temp_value)
            domain_paths["Math2"] = math2_path if os.path.isdir(math2_path) else None
        if math3_tpl:
            math3_path = math3_tpl.format(T=temp_value)
            domain_paths["Math3"] = math3_path if os.path.isdir(math3_path) else None
        if carpark_tpl:
            carpark_path = carpark_tpl.format(T=temp_value)
            domain_paths["Carpark"] = carpark_path if os.path.isdir(carpark_path) else None
        roots_by_temp[float(temp_value)] = {
            domain_name: domain_path for domain_name, domain_path in domain_paths.items() if domain_path
        }
    return roots_by_temp


def add_results_root_and_split_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the common ``results_root`` positional and optional ``--split`` flag.

    Used by several RQ entrypoints to keep CLI surfaces consistent.
    """
    parser.add_argument(
        "results_root",
        help=("Root directory containing step*/.../*.jsonl (e.g., artifacts/results/GRPO-1.5B-math-temp-0.05-3)."),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )


def add_optional_results_root_and_split_args(parser: argparse.ArgumentParser) -> None:
    """
    Add an optional ``results_root`` positional and ``--split`` flag.

    This matches the looser CLI used by some plotting scripts where
    ``results_root`` can be omitted in favor of explicit per-domain roots.
    """
    parser.add_argument("results_root", nargs="?", default=None)
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )


def add_results_root_split_and_output_args(
    parser: argparse.ArgumentParser,
    *,
    dataset_default: str,
    model_default: str,
    results_root_optional: bool = False,
) -> None:
    """
    Add a ``results_root`` positional (required or optional), ``--split``,
    and basic output metadata flags used by multiple RQ-style scripts.

    This combines :func:`add_results_root_and_split_args` /
    :func:`add_optional_results_root_and_split_args` with shared
    ``--out_dir``, ``--dataset_name`` and ``--model_name`` arguments.
    """
    if results_root_optional:
        add_optional_results_root_and_split_args(parser)
    else:
        add_results_root_and_split_args(parser)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default=dataset_default)
    parser.add_argument("--model_name", default=model_default)


def add_split_and_out_dir_args(
    parser: argparse.ArgumentParser,
    out_dir_help: str,
) -> None:
    """
    Add shared ``--split`` and ``--out_dir`` arguments used by RQ-style drivers.
    """
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help=out_dir_help,
    )


def add_standard_domain_root_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared ``--root_*`` domain root arguments plus an optional
    ``results_root`` fallback used by several histogram/conditional scripts.
    """
    parser.add_argument("--root_crossword", type=str, default=None)
    parser.add_argument("--root_math", type=str, default=None)
    parser.add_argument(
        "--root_math2",
        type=str,
        default=None,
        help="Optional second Math model folder.",
    )
    parser.add_argument("--root_carpark", type=str, default=None)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=None,
        help="Fallback single root if domain-specific roots are not provided.",
    )


def add_mixed_results_root_and_domain_args(
    parser: argparse.ArgumentParser,
    *,
    dataset_default: str = "MIXED",
    model_default: str = "Qwen2.5-1.5B",
) -> None:
    """
    Add shared domain root arguments plus optional ``results_root``, ``--split``,
    and basic output metadata for mixed-domain plots.
    """
    add_standard_domain_root_args(parser)
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default=dataset_default)
    parser.add_argument("--model_name", default=model_default)


def build_mixed_root_arg_parser() -> argparse.ArgumentParser:
    """
    Construct a base ArgumentParser with mixed-domain root and output args.
    """
    parser = argparse.ArgumentParser()
    add_mixed_results_root_and_domain_args(parser)
    return parser


def add_gpt_mode_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared ``--gpt_mode`` argument used across analysis scripts.
    """
    parser.add_argument(
        "--gpt_mode",
        choices=["canonical", "broad"],
        default="canonical",
    )


def add_split_arg(
    parser: argparse.ArgumentParser,
    *,
    default: Optional[str] = "test",
    help_text: str = "Dataset split substring to filter filenames (e.g., 'test').",
) -> None:
    """
    Add a ``--split`` argument with a configurable default and help text.

    Most ad-hoc analysis scripts need the same ``--split`` flag; centralizing
    it avoids repeating the literal ``add_argument`` block everywhere.
    """
    parser.add_argument("--split", type=str, default=default, help=help_text)


def add_split_and_gpt_mode_args(
    parser: argparse.ArgumentParser,
    *,
    split_default: Optional[str] = "test",
) -> None:
    """
    Convenience helper that adds both ``--split`` and ``--gpt_mode`` arguments.
    """
    add_split_arg(parser, default=split_default)
    add_gpt_mode_arg(parser)


def add_carpark_threshold_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared Carpark soft-reward threshold arguments.
    """
    parser.add_argument(
        "--carpark_success_op",
        choices=["gt", "ge", "eq"],
        default="gt",
    )
    parser.add_argument(
        "--carpark_soft_threshold",
        type=float,
        default=0.0,
    )


def add_gpt_step_and_carpark_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared GPT / step-range / Carpark threshold arguments used by
    multiple plotting scripts (graph_1, temp_graph).
    """
    add_gpt_mode_arg(parser)
    parser.add_argument("--no_gpt_subset_native", action="store_true")
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    add_carpark_threshold_args(parser)


def add_gpt_label_policy_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared GPT label policy arguments used by H1/H3-style analyses.
    """
    add_gpt_mode_arg(parser)
    parser.add_argument(
        "--no_gate_gpt_by_words",
        action="store_true",
        help=("If set, GPT shifts are NOT restricted to samples that also have Words-cue."),
    )


def add_run_rq2_flag(parser: argparse.ArgumentParser, help_suffix: str) -> None:
    """
    Add the shared ``--run_rq2`` flag used by figure-style scripts that can
    optionally invoke the RQ2 analysis first.
    """
    parser.add_argument(
        "--run_rq2",
        action="store_true",
        help=("Also run the RQ2 training-stage analysis (src.analysis.rq2_analysis) " + help_suffix),
    )


def add_formal_threshold_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared formal-aha threshold arguments (δ1, δ2, min_prior_steps).
    """
    parser.add_argument("--delta1", type=float, default=0.13)
    parser.add_argument("--delta2", type=float, default=0.13)
    parser.add_argument("--min_prior_steps", type=int, default=2)


def add_uncertainty_field_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared ``--unc_field`` argument used by uncertainty plots.

    The field controls which entropy-style quantity to treat as the primary
    uncertainty measure: answer, overall, or think.
    """
    parser.add_argument(
        "--unc_field",
        choices=["answer", "overall", "think"],
        default="answer",
    )


def add_common_plot_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared output/figure arguments used by multiple analysis scripts.

    These are:
      --out_dir, --dataset_name, --model_name, --dpi, --make_plot
    """
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="MIXED")
    parser.add_argument("--model_name", type=str, default="MODEL")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--make_plot", action="store_true")


def add_temp_scan_args(
    parser: argparse.ArgumentParser,
    include_math3: bool = True,
) -> None:
    """
    Add shared temperature / template CLI arguments used by temp_graph and
    temperature_effects.
    """
    parser.add_argument("--temps", nargs="+", type=float, required=True)
    parser.add_argument("--scan_root", type=str, default=None)

    # Template inputs (optional)
    parser.add_argument("--crossword_tpl", type=str, default=None)
    parser.add_argument("--math_tpl", type=str, default=None)
    parser.add_argument("--math2_tpl", type=str, default=None)
    if include_math3:
        parser.add_argument("--math3_tpl", type=str, default=None)
    parser.add_argument("--carpark_tpl", type=str, default=None)


def build_results_root_argv(results_root: str, split: Optional[str]) -> List[str]:
    """
    Helper to build a standard [results_root, --split ...] argv list for
    delegating into H1/H3-style scripts.
    """
    argv: List[str] = [results_root]
    if split:
        argv += ["--split", split]
    return argv


def run_module_main_with_argv(module_main, argv: List[str], prog: str) -> None:
    """
    Invoke a module-style ``main`` with a synthetic ``sys.argv``, preserving
    the caller's arguments.

    Shared by RQ-style orchestration modules so they do not need to reimplement
    the same wrapper.
    """
    old_argv = list(sys.argv)
    sys.argv = [prog] + argv
    try:
        module_main()
    finally:
        sys.argv = old_argv


def load_legacy_main_from_path(
    script_path: Path,
    import_name: str,
) -> Callable[[], Any]:
    """
    Shared helper for legacy wrappers (figure-*, math_plots, h4_analysis).

    Loads a legacy script by path and returns its ``main`` callable, raising
    ``SystemExit`` with a clear message if the script cannot be imported or
    does not define ``main``.
    """
    if not script_path.is_file():
        raise SystemExit(f"Legacy script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(import_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    if not hasattr(module, "main"):
        raise SystemExit(f"{script_path} does not define a main() function.")
    return getattr(module, "main")


def parse_passes_argument(passes_arg: str) -> List[str]:
    """
    Parse a comma-separated ``--passes`` argument into a non-empty list.
    """
    passes = [token.strip() for token in (passes_arg or "").split(",") if token.strip()]
    if not passes:
        raise SystemExit("Must specify at least one pass key via --passes.")
    return passes


def get_problem_id(rec: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic problem identifier that works across math/xword/carpark.

    Prefers explicit problem-like keys; falls back to sample index when
    available.
    """
    for key in ("problem_id", "example_id", "id", "question", "problem", "clue", "title", "uid"):
        value = rec.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return str(value)
    sample_index = rec.get("sample_idx")
    return None if sample_index is None else f"sample_{sample_index}"


def problem_key_from_record(rec: Dict[str, Any], missing_default: str) -> str:
    """
    Robust problem key for math-style logs, matching the conventions used in
    H1/H3 analyses.
    """
    problem_key = rec.get("problem") or rec.get("clue") or rec.get("row_key")
    if problem_key is None:
        dataset_index = rec.get("dataset_index")
        problem_key = f"idx:{dataset_index}" if dataset_index is not None else missing_default
    return str(problem_key)


def step_within_bounds(
    step: int,
    min_step: Optional[int],
    max_step: Optional[int],
) -> bool:
    """
    Return True if ``step`` lies within the optional [min_step, max_step] bounds.
    """
    if min_step is not None and step < min_step:
        return False
    if max_step is not None and step > max_step:
        return False
    return True


def record_matches_split(
    record: Dict[str, Any],
    split_value: Optional[str],
) -> bool:
    """
    Return True if a record is compatible with a requested ``split`` value.

    When ``split_value`` is truthy, records that carry a non-empty ``split``
    field must match it exactly; records without a split label are kept.
    """
    if not split_value:
        return True
    rec_split = record.get("split")
    if rec_split:
        return str(rec_split) == str(split_value)
    return True


def step_from_record_if_within_bounds(
    record: Dict[str, Any],
    path: str,
    split_value: Optional[str],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[int]:
    """
    Combined split / step helper used by entropy-style scripts.

    Returns an integer step inferred via :func:`step_from_rec_or_path`
    when the record passes an optional ``split`` filter and lies within
    the ``[min_step, max_step]`` bounds; otherwise returns ``None``.
    """
    if not record_matches_split(record, split_value):
        return None
    step = step_from_rec_or_path(record, path)
    if not step_within_bounds(step, min_step, max_step):
        return None
    return step


def first_nonempty_str(*xs: Any) -> Optional[str]:
    """
    Return the first non-empty string among the arguments, or None.
    """
    for candidate in xs:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
    return None


def canon_equal(pred_canon: Optional[str], gold: Any) -> Optional[int]:
    """
    Compare a canonical predicted answer to gold.

    - If gold is a string, require exact match after stripping.
    - If gold is a list/tuple/set of strings, treat it as a set.
    - Otherwise, return None.
    """
    if pred_canon is None or gold is None:
        return None
    if isinstance(gold, (list, tuple, set)):
        gold_set = {str(g).strip() for g in gold if isinstance(g, str)}
        return int(pred_canon.strip() in gold_set) if gold_set else None
    if isinstance(gold, str):
        return int(pred_canon.strip() == gold.strip())
    return None
