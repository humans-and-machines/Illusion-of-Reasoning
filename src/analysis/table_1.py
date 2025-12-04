#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha Conditionals (Domain-level) — robust to Llama-style rows
------------------------------------------------------------
For each provided folder/domain (e.g., --root_crossword, --root_math, --root_carpark),
compute:

  • P(success | shift) and P(success | no shift)
    - Crosswords/Math: success = accuracy (is_correct_pred)
    - Carpark: success = (soft_reward OP threshold), default OP='>' and threshold=0.0

Key robustness fixes
--------------------
• Read shift/cue fields from pass1 OR top level (handles dumps where fields moved).
• Optional: in --gpt_mode=broad, treat non-empty marker lists
  (shift_markers_v1, _shift_prefilter_markers) as SHIFT=1 even if booleans are false.
• Optional: --allow_judge_cues_non_xword makes judge-time cues open the gate on non-crossword.
• Success fallback: if pass1.is_correct_pred is missing, use top-level is_correct_pred.

Output
------
CSV at <out_dir>/aha_conditionals__<dataset>__<model>.csv with columns:
    domain, n_total,
    k_correct_shift, n_shift, p_correct_given_shift,
    k_correct_noshift, n_noshift, p_correct_given_noshift,
    shift_rate
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.aha_utils import any_keys_true, cue_gate_for_llm
from src.analysis.io import build_jsonl_files_by_domain, iter_records_from_file, scan_jsonl_files
from src.analysis.metrics import make_carpark_success_fn
from src.analysis.utils import (
    add_split_and_out_dir_args,
    add_standard_domain_root_args,
    both_get,
    coerce_bool,
    coerce_float,
    extract_pass1_and_step,
    gpt_keys_for_mode,
    nat_step_from_path,
    step_within_bounds,
)


@dataclass
class GptAhaConfig:
    """
    Configuration for GPT/LLM-labeled Aha detection.

    :param gpt_keys: Candidate boolean shift keys to inspect.
    :param gpt_subset_native: Whether to gate GPT shifts by native cues.
    :param allow_judge_cues_non_xword: Whether judge-time cues apply to non-crossword domains.
    :param broad_counts_marker_lists: Whether non-empty marker lists count as positive shifts.
    """

    gpt_keys: List[str]
    gpt_subset_native: bool
    allow_judge_cues_non_xword: bool
    broad_counts_marker_lists: bool


def _aha_gpt_for_rec(
    pass1_fields: Dict[str, Any],
    rec: Dict[str, Any],
    domain: Optional[str],
    config: GptAhaConfig,
) -> int:
    """
    Shift-positive if any gpt_keys are truthy. In 'broad' mode
    (broad_counts_marker_lists=True), treat non-empty 'shift_markers_v1'
    / '_shift_prefilter_markers' as positive too.
    """
    raw = any_keys_true(pass1_fields, rec, config.gpt_keys)

    if config.broad_counts_marker_lists and raw == 0:
        # Consider marker lists as positive signals in broad mode
        if coerce_bool(both_get(pass1_fields, rec, "shift_markers_v1")) == 1:
            raw = 1
        elif (
            coerce_bool(
                both_get(pass1_fields, rec, "_shift_prefilter_markers"),
            )
            == 1
        ):
            raw = 1

    if not config.gpt_subset_native:
        return int(raw)

    gate = cue_gate_for_llm(
        pass1_fields,
        rec,
        domain,
        config.allow_judge_cues_non_xword,
    )
    return int(raw & gate)


# ----------------- Core loading -----------------
def _extract_soft_reward(
    rec: Dict[str, Any],
    pass1_fields: Dict[str, Any],
) -> Optional[float]:
    return coerce_float(both_get(pass1_fields, rec, "soft_reward"))


@dataclass
class LoadRowsConfig:
    """
    Configuration for loading per-sample rows across domains.

    :param gpt_config: GPT/LLM Aha detection configuration.
    :param min_step: Optional minimum training step to keep.
    :param max_step: Optional maximum training step to keep.
    :param carpark_success_fn: Function mapping raw rewards to success labels.
    :param debug: Whether to print per-domain debug counts.
    """

    gpt_config: GptAhaConfig
    min_step: Optional[int]
    max_step: Optional[int]
    carpark_success_fn: Callable[[Any], Optional[int]]
    debug: bool = False


def _build_row_for_record(
    domain: str,
    record: Dict[str, Any],
    step_from_name: Optional[int],
    config: LoadRowsConfig,
) -> Optional[Dict[str, int]]:
    """
    Build a single (domain, step, correct, shift) row for one record.

    :param domain: Domain label (for example, ``\"Math\"`` or ``\"Carpark\"``).
    :param record: JSON record parsed from a PASS-1-style file.
    :param step_from_name: Optional step inferred from the filename.
    :param config: Row-loading configuration.
    :returns: Row dictionary or ``None`` if the record should be skipped.
    """
    pass1_fields, step_int = extract_pass1_and_step(record, step_from_name)
    if not pass1_fields or step_int is None:
        return None

    if not step_within_bounds(step_int, config.min_step, config.max_step):
        return None

    domain_lower = domain.lower()
    if domain_lower == "carpark":
        soft_reward = _extract_soft_reward(record, pass1_fields)
        correct = config.carpark_success_fn(soft_reward)  # 1/0 or None
    else:
        correct = coerce_bool(both_get(pass1_fields, record, "is_correct_pred"))

    if correct is None:
        return None

    aha_llm = _aha_gpt_for_rec(
        pass1_fields,
        record,
        domain,
        config.gpt_config,
    )

    return {
        "domain": domain,
        "step": int(step_int),
        "correct": int(correct),
        "shift": int(aha_llm),
    }


def load_rows(
    files_by_domain: Dict[str, List[str]],
    config: LoadRowsConfig,
) -> pd.DataFrame:
    """
    Load per-sample rows with domain, step, correctness, and GPT Aha flags.

    :param files_by_domain: Mapping from domain label to list of JSONL paths.
    :param config: Structured configuration controlling filtering and labeling.
    :returns: DataFrame with columns ``domain``, ``step``, ``correct``, ``shift``.
    """
    rows = []
    for dom, files in files_by_domain.items():
        domain_str = str(dom)
        for path in files:
            step_from_name = nat_step_from_path(path)
            for rec in iter_records_from_file(path):
                row = _build_row_for_record(
                    domain_str,
                    rec,
                    step_from_name,
                    config,
                )
                if row is not None:
                    rows.append(row)
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise SystemExit("No rows found. Check roots and/or --split / step filters.")
    if config.debug:
        for dom, sub in rows_df.groupby("domain"):
            print(f"[debug] {dom:>12}: n={len(sub):6d}, shifts={int(sub['shift'].sum()):6d}")
    return rows_df


# ----------------- Aggregation -----------------


def agg_domain_conditionals(rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per domain, preserving the input order.
    """
    out_rows = []
    domain_order = list(dict.fromkeys(rows_df["domain"].tolist()))
    for dom in domain_order:
        sub = rows_df[rows_df["domain"] == dom]
        m_shift = sub["shift"] == 1
        m_noshift = sub["shift"] == 0

        n_shift = int(m_shift.sum())
        n_noshift = int(m_noshift.sum())
        n_total = int(len(sub))

        k_shift = int(sub.loc[m_shift, "correct"].sum()) if n_shift > 0 else 0
        k_noshift = int(sub.loc[m_noshift, "correct"].sum()) if n_noshift > 0 else 0

        p_c_given_shift = (k_shift / n_shift) if n_shift > 0 else np.nan
        p_c_given_noshift = (k_noshift / n_noshift) if n_noshift > 0 else np.nan
        shift_rate = (n_shift / n_total) if n_total > 0 else np.nan

        out_rows.append(
            {
                "domain": dom,
                "n_total": n_total,
                "k_correct_shift": k_shift,
                "n_shift": n_shift,
                "p_correct_given_shift": p_c_given_shift,
                "k_correct_noshift": k_noshift,
                "n_noshift": n_noshift,
                "p_correct_given_noshift": p_c_given_noshift,
                "shift_rate": shift_rate,
            }
        )
    return pd.DataFrame(out_rows)


HARD_STEP_CAP = 1000


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct an argument parser for the Aha conditionals CLI.

    :returns: Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser()

    # Standard domain roots plus optional fallback results_root
    add_standard_domain_root_args(parser)

    # Second & third Math roots (e.g., different models)
    parser.add_argument(
        "--label_math2",
        type=str,
        default="Math (Qwen-7B)",
        help="Row label for the second Math root.",
    )

    parser.add_argument(
        "--root_math3",
        type=str,
        default=None,
        help="Path to a third Math results folder (e.g., Llama-8B).",
    )
    parser.add_argument(
        "--label_math3",
        type=str,
        default="Math (Llama-8B)",
        help="Row label for the third Math root.",
    )

    add_split_and_out_dir_args(
        parser,
        out_dir_help=("Base output directory (default: <first_root>/aha_conditionals)."),
    )
    parser.add_argument("--dataset_name", default="MIXED")
    parser.add_argument("--model_name", default="MIXED_MODELS")

    # GPT label policy
    parser.add_argument(
        "--gpt_mode",
        choices=["canonical", "broad"],
        default="canonical",
        help=("canonical: boolean keys only; broad: also count marker lists as positive."),
    )
    parser.add_argument(
        "--no_gpt_subset_native",
        action="store_true",
        help="If set, DO NOT apply cue gating ('native subset').",
    )

    # Allow judge-time cues for non-crossword (off by default to preserve prior behavior)
    parser.add_argument(
        "--allow_judge_cues_non_xword",
        action="store_true",
        help="Open the cue gate for judge-time cues in non-crossword domains.",
    )

    # Step filters (hard cap at 1000 is enforced below)
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Upper bound on step. Note: hard cap of 1000 is applied.",
    )

    # --- Carpark success policy ---
    parser.add_argument(
        "--carpark_soft_threshold",
        type=float,
        default=0.1,
        help="Carpark success if soft_reward OP threshold (default OP='gt').",
    )
    parser.add_argument(
        "--carpark_success_op",
        choices=["gt", "ge", "eq"],
        default="gt",
        help="Comparison operator for Carpark success against soft_reward.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print domain counts after loading.",
    )

    return parser


def _build_files_by_domain(args: argparse.Namespace) -> tuple[Dict[str, List[str]], str]:
    """
    Build a mapping from domain name to JSONL files and return the first root.

    :param args: Parsed CLI arguments.
    :returns: Tuple ``(files_by_domain, first_root)``.
    :raises SystemExit: If no files can be found under the provided roots.
    """
    files_by_domain: Dict[str, List[str]] = {}
    first_root: Optional[str] = None

    domain_roots: Dict[str, Optional[str]] = {}
    if args.root_crossword:
        domain_roots["Crossword"] = args.root_crossword
    if args.root_math:
        domain_roots["Math"] = args.root_math
    if args.root_math2:
        label2 = (args.label_math2 or "Math2").strip()
        if label2 in files_by_domain:
            base_label = label2
            idx = 2
            while label2 in files_by_domain:
                label2 = f"{base_label}-{idx}"
                idx += 1
        domain_roots[label2] = args.root_math2
    if args.root_math3:
        label3 = (args.label_math3 or "Math3").strip()
        if label3 in files_by_domain:
            base_label = label3
            idx = 2
            while label3 in files_by_domain:
                label3 = f"{base_label}-{idx}"
                idx += 1
        domain_roots[label3] = args.root_math3
    if args.root_carpark:
        domain_roots["Carpark"] = args.root_carpark

    files_by_domain, first_root = build_jsonl_files_by_domain(domain_roots, args.split)

    if not files_by_domain:
        if not args.results_root:
            raise SystemExit(
                "Provide --root_crossword/--root_math/--root_math2/--root_math3/"
                "--root_carpark, or a fallback results_root.",
            )
        files_by_domain["All"] = scan_jsonl_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    assert first_root is not None  # for type checkers
    return files_by_domain, first_root


def _build_gpt_config(args: argparse.Namespace) -> GptAhaConfig:
    """
    Construct a :class:`GptAhaConfig` from parsed CLI arguments.

    :param args: Parsed CLI arguments.
    :returns: GPT Aha detection configuration.
    """
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = gpt_keys_for_mode(args.gpt_mode)
    broad_counts_marker_lists = args.gpt_mode != "canonical"

    return GptAhaConfig(
        gpt_keys=gpt_keys,
        gpt_subset_native=gpt_subset_native,
        allow_judge_cues_non_xword=args.allow_judge_cues_non_xword,
        broad_counts_marker_lists=broad_counts_marker_lists,
    )


def _compute_effective_max_step(args: argparse.Namespace) -> int:
    """
    Compute the effective maximum step, respecting the global hard cap.

    :param args: Parsed CLI arguments.
    :returns: Effective maximum step value.
    """
    effective_max_step = HARD_STEP_CAP if args.max_step is None else min(args.max_step, HARD_STEP_CAP)
    if args.max_step is None or args.max_step > HARD_STEP_CAP:
        print(
            f"[info] Capping max_step to {effective_max_step} (hard cap = {HARD_STEP_CAP}).",
        )
    return effective_max_step


def main() -> None:
    """
    Entry point for computing domain-level Aha conditionals and writing Table 1 CSV.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    files_by_domain, first_root = _build_files_by_domain(args)

    out_dir = args.out_dir or os.path.join(first_root, "aha_conditionals")
    os.makedirs(out_dir, exist_ok=True)

    gpt_config = _build_gpt_config(args)
    effective_max_step = _compute_effective_max_step(args)

    load_config = LoadRowsConfig(
        gpt_config=gpt_config,
        min_step=args.min_step,
        max_step=effective_max_step,
        carpark_success_fn=make_carpark_success_fn(
            args.carpark_success_op,
            args.carpark_soft_threshold,
        ),
        debug=args.debug,
    )

    # Load rows and aggregate domain-level conditionals (preserve input order)
    table = agg_domain_conditionals(load_rows(files_by_domain, load_config))

    # Save + print
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_csv = os.path.join(out_dir, f"aha_conditionals__{slug}.csv")
    table.to_csv(out_csv, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nAha conditionals (domain-level):")
        print(table.to_string(index=False))
        print(f"\nSaved CSV -> {out_csv}")


if __name__ == "__main__":
    main()
