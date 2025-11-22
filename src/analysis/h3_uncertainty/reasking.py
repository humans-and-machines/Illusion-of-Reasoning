"""
Re-asking table construction and Wilson CI helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccuracyGroupSpec:
    """Column names used to compute pass-level accuracy deltas."""

    size_col: str
    pass1_col: str
    pass2_col: str


QUESTION_GROUP_SPEC = AccuracyGroupSpec(
    size_col="n_problems",
    pass1_col="any_pass1",
    pass2_col="any_pass2",
)
PROMPT_GROUP_SPEC = AccuracyGroupSpec(
    size_col="n_pairs",
    pass1_col="correct1",
    pass2_col="correct2",
)


def wilson_ci(num_success: int, num_trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute the Wilson binomial proportion interval for ``num_success/num_trials``.
    """
    if num_trials <= 0:
        return (float("nan"), float("nan"))
    if abs(alpha - 0.05) < 1e-12:
        z_score = 1.959963984540054
    else:
        z_score = float(np.abs(np.sqrt(2) * np.erfcinv(alpha)))
    proportion = num_success / num_trials
    denom = 1 + z_score * z_score / num_trials
    center = (proportion + z_score * z_score / (2 * num_trials)) / denom
    half_width = (
        z_score
        * math.sqrt(
            proportion * (1 - proportion) / num_trials
            + z_score * z_score / (4 * num_trials * num_trials),
        )
        / denom
    )
    lower_bound, upper_bound = center - half_width, center + half_width
    return max(0.0, lower_bound), min(1.0, upper_bound)


def _build_pairs_table(
    merged_df: pd.DataFrame,
    bucket_column: str,
) -> pd.DataFrame:
    pairs = merged_df[
        [
            "pair_id",
            "problem",
            "step",
            "prompt_key1",
            "prompt_key2",
            "correct1",
            "correct2",
            "forced_insight2",
        ]
    ].rename(
        columns={
            "prompt_key1": "prompt_key_pass1",
            "prompt_key2": "prompt_key_pass2",
            "forced_insight2": "forced_insight",
        },
    )
    pairs["delta"] = pairs["correct2"] - pairs["correct1"]
    if bucket_column in merged_df.columns:
        pairs[bucket_column] = merged_df[bucket_column].to_numpy()
    return pairs


def _build_problem_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    problem_summary = (
        merged_df.groupby("problem", as_index=False)
        .agg(
            n_pairs=("pair_id", "size"),
            any_pass1=("correct1", "max"),
            any_pass2=("correct2", "max"),
            mean_pass1=("correct1", "mean"),
            mean_pass2=("correct2", "mean"),
        )
        .copy()
    )
    problem_summary["delta_any"] = (
        problem_summary["any_pass2"] - problem_summary["any_pass1"]
    )
    problem_summary["delta_mean"] = (
        problem_summary["mean_pass2"] - problem_summary["mean_pass1"]
    )
    return problem_summary


def _build_forced_condition_table(
    merged_df: pd.DataFrame,
    condition_fn,
) -> pd.DataFrame:
    if "forced_insight2" not in merged_df.columns:
        return pd.DataFrame()
    forced_rows: List[Dict[str, Any]] = []
    grouped_pairs = merged_df.groupby("forced_insight2")
    for forced_value, forced_subset in grouped_pairs:
        if forced_subset.empty:
            continue
        forced_problem_summary = (
            forced_subset.groupby("problem", as_index=False)
            .agg(
                n_pairs=("pair_id", "size"),
                any_pass1=("correct1", "max"),
                any_pass2=("correct2", "max"),
                mean_pass1=("correct1", "mean"),
                mean_pass2=("correct2", "mean"),
            )
            .copy()
        )
        forced_problem_summary["delta_mean"] = (
            forced_problem_summary["mean_pass2"] - forced_problem_summary["mean_pass1"]
        )
        for condition_label, mask in (
            ("pass1_any_correct==1", forced_problem_summary["any_pass1"] == 1),
            ("pass1_any_correct==0", forced_problem_summary["any_pass1"] == 0),
        ):
            subset_probs = forced_problem_summary[mask]
            if subset_probs.empty:
                continue
            summary_row = condition_fn(
                matched_rows=forced_subset,
                probs_subset=subset_probs,
                label=condition_label,
                forced_insight=int(forced_value),
            )
            forced_rows.append(summary_row)
    return pd.DataFrame(forced_rows)


def compute_reasking_tables(
    df_all: pd.DataFrame,
    pass1_bucket_col: str = "perplexity_bucket",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build paired and aggregated tables summarizing re-asking outcomes.
    """
    pass1_df = df_all[df_all["pass_id"] == 1].copy()
    pass2_df = df_all[df_all["pass_id"] == 2].copy()
    if pass2_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged_df = pass1_df.merge(
        pass2_df,
        on=["pair_id", "problem", "step"],
        suffixes=("1", "2"),
    )
    if merged_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if pass1_bucket_col in pass1_df.columns:
        merged_df = merged_df.merge(
            pass1_df[["pair_id", pass1_bucket_col]].drop_duplicates(),
            on="pair_id",
            how="left",
        )

    pairs_df = _build_pairs_table(merged_df, pass1_bucket_col)
    probs_df = _build_problem_summary(merged_df)

    def _condition_summary(
        matched_rows: pd.DataFrame,
        probs_subset: pd.DataFrame,
        label: str,
        *,
        share_keep_any: bool = False,
        forced_insight: Optional[int] = None,
    ) -> Dict[str, Any]:
        problem_count = len(probs_subset)
        subset = matched_rows[matched_rows["problem"].isin(probs_subset["problem"])]
        micro_pass1 = float(subset["correct1"].mean()) if not subset.empty else np.nan
        micro_pass2 = float(subset["correct2"].mean()) if not subset.empty else np.nan
        micro_delta = (
            float(micro_pass2 - micro_pass1) if not subset.empty else np.nan
        )
        macro_mean_pass1 = (
            float(probs_subset["mean_pass1"].mean()) if problem_count else np.nan
        )
        macro_mean_pass2 = (
            float(probs_subset["mean_pass2"].mean()) if problem_count else np.nan
        )
        macro_delta = (
            float(probs_subset["delta_mean"].mean()) if problem_count else np.nan
        )
        result = {
            "condition": label,
            "n_problems": int(problem_count),
            "macro_mean_acc_pass1": macro_mean_pass1,
            "macro_mean_acc_pass2": macro_mean_pass2,
            "macro_delta_mean": macro_delta,
            "micro_acc_pass1": micro_pass1,
            "micro_acc_pass2": micro_pass2,
            "micro_delta": micro_delta,
        }
        share_any = float(probs_subset["any_pass2"].mean()) if problem_count else np.nan
        result["share_any_pass2"] = share_any
        if share_keep_any:
            result["share_keep_any_correct_in_pass2"] = share_any
        if forced_insight is not None:
            result["forced_insight"] = forced_insight
        return result

    cond_rows = []
    mask_any = probs_df["any_pass1"] == 1
    cond_rows.append(
        _condition_summary(
            matched_rows=merged_df,
            probs_subset=probs_df[mask_any],
            label="pass1_any_correct==1",
            share_keep_any=True,
        ),
    )
    cond_rows.append(
        _condition_summary(
            matched_rows=merged_df,
            probs_subset=probs_df[~mask_any],
            label="pass1_any_correct==0",
        ),
    )
    cond_df = pd.DataFrame(cond_rows)

    cond_forced_df = _build_forced_condition_table(merged_df, _condition_summary)

    return pairs_df, probs_df, cond_df, cond_forced_df


def split_reasking_by_aha(
    pairs_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    pass1_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split prompt/problem summaries by Aha variant based on pass1 flags.
    """
    aha_variants = [
        ("aha_words", "words"),
        ("aha_gpt", "gpt"),
        ("aha_formal", "formal"),
    ]
    flag_cols = [col for col, _ in aha_variants]
    flags = (
        pass1_df[["pair_id"] + flag_cols]
        .drop_duplicates("pair_id")
        .copy()
    )
    if flags.empty:
        return pd.DataFrame(), pd.DataFrame()
    prompt_parts = []
    problem_parts = []
    merged = pairs_df.merge(flags, on="pair_id", how="left")
    for col, label in aha_variants:
        subset = merged[merged[col] == 1].copy()
        if subset.empty:
            continue
        subset["aha_variant"] = label
        prompt_parts.append(subset)
        problems = probs_df[probs_df["problem"].isin(subset["problem"])].copy()
        if problems.empty:
            continue
        problems["aha_variant"] = label
        problem_parts.append(problems)

    prompt_by_aha = (
        pd.concat(prompt_parts, ignore_index=True)
        if prompt_parts
        else pd.DataFrame()
    )
    problem_by_aha = (
        pd.concat(problem_parts, ignore_index=True)
        if problem_parts
        else pd.DataFrame()
    )
    return prompt_by_aha, problem_by_aha


def _ensure_groupcol(frame: pd.DataFrame, group_keys: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if group_keys:
        return frame.copy(), group_keys
    augmented_frame = frame.copy()
    augmented_frame["__all__"] = "all"
    return augmented_frame, ["__all__"]


def question_level_any_with_ci(
    m_pairs: pd.DataFrame,
    group_keys: List[str],
) -> pd.DataFrame:
    """
    Per-problem accuracy with Wilson confidence intervals.
    """
    data, keys = _ensure_groupcol(m_pairs, group_keys)
    per_problem = _per_problem_aggregate(data, keys)
    rows = [
        _pairwise_accuracy_delta(
            subset=subset,
            keys=keys,
            group_vals=group_vals,
            spec=QUESTION_GROUP_SPEC,
        )
        for group_vals, subset in per_problem.groupby(keys)
    ]
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def prompt_level_acc_with_ci(
    m_pairs: pd.DataFrame,
    group_keys: List[str],
) -> pd.DataFrame:
    """
    Per-prompt accuracy with Wilson confidence intervals.
    """
    data, keys = _ensure_groupcol(m_pairs, group_keys)
    rows = [
        _pairwise_accuracy_delta(
            subset=subset,
            keys=keys,
            group_vals=group_vals,
            spec=PROMPT_GROUP_SPEC,
        )
        for group_vals, subset in data.groupby(keys)
    ]
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
def _per_problem_aggregate(data: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Aggregate pair-level rows into per-problem any-pass flags."""
    return (
        data.groupby(keys + ["problem"], as_index=False)
        .agg(any_pass1=("correct1", "max"), any_pass2=("correct2", "max"))
    )


def _pairwise_accuracy_delta(
    subset: pd.DataFrame,
    keys: List[str],
    group_vals: Any,
    spec: AccuracyGroupSpec,
) -> Dict[str, Any]:
    """
    Compute accuracy deltas and Wilson CIs for a single grouping subset.
    """
    normalized_keys = group_vals if isinstance(group_vals, tuple) else (group_vals,)
    pair_count = int(len(subset))
    row = dict(zip(keys, normalized_keys))
    row[spec.size_col] = pair_count

    for column in (spec.pass1_col, spec.pass2_col):
        successes = int(subset[column].sum())
        accuracy = successes / pair_count if pair_count else float("nan")
        lower, upper = wilson_ci(successes, pair_count)
        row[column] = accuracy
        row[f"{column}_lo"] = lower
        row[f"{column}_hi"] = upper

    row["delta"] = (
        row[spec.pass2_col] - row[spec.pass1_col] if pair_count else float("nan")
    )
    return row
