#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2: Temperature x Aha! analysis (+ stages of training, + three Aha definitions)
-------------------------------------------------------------------------------
Evaluate whether temperature (default vs low-temp=0.3) and Aha markers affect accuracy
across training stages, using only overlapping (problem, step[, sample_idx]) between runs.

Aha definitions:
  • words  : pass1.has_reconsider_cue (excl. injected)
  • gpt    : canonical LLM shift (optionally gated by words)
  • formal : problem–step level: prior failure & prior shift-stability & shift now,
             computed within each temperature run, then AND-gated with sample-level GPT Aha

CLI:
  --aha {words|gpt|gpt_broad|formal|none|all}
  --gate_gpt_by_words (for gpt/gpt_broad)
  --formal_delta1 0.13  --formal_delta2 0.13  --formal_min_prior_steps 2

Stage bucketing:
  --stage_mode {quantile|fixed}
  --stage_quantiles 0.33 0.66
  --stage_bounds lo hi

Outputs (per Aha mode; under <out_dir>/<aha_mode>/ when --aha all):
  - h2_glm_summary.txt
  - h2_glm_coefficients.csv
  - h2_group_accuracy.csv
  - h2_group_accuracy_by_step.csv
  - h2_stage_group_accuracy.csv
  - h2_stage_aha_counts.csv
  - h2_stage_glm_summary.txt
  - h2_stage_glm_coefficients.csv
  - h2_stage_info.json
  - h2_overlap_summary.json (kept once at top-level when single mode; per-mode when --aha all)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.core import add_standard_formal_flags, build_problem_step_for_formal, iter_pass1_records
from src.analysis.io import scan_jsonl_files
from src.analysis.labels import aha_gpt_broad, aha_gpt_canonical, aha_words
from src.analysis.metrics import add_step_std_column, lazy_import_statsmodels
from src.analysis.utils import coerce_bool, extract_pass1_and_step, problem_key_from_record


# ---------- file scanning & misc ----------

# ---------- load a root into per-sample rows ----------


def _compute_aha_flags(
    pass1_data: Dict[str, Any],
    record: Dict[str, Any],
    aha_mode: str,
    gate_gpt_by_words: bool,
) -> Tuple[int, int, int]:
    """
    Compute (words, gpt, aha) flags for a single pass-1 record.

    ``aha_mode`` controls which signal populates the generic ``aha`` field.
    """
    words_flag = aha_words(pass1_data)
    if aha_mode == "gpt":
        raw_gpt = aha_gpt_canonical(pass1_data, record)
    elif aha_mode == "gpt_broad":
        raw_gpt = aha_gpt_broad(pass1_data, record)
    else:
        raw_gpt = 0
    if aha_mode.startswith("gpt") and gate_gpt_by_words:
        gpt_flag = int(raw_gpt and words_flag)
    else:
        gpt_flag = int(raw_gpt)

    if aha_mode == "words":
        aha_flag = words_flag
    elif aha_mode.startswith("gpt"):
        aha_flag = gpt_flag
    else:
        aha_flag = 0
    return int(words_flag), int(gpt_flag), int(aha_flag)


def load_samples(
    root: str,
    split_filter: Optional[str],
    aha_mode: str = "gpt",
    gate_gpt_by_words: bool = True,
) -> pd.DataFrame:
    """
    Returns per-sample rows with columns:
      dataset, model, split, problem, step, sample_idx, correct, aha_words, aha_gpt, aha
    """
    # Path-level split filtering happens via ``split_filter`` on ``rec['split']``.
    files = scan_jsonl_files(root, split_substr=None)
    rows: List[Dict[str, Any]] = []

    for _, step_from_name, rec in iter_pass1_records(files):
        if split_filter is not None:
            if str(rec.get("split", "")).lower() != str(split_filter).lower():
                continue

        pass1_data, step = extract_pass1_and_step(rec, step_from_name)
        if not pass1_data or step is None:
            continue

        problem_key = problem_key_from_record(rec, missing_default="unknown")

        correct = coerce_bool(
            pass1_data.get("is_correct_pred")
            if "is_correct_pred" in pass1_data
            else pass1_data.get("is_correct", pass1_data.get("correct")),
        )
        if correct is None:
            continue

        aha_words_flag, aha_gpt_flag, aha_flag = _compute_aha_flags(
            pass1_data,
            rec,
            aha_mode,
            gate_gpt_by_words,
        )

        rows.append(
            {
                "dataset": rec.get("dataset"),
                "model": rec.get("model"),
                "split": rec.get("split"),
                "problem": str(problem_key),
                "step": int(step),
                "sample_idx": rec.get(
                    "sample_idx",
                    rec.get("sample_index", rec.get("idx", rec.get("i", None))),
                ),
                "correct": int(correct),
                "aha_words": aha_words_flag,
                "aha_gpt": aha_gpt_flag,
                "aha": aha_flag,  # generic slot for selected aha
            },
        )
    return pd.DataFrame(rows)


def _apply_step_bounds(df: pd.DataFrame, min_step: Optional[int], max_step: Optional[int]) -> pd.DataFrame:
    """
    Restrict rows to the requested step range.
    """
    if min_step is not None:
        df = df[df["step"] >= min_step]
    if max_step is not None:
        df = df[df["step"] <= max_step]
    if df.empty:
        raise SystemExit("No rows remain after applying step bounds.")
    return df


# ---------- overlap + merge helpers ----------


def restrict_to_overlap(
    df_high: pd.DataFrame,
    df_low: pd.DataFrame,
    use_sample_idx: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Intersect on keys. Always require (problem, step); include sample_idx if present on both.
    Returns filtered (dfA, dfB, keys_used).
    """
    keys = ["problem", "step"]
    if use_sample_idx:
        keys.append("sample_idx")

    keys_high = df_high[keys].drop_duplicates()
    keys_low = df_low[keys].drop_duplicates()
    common = keys_high.merge(keys_low, on=keys, how="inner")
    if common.empty:
        raise SystemExit("No overlapping (problem, step[, sample_idx]) between runs.")

    merged_high = common.merge(df_high, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    merged_low = common.merge(df_low, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    return merged_high, merged_low, keys


def restrict_to_overlap_many(dfs: List[pd.DataFrame], use_sample_idx: bool) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Intersect multiple DataFrames on (problem, step[, sample_idx]).
    """
    if len(dfs) < 2:
        raise SystemExit("Need at least two runs to compute overlap.")
    keys = ["problem", "step"]
    if use_sample_idx:
        keys.append("sample_idx")

    common = dfs[0][keys].drop_duplicates()
    for df in dfs[1:]:
        common = common.merge(df[keys].drop_duplicates(), on=keys, how="inner")
        if common.empty:
            raise SystemExit("No overlapping (problem, step[, sample_idx]) among runs.")

    filtered = [
        common.merge(df, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
        for df in dfs
    ]
    return filtered, keys


# ---------- stage bucketing ----------


def assign_stage(
    input_df: pd.DataFrame,
    mode: str = "quantile",
    quantiles: Tuple[float, float] = (0.33, 0.66),
    bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assign training-stage labels (early/mid/late) based on step values.

    When ``mode == "fixed"``, uses explicit integer ``bounds``; otherwise uses
    quantiles of the observed steps.
    """
    stage_df = input_df.copy()
    steps = np.sort(stage_df["step"].unique())
    info: Dict[str, Any] = {"mode": mode}

    if mode == "fixed":
        if not bounds or len(bounds) != 2:
            raise SystemExit("--stage_mode fixed requires --stage_bounds lo hi")
        lower_bound, upper_bound = int(bounds[0]), int(bounds[1])
        info.update({"bounds": [lower_bound, upper_bound]})
    else:
        quantile_1, quantile_2 = quantiles
        lower_bound = int(np.quantile(steps, quantile_1))
        upper_bound = int(np.quantile(steps, quantile_2))
        info.update(
            {
                "quantiles": [float(quantile_1), float(quantile_2)],
                "cutpoints": [lower_bound, upper_bound],
            },
        )

    def _stage_label(step_value: int) -> str:
        if step_value <= lower_bound:
            return "early"
        if step_value <= upper_bound:
            return "mid"
        return "late"

    stage_df["stage"] = stage_df["step"].apply(_stage_label)
    return stage_df, info


# ---------- FORMAL Aha (problem–step) ----------
def attach_formal_sample_level(
    samples_df: pd.DataFrame,
    delta1: float,
    delta2: float,
    min_prior_steps: int,
) -> pd.DataFrame:
    """
    Compute formal at (run_label, problem, step) level; merge to samples and
    AND-gate with sample-level GPT Aha.

    Returns df with new columns: ``aha_formal_ps`` and ``aha_formal``.
    """
    # Reuse the shared helper that aggregates per-(problem, step) statistics,
    # treating ``run_label`` as a domain key for the purposes of Formal Aha.
    tmp_samples = samples_df.rename(columns={"run_label": "domain"})
    problem_step_df = build_problem_step_for_formal(tmp_samples).rename(
        columns={"domain": "run_label"},
    )
    problem_step_df = add_standard_formal_flags(
        problem_step_df,
        out_column="aha_formal_ps",
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
        group_keys=["run_label", "problem"],
    )
    merged_df = samples_df.merge(
        problem_step_df[["run_label", "problem", "step", "aha_formal_ps"]],
        on=["run_label", "problem", "step"],
        how="left",
    ).fillna({"aha_formal_ps": 0})
    merged_df["aha_formal_ps"] = merged_df["aha_formal_ps"].astype(int)
    # sample-level formal = problem-step formal AND sample's GPT Aha
    merged_df["aha_formal"] = (merged_df["aha_formal_ps"] & merged_df["aha_gpt"]).astype(
        int,
    )
    return merged_df


# ---------- modeling ----------


def fit_glm_binomial(
    input_df: pd.DataFrame,
    aha_col: Optional[str],
    cluster_by: str = "problem",
    out_txt: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Fit a baseline GLM ``correct ~ temp_low (+ aha + interaction)`` with robust SEs.
    """
    stats_module, stats_formula = lazy_import_statsmodels()
    glm_df = add_step_std_column(
        input_df,
        step_column="step",
        std_column="step_std",
    )

    formula_terms = ["C(problem)", "step_std", "temp_low"]
    if aha_col:
        formula_terms += [aha_col, f"temp_low:{aha_col}"]
    formula = "correct ~ " + " + ".join(formula_terms)

    if cluster_by == "problem":
        cov_type, cov_kwds = (
            "cluster",
            {
                "groups": pd.Categorical(glm_df["problem"]).codes,
                "use_correction": True,
                "df_correction": True,
            },
        )
    else:
        cov_type, cov_kwds = "HC1", {}

    try:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except TypeError:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type)

    coef_df = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "z": (res.params.values / np.where(res.bse.values == 0, np.nan, res.bse.values)),
            "p": res.pvalues.values,
        },
    )

    if out_txt:
        with open(out_txt, "w", encoding="utf-8") as handle:
            handle.write(res.summary().as_text())
            handle.write(
                f"\n\nFormula: {formula}\nCovariance: {cov_type} (clustered by {cluster_by})\n",
            )

    out = {"formula": formula, "N": int(len(glm_df))}
    out["coef_temp_low"], out["p_temp_low"] = (
        float(res.params.get("temp_low", np.nan)),
        float(res.pvalues.get("temp_low", np.nan)),
    )
    if aha_col:
        out["coef_aha"], out["p_aha"] = (
            float(res.params.get(aha_col, np.nan)),
            float(res.pvalues.get(aha_col, np.nan)),
        )
        out["coef_interaction"], out["p_interaction"] = (
            float(res.params.get(f"temp_low:{aha_col}", np.nan)),
            float(res.pvalues.get(f"temp_low:{aha_col}", np.nan)),
        )
    return out, coef_df


def fit_glm_stage_interaction(
    input_df: pd.DataFrame,
    aha_col: Optional[str],
    cluster_by: str,
    out_txt: Optional[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Fit a GLM with stage×temp_low (and optional stage×Aha) interactions.
    """
    stats_module, stats_formula = lazy_import_statsmodels()
    glm_df = input_df.copy()
    glm_df["step_std"] = (glm_df["step"] - glm_df["step"].mean()) / (glm_df["step"].std(ddof=0) + 1e-8)
    base_terms = ["C(problem)", "step_std", "temp_low", "C(stage)"]
    if aha_col:
        base_terms += [aha_col, f"temp_low:{aha_col}"]
    base_terms += ["temp_low:C(stage)"]
    if aha_col:
        base_terms += [f"{aha_col}:C(stage)"]

    formula = "correct ~ " + " + ".join(base_terms)
    if cluster_by == "problem":
        cov_type, cov_kwds = (
            "cluster",
            {
                "groups": pd.Categorical(glm_df["problem"]).codes,
                "use_correction": True,
                "df_correction": True,
            },
        )
    else:
        cov_type, cov_kwds = "HC1", {}

    try:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except TypeError:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type)

    coef_df = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "z": (res.params.values / np.where(res.bse.values == 0, np.nan, res.bse.values)),
            "p": res.pvalues.values,
        },
    )

    if out_txt:
        with open(out_txt, "a", encoding="utf-8") as handle:
            handle.write("\n\n=== Stage-interaction GLM ===\n")
            handle.write(res.summary().as_text())
            handle.write(
                f"\n\nFormula: {formula}\nCovariance: {cov_type} (clustered by {cluster_by})\n",
            )

    out = {"formula": formula, "N": int(len(glm_df))}
    out["coef_temp_low"], out["p_temp_low"] = (
        float(res.params.get("temp_low", np.nan)),
        float(res.pvalues.get("temp_low", np.nan)),
    )
    if aha_col:
        out["coef_aha"], out["p_aha"] = (
            float(res.params.get(aha_col, np.nan)),
            float(res.pvalues.get(aha_col, np.nan)),
        )
        out["coef_temp_low_x_aha"], out["p_temp_low_x_aha"] = (
            float(res.params.get(f"temp_low:{aha_col}", np.nan)),
            float(res.pvalues.get(f"temp_low:{aha_col}", np.nan)),
        )
    return out, coef_df


# ---------- group accuracy tables ----------


def compute_group_acc(
    input_df: pd.DataFrame,
    aha_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute overall and per-step accuracy tables, optionally split by Aha flag.
    """
    grp_cols = ["temp_low"] + ([aha_col] if aha_col else [])
    overall = input_df.groupby(grp_cols, as_index=False).agg(n=("correct", "size"), k=("correct", "sum"))
    overall["accuracy"] = overall["k"] / overall["n"]
    by_step = input_df.groupby(["step"] + grp_cols, as_index=False).agg(
        n=("correct", "size"),
        k=("correct", "sum"),
    )
    by_step["accuracy"] = by_step["k"] / by_step["n"]
    return overall, by_step


def compute_stage_group_acc(
    stage_df: pd.DataFrame,
    aha_col: Optional[str],
) -> pd.DataFrame:
    """
    Compute per-stage, per-temp accuracy tables, optionally split by Aha flag.
    """
    grp_cols = ["stage", "temp_low"] + ([aha_col] if aha_col else [])
    grouped = stage_df.groupby(grp_cols, as_index=False).agg(n=("correct", "size"), k=("correct", "sum"))
    grouped["accuracy"] = grouped["k"] / grouped["n"]
    return grouped


def compute_stage_aha_counts(
    stage_df: pd.DataFrame,
    aha_col: Optional[str],
) -> pd.DataFrame:
    """
    Compute counts of Aha-labeled samples per (stage, temp_low) combination.
    """
    if aha_col is None:
        return pd.DataFrame()
    sub = stage_df[stage_df[aha_col] == 1]
    counts = sub.groupby(["stage", "temp_low"], as_index=False).agg(n_aha=("correct", "size"))
    return counts


# ---------- per-stage GLMs & outputs ----------


def _run_stage_specific_glms(
    mode_df: pd.DataFrame,
    aha_col: Optional[str],
    cluster_by: str,
    out_dir: str,
) -> None:
    """
    Run separate GLMs within each training stage and persist coefficients.
    """
    if "stage" not in mode_df.columns:
        return

    headline_rows: List[Dict[str, Any]] = []
    coef_frames: List[pd.DataFrame] = []

    for stage_label in sorted(mode_df["stage"].unique()):
        stage_subset = mode_df[mode_df["stage"] == stage_label].copy()
        if stage_subset.empty:
            continue

        stage_summary, stage_coef = fit_glm_binomial(
            stage_subset,
            aha_col=aha_col,
            cluster_by=cluster_by,
            out_txt=None,
        )
        stage_summary["stage"] = stage_label
        headline_rows.append(stage_summary)

        stage_coef["stage"] = stage_label
        coef_frames.append(stage_coef)

    if coef_frames:
        stage_coef_df = pd.concat(coef_frames, ignore_index=True)
        coef_path = os.path.join(out_dir, "h2_stage_specific_glm_coefficients.csv")
        stage_coef_df.to_csv(coef_path, index=False)

    if headline_rows:
        summary_path = os.path.join(out_dir, "h2_stage_specific_glm_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(headline_rows, handle, indent=2)


def _write_group_outputs(
    mode_df: pd.DataFrame,
    aha_col: Optional[str],
    aha_mode_name: str,
    out_dir: str,
    glm_summaries: Dict[str, Dict[str, Any]],
) -> None:
    """
    Persist group accuracy tables and stage info, and print a console summary.
    """
    overall_acc, step_acc = compute_group_acc(mode_df, aha_col=aha_col)
    overall_acc.to_csv(os.path.join(out_dir, "h2_group_accuracy.csv"), index=False)
    step_acc.to_csv(
        os.path.join(out_dir, "h2_group_accuracy_by_step.csv"),
        index=False,
    )

    stage_acc = compute_stage_group_acc(mode_df, aha_col=aha_col)
    stage_acc.to_csv(
        os.path.join(out_dir, "h2_stage_group_accuracy.csv"),
        index=False,
    )

    stage_aha = compute_stage_aha_counts(mode_df, aha_col=aha_col)
    if not stage_aha.empty:
        stage_aha.to_csv(
            os.path.join(out_dir, "h2_stage_aha_counts.csv"),
            index=False,
        )

    summ_base = glm_summaries["baseline"]
    summ_int = glm_summaries["stage_interaction"]

    info = {
        "aha_mode": aha_mode_name,
        "glm_headline": summ_base,
        "stage_interaction_headline": summ_int,
    }
    with open(
        os.path.join(out_dir, "h2_stage_info.json"),
        "w",
        encoding="utf-8",
    ) as info_file:
        json.dump(info, info_file, indent=2)

    # Console snippet
    print(
        f"\n[{aha_mode_name.upper()}] Baseline GLM: {summ_base['formula']}  N={summ_base['N']}",
    )
    print(
        f"  temp_low coef={summ_base.get('coef_temp_low', np.nan):+.4f}  p={summ_base.get('p_temp_low', np.nan):.3g}",
    )
    if aha_col:
        print(
            f"  {aha_col} coef={summ_base.get('coef_aha', np.nan):+.4f}  p={summ_base.get('p_aha', np.nan):.3g}",
        )
        print(
            f"  temp_low:{aha_col} coef="
            f"{summ_base.get('coef_interaction', np.nan):+.4f}  "
            f"p={summ_base.get('p_interaction', np.nan):.3g}",
        )
    print("  Wrote ->", out_dir)


# ---------- Orchestrate one Aha mode ----------


def evaluate_for_aha_mode(
    combined_in: pd.DataFrame,
    aha_mode_name: str,
    out_dir: str,
    cluster_by: str,
    formal_cfg: Dict[str, Any],
) -> None:
    """
    Run baseline + stage-interaction + stage-specific GLMs and write group accuracies
    for a given Aha mode: 'words' | 'gpt' | 'gpt_broad' | 'formal' | 'none'
    """
    os.makedirs(out_dir, exist_ok=True)
    mode_df = combined_in.copy()

    # Select aha column name
    if aha_mode_name == "words":
        aha_col = "aha_words"
    elif aha_mode_name in ("gpt", "gpt_broad"):
        aha_col = "aha"
    elif aha_mode_name == "formal":
        # formal uses problem-step flags; compute by run and AND with sample GPT
        mode_df = attach_formal_sample_level(
            mode_df,
            delta1=float(formal_cfg["delta1"]),
            delta2=float(formal_cfg["delta2"]),
            min_prior_steps=int(formal_cfg["min_prior_steps"]),
        )
        aha_col = "aha_formal"
    else:
        aha_col = None  # 'none'

    # -------- Baseline GLM --------
    summ_base, coef_base = fit_glm_binomial(
        mode_df,
        aha_col=aha_col,
        cluster_by=cluster_by,
        out_txt=os.path.join(out_dir, "h2_glm_summary.txt"),
    )
    coef_base.to_csv(os.path.join(out_dir, "h2_glm_coefficients.csv"), index=False)

    # -------- Stage-interaction GLM --------
    summ_int, coef_int = fit_glm_stage_interaction(
        mode_df,
        aha_col=aha_col,
        cluster_by=cluster_by,
        out_txt=os.path.join(out_dir, "h2_stage_glm_summary.txt"),
    )
    coef_int["which"] = "interaction"
    coef_int.to_csv(os.path.join(out_dir, "h2_stage_glm_coefficients.csv"), index=False)

    # -------- Stage-specific GLMs --------
    _run_stage_specific_glms(
        mode_df,
        aha_col=aha_col,
        cluster_by=cluster_by,
        out_dir=out_dir,
    )

    # -------- Group accuracies, counts, and console summary --------
    _write_group_outputs(
        mode_df=mode_df,
        aha_col=aha_col,
        aha_mode_name=aha_mode_name,
        out_dir=out_dir,
        glm_summaries={
            "baseline": summ_base,
            "stage_interaction": summ_int,
        },
    )


# ---------- main ----------


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct and return the CLI argument parser for H2 temp×Aha analysis.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_high",
        nargs="?",
        default=None,
        help="Results root for baseline temp (e.g., GRPO-1.5B).",
    )
    parser.add_argument(
        "root_low",
        nargs="?",
        default=None,
        help="Results root for low-temp run (e.g., GRPO-1.5B-low-temp).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="Optional list of >=2 result roots at different temperatures. "
        "When provided, all listed runs are intersected and analyzed jointly "
        "with a standardized temperature regressor.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Record-level split filter (e.g., 'test').",
    )
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <root_high>/h2_temp_aha).",
    )
    parser.add_argument(
        "--ignore_sample_idx_overlap",
        action="store_true",
        help="When intersecting runs, ignore sample_idx and match only on (problem, step).",
    )

    # Aha selection
    parser.add_argument(
        "--aha",
        choices=["words", "gpt", "gpt_broad", "formal", "none", "all"],
        default="gpt",
        help="Aha feature(s) to evaluate.",
    )
    parser.add_argument(
        "--gate_gpt_by_words",
        action="store_true",
        help="Require GPT Aha to also have Words cue (aha_gpt ⊆ aha_words).",
    )

    # Formal thresholds
    parser.add_argument(
        "--formal_delta1",
        type=float,
        default=0.13,
        help=("Formal prior failure threshold on freq_correct (default 0.13)."),
    )
    parser.add_argument(
        "--formal_delta2",
        type=float,
        default=0.13,
        help=("Formal prior shift-stability threshold on aha_rate_gpt (default 0.13)."),
    )
    parser.add_argument(
        "--formal_min_prior_steps",
        type=int,
        default=2,
        help="Formal minimum prior steps before eligibility (default 2).",
    )

    # Stage bucketing
    parser.add_argument(
        "--stage_mode",
        choices=["quantile", "fixed"],
        default="quantile",
        help="Stage definition: tertiles (quantile) or fixed cutpoints.",
    )
    parser.add_argument(
        "--stage_quantiles",
        nargs=2,
        type=float,
        default=(0.33, 0.66),
        help="Quantiles for early/mid/late when stage_mode=quantile.",
    )
    parser.add_argument(
        "--stage_bounds",
        nargs=2,
        type=int,
        default=None,
        help="Fixed cutpoints (lo hi) when stage_mode=fixed.",
    )

    # GLM cov type
    parser.add_argument(
        "--cluster_by",
        choices=["problem", "none"],
        default="problem",
        help="Covariance type for GLM (default: cluster by problem).",
    )
    return parser


def _load_and_overlap_runs(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load high/low temperature runs, restrict to overlapping units, and combine.
    """
    df_high = load_samples(
        args.root_high,
        args.split,
        aha_mode="gpt",
        gate_gpt_by_words=args.gate_gpt_by_words,
    )
    df_low = load_samples(
        args.root_low,
        args.split,
        aha_mode="gpt",
        gate_gpt_by_words=args.gate_gpt_by_words,
    )
    df_high = _apply_step_bounds(df_high, args.min_step, args.max_step)
    df_low = _apply_step_bounds(df_low, args.min_step, args.max_step)
    if df_high.empty or df_low.empty:
        raise SystemExit(
            "One directory has no usable sample-level rows. Check --split and file structure.",
        )

    df_high_overlap, df_low_overlap, overlap_keys = restrict_to_overlap(
        df_high,
        df_low,
        use_sample_idx=not args.ignore_sample_idx_overlap,
    )

    df_high_overlap["temp_low"] = 0
    df_high_overlap["run_label"] = "high"
    df_low_overlap["temp_low"] = 1
    df_low_overlap["run_label"] = "low"

    columns = [
        "dataset",
        "model",
        "split",
        "problem",
        "step",
        "sample_idx",
        "correct",
        "aha_words",
        "aha_gpt",
        "aha",
        "temp_low",
        "run_label",
    ]
    combined_df = pd.concat(
        [df_high_overlap[columns], df_low_overlap[columns]],
        ignore_index=True,
    )
    combined_df["correct"] = combined_df["correct"].astype(int)
    combined_df["temp_low"] = combined_df["temp_low"].astype(int)
    return combined_df, df_high_overlap, overlap_keys


def _assign_stage_labels(
    combined_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attach stage labels (early/mid/late) based on the chosen stage mode.
    """
    if args.stage_mode == "fixed":
        stage_df, stage_info = assign_stage(
            combined_df,
            mode="fixed",
            bounds=tuple(args.stage_bounds) if args.stage_bounds else None,
        )
    else:
        quantile_1, quantile_2 = tuple(args.stage_quantiles)
        stage_df, stage_info = assign_stage(
            combined_df,
            mode="quantile",
            quantiles=(quantile_1, quantile_2),
        )
    return stage_df, stage_info


def _resolve_aha_modes(args: argparse.Namespace) -> List[str]:
    """
    Determine which Aha modes to run based on CLI arguments.
    """
    if args.aha == "all":
        return ["words", "gpt", "formal"]
    if args.aha == "gpt_broad":
        return ["gpt_broad"]
    if args.aha == "none":
        return ["none"]
    return [args.aha]


def _parse_temp_from_path(path_str: str) -> float:
    """
    Try to extract a numeric temperature value from a results directory name.
    """
    match = re.search(r"temp[-_]?([0-9.]+)", path_str)
    if not match:
        raise SystemExit(f"Unable to parse temperature from path: {path_str}")
    return float(match.group(1))


def _run_modes_for_combined(
    combined_df: pd.DataFrame,
    modes: List[str],
    args: argparse.Namespace,
    out_root: str,
    formal_cfg: Dict[str, Any],
) -> None:
    """
    Loop over requested modes, optionally recomputing broad GPT Aha flags.
    """
    for mode_name in modes:
        mode_out_dir = os.path.join(out_root, mode_name) if len(modes) > 1 else out_root
        mode_df = combined_df.copy()

        if mode_name == "gpt_broad":
            df_high_broad = load_samples(
                args.root_high,
                args.split,
                aha_mode="gpt_broad",
                gate_gpt_by_words=args.gate_gpt_by_words,
            )
            df_low_broad = load_samples(
                args.root_low,
                args.split,
                aha_mode="gpt_broad",
                gate_gpt_by_words=args.gate_gpt_by_words,
            )
            df_high_broad, df_low_broad, _ = restrict_to_overlap(
                df_high_broad,
                df_low_broad,
                use_sample_idx=not args.ignore_sample_idx_overlap,
            )
            df_high_broad["temp_low"] = 0
            df_high_broad["run_label"] = "high"
            df_low_broad["temp_low"] = 1
            df_low_broad["run_label"] = "low"
            broad_columns = ["problem", "step", "sample_idx", "aha_words", "aha_gpt"]
            mode_df = mode_df.drop(columns=["aha_words", "aha_gpt"], errors="ignore").merge(
                pd.concat(
                    [df_high_broad[broad_columns], df_low_broad[broad_columns]],
                ),
                on=["problem", "step", "sample_idx"],
                how="left",
            )

        evaluate_for_aha_mode(
            mode_df,
            mode_name,
            mode_out_dir,
            cluster_by=args.cluster_by,
            formal_cfg=formal_cfg,
        )


def main() -> None:
    """
    CLI entry point for H2 temperature×Aha (+ stages, overlap-only) analysis.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.roots:
        run_multi_temperature_mode(args)
        return
    if not args.root_high or not args.root_low:
        parser.error("root_high and root_low are required unless --roots is provided.")
    out_root = args.out_dir or os.path.join(args.root_high, "h2_temp_aha")
    os.makedirs(out_root, exist_ok=True)

    combined_df, df_high_overlap, overlap_keys = _load_and_overlap_runs(
        args,
    )
    combined_df, stage_info = _assign_stage_labels(combined_df, args)

    info_header = {
        "keys_used": overlap_keys,
        "n_overlap_units_per_run": int(len(df_high_overlap)),
        "n_combined_rows": int(len(combined_df)),
        "root_high": args.root_high,
        "root_low": args.root_low,
        "stage_info": stage_info,
    }
    with open(
        os.path.join(out_root, "h2_stage_info.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(info_header, handle, indent=2)

    formal_cfg = {
        "delta1": args.formal_delta1,
        "delta2": args.formal_delta2,
        "min_prior_steps": args.formal_min_prior_steps,
    }

    modes = _resolve_aha_modes(args)
    _run_modes_for_combined(
        combined_df=combined_df,
        modes=modes,
        args=args,
        out_root=out_root,
        formal_cfg=formal_cfg,
    )

    print("\n=== H2 Temperature x Aha! (+ stages; overlap only) ===")
    print(f"Keys used for overlap: {overlap_keys}")
    print(f"N units per run after overlap: {len(df_high_overlap)}")
    print(f"N combined rows (both temps): {len(combined_df)}")
    if stage_info.get("mode") == "fixed":
        print(
            f"Stage cuts (fixed): early <= {stage_info['bounds'][0]} < mid <= {stage_info['bounds'][1]} < late",
        )
    else:
        cutpoints = stage_info.get("cutpoints", [])
        quantiles = stage_info.get("quantiles", [])
    print(
        f"Stage cuts (quantiles {quantiles}): early <= {cutpoints[0]} < mid <= {cutpoints[1]} < late",
    )
    print("Output root:", out_root)


def _load_multi_temp_frames(
    root_paths: List[str],
    split_filter: Optional[str],
    aha_mode_for_loading: str,
    gate_gpt_by_words: bool,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Tuple[List[pd.DataFrame], List[float]]:
    """
    Load multiple result roots and attach their temperature values.
    """
    frames: List[pd.DataFrame] = []
    temps: List[float] = []
    for root in root_paths:
        df = load_samples(
            root,
            split_filter,
            aha_mode=aha_mode_for_loading,
            gate_gpt_by_words=gate_gpt_by_words,
        )
        if df.empty:
            raise SystemExit(f"No usable rows found under {root}")
        df = _apply_step_bounds(df, min_step, max_step)
        temp_value = _parse_temp_from_path(root)
        df["run_label"] = Path(root).name
        df["temp_value"] = temp_value
        frames.append(df)
        temps.append(temp_value)
        print(f"[info] Loaded {len(df):,} rows from {root} (temp={temp_value})")
    return frames, temps


def _fit_multi_temp_glm(
    mode_df: pd.DataFrame,
    aha_col: Optional[str],
    cluster_by: str,
    out_dir: str,
    mode_name: str,
) -> None:
    """
    Fit correct ~ C(problem) + temp_std (+ aha) for the stacked multi-temp data.
    """
    stats_module, stats_formula = lazy_import_statsmodels()
    glm_df = mode_df.copy()
    if glm_df["temp_value"].nunique() < 2:
        raise SystemExit("Need at least two distinct temperatures for multi-temp GLM.")
    glm_df["temp_std"] = (glm_df["temp_value"] - glm_df["temp_value"].mean()) / (
        glm_df["temp_value"].std(ddof=0) + 1e-8
    )

    formula_terms = ["C(problem)", "temp_std"]
    if aha_col:
        formula_terms.append(aha_col)
    formula = "correct ~ " + " + ".join(formula_terms)

    if cluster_by == "problem":
        cov_type, cov_kwds = (
            "cluster",
            {
                "groups": pd.Categorical(glm_df["problem"]).codes,
                "use_correction": True,
                "df_correction": True,
            },
        )
    else:
        cov_type, cov_kwds = "HC1", {}

    try:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except TypeError:
        res = stats_formula.glm(
            formula,
            data=glm_df,
            family=stats_module.families.Binomial(),
        ).fit(cov_type=cov_type)

    mode_dir = os.path.join(out_dir, mode_name)
    os.makedirs(mode_dir, exist_ok=True)
    with open(
        os.path.join(mode_dir, "multi_temp_glm_summary.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(res.summary().as_text())
        handle.write(f"\n\nFormula: {formula}\nCovariance: {cov_type}\n")

    coef_df = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "se": res.bse.values,
            "z": (res.params.values / np.where(res.bse.values == 0, np.nan, res.bse.values)),
            "p": res.pvalues.values,
        },
    )
    coef_df.to_csv(os.path.join(mode_dir, "multi_temp_glm_coefficients.csv"), index=False)

    metrics: Dict[str, Any] = {
        "N": int(len(glm_df)),
        "coef_temp_std": float(res.params.get("temp_std", float("nan"))),
        "p_temp_std": float(res.pvalues.get("temp_std", float("nan"))),
    }

    if aha_col:
        mask_aha1 = glm_df[aha_col] == 1
        mask_aha0 = glm_df[aha_col] == 0
        acc1 = float(glm_df.loc[mask_aha1, "correct"].mean()) if mask_aha1.any() else float("nan")
        acc0 = float(glm_df.loc[mask_aha0, "correct"].mean()) if mask_aha0.any() else float("nan")
        metrics.update(
            {
                "share_shift": float(glm_df[aha_col].mean()),
                "acc_shift": acc1,
                "acc_no_shift": acc0,
                "delta_pp": (acc1 - acc0) * 100.0 if np.isfinite(acc1) and np.isfinite(acc0) else float("nan"),
                "coef_shift": float(res.params.get(aha_col, float("nan"))),
                "p_shift": float(res.pvalues.get(aha_col, float("nan"))),
            },
        )
        data_with_aha1 = glm_df.copy()
        data_with_aha1[aha_col] = 1
        data_with_aha0 = glm_df.copy()
        data_with_aha0[aha_col] = 0
        metrics["AME"] = float(np.mean(res.predict(data_with_aha1) - res.predict(data_with_aha0)))
    else:
        metrics.update(
            {
                "share_shift": float("nan"),
                "acc_shift": float("nan"),
                "acc_no_shift": float("nan"),
                "delta_pp": float("nan"),
                "coef_shift": float("nan"),
                "p_shift": float("nan"),
                "AME": float("nan"),
            },
        )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(mode_dir, "multi_temp_glm_metrics.csv"),
        index=False,
    )


def run_multi_temperature_mode(args: argparse.Namespace) -> None:
    """
    Load >=2 temperature runs and fit correct ~ C(problem) + temp_std + shift.
    """
    root_paths = args.roots
    if len(root_paths) < 2:
        raise SystemExit("Provide at least two --roots directories for multi-temperature mode.")

    load_mode = "gpt_broad" if args.aha == "gpt_broad" else "gpt"
    frames, temps = _load_multi_temp_frames(
        root_paths,
        args.split,
        load_mode,
        args.gate_gpt_by_words,
        args.min_step,
        args.max_step,
    )
    filtered_frames, overlap_keys = restrict_to_overlap_many(
        frames,
        use_sample_idx=not args.ignore_sample_idx_overlap,
    )
    combined_df = pd.concat(filtered_frames, ignore_index=True)
    combined_df["temp_value"] = combined_df["temp_value"].astype(float)
    print(f"[info] Keys used for overlap: {overlap_keys}")
    print(f"[info] Combined rows (all temps): {len(combined_df):,}")

    out_root = args.out_dir or os.path.join(root_paths[0], "h2_temp_aha_multi")
    os.makedirs(out_root, exist_ok=True)

    formal_cfg = {
        "delta1": args.formal_delta1,
        "delta2": args.formal_delta2,
        "min_prior_steps": args.formal_min_prior_steps,
    }

    for mode_name in _resolve_aha_modes(args):
        if mode_name == "gpt_broad" and load_mode != "gpt_broad":
            raise SystemExit("Multi-temperature mode currently supports gpt_broad only when --aha gpt_broad.")
        mode_df = combined_df.copy()
        if mode_name == "formal":
            mode_df = attach_formal_sample_level(
                mode_df,
                delta1=formal_cfg["delta1"],
                delta2=formal_cfg["delta2"],
                min_prior_steps=formal_cfg["min_prior_steps"],
            )
            aha_col = "aha_formal"
        elif mode_name == "words":
            aha_col = "aha_words"
        elif mode_name in ("gpt", "gpt_broad"):
            aha_col = "aha_gpt"
        elif mode_name == "none":
            aha_col = None
        else:
            aha_col = "aha"
        _fit_multi_temp_glm(
            mode_df=mode_df,
            aha_col=aha_col,
            cluster_by=args.cluster_by,
            out_dir=out_root,
            mode_name=mode_name,
        )
    print(f"[info] Multi-temperature outputs -> {out_root}")


if __name__ == "__main__":
    main()
