#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entropy bin regression (per-domain GLM + optional plots/tables).

Builds rows with domain, problem, step, sample, shift_at_1, correct_at_1,
entropy_at_1, entropy_bin_at_1, entropy_bin_label, correct_at_2.

Per-domain regressions (run twice): correct_at_2 ~ correct_at_1 +
  C(entropy_bin_label) + C(problem). Uses cluster-robust SEs; HC1 fallback.

Binning options: --fixed_bins takes precedence; otherwise --binning
  {uniform,quantile} with --bins K, or --equal_n_bins with bin_scope, tie_break,
  random_seed controls.

Entropy modes: sum (think+answer), think, answer, combined.

Outputs per domain/mode: rows__<slug>__<domain>__<mode>.csv,
  model_{none|false}__<slug>__<domain>__<mode>.txt,
  bin_contrasts__{none|false}__<slug>__<domain>__<mode>.csv,
  bin_contrasts__<slug>__<domain>__<mode>.{png,pdf}
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
except (ImportError, ModuleNotFoundError):
    # Optional dependency; functions that need statsmodels will raise when called.
    sm = None  # type: ignore[assignment]
    smf = None  # type: ignore[assignment]
    PerfectSeparationError = Exception  # type: ignore[assignment]

try:
    import src.analysis.io as _io_mod

    scan_files_step_only = getattr(_io_mod, "scan_files_step_only")
    iter_records_from_file = getattr(_io_mod, "iter_records_from_file")
    if scan_files_step_only is None or iter_records_from_file is None:
        raise AttributeError("missing io helpers")
except (ImportError, AttributeError):  # pragma: no cover - tests stub this

    def scan_files_step_only(*_args, **_kwargs):
        """Stub scan_files_step_only used when src.analysis.io is unavailable."""
        raise ImportError("scan_files_step_only unavailable")

    def iter_records_from_file(*_args, **_kwargs):
        """Stub iter_records_from_file used when src.analysis.io is unavailable."""
        return []


from src.analysis.metrics import carpark_success_from_soft_reward, extract_correct
from src.analysis.utils import (
    add_common_plot_args,
    coerce_bool,
    coerce_float,
    gpt_keys_for_mode,
    step_from_rec_or_path,
)


@dataclass(frozen=True)
class BinContrastPlotInputs:
    """Container for per-domain AME data and plot destinations."""

    ame_none: pd.DataFrame
    ame_false: pd.DataFrame
    out_png: str
    out_pdf: str
    title: str


@dataclass(frozen=True)
class DomainRunContext:
    """Metadata for per-domain runs within a particular entropy mode."""

    domain: str
    entropy_mode: str
    slug_mode: str
    out_dir: str


@dataclass(frozen=True)
class RowBuildConfig:
    """Configuration governing how we assemble the modeling rows."""

    carpark_op: str
    carpark_thr: float
    gpt_mode: str
    min_step: Optional[int]
    max_step: Optional[int]


# ------------------------------- Path & step helpers -------------------------------

SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache", "__pycache__"}
MODE_TAG = {"sum": "eSUM", "think": "eTHINK", "answer": "eANSWER", "combined": "eCOMB"}


def scan_files(root: str, split_substr: Optional[str], skip_substrings: Set[str]) -> List[str]:
    """Proxy to the shared scan function so callers retain the same signature."""
    return scan_files_step_only(root, split_substr, skip_substrings)


# ------------------------------- Coercion utils -------------------------------


def both_get(pass_dict: Dict[str, Any], rec: Dict[str, Any], key: str, default=None):
    """Return a value from the pass dict if present, otherwise fall back to rec."""
    value = pass_dict.get(key, None)
    return value if value is not None else rec.get(key, default)


def parse_comma_list(text: Optional[str]) -> List[str]:
    """Return a list of trimmed tokens from a comma-separated string."""
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


# ------------------------------- Domain / record parsing -------------------------------


def domain_from_path(path: str) -> str:
    """Return a coarse domain label inferred from a file path."""
    lower_path = path.lower()
    if ("xword" in lower_path) or ("crossword" in lower_path):
        return "Crossword"
    if ("carpark" in lower_path) or ("rush" in lower_path) or ("parking" in lower_path):
        return "Carpark"
    return "Math"


def get_problem(rec: Dict[str, Any]) -> Optional[str]:
    """Return a stable problem identifier string for logging/aggregation."""
    for key in (
        "problem_id",
        "example_id",
        "id",
        "uid",
        "question",
        "clue",
        "title",
        "problem",
    ):
        value = rec.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            return f"{key}:{value}"
    index = rec.get("sample_idx")
    return f"sample_{index}" if index is not None else None


def get_sample(rec: Dict[str, Any]) -> Optional[str]:
    """Return a short sample identifier (if present)."""
    sample_index = rec.get("sample_idx")
    if sample_index is None:
        return None
    try:
        return f"s{int(sample_index)}"
    except (TypeError, ValueError):
        return f"s{str(sample_index)}"


# ------------------------------- Entropy at pass-1 -------------------------------


def entropy_from_pass1(pass_one: Dict[str, Any], mode: str = "sum") -> Optional[float]:
    """
    Compute entropy_at_1 from pass-1:
      - sum      : entropy_think + entropy_answer (parts first; else fallback to 'entropy')
      - think    : entropy_think (fallback to 'entropy')
      - answer   : entropy_answer (fallback to 'entropy')
      - combined : 'entropy' as-is (fallback to avg(parts) if both available)
    """
    entropy_think_value = coerce_float(pass_one.get("entropy_think"))
    entropy_answer_value = coerce_float(pass_one.get("entropy_answer"))
    entropy_combined_value = coerce_float(pass_one.get("entropy"))

    entropy_estimate = entropy_combined_value
    if mode == "sum":
        parts = [value for value in (entropy_think_value, entropy_answer_value) if value is not None]
        entropy_estimate = float(sum(parts)) if parts else entropy_combined_value
    elif mode == "think":
        entropy_estimate = entropy_think_value if entropy_think_value is not None else entropy_combined_value
    elif mode == "answer":
        entropy_estimate = entropy_answer_value if entropy_answer_value is not None else entropy_combined_value
    elif mode == "combined":
        if entropy_combined_value is not None:
            entropy_estimate = entropy_combined_value
        elif entropy_think_value is not None and entropy_answer_value is not None:
            entropy_estimate = 0.5 * (entropy_think_value + entropy_answer_value)
        else:
            entropy_estimate = entropy_think_value if entropy_think_value is not None else entropy_answer_value

    return entropy_estimate


# shift_at_1 (canonical by default)
def compute_shift_at_1(
    pass1_dict: Dict[str, Any],
    rec: Dict[str, Any],
    gpt_mode: str = "canonical",
) -> Optional[int]:
    """Return 1/0/None according to GPT shift flags for the requested mode."""
    keys = gpt_keys_for_mode(gpt_mode)
    observed_flags: List[int] = []
    for key in keys:
        value = both_get(pass1_dict, rec, key, None)
        if value is not None:
            coerced = coerce_bool(value)
            if coerced is not None:
                observed_flags.append(coerced)
    if not observed_flags:
        return None
    return 1 if any(observed_flags) else 0


# ------------------------------- Build rows (no binning yet) -------------------------------


def _build_row_for_record(
    record: Dict[str, Any],
    domain: str,
    step: int,
    entropy_mode: str,
    config: RowBuildConfig,
) -> Dict[str, Any]:
    """Return a single modeling row constructed from a record payload."""
    pass1_payload = record.get("pass1") or {}
    pass2_payload = record.get("pass2") or {}
    problem = get_problem(record)
    sample = get_sample(record)

    correct_pass1 = extract_correct(pass1_payload, record)
    if correct_pass1 is None:
        correct_pass1 = carpark_success_from_soft_reward(
            record,
            pass1_payload,
            config.carpark_op,
            config.carpark_thr,
        )

    correct_pass2 = None
    if isinstance(pass2_payload, dict) and pass2_payload:
        correct_pass2 = extract_correct(pass2_payload, record)
        if correct_pass2 is None and domain == "Carpark":
            correct_pass2 = carpark_success_from_soft_reward(
                record,
                pass2_payload,
                config.carpark_op,
                config.carpark_thr,
            )
    if correct_pass2 is None:
        correct_pass2 = coerce_bool(record.get("is_correct_after_reconsideration"))

    shift1 = compute_shift_at_1(pass1_payload, record, gpt_mode=config.gpt_mode)
    entropy_value = entropy_from_pass1(pass1_payload, mode=entropy_mode)
    entropy_think = coerce_float(pass1_payload.get("entropy_think"))
    entropy_answer = coerce_float(pass1_payload.get("entropy_answer"))
    return {
        "domain": domain,
        "problem": problem,
        "step": step,
        "sample": sample,
        "shift_at_1": shift1,
        "correct_at_1": correct_pass1,
        "entropy_at_1": entropy_value,
        "entropy_think_at_1": entropy_think,
        "entropy_answer_at_1": entropy_answer,
        "correct_at_2": correct_pass2,
    }


def build_rows(
    files: List[str],
    entropy_mode: str,
    config: RowBuildConfig,
) -> pd.DataFrame:
    """Build the per-row dataset used by the downstream binning and regressions."""
    rows = []
    for path in files:
        domain = domain_from_path(path)
        for record in iter_records_from_file(path):
            step = step_from_rec_or_path(record, path)
            if config.min_step is not None and step < config.min_step:
                continue
            if config.max_step is not None and step > config.max_step:
                continue
            rows.append(
                _build_row_for_record(
                    record,
                    domain,
                    step,
                    entropy_mode,
                    config,
                ),
            )

    model_df = pd.DataFrame(rows)
    # keep rows needed for modeling
    model_df = model_df[
        model_df["domain"].notna()
        & model_df["problem"].notna()
        & model_df["correct_at_1"].notna()
        & model_df["correct_at_2"].notna()
        & model_df["entropy_at_1"].notna()
    ].copy()
    model_df["correct_at_1"] = model_df["correct_at_1"].astype(int)
    model_df["correct_at_2"] = model_df["correct_at_2"].astype(int)
    return model_df


# ------------------------------- Binning --------------------------------------


def parse_fixed_bins(bin_text: Optional[str]) -> Optional[List[float]]:
    """Return explicit bin edges parsed from the CLI string or None when unset."""
    if not bin_text:
        return None
    out = []
    for token in bin_text.split(","):
        normalized = token.strip().lower()
        if normalized in ("inf", "+inf"):
            out.append(float("inf"))
            continue
        if normalized == "-inf":
            out.append(float("-inf"))
            continue
        out.append(float(normalized))
    if len(out) < 2:
        raise SystemExit("--fixed_bins must provide at least two edges (e.g., 0,1,2,inf).")
    return out


def compute_edges(
    values: np.ndarray,
    binning: str,
    bins: int,
    fixed: Optional[List[float]],
) -> List[float]:
    """Return bin edges using uniform/quantile logic or user-specified edges."""
    if fixed is not None:
        return fixed
    if binning == "uniform":
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise SystemExit("Uniform binning requires at least one finite value.")
        low_edge, high_edge = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
        if not np.isfinite(low_edge) or not np.isfinite(high_edge):
            raise SystemExit("Non-finite min/max for uniform bins.")
        if high_edge <= low_edge:
            high_edge = low_edge + 1e-9
        return list(np.linspace(low_edge, high_edge, bins + 1))
    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = list(np.quantile(values, quantiles))
        eps = 1e-9
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        return edges
    raise SystemExit(f"Unknown --binning '{binning}'.")


def label_interval(interval):
    """Return a string label for a pandas Interval or propagate NaN."""
    if pd.isna(interval):
        return np.nan
    if isinstance(interval, str):
        return interval
    try:
        return f"[{interval.left:g},{interval.right:g})"
    except AttributeError:
        return np.nan


def apply_binning_cut(
    data_frame: pd.DataFrame,
    edges: List[float],
    scope: str = "global",
) -> pd.DataFrame:
    """Apply pandas.cut binning globally or per-domain and label categories."""
    data_frame = data_frame.copy()
    if scope == "domain":
        parts = []
        for _, sub in data_frame.groupby("domain"):
            parts.append(pd.cut(sub["entropy_at_1"], bins=edges, right=False, include_lowest=True))
        data_frame["entropy_bin_at_1"] = pd.concat(parts).sort_index()
    else:
        data_frame["entropy_bin_at_1"] = pd.cut(
            data_frame["entropy_at_1"], bins=edges, right=False, include_lowest=True
        )
    cats = data_frame["entropy_bin_at_1"].cat.categories
    labels = [label_interval(interval) for interval in cats]
    data_frame["entropy_bin_label"] = data_frame["entropy_bin_at_1"].cat.rename_categories(labels)
    data_frame["entropy_bin_label"] = data_frame["entropy_bin_label"].astype(
        pd.api.types.CategoricalDtype(categories=labels, ordered=True)
    )
    return data_frame


# -------- Equal-N (rank-based) binning with tie-breaking (global or per-domain) --------


def _rank_based_bins(series_values: pd.Series, bins: int, tie_break: str, seed: int) -> np.ndarray:
    """Assign rank-based bin ids with optional jitter tie-breaking."""
    series_copy = series_values.copy()
    mask = series_copy.notna()
    valid_count = int(mask.sum())
    if valid_count == 0:
        return np.full(len(series_copy), -1, dtype=int)
    if tie_break == "random":
        rng = np.random.default_rng(seed)
        scale = max(1.0, float(np.nanstd(series_copy.values)))
        jitter = rng.uniform(-1e-9 * scale, 1e-9 * scale, size=valid_count)
        vals = series_copy[mask].values.astype(float) + jitter
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(valid_count, dtype=float)
        ranks[order] = np.arange(1, valid_count + 1, dtype=float)
    else:  # stable
        ranks = series_copy[mask].rank(method="first").values
    idx = np.floor((ranks - 1) * bins / valid_count).astype(int)
    idx[idx < 0] = 0
    idx[idx >= bins] = bins - 1
    out = np.full(len(series_copy), -1, dtype=int)
    out[mask.values] = idx
    return out


def apply_equal_n_binning(
    data_frame: pd.DataFrame,
    bins: int,
    scope: str = "global",
    tie_break: str = "stable",
    seed: int = 42,
) -> pd.DataFrame:
    """Assign equal-count bins either globally or per-domain with tie-breaking."""
    data_frame = data_frame.copy()

    def _apply(sub: pd.DataFrame) -> pd.DataFrame:
        idx = _rank_based_bins(sub["entropy_at_1"], bins=bins, tie_break=tie_break, seed=seed)
        sub = sub.copy()
        sub["_bin_id"] = idx
        labels = []
        for bin_index in range(bins):
            values = sub.loc[sub["_bin_id"] == bin_index, "entropy_at_1"]
            if values.empty:
                labels.append(f"Q{bin_index + 1} [∅]")
            else:
                lower_bound = float(np.min(values))
                upper_bound = float(np.max(values))
                labels.append(f"Q{bin_index + 1} [{lower_bound:g},{upper_bound:g})")
        cat = pd.api.types.CategoricalDtype(categories=labels, ordered=True)
        label_lookup = [labels[bin_id] if bin_id >= 0 else np.nan for bin_id in sub["_bin_id"]]
        sub["entropy_bin_label"] = pd.Categorical(label_lookup, dtype=cat)
        id_cat = pd.api.types.CategoricalDtype(categories=list(range(bins)), ordered=True)
        id_lookup = [bin_id if bin_id >= 0 else np.nan for bin_id in sub["_bin_id"]]
        sub["entropy_bin_at_1"] = pd.Categorical(id_lookup, dtype=id_cat)
        sub.drop(columns=["_bin_id"], inplace=True)
        return sub

    if scope == "domain":
        parts = []
        for _, group_df in data_frame.groupby("domain"):
            parts.append(_apply(group_df))
        data_frame = pd.concat(parts, axis=0).sort_index()
    else:
        data_frame = _apply(data_frame)
    return data_frame


# ------------------------------- Modeling helpers -------------------------------


def prune_subset(sub: pd.DataFrame, min_rows_per_problem: int = 2) -> pd.DataFrame:
    """Remove problems lacking sufficient rows or outcome variation."""
    if sub.empty:
        return sub
    grouped = sub.groupby("problem")["correct_at_2"].agg(n="size", nunq="nunique").reset_index()
    keep = grouped[(grouped["n"] >= min_rows_per_problem) & (grouped["nunq"] > 1)]["problem"]
    dropped = len(grouped) - len(keep)
    if dropped > 0:
        print(
            f"[prune] Dropping {dropped} problem(s) with <{min_rows_per_problem} rows or no outcome variation.",
        )
    sub = sub[sub["problem"].isin(keep)].copy()
    if "entropy_bin_label" in sub.columns and hasattr(sub["entropy_bin_label"], "cat"):
        sub["entropy_bin_label"] = sub["entropy_bin_label"].cat.remove_unused_categories()
    return sub


def fit_clustered_glm(
    data_frame: pd.DataFrame,
    formula: str,
    cluster_col: str,
):
    """Fit a GLM with cluster-robust SEs and fallback to HC1 if needed."""
    if sm is None or smf is None:
        raise ImportError("statsmodels is required: pip install statsmodels")
    model = smf.glm(formula=formula, data=data_frame, family=sm.families.Binomial())
    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": data_frame[cluster_col]})
        if not np.isfinite(result.bse).all():
            print("[warn] Cluster SEs contain non-finite values; falling back to HC1 robust.")
            result = model.fit(cov_type="HC1")
        covariance_used = result.cov_type
    except (np.linalg.LinAlgError, ValueError, PerfectSeparationError) as exc:
        print(f"[warn] Cluster sandwich failed ({exc}). Falling back to HC1 robust.")
        result = model.fit(cov_type="HC1")
        covariance_used = "HC1"
    return result, result.summary().as_text(), covariance_used


def compute_bin_ame(
    result,
    df_model: pd.DataFrame,
    bin_col: str,
    baseline_label: str,
) -> pd.DataFrame:
    """Compute average marginal effects for each bin relative to baseline."""
    if not isinstance(df_model[bin_col].dtype, CategoricalDtype):
        df_model = df_model.copy()
        df_model[bin_col] = df_model[bin_col].astype("category")
    cats = df_model[bin_col].cat.categories
    cat_dtype = CategoricalDtype(categories=cats, ordered=True)

    base = df_model.copy()
    base[bin_col] = pd.Categorical([baseline_label] * len(base), dtype=cat_dtype)
    p_base = result.predict(base)

    out_rows = []
    for bin_label in cats:
        cur = df_model.copy()
        cur[bin_col] = pd.Categorical([bin_label] * len(cur), dtype=cat_dtype)
        p_cur = result.predict(cur)
        ame = float(np.mean(p_cur - p_base))
        out_rows.append({"bin": bin_label, "ame": ame, "n_rows": int(len(cur))})
    return pd.DataFrame(out_rows)


# ------------------------------- Plotting -------------------------------


def plot_bin_contrasts(plot_inputs: BinContrastPlotInputs, dpi: int = 300) -> None:
    """Render AME bar plots comparing bin contrasts for two shift subsets."""
    bins = list(plot_inputs.ame_none["bin"]) if not plot_inputs.ame_none.empty else list(plot_inputs.ame_false["bin"])
    bin_positions = np.arange(len(bins))
    width = 0.38

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    bar_fn = getattr(axis, "bar", None)
    if bar_fn and not plot_inputs.ame_none.empty:
        bar_fn(
            bin_positions - width / 2,
            plot_inputs.ame_none["ame"].values,
            width,
            label="shift_at_1 is None",
        )
    if bar_fn and not plot_inputs.ame_false.empty:
        bar_fn(
            bin_positions + width / 2,
            plot_inputs.ame_false["ame"].values,
            width,
            label="shift_at_1 == 0",
        )

    xticks_fn = getattr(axis, "set_xticks", None)
    if xticks_fn:
        xticks_fn(bin_positions, bins, rotation=0)
    ylabel_fn = getattr(axis, "set_ylabel", None)
    if ylabel_fn:
        ylabel_fn("AME (Δ pp vs baseline bin)")
    title_fn = getattr(axis, "set_title", None)
    if title_fn:
        title_fn(plot_inputs.title)
    grid_fn = getattr(axis, "grid", None)
    if grid_fn:
        grid_fn(True, axis="y", alpha=0.3)
    axhline_fn = getattr(axis, "axhline", None)
    if axhline_fn:
        axhline_fn(0.0, linewidth=1, color="black")

    legend_fn = getattr(axis, "legend", None)
    if legend_fn:
        legend_fn(loc="best")
    fig.tight_layout()
    fig.savefig(plot_inputs.out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(plot_inputs.out_pdf, bbox_inches="tight")
    plt.close(fig)


# ------------------------------- Main ---------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for entropy-bin regression analyses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_root", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)

    # Path filters (e.g., GRPO-1.5B & temp-0.7)
    parser.add_argument(
        "--path_include",
        type=str,
        default=None,
        help="Comma-separated substrings; all must appear in PATH (e.g., 'GRPO-1.5B,temp-0.7').",
    )
    parser.add_argument(
        "--path_exclude",
        type=str,
        default=None,
        help="Comma-separated substrings; any match excludes the file.",
    )

    # Binning
    parser.add_argument("--bin_scope", choices=["global", "domain"], default="global")
    parser.add_argument(
        "--binning",
        choices=["fixed", "uniform", "quantile"],
        default="uniform",
        help="Ignored if --fixed_bins or --equal_n_bins is set.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of bins for uniform/quantile/equal_n.",
    )
    parser.add_argument(
        "--fixed_bins",
        type=str,
        default=None,
        help="Comma-separated edges, e.g., '0,0.75,1.25,2,inf' (overrides --binning).",
    )

    # Equal-N rank binning
    parser.add_argument(
        "--equal_n_bins",
        action="store_true",
        help="Force rank-based equal-count bins (exact quartiles when --bins 4).",
    )
    parser.add_argument(
        "--tie_break",
        choices=["stable", "random"],
        default="stable",
        help="Tie handling for equal-N bins: stable (deterministic) or random (tiny jitter).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for --tie_break random.",
    )

    # Carpark & shift
    parser.add_argument("--carpark_success_op", choices=["gt", "ge", "eq"], default="gt")
    parser.add_argument("--carpark_soft_threshold", type=float, default=0.0)
    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")

    # Domains
    parser.add_argument("--domains", type=str, default="Crossword,Math,Carpark")

    # Entropy modes
    parser.add_argument(
        "--entropy_mode",
        choices=["sum", "think", "answer", "combined"],
        default=None,
        help="Run a single entropy mode.",
    )
    parser.add_argument(
        "--entropy_modes",
        nargs="+",
        choices=["sum", "think", "answer", "combined"],
        default=None,
        help="Run multiple entropy modes in one go (e.g., --entropy_modes sum think answer).",
    )

    # Pruning
    parser.add_argument(
        "--min_rows_per_problem",
        type=int,
        default=2,
        help="Drop problems with fewer rows than this (prevents separation).",
    )

    # Output / misc
    add_common_plot_args(parser)
    parser.add_argument("--debug", action="store_true")
    return parser


def _resolve_entropy_modes(args: argparse.Namespace) -> List[str]:
    """Return the entropy modes requested on the CLI (defaulting to all)."""
    if args.entropy_modes:
        return args.entropy_modes
    if args.entropy_mode:
        return [args.entropy_mode]
    return ["sum", "think", "answer"]


def _scan_and_filter_files(args: argparse.Namespace) -> List[str]:
    """Scan for candidate files and apply include/exclude filters."""
    skip_set = {s.lower() for s in SKIP_DIR_DEFAULT}
    files_all = scan_files_step_only(args.scan_root, args.split, skip_set)
    if not files_all:
        sys.exit("No files found. Check --scan_root / --split.")

    include_filters = [pattern.lower() for pattern in parse_comma_list(args.path_include)]
    exclude_filters = [pattern.lower() for pattern in parse_comma_list(args.path_exclude)]

    filtered: List[str] = []
    for file_path in files_all:
        normalized = file_path.lower()
        if include_filters and not all(pattern in normalized for pattern in include_filters):
            continue
        if exclude_filters and any(pattern in normalized for pattern in exclude_filters):
            continue
        filtered.append(file_path)
    if not filtered:
        sys.exit("No files left after path filters. Adjust --path_include/--path_exclude.")
    return filtered


def _apply_binning_strategy(
    df_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, str]:
    """Apply the requested binning method and return descriptive metadata."""
    fixed_edges = parse_fixed_bins(args.fixed_bins)
    if fixed_edges is not None:
        df_rows = apply_binning_cut(df_rows, fixed_edges, scope=args.bin_scope)
        return df_rows, f"fixed edges: {fixed_edges}"
    if args.equal_n_bins:
        df_rows = apply_equal_n_binning(
            df_rows,
            bins=args.bins,
            scope=args.bin_scope,
            tie_break=args.tie_break,
            seed=args.random_seed,
        )
        bins_info = (
            f"equal_n_bins scope={args.bin_scope}, bins={args.bins}, "
            f"tie_break={args.tie_break}, seed={args.random_seed}"
        )
        return df_rows, bins_info
    if args.binning == "fixed":
        sys.exit("Provide --fixed_bins when --binning fixed; or use --equal_n_bins.")
    edges = compute_edges(df_rows["entropy_at_1"].to_numpy(), args.binning, args.bins, None)
    df_rows = apply_binning_cut(df_rows, edges, scope=args.bin_scope)
    return df_rows, f"{args.binning} edges: {edges}"


def _fit_subset_and_save(
    subset_tag: str,
    subset_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    args: argparse.Namespace,
    context: DomainRunContext,
) -> pd.DataFrame:
    """Prune, fit, and persist a GLM for a specific subset."""
    if subset_df.empty:
        print(
            f"[{context.domain}:{context.entropy_mode}:{subset_tag}] subset empty; skipping model.",
        )
        return subset_df

    subset_df["entropy_bin_label"] = pd.Categorical(
        subset_df["entropy_bin_label"],
        categories=domain_df["entropy_bin_label"].cat.categories,
        ordered=True,
    )

    rows_before = len(subset_df)
    subset_df = prune_subset(subset_df, min_rows_per_problem=args.min_rows_per_problem)
    print(
        f"[{context.domain}:{context.entropy_mode}:{subset_tag}] "
        f"kept {len(subset_df)}/{rows_before} rows after pruning",
    )

    subset_df["entropy_bin_label"] = subset_df["entropy_bin_label"].cat.remove_unused_categories()

    if subset_df.empty or subset_df["problem"].nunique() < 2:
        print(
            f"[{context.domain}:{context.entropy_mode}:{subset_tag}] "
            "Not enough variation after pruning; skipping model.",
        )
        return subset_df

    formula = "correct_at_2 ~ correct_at_1 + C(entropy_bin_label) + C(problem)"
    result, summary_text, cov_used = fit_clustered_glm(subset_df, formula, cluster_col="problem")
    txt_path = os.path.join(
        context.out_dir,
        f"model_{subset_tag}__{context.slug_mode}__{context.domain}.txt",
    )
    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"# cov_type: {cov_used}\n")
        file_obj.write(summary_text + "\n")
    print(f"[saved] {txt_path}")

    baseline_label = subset_df["entropy_bin_label"].cat.categories[0]
    ame_df = compute_bin_ame(result, subset_df, "entropy_bin_label", baseline_label)
    ame_csv = os.path.join(
        context.out_dir,
        f"bin_contrasts__{subset_tag}__{context.slug_mode}__{context.domain}.csv",
    )
    ame_df.to_csv(ame_csv, index=False)
    print(f"[saved] {ame_csv}")

    return subset_df


def _process_domain(
    context: DomainRunContext,
    args: argparse.Namespace,
    df_dom: pd.DataFrame,
) -> None:
    """Handle modelling and plotting for a single domain."""
    out_dir = context.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if not isinstance(df_dom["entropy_bin_label"].dtype, CategoricalDtype):
        cats_dom = df_dom["entropy_bin_label"].dropna().drop_duplicates().tolist()
        df_dom["entropy_bin_label"] = pd.Categorical(
            df_dom["entropy_bin_label"],
            categories=cats_dom,
            ordered=True,
        )

    rows_csv = os.path.join(out_dir, f"rows__{context.slug_mode}__{context.domain}.csv")
    df_dom.to_csv(rows_csv, index=False)
    print(f"[{context.domain}:{context.entropy_mode}] rows: {len(df_dom):d}  -> {rows_csv}")

    subsets = {
        "none": df_dom[df_dom["shift_at_1"].isna()].copy(),
        "false": df_dom[df_dom["shift_at_1"] == 0].copy(),
        "true": df_dom[df_dom["shift_at_1"] == 1].copy(),
    }

    if args.debug:
        print(f"[{context.domain}:{context.entropy_mode}] shift_at_1 counts (NaN/0/1):")
        print(df_dom["shift_at_1"].value_counts(dropna=False).to_string())
        for tag in ("none", "false", "true"):
            print(
                f"[{context.domain}:{context.entropy_mode}:{tag}] subset rows: {len(subsets[tag])}",
            )

    for tag in ("none", "false", "true"):
        subsets[tag] = _fit_subset_and_save(
            subset_tag=tag,
            subset_df=subsets[tag],
            domain_df=df_dom,
            args=args,
            context=context,
        )

    if not args.make_plot:
        return

    def _load(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=["bin", "ame", "n_rows"])

    ame_none = _load(
        os.path.join(
            out_dir,
            f"bin_contrasts__none__{context.slug_mode}__{context.domain}.csv",
        ),
    )
    ame_false = _load(
        os.path.join(
            out_dir,
            f"bin_contrasts__false__{context.slug_mode}__{context.domain}.csv",
        ),
    )
    if ame_none.empty and ame_false.empty:
        return

    png = os.path.join(
        out_dir,
        f"bin_contrasts__{context.slug_mode}__{context.domain}.png",
    )
    pdf = os.path.join(
        out_dir,
        f"bin_contrasts__{context.slug_mode}__{context.domain}.pdf",
    )
    title = f"{context.domain} — Entropy-bin contrasts (Δ pp vs baseline) — {args.model_name} [{context.entropy_mode}]"
    plot_data = BinContrastPlotInputs(
        ame_none=ame_none,
        ame_false=ame_false,
        out_png=png,
        out_pdf=pdf,
        title=title,
    )
    plot_bin_contrasts(plot_data, dpi=args.dpi)
    print(f"[saved] {png}\n[saved] {pdf}")


def _run_entropy_mode(
    emode: str,
    args: argparse.Namespace,
    files: List[str],
    keep_domains: Optional[Set[str]],
    out_root_base: str,
) -> None:
    """Process one entropy mode end-to-end."""
    print(f"\n=== Running entropy mode: {emode} ===")
    row_config = RowBuildConfig(
        carpark_op=args.carpark_success_op,
        carpark_thr=args.carpark_soft_threshold,
        gpt_mode=args.gpt_mode,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    df_rows = build_rows(
        files=files,
        entropy_mode=emode,
        config=row_config,
    )
    if args.debug:
        print(f"[info] built rows: n={len(df_rows)}")

    if keep_domains:
        df_rows = df_rows[df_rows["domain"].isin(keep_domains)]
    if df_rows.empty:
        print("[warn] No rows after --domains filter; skipping this mode.")
        return

    df_rows, bins_info = _apply_binning_strategy(df_rows, args)

    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    slug_mode = f"{slug}__{MODE_TAG[emode]}"
    out_root = os.path.join(out_root_base, MODE_TAG[emode])
    os.makedirs(out_root, exist_ok=True)

    for dom, df_dom in df_rows.groupby("domain", sort=False):
        domain_out_dir = os.path.join(out_root, dom.replace(" ", "_"))
        context = DomainRunContext(
            domain=dom,
            entropy_mode=emode,
            slug_mode=slug_mode,
            out_dir=domain_out_dir,
        )
        _process_domain(
            context=context,
            args=args,
            df_dom=df_dom,
        )

    if args.debug:
        with pd.option_context("display.width", 160):
            print("\n[Rows head]")
            print(df_rows.head(6).to_string(index=False))
            print("\nBins used:", bins_info)


def main() -> None:
    """CLI entrypoint for entropy-bin regression analysis."""
    parser = build_arg_parser()
    args = parser.parse_args()

    modes = _resolve_entropy_modes(args)
    files = _scan_and_filter_files(args)
    keep_domains = set(parse_comma_list(args.domains)) if args.domains else set()

    out_root_base = args.out_dir or os.path.join(args.scan_root, "entropy_bin_reg")
    os.makedirs(out_root_base, exist_ok=True)

    for emode in modes:
        _run_entropy_mode(
            emode=emode,
            args=args,
            files=files,
            keep_domains=keep_domains or None,
            out_root_base=out_root_base,
        )


if __name__ == "__main__":
    main()
