#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temperature_effects.py  (parallel; updated to support Math2/7B & joint plotting)
-------------------------------------------------------------------------------
Temperature-focused shift table (raw effect + AME) with fast parallel loading.

New in this version
-------------------
• Supports a second Math series (e.g., Qwen-7B) via --math2_tpl/--label_math2
• Per-temperature raw-effect aggregation includes Math2
• Plot shows Math and Math2 as separate series (others are skipped if absent)
• Backward compatible with older usage (Crossword/Math/Carpark only)
• Robust tick handling: x-axis uses the exact temperatures you passed

Parallelism
-----------
  --workers 40          number of processes/threads (default: 40)
  --parallel process    process (default) or thread pool
  --chunksize N         batching for executor.map

Everything else is as before (step caps, robust correctness, per-temp CSV+plot).
"""

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import concurrent.futures as cf
import importlib
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from src.analysis.core import discover_roots_for_temp_args
from src.analysis.io import scan_files_step_only, iter_records_from_file
from src.analysis.labels import aha_gpt_for_rec
from src.analysis.metrics import (
    extract_correct,
    make_carpark_success_fn,
    shift_conditional_counts,
)
from src.analysis.temperature_effects_cli import build_temperature_effects_arg_parser
from src.analysis.utils import (
    coerce_float,
    extract_pass1_and_step,
    get_problem_id as _get_problem_id,
    gpt_keys_for_mode,
    nat_step_from_path,
    step_from_record_if_within_bounds,
)

# Matplotlib + global styling (lazy import to avoid hard dependency at import time)
mpl = importlib.import_module("matplotlib")
plt = importlib.import_module("matplotlib.pyplot")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,   # embed TrueType
    "ps.fonttype": 42,
})

# ---------- constants ----------

HARD_MAX_STEP = 1000

# Directories that almost never contain user data of interest
SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache", "__pycache__"}

# ---------- helpers ----------


def _lazy_import_statsmodels():
    """
    Lazily import statsmodels pieces needed for GLM fits.

    Using importlib avoids hard import-time dependencies in environments
    where statsmodels is not installed, while still failing clearly at
    runtime if these helpers are used without the package.
    """
    sm_module = importlib.import_module("statsmodels.api")
    smf_module = importlib.import_module("statsmodels.formula.api")
    links_module = importlib.import_module("statsmodels.genmod.families.links")
    tools_module = importlib.import_module("statsmodels.tools.sm_exceptions")
    logit_cls = getattr(links_module, "Logit")
    perfect_sep_err = getattr(tools_module, "PerfectSeparationError")
    return sm_module, smf_module, logit_cls, perfect_sep_err


# ---------- correctness / soft reward helpers ----------


def _extract_correct(pass1_data: Dict[str, Any], record: Dict[str, Any]) -> Optional[int]:
    """
    Thin wrapper around the shared correctness extractor from metrics.
    """
    return extract_correct(pass1_data, record)


def _extract_soft_reward(record: Dict[str, Any], pass1_data: Dict[str, Any]) -> Optional[float]:
    """
    Extract a soft_reward value from either the record or the pass-1 object.
    """
    return coerce_float(record.get("soft_reward", pass1_data.get("soft_reward")))

# ---------- stats ----------


def _prepare_glm_frame(sub_all_temps: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise temperature and coerce fields needed for the GLM.
    """
    glm_df = sub_all_temps.copy()
    temp_values = glm_df["temp"].to_numpy(dtype=float)
    mean_temp = float(temp_values.mean())
    std_temp = float(temp_values.std(ddof=0) + 1e-12)
    glm_df["temp_std"] = (temp_values - mean_temp) / std_temp
    glm_df["shift"] = glm_df["shift"].astype(float)
    glm_df["correct"] = glm_df["correct"].astype(float)
    return glm_df


def _fit_shift_glm(glm_df: pd.DataFrame):
    """
    Fit the temperature/shift GLM and return the fitted result.
    """
    sm_module, smf_module, logit_cls, perfect_sep_err = _lazy_import_statsmodels()

    model = smf_module.glm(
        formula="correct ~ C(problem_id) + temp_std + shift",
        data=glm_df,
        family=sm_module.families.Binomial(link=logit_cls()),
    )
    try:
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": glm_df["problem_id"]},
            maxiter=200,
        )
    except (perfect_sep_err, LinAlgError, ValueError):
        res = model.fit()
        res = res.get_robustcov_results(
            cov_type="cluster",
            groups=glm_df["problem_id"],
        )
    return res


def _average_marginal_effect_for_shift(res) -> Tuple[float, float]:
    """
    Compute the AME for the ``shift`` regressor and its p-value.
    """

    exog_names = list(res.model.exog_names)
    try:
        shift_index = exog_names.index("shift")
    except ValueError:
        shift_index = max(
            idx for idx, name in enumerate(exog_names) if "shift" in name
        )

    design_matrix = res.model.exog.copy()
    params_vector = res.params.to_numpy()
    logits_with_shift = design_matrix @ params_vector
    probs_with_shift = 1.0 / (1.0 + np.exp(-logits_with_shift))
    design_matrix[:, shift_index] = 0.0
    logits_without_shift = design_matrix @ params_vector
    probs_without_shift = 1.0 / (1.0 + np.exp(-logits_without_shift))
    delta_probs = probs_with_shift - probs_without_shift
    ame = float(np.mean(delta_probs))
    p_shift = float(res.pvalues.get("shift", np.nan))
    return ame, p_shift


def ame_glm_temp(sub_all_temps: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit a GLM and return the AME of shift along with its p-value.
    """
    glm_df = _prepare_glm_frame(sub_all_temps)
    res = _fit_shift_glm(glm_df)
    return _average_marginal_effect_for_shift(res)

# ---------- PARALLEL WORKER ----------


class ParallelConfig(NamedTuple):
    """
    Configuration for parallel loading and gating of records.
    """
    gpt_keys: List[str]
    gpt_subset_native: bool
    min_step: Optional[int]
    max_step: Optional[int]
    carpark_op: str
    carpark_thr: float
    temp_value: float
    workers: int
    parallel_mode: str
    chunksize: int


def _compute_correct(
    dom_lower: str,
    record: Dict[str, Any],
    pass1_data: Dict[str, Any],
    carpark_success_fn: Callable[[Any], Optional[int]],
) -> Optional[int]:
    """
    Compute correctness indicator for a record across domains.
    """
    if dom_lower.startswith("carpark"):
        soft_reward = _extract_soft_reward(record, pass1_data)
        ok_flag = carpark_success_fn(soft_reward)
        if ok_flag is None:
            return None
        return int(ok_flag)
    correct_flag = _extract_correct(pass1_data, record)
    if correct_flag is None:
        return None
    return int(correct_flag)


def _process_file_worker(task: Tuple[str, str, ParallelConfig]) -> List[Dict[str, Any]]:
    """
    One file -> list of rows.

    task = (dom, path, parallel_cfg)
    """
    dom, path, parallel_cfg = task
    return _rows_for_file(dom, path, parallel_cfg)


def _rows_for_file(
    domain: str,
    path: str,
    parallel_cfg: ParallelConfig,
) -> List[Dict[str, Any]]:
    """
    Load and gate rows for a single (domain, path) pair.
    """
    rows: List[Dict[str, Any]] = []
    carpark_success_fn = make_carpark_success_fn(
        parallel_cfg.carpark_op,
        parallel_cfg.carpark_thr,
    )
    dom_lower = domain.lower()
    step_from_name = nat_step_from_path(path)

    for rec in iter_records_from_file(path):
        pass1_data, _ = extract_pass1_and_step(rec, step_from_name)
        if not pass1_data:
            continue

        step = step_from_record_if_within_bounds(
            rec,
            path,
            split_value=None,
            min_step=parallel_cfg.min_step,
            max_step=parallel_cfg.max_step,
        )
        if step is None:
            continue

        correct = _compute_correct(dom_lower, rec, pass1_data, carpark_success_fn)
        if correct is None:
            continue

        problem_id = _get_problem_id(rec)
        if problem_id is None:
            continue

        shift = aha_gpt_for_rec(
            pass1_data,
            rec,
            parallel_cfg.gpt_subset_native,
            parallel_cfg.gpt_keys,
            domain,
        )

        rows.append(
            {
                "domain": str(domain),
                "problem_id": f"{domain}::{problem_id}",
                "step": int(step),
                "temp": float(parallel_cfg.temp_value),
                "correct": int(correct),
                "shift": int(shift),
            }
        )

    return rows


def load_rows_parallel(
    files_by_domain: Dict[str, List[str]],
    parallel_cfg: ParallelConfig,
) -> pd.DataFrame:
    """
    Load and gate rows in parallel across all domains.
    """
    tasks: List[Tuple[str, str, ParallelConfig]] = []
    for dom, files in files_by_domain.items():
        for path in files:
            tasks.append((dom, path, parallel_cfg))
    if not tasks:
        return pd.DataFrame(
            columns=["domain", "problem_id", "step", "temp", "correct", "shift"],
        )

    exec_cls = (
        cf.ProcessPoolExecutor
        if parallel_cfg.parallel_mode == "process"
        else cf.ThreadPoolExecutor
    )
    rows_all: List[Dict[str, Any]] = []
    chunksize = parallel_cfg.chunksize
    workers = parallel_cfg.workers
    effective_chunksize = max(
        1,
        chunksize if chunksize > 0 else len(tasks) // (workers * 4) or 1,
    )
    with exec_cls(max_workers=workers) as executor:
        for res in executor.map(
            _process_file_worker,
            tasks,
            chunksize=effective_chunksize,
        ):
            if res:
                rows_all.extend(res)
    return pd.DataFrame(rows_all)

# ---------- NEW: per-temperature raw-effect computation & plotting ----------


def per_temp_delta(df_temp_dom: pd.DataFrame) -> Tuple[float, float, int, int]:
    """
    Compute per-temperature delta (pp) and its standard error.
    """
    _, _, prob_shift, prob_no_shift = shift_conditional_counts(df_temp_dom)
    n_shift = int((df_temp_dom["shift"] == 1).sum())
    n_no_shift = int((df_temp_dom["shift"] == 0).sum())
    if (
        not (np.isfinite(prob_shift) and np.isfinite(prob_no_shift))
        or n_shift == 0
        or n_no_shift == 0
    ):
        return (np.nan, np.nan, n_shift, n_no_shift)
    delta = (prob_shift - prob_no_shift) * 100.0
    se_pp = 100.0 * float(
        np.sqrt(
            (prob_shift * (1 - prob_shift)) / n_shift
            + (prob_no_shift * (1 - prob_no_shift)) / n_no_shift,
        ),
    )
    return (delta, se_pp, n_shift, n_no_shift)


class PlotConfig(NamedTuple):
    """
    Configuration for plotting raw effects vs temperature.
    """
    pertemp_df: pd.DataFrame
    out_png: str
    out_pdf: Optional[str]
    title: str
    x_temps_sorted: List[float]
    label_map: Dict[str, str]
    dpi: int = 300


def make_plot(config: PlotConfig) -> None:
    """
    Plot raw effects vs temperature for each domain.
    """
    fig, axis = plt.subplots(figsize=(5, 3), constrained_layout=True)

    # Series order; empty ones are skipped automatically
    series = [
        ("Math", "C2", "o"),
        ("Math2", "C4", "s"),  # second math series (e.g., Qwen-7B)
        ("Crossword", "C0", "o"),
        ("Carpark", "C3", "o"),
    ]
    for dom_key, color, marker in series:
        sub = config.pertemp_df[
            config.pertemp_df["domain_key"] == dom_key
        ].copy()
        if sub.empty:
            continue
        sub = (
            sub.set_index("temp")
            .reindex(sorted(set(config.x_temps_sorted)))
            .reset_index()
        )
        axis.errorbar(
            sub["temp"],
            sub["delta_pp"],
            yerr=sub["se_pp"],
            fmt=f"{marker}-",
            capsize=4,
            elinewidth=1,
            linewidth=2,
            label=config.label_map.get(dom_key, dom_key),
            color=color,
        )

    # Helpful x ticks: exactly the temperatures you passed; pad edges
    xticks = sorted(set(config.x_temps_sorted))
    axis.set_xticks(xticks)
    axis.set_xticklabels(
        [f"{temp_value:.2f}".rstrip("0").rstrip(".") for temp_value in xticks],
    )
    if len(xticks) >= 2:
        pad = 0.05 * (max(xticks) - min(xticks))
    else:
        pad = 0.05
    axis.set_xlim(min(xticks) - pad, max(xticks) + pad)

    axis.axhline(0.0, linewidth=1, linestyle="--", color="0.4")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Raw effect of shift on accuracy (pp)")
    axis.set_title(config.title)
    axis.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )
    axis.grid(True, axis="y", alpha=0.25)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(config.out_png, dpi=config.dpi, bbox_inches="tight")
    if config.out_pdf:
        fig.savefig(config.out_pdf, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for this script.
    """
    return build_temperature_effects_arg_parser()


def _compute_out_dir(args: argparse.Namespace) -> str:
    """
    Determine the base output directory for this run.
    """
    if args.out_dir:
        return args.out_dir
    guess = args.scan_root or (
        args.crossword_tpl
        or args.math_tpl
        or args.math2_tpl
        or args.carpark_tpl
        or "."
    )
    return os.path.join(guess if isinstance(guess, str) else ".", "temperature_effects")


def _compute_gpt_keys(args: argparse.Namespace) -> Tuple[bool, List[str]]:
    """
    Build GPT gating configuration from CLI args.
    """
    gpt_subset_native = not args.no_gpt_subset_native
    keys = gpt_keys_for_mode(args.gpt_mode)
    return gpt_subset_native, keys


def _discover_roots(
    args: argparse.Namespace,
    skip_set: set,
) -> Dict[float, Dict[str, str]]:
    """
    Construct the mapping {temp -> domain -> root_dir}.
    """
    return discover_roots_for_temp_args(
        args,
        skip_set,
        include_math3=False,
    )


class AnalysisContext(NamedTuple):
    """
    Immutable configuration for the main aggregation loop.
    """

    args: argparse.Namespace
    gpt_subset_native: bool
    gpt_keys: List[str]
    max_step_eff: int
    carpark_op: str
    carpark_thr: float
    skip_set: set


def _discover_files_for_temp(
    domain_map: Dict[str, str],
    args: argparse.Namespace,
    skip_set: set,
) -> Dict[str, List[str]]:
    """
    Build the per-domain file list for a single temperature.
    """
    files_by_domain: Dict[str, List[str]] = {}
    discover_keys = (
        ["Crossword", "Math", "Math2", "Carpark"]
        if args.include_math2
        else ["Crossword", "Math", "Carpark"]
    )
    for dom in discover_keys:
        path = domain_map.get(dom)
        if not path:
            continue
        files = scan_files_step_only(path, args.split, skip_set)
        if files:
            files_by_domain[dom] = files
    return files_by_domain


def _load_df_for_temp(
    temp_value: float,
    files_by_domain: Dict[str, List[str]],
    ctx: "AnalysisContext",
) -> pd.DataFrame:
    """
    Use ParallelConfig + load_rows_parallel to load rows for a temperature.
    """
    args = ctx.args
    parallel_cfg = ParallelConfig(
        gpt_keys=ctx.gpt_keys,
        gpt_subset_native=ctx.gpt_subset_native,
        min_step=args.min_step,
        max_step=ctx.max_step_eff,
        carpark_op=ctx.carpark_op,
        carpark_thr=ctx.carpark_thr,
        temp_value=temp_value,
        workers=args.workers,
        parallel_mode=args.parallel,
        chunksize=args.chunksize,
    )
    return load_rows_parallel(
        files_by_domain=files_by_domain,
        parallel_cfg=parallel_cfg,
    )


def _log_no_rows_for_temp(
    temp_value: float,
    files_by_domain: Dict[str, List[str]],
    domain_map: Dict[str, str],
) -> None:
    """
    Emit a warning when no rows are loaded for a temperature.
    """
    print(f"[warn] T={temp_value}: no rows loaded after parsing/gating.")
    for dom_dbg, flist in files_by_domain.items():
        print(
            f"    files[{dom_dbg}]={len(flist)} under {domain_map.get(dom_dbg)}",
        )


def _log_loaded_rows(df_temp: pd.DataFrame, temp_value: float) -> None:
    """
    Emit a compact summary of loaded rows by domain for a temperature.
    """
    summary_parts: List[str] = []
    for dom_show in ["Crossword", "Math", "Math2", "Carpark"]:
        n_dom = int((df_temp["domain"] == dom_show).sum())
        summary_parts.append(f"{dom_show}={n_dom}")
    print(
        f"[info] loaded rows @ T={temp_value}: "
        + ", ".join(summary_parts),
    )


def _append_domain_and_pertemp(
    df_temp: pd.DataFrame,
    temp_value: float,
    domain_frames: Dict[str, List[pd.DataFrame]],
    pertemp_rows: List[Dict[str, Any]],
) -> None:
    """
    Update per-domain frames and per-temperature aggregates.
    """
    for dom in ["Crossword", "Math", "Math2", "Carpark"]:
        sub = df_temp[df_temp["domain"] == dom]
        if not sub.empty:
            domain_frames[dom].append(sub)

    # per-temp raw effects (include Math2)
    for dom in ["Crossword", "Math", "Math2", "Carpark"]:
        sub = df_temp[df_temp["domain"] == dom]
        if sub.empty:
            continue
        delta_pp, se_pp, n_shift, n_no_shift = per_temp_delta(sub)
        pertemp_rows.append(
            {
                "temp": float(temp_value),
                "domain_key": dom,
                "domain": dom,
                "delta_pp": delta_pp,
                "se_pp": se_pp,
                "n_shift": n_shift,
                "n_noshift": n_no_shift,
            }
        )


def _collect_domain_and_pertemp(
    ctx: "AnalysisContext",
    roots_by_temp: Dict[float, Dict[str, str]],
) -> Tuple[Dict[str, List[pd.DataFrame]], List[Dict[str, Any]], List[float]]:
    """
    Load rows for each temperature and domain and build aggregates.
    """
    args = ctx.args
    domain_frames: Dict[str, List[pd.DataFrame]] = {
        "Crossword": [],
        "Math": [],
        "Math2": [],
        "Carpark": [],
    }
    pertemp_rows: List[Dict[str, Any]] = []
    x_temps_sorted = sorted(roots_by_temp.keys())

    for temp_value, domain_map in roots_by_temp.items():
        files_by_domain = _discover_files_for_temp(
            domain_map,
            args,
            ctx.skip_set,
        )
        if not files_by_domain:
            continue

        df_temp = _load_df_for_temp(temp_value, files_by_domain, ctx)
        if df_temp.empty:
            _log_no_rows_for_temp(temp_value, files_by_domain, domain_map)
            continue

        _log_loaded_rows(df_temp, temp_value)
        _append_domain_and_pertemp(
            df_temp,
            temp_value,
            domain_frames,
            pertemp_rows,
        )

    return domain_frames, pertemp_rows, x_temps_sorted


def _summarize_domains(
    domain_frames: Dict[str, List[pd.DataFrame]],
    label_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Build per-domain aggregate table with AMEs.
    """
    rows: List[Dict[str, Any]] = []
    for dom in ["Crossword", "Math", "Math2", "Carpark"]:
        if not domain_frames[dom]:
            continue
        domain_df = pd.concat(domain_frames[dom], ignore_index=True)
        total_count, share, acc_shift, acc_no_shift = shift_conditional_counts(domain_df)
        if np.isfinite(acc_shift) and np.isfinite(acc_no_shift):
            delta_pp = (acc_shift - acc_no_shift) * 100.0
        else:
            delta_pp = np.nan
        ame, p_shift = ame_glm_temp(domain_df)
        rows.append(
            {
                "domain_key": dom,
                "domain": label_map[dom],
                "N": total_count,
                "share_shift": share,
                "acc_shift": acc_shift,
                "delta_pp": delta_pp,
                "AME": ame,
                "p": p_shift,
            }
        )

    if not rows:
        sys.exit("No per-domain aggregates available.")

    return pd.DataFrame(
        rows,
        columns=[
            "domain",
            "N",
            "share_shift",
            "acc_shift",
            "delta_pp",
            "AME",
            "p",
        ],
    )


def _emit_domain_table(
    tab: pd.DataFrame,
    out_csv: str,
) -> None:
    """
    Pretty-print and save the per-domain aggregate table.
    """
    tab.to_csv(out_csv, index=False)
    print("[info] Aggregates by domain:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        disp = tab.copy()
        disp["share_shift"] = disp["share_shift"].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "--",
        )
        disp["acc_shift"] = disp["acc_shift"].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "--",
        )
        disp["delta_pp"] = disp["delta_pp"].map(
            lambda value: f"{value:+.2f}" if pd.notna(value) else "--",
        )
        disp["AME"] = disp["AME"].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "--",
        )
        disp["p"] = disp["p"].map(
            lambda value: f"{value:.3g}" if pd.notna(value) else "--",
        )
        print(disp.to_string(index=False))
    print(f"\nSaved CSV -> {out_csv}")


def _write_outputs(
    args: argparse.Namespace,
    domain_frames: Dict[str, List[pd.DataFrame]],
    pertemp_rows: List[Dict[str, Any]],
    x_temps_sorted: List[float],
) -> None:
    """
    Materialize per-domain and per-temperature outputs (tables + plots).
    """
    label_map = {
        "Crossword": args.label_crossword,
        "Math": args.label_math,
        "Math2": args.label_math2,
        "Carpark": args.label_carpark,
    }

    tab = _summarize_domains(domain_frames, label_map)

    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_dir_final = (
        args.out_dir
        or os.path.join(args.scan_root or ".", "temperature_effects")
    )
    os.makedirs(out_dir_final, exist_ok=True)
    out_csv = os.path.join(out_dir_final, f"temperature_shift_table__{slug}.csv")
    _emit_domain_table(tab, out_csv)

    if not pertemp_rows:
        return

    pertemp = pd.DataFrame(pertemp_rows)
    pertemp["domain"] = pertemp["domain_key"].map(label_map).fillna(
        pertemp["domain_key"],
    )
    pertemp_csv = os.path.join(
        out_dir_final,
        f"temperature_shift_raw_effects__{slug}.csv",
    )
    pertemp.sort_values(["domain_key", "temp"]).to_csv(
        pertemp_csv,
        index=False,
    )
    print(f"Saved per-temperature raw effects CSV -> {pertemp_csv}")

    if not args.make_plot:
        return

    png_path = os.path.join(
        out_dir_final,
        f"temperature_shift_plot__{slug}.png",
    )
    pdf_path = os.path.join(
        out_dir_final,
        f"temperature_shift_plot__{slug}.pdf",
    )
    make_plot(
        PlotConfig(
            pertemp_df=pertemp,
            out_png=png_path,
            out_pdf=pdf_path,
            title=(
                args.plot_title
                or f"Raw effect vs temperature — {args.model_name}"
            ),
            x_temps_sorted=x_temps_sorted,
            label_map=label_map,
            dpi=args.dpi,
        )
    )
    print(f"Saved plot PNG -> {png_path}")
    print(f"Saved plot PDF -> {pdf_path}")


def main() -> None:
    """
    Entry point: parse CLI args, run analysis, and emit outputs.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_dir = _compute_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    gpt_subset_native, gpt_keys = _compute_gpt_keys(args)

    # Step cap (hard cap 1000)
    max_step_eff = (
        HARD_MAX_STEP
        if args.max_step is None
        else min(args.max_step, HARD_MAX_STEP)
    )
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(
            f"[info] Capping max_step to {max_step_eff} "
            f"(hard cap = {HARD_MAX_STEP}).",
        )

    carpark_op = args.carpark_success_op
    carpark_thr = args.carpark_soft_threshold
    skip_set = {substr.lower() for substr in args.skip_substr} | SKIP_DIR_DEFAULT

    roots_by_temp = _discover_roots(args, skip_set)
    if not roots_by_temp:
        sys.exit(
            "No usable folders discovered. "
            "Check --scan_root or templates + temps.",
        )

    ctx = AnalysisContext(
        args=args,
        gpt_subset_native=gpt_subset_native,
        gpt_keys=gpt_keys,
        max_step_eff=max_step_eff,
        carpark_op=carpark_op,
        carpark_thr=carpark_thr,
        skip_set=skip_set,
    )
    domain_frames, pertemp_rows, x_temps_sorted = _collect_domain_and_pertemp(
        ctx,
        roots_by_temp,
    )
    _write_outputs(args, domain_frames, pertemp_rows, x_temps_sorted)


if __name__ == "__main__":
    main()
