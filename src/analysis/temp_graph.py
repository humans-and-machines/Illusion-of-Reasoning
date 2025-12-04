#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temperature_raw_effects.py  (UPDATED)

Raw Δpp-by-temperature (plot + CSV) for:
  • Crossword (Xword)
  • Math (1.5B)          <-- base series (kept as-is)
  • Math2 (extra math)   <-- e.g., Qwen-7B
  • Math3 (extra math)   <-- e.g., Llama-8B
  • Carpark (Rush Hour)

New CLI:
  --math2_tpl  "path/GRPO-7B-math-temp-{T}"   --label_math2  "Qwen-7B-Math"
  --math3_tpl  "path/GRPO-Llama8B-math-temp-{T}" --label_math3 "Llama-8B-Math"

If templates are omitted, behavior is unchanged. If --scan_root is used,
discovery remains conservative (base series only), so use templates to inject
7B/8B series without disturbing prior runs.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.core import LoadRowsConfig, discover_roots_for_temp_args
from src.analysis.io import build_files_by_domain, iter_records_from_file
from src.analysis.labels import aha_gpt_for_rec
from src.analysis.metrics import extract_correct, make_carpark_success_fn, shift_conditional_counts

# ---- Matplotlib (Times New Roman, 14pt, 5" wide figure) ----
from src.analysis.plotting import apply_default_style
from src.analysis.utils import (
    add_gpt_step_and_carpark_args,
    add_temp_scan_args,
    coerce_float,
    get_problem_id,
    gpt_keys_for_mode,
    nat_step_from_path,
)


apply_default_style(
    {
        "legend.fontsize": 12,
        # Keep requested canvas size (no tight-cropping)
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.02,
    },
)

# ---------- patterns ----------
SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache"}


def _extract_correct(pass1_obj: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """Wrapper around the shared correctness extractor from metrics."""
    return extract_correct(pass1_obj, rec)


def _extract_soft_reward(rec: Dict[str, Any], pass1_obj: Dict[str, Any]) -> Optional[float]:
    """Extract a soft_reward value from either the record or the pass-1 object."""
    return coerce_float(rec.get("soft_reward", pass1_obj.get("soft_reward")))


# ---------- stats ----------
def per_temp_delta(df_temp_domain: pd.DataFrame) -> Tuple[float, float, int, int, float, float]:
    """
    Return (delta_pp, se_pp, n_shift, n_noshift, p1, p0) for a single temp×domain slice.

    Δpp = (p1 - p0) * 100. SE via normal approx:
    ``100*sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)``.
    """
    _, _, p_shift, p_no_shift = shift_conditional_counts(df_temp_domain)
    num_shift = int((df_temp_domain["shift"] == 1).sum())
    num_no_shift = int((df_temp_domain["shift"] == 0).sum())
    if not (np.isfinite(p_shift) and np.isfinite(p_no_shift)) or num_shift == 0 or num_no_shift == 0:
        return (np.nan, np.nan, num_shift, num_no_shift, p_shift, p_no_shift)
    delta = (p_shift - p_no_shift) * 100.0
    se_pp = 100.0 * float(
        np.sqrt(
            (p_shift * (1 - p_shift)) / num_shift + (p_no_shift * (1 - p_no_shift)) / num_no_shift,
        ),
    )
    return (delta, se_pp, num_shift, num_no_shift, p_shift, p_no_shift)


# ---------- loading ----------
def _iter_rows_for_domain(
    domain_name: str,
    files: List[str],
    config: LoadRowsConfig,
    temp_value: float,
    bump: Callable[[str, str], None],
) -> Dict[str, Any]:
    """
    Yield per-sample row dictionaries for a single domain and temperature.
    """
    for path in files:
        step_from_name = nat_step_from_path(path)
        for rec in iter_records_from_file(path):
            pass1_obj = rec.get("pass1") or {}
            if not isinstance(pass1_obj, dict):
                pass1_obj = {}

            raw_step = rec.get("step") or rec.get("global_step") or rec.get("training_step") or step_from_name
            try:
                step_int = int(raw_step) if raw_step is not None else 0
            except (TypeError, ValueError):
                step_int = 0

            if config.min_step is not None and step_int < config.min_step:
                bump(domain_name, "step<min")
                continue
            if config.max_step is not None and step_int > config.max_step:
                bump(domain_name, "step>max")
                continue

            if str(domain_name).lower().startswith("carpark"):
                correct_flag = config.carpark_success_fn(
                    _extract_soft_reward(rec, pass1_obj),
                )
                if correct_flag is None:
                    bump(domain_name, "soft_reward_missing")
                    continue
            else:
                correct_flag = _extract_correct(pass1_obj, rec)
                if correct_flag is None:
                    bump(domain_name, "correct_missing")
                    continue
            correct_int = int(correct_flag)

            problem_id = get_problem_id(rec)
            if problem_id is None:
                bump(domain_name, "problem_id_missing")
                continue

            shift = aha_gpt_for_rec(
                pass1_obj,
                rec,
                config.gpt_subset_native,
                config.gpt_keys,
                domain_name,
            )

            yield {
                "domain": str(domain_name),
                "problem_id": f"{domain_name}::{problem_id}",
                "step": step_int,
                "temp": float(temp_value),
                "correct": correct_int,
                "shift": int(shift),
            }


def load_rows(
    files_by_domain: Dict[str, List[str]],
    config: LoadRowsConfig,
    temp_value: float,
) -> pd.DataFrame:
    """
    Load per-sample rows for a given temperature across all domains.

    Returns a DataFrame with columns: domain, problem_id, step, temp, correct, shift.
    """
    rows: List[Dict[str, Any]] = []
    skips: Dict[str, Dict[str, int]] = {}

    def bump(domain_name: str, key: str) -> None:
        if domain_name not in skips:
            skips[domain_name] = {}
        skips[domain_name][key] = skips[domain_name].get(key, 0) + 1

    for domain_name, files in files_by_domain.items():
        for row in _iter_rows_for_domain(
            domain_name,
            files,
            config,
            temp_value,
            bump,
        ):
            rows.append(row)

    for dom, domain_skip_counts in skips.items():
        total = sum(domain_skip_counts.values())
        if total:
            parts = ", ".join(f"{key}={value}" for key, value in sorted(domain_skip_counts.items()))
            print(f"[debug] skips[{dom}]: {parts}")
    return pd.DataFrame(rows)


# ---------- plotting ----------
_SERIES_ORDER = ["Crossword", "Math", "Math2", "Math3", "Carpark"]
_SERIES_STYLE = {
    "Crossword": {"color": "C0", "marker": "o"},
    "Math": {"color": "C2", "marker": "o"},  # green
    "Math2": {"color": "C4", "marker": "s"},  # Qwen-7B
    "Math3": {"color": "#2ca02c", "marker": "^"},  # Llama-8B = same green
    "Carpark": {"color": "C3", "marker": "o"},
}


@dataclass
class PlotIOConfig:
    """Output paths and rendering configuration for the temperature plot."""

    png_path: str
    pdf_path: str
    dpi: int = 300


def make_plot(
    pertemp_df: pd.DataFrame,
    title: str,
    x_temps_sorted: List[float],
    label_map: Dict[str, str],
    io_config: PlotIOConfig,
) -> None:
    """
    Plot Δpp vs temperature for any present series among
    Crossword, Math, Math2, Math3, Carpark.
    """
    figure, axis = plt.subplots(figsize=(5, 3), constrained_layout=False)

    present = [series for series in _SERIES_ORDER if series in set(pertemp_df["domain_key"])]
    for dom_key in present:
        sub = pertemp_df[pertemp_df["domain_key"] == dom_key].copy()
        if sub.empty:
            continue
        sub = sub.set_index("temp").reindex(x_temps_sorted).reset_index()
        style = _SERIES_STYLE.get(dom_key, {"color": "C0", "marker": "o"})
        axis.errorbar(
            sub["temp"],
            sub["delta_pp"],
            yerr=sub["se_pp"],
            fmt=style["marker"] + "-",
            capsize=4,
            elinewidth=1,
            linewidth=2,
            label=label_map.get(dom_key, dom_key),
            color=style["color"],
        )

    axis.axhline(0.0, linewidth=1, linestyle="--", color="0.4")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Raw effect of shift\non accuracy (pp)")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=min(len(present), 4),
        frameon=True,
        borderaxespad=0.0,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    figure.subplots_adjust(bottom=0.32)
    figure.savefig(io_config.png_path, dpi=io_config.dpi, bbox_inches="tight")
    figure.savefig(io_config.pdf_path, dpi=io_config.dpi, bbox_inches="tight")
    plt.close(figure)


# ---------- main ----------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the temperature graph script."""
    arg_parser = argparse.ArgumentParser()
    add_temp_scan_args(arg_parser, include_math3=True)

    # Labels
    arg_parser.add_argument("--label_crossword", type=str, default="Xword")
    arg_parser.add_argument("--label_math", type=str, default="Qwen-1.5B-Math")
    arg_parser.add_argument("--label_math2", type=str, default="Qwen-7B-Math")
    arg_parser.add_argument("--label_math3", type=str, default="Llama-8B-Math")
    arg_parser.add_argument("--label_carpark", type=str, default="Rush Hour")

    arg_parser.add_argument(
        "--split",
        default=None,
        help="Only include files whose NAMES contain this substring (e.g., 'test').",
    )
    arg_parser.add_argument("--out_dir", default=None)
    arg_parser.add_argument("--dataset_name", default="MIXED")
    arg_parser.add_argument("--model_name", default="Qwen-1.5B")
    add_gpt_step_and_carpark_args(arg_parser)

    arg_parser.add_argument("--low_alias", type=float, default=0.3)
    arg_parser.add_argument(
        "--skip_substr",
        nargs="*",
        default=["compare-1shot", "1shot", "hf_cache"],
    )

    arg_parser.add_argument("--make_plot", action="store_true")
    arg_parser.add_argument("--plot_title", type=str, default=None)
    arg_parser.add_argument("--dpi", type=int, default=300)
    return arg_parser


def _resolve_output_dir(args: argparse.Namespace) -> str:
    """Resolve the output directory for CSVs and plots."""
    if args.out_dir:
        return args.out_dir
    guess = (
        args.scan_root
        or args.crossword_tpl
        or args.math_tpl
        or args.math2_tpl
        or args.math3_tpl
        or args.carpark_tpl
        or "."
    )
    base = guess if isinstance(guess, str) else "."
    return os.path.join(base, "temperature_raw_effects")


def _effective_max_step(args: argparse.Namespace) -> int:
    """Compute the effective max_step with a hard safety cap."""
    hard_max_step = 1000
    if args.max_step is None:
        max_step_eff = hard_max_step
    else:
        max_step_eff = min(args.max_step, hard_max_step)
    if args.max_step is None or args.max_step > hard_max_step:
        print(
            f"[info] Capping max_step to {max_step_eff} (hard cap = {hard_max_step}).",
        )
    return max_step_eff


def _discover_roots(
    args: argparse.Namespace,
    skip_set: set[str],
) -> Dict[float, Dict[str, str]]:
    """Discover per-temperature root directories from scan_root or templates."""
    return discover_roots_for_temp_args(
        args,
        skip_set,
        include_math3=True,
    )


def _build_label_map(args: argparse.Namespace) -> Dict[str, str]:
    """Build a mapping from domain keys to human-readable labels."""
    return {
        "Crossword": args.label_crossword,
        "Math": args.label_math,
        "Math2": args.label_math2,
        "Math3": args.label_math3,
        "Carpark": args.label_carpark,
    }


def _compute_pertemp_dataframe(
    args: argparse.Namespace,
    roots_by_temp: Dict[float, Dict[str, str]],
    max_step_eff: int,
    skip_set: set[str],
) -> Tuple[pd.DataFrame, List[float]]:
    """Load rows per temperature and compute per-temperature raw effects."""
    pertemp_rows: List[Dict[str, Any]] = []
    x_temps_sorted = sorted(roots_by_temp.keys())

    load_config = LoadRowsConfig(
        gpt_keys=gpt_keys_for_mode(args.gpt_mode),
        gpt_subset_native=not args.no_gpt_subset_native,
        min_step=args.min_step,
        max_step=max_step_eff,
        carpark_success_fn=make_carpark_success_fn(
            args.carpark_success_op,
            args.carpark_soft_threshold,
        ),
    )

    for temp_value, domain_paths in roots_by_temp.items():
        rows_df = _load_rows_for_temp(
            temp_value=temp_value,
            domain_paths=domain_paths,
            load_config=load_config,
            args=args,
            skip_set=skip_set,
        )
        if rows_df is None:
            continue
        pertemp_rows.extend(
            _compute_pertemp_rows_for_temp(temp_value=temp_value, rows_df=rows_df),
        )

    if not pertemp_rows:
        sys.exit("No per-temperature aggregates available.")

    pertemp_df = pd.DataFrame(pertemp_rows)
    return pertemp_df, x_temps_sorted


def _load_rows_for_temp(
    temp_value: float,
    domain_paths: Dict[str, str],
    load_config: LoadRowsConfig,
    args: argparse.Namespace,
    skip_set: set[str],
) -> Optional[pd.DataFrame]:
    """
    Load and log per-sample rows for a single temperature value.
    """
    files_by_domain = build_files_by_domain(
        domain_paths,
        ["Crossword", "Math", "Math2", "Math3", "Carpark"],
        args.split,
        skip_set,
    )
    if not files_by_domain:
        return None

    rows_df = load_rows(files_by_domain, load_config, temp_value=temp_value)
    if rows_df.empty:
        print(f"[warn] T={temp_value}: no rows loaded after parsing/gating.")
        for domain_name, file_list in files_by_domain.items():
            print(
                f"    files[{domain_name}]={len(file_list)} under {domain_paths.get(domain_name)}",
            )
        return None

    counts: List[str] = []
    for domain_name in ["Crossword", "Math", "Math2", "Math3", "Carpark"]:
        num_rows = int((rows_df["domain"] == domain_name).sum())
        counts.append(f"{domain_name}={num_rows}")
    print(f"[info] loaded rows @ T={temp_value}: " + ", ".join(counts))

    return rows_df


def _compute_pertemp_rows_for_temp(
    temp_value: float,
    rows_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Compute per-temperature aggregates for all domains at a single temperature.
    """
    pertemp_rows: List[Dict[str, Any]] = []
    for domain_name in ["Crossword", "Math", "Math2", "Math3", "Carpark"]:
        domain_df = rows_df[rows_df["domain"] == domain_name]
        if domain_df.empty:
            continue
        (
            delta_pp,
            se_pp,
            num_shift,
            num_no_shift,
            p_shift,
            p_no_shift,
        ) = per_temp_delta(domain_df)
        pertemp_rows.append(
            {
                "temp": float(temp_value),
                "domain_key": domain_name,
                "domain": domain_name,
                "delta_pp": delta_pp,
                "se_pp": se_pp,
                "n_shift": num_shift,
                "n_noshift": num_no_shift,
                "p1": p_shift,
                "p0": p_no_shift,
            },
        )
    return pertemp_rows


def _save_outputs(
    args: argparse.Namespace,
    out_dir: str,
    pertemp_df: pd.DataFrame,
    x_temps_sorted: List[float],
    label_map: Dict[str, str],
) -> None:
    """Save per-temperature CSV, optional plot, and console preview."""
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_csv = os.path.join(out_dir, f"raw_effects_by_temperature__{slug}.csv")

    # Save per-temperature CSV
    pertemp_df = pertemp_df.copy()
    pertemp_df["domain"] = (
        pertemp_df["domain_key"]
        .map(label_map)
        .fillna(
            pertemp_df["domain_key"],
        )
    )
    pertemp_df = pertemp_df.sort_values(["domain_key", "temp"])
    pertemp_df.to_csv(out_csv, index=False)
    print(f"Saved per-temperature CSV -> {out_csv}")

    # Plot (optional)
    if args.make_plot:
        png_path = os.path.join(out_dir, f"raw_effects_plot__{slug}.png")
        pdf_path = os.path.join(out_dir, f"raw_effects_plot__{slug}.pdf")
        present_keys = [key for key in _SERIES_ORDER if key in set(pertemp_df["domain_key"])]
        if not present_keys:
            print("[warn] Nothing to plot.")
        else:
            if args.plot_title:
                plot_title = args.plot_title
            else:
                # Compact title listing present series
                shown = ", ".join(label_map[key] for key in present_keys)
                plot_title = f"Raw Effect vs Temperature — {shown}"
            io_config = PlotIOConfig(
                png_path=png_path,
                pdf_path=pdf_path,
                dpi=args.dpi,
            )
            make_plot(
                pertemp_df=pertemp_df,
                title=plot_title,
                x_temps_sorted=x_temps_sorted,
                label_map=label_map,
                io_config=io_config,
            )
            print(f"Saved plot PNG -> {png_path}")
            print(f"Saved plot PDF -> {pdf_path}")

    _print_console_preview(pertemp_df)


def _print_console_preview(pertemp_df: pd.DataFrame) -> None:
    """Pretty-print a console summary of per-temperature raw effects."""
    with pd.option_context("display.width", 140):
        display_df = pertemp_df.copy()
        for column_name in ["delta_pp"]:
            display_df[column_name] = display_df[column_name].map(
                lambda val: f"{val:+.2f}" if pd.notna(val) else "--",
            )
        for column_name in ["se_pp"]:
            display_df[column_name] = display_df[column_name].map(
                lambda val: f"{val:.2f}" if pd.notna(val) else "--",
            )
        for column_name in ["p1", "p0"]:
            display_df[column_name] = display_df[column_name].map(
                lambda val: f"{val:.4f}" if pd.notna(val) else "--",
            )
        print("\n[info] Per-temperature Raw Effects:\n")
        cols = [
            "domain",
            "temp",
            "delta_pp",
            "se_pp",
            "n_shift",
            "n_noshift",
            "p1",
            "p0",
        ]
        print(display_df[cols].to_string(index=False))


def main() -> None:
    """CLI entry point for the temperature raw-effects graph."""
    args = _build_arg_parser().parse_args()

    out_dir = _resolve_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    max_step_eff = _effective_max_step(args)
    skip_set = {s.lower() for s in args.skip_substr} | SKIP_DIR_DEFAULT
    roots_by_temp = _discover_roots(args, skip_set)

    if not roots_by_temp:
        sys.exit("No usable folders discovered. Check --scan_root or templates + temps.")

    pertemp_df, x_temps_sorted = _compute_pertemp_dataframe(
        args,
        roots_by_temp,
        max_step_eff,
        skip_set,
    )
    label_map = _build_label_map(args)
    _save_outputs(args, out_dir, pertemp_df, x_temps_sorted, label_map)


if __name__ == "__main__":
    main()
