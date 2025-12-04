#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pass2_effects.py  —  Raw effect of Pass-2 by domain, split by Pass-1 solvability

What it does
------------
• Loads PASS1/PASS2 records from your JSONLs (same format as your other scripts).
• Groups at the (domain, step, problem) level, averaging over the 8 samples:
    p1_acc = mean(correct_pass1 over samples)
    p2_acc = mean(correct_pass2 over samples)
    raw_effect = p2_acc - p1_acc
• Splits problems into two buckets:
    - P1_ANY  : p1_acc > 0  (at least one of 8 correct in Pass-1)
    - P1_NONE : p1_acc = 0  (none correct in Pass-1)
• Produces a 3-panel figure (Carpark / Crossword / Math):
    bar = mean(raw_effect) per bucket, with 95% bootstrap CI; n annotated
• Writes CSVs for per-problem rows and domain summaries.

Outputs
-------
graphs/
  pass2_raw_effects_{tag}.png, .pdf
  tables/pass2_per_problem_{tag}.csv
  tables/pass2_summary_{tag}.csv

Notes
-----
• Carpark correctness uses soft reward thresholds (configurable).
• You can pool across steps per problem via --pool_across_steps (optional).
"""

import argparse
import builtins
import glob
import os
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.common.mpl_stub_helpers import AxisSettersMixin, coerce_axes_sequence
from src.analysis.common.parser_helpers import add_carpark_softscore_args
from src.analysis.io import iter_records_from_file
from src.analysis.plotting import apply_entropy_plot_style
from src.analysis.utils import add_domain_root_args, truthy_flag


_ORIG_ISINSTANCE = builtins.isinstance
_ORIG_LIST_TYPE = list

_PANDAS_ORIG_INIT_ATTR = "_analysis_orig_dataframe_init"
if hasattr(pd.DataFrame, _PANDAS_ORIG_INIT_ATTR):
    # Module has been imported before in this interpreter; reuse the original
    # constructor saved on the class and avoid re-patching.
    _PANDAS_DATAFRAME_INIT = getattr(pd.DataFrame, _PANDAS_ORIG_INIT_ATTR)
else:
    # First import: capture the original constructor so subsequent reloads do
    # not accidentally wrap our patched version. We persist the original
    # constructor on the class so later imports can find it.
    _PANDAS_DATAFRAME_INIT = getattr(pd.DataFrame, "__init__", None)
    if _PANDAS_DATAFRAME_INIT is not None:
        setattr(pd.DataFrame, _PANDAS_ORIG_INIT_ATTR, _PANDAS_DATAFRAME_INIT)


def _patched_dataframe_init(self, *args, **kwargs):
    """
    Ensure pandas sees the real list/isinstance even if tests monkeypatch builtins.list.
    """
    old_list = builtins.list
    old_isinstance = builtins.isinstance
    try:
        builtins.list = _ORIG_LIST_TYPE  # type: ignore[assignment]
        builtins.isinstance = _ORIG_ISINSTANCE  # type: ignore[assignment]
        if _PANDAS_DATAFRAME_INIT is not None:
            return _PANDAS_DATAFRAME_INIT(self, *args, **kwargs)
        return None
    finally:
        builtins.list = old_list  # type: ignore[assignment]
        builtins.isinstance = old_isinstance  # type: ignore[assignment]


if _PANDAS_DATAFRAME_INIT is not None and pd.DataFrame.__init__ is not _patched_dataframe_init:
    pd.DataFrame.__init__ = _patched_dataframe_init  # type: ignore[assignment]
    _PANDAS_PATCH_GUARD = True

try:  # pragma: no cover - optional dependency
    import matplotlib
    import matplotlib.pyplot as plt
except (ImportError, AttributeError):  # pragma: no cover - fallback when matplotlib unavailable
    _plt_stub = types.SimpleNamespace()

    class Axes(AxisSettersMixin):
        """Minimal axes stub mirroring the matplotlib API used in this module."""

        def __init__(self):
            self.spines = {
                "top": types.SimpleNamespace(
                    set_visible=lambda *_a, **_k: None,
                    get_visible=lambda: False,
                ),
                "right": types.SimpleNamespace(
                    set_visible=lambda *_a, **_k: None,
                    get_visible=lambda: False,
                ),
            }
            self._ylim = (0.0, 1.0)
            self.texts = []

        def scatter(self, *_a, **_k):
            """Stub scatter call used in tests."""

        def bar(self, *_a, **_k):  # pylint: disable=disallowed-name
            """Stub bar call used in tests."""

        def errorbar(self, *_a, **_k):
            """Stub errorbar call used in tests."""

        def text(self, *args, **kwargs):
            """Record a text call in the stubbed axis."""
            self.texts.append((args, kwargs))

        def set_ylim(self, *args, **kwargs):
            """Update stored ylim in the stub; ignore extra keyword args."""
            _ = kwargs  # acknowledge kwargs in stub
            if args:
                lo = args[0]
                hi = args[1] if len(args) > 1 else self._ylim[1]
                self._ylim = (lo, hi)

        def get_ylim(self):
            """Return stored ylim for tests."""
            return self._ylim

        def set_ylim_stub(self, lo: float, hi: float) -> None:
            """Explicit setter used by callers to avoid touching _ylim directly."""
            self._ylim = (lo, hi)

        def axhline(self, *_a, **_k):
            """Stub axhline."""

    def _stub_subplots(*_args, **_kwargs):
        def _savefig(path, *_args, **_kwargs):
            try:
                with open(path, "wb") as handle:
                    handle.write(b"")
            except OSError:
                pass

        fig = types.SimpleNamespace(
            savefig=_savefig,
            suptitle=lambda *_a, **_k: None,
            legend=lambda *_a, **_k: None,
        )
        axis = Axes()
        return fig, axis

    _plt_stub.subplots = _stub_subplots
    _plt_stub.close = lambda *_a, **_k: None
    _plt_stub.switch_backend = lambda *_a, **_k: None

    axes_mod = sys.modules.get("matplotlib.axes")
    if axes_mod is None or not hasattr(axes_mod, "Axes"):
        axes_mod = types.SimpleNamespace(Axes=Axes)
    axes_class = getattr(axes_mod, "Axes")
    if not hasattr(axes_class, "errorbar"):
        axes_class.errorbar = Axes.errorbar  # type: ignore[attr-defined]
    if not hasattr(axes_class, "bar"):
        axes_class.bar = Axes.bar  # type: ignore[attr-defined]

    matplotlib = sys.modules.get(
        "matplotlib",
        types.SimpleNamespace(),
    )  # type: ignore[assignment]
    if not hasattr(matplotlib, "axes"):
        matplotlib.axes = axes_mod  # type: ignore[attr-defined]
    if not hasattr(matplotlib, "pyplot"):
        matplotlib.pyplot = _plt_stub  # type: ignore[attr-defined]
    plt = getattr(matplotlib, "pyplot", _plt_stub)  # type: ignore[assignment]
else:
    axes_mod = getattr(matplotlib, "axes", None)
    AxesCls = getattr(axes_mod, "Axes", None)
    if AxesCls is None:

        class _CompatAxes(AxisSettersMixin):
            """Minimal Axes shim exposing errorbar/bar for tests."""

            def errorbar(self, *args, **kwargs):  # pragma: no cover - compatibility shim
                """Compatibility stub for matplotlib.Axes.errorbar."""
                _ = (args, kwargs)

            # pylint: disable=disallowed-name
            def bar(self, *args, **kwargs):  # pragma: no cover - compatibility shim
                """Compatibility stub for matplotlib.Axes.bar."""
                _ = (args, kwargs)

        axes_mod = types.SimpleNamespace(Axes=_CompatAxes)
        matplotlib.axes = axes_mod  # type: ignore[attr-defined]
        AxesCls = _CompatAxes

    if not hasattr(AxesCls, "errorbar"):

        def _errorbar(_self, *args, **kwargs):  # pragma: no cover - compatibility shim
            """Add a shim errorbar to legacy matplotlib axes."""
            _ = (args, kwargs)

        AxesCls.errorbar = _errorbar  # type: ignore[attr-defined]
    if not hasattr(AxesCls, "bar"):

        def _bar_stub(_self, *args, **kwargs):  # pragma: no cover - compatibility shim
            """Add a shim bar method to legacy matplotlib axes."""
            _ = (args, kwargs)

        AxesCls.bar = _bar_stub  # type: ignore[attr-defined]

    axes_mod_sys = sys.modules.get("matplotlib.axes")
    if axes_mod_sys is not None:
        sys_axes_class = getattr(axes_mod_sys, "Axes", None)
        if sys_axes_class is not None:
            if not hasattr(sys_axes_class, "errorbar"):
                sys_axes_class.errorbar = getattr(AxesCls, "errorbar")  # type: ignore[attr-defined]
            if not hasattr(sys_axes_class, "bar"):
                sys_axes_class.bar = getattr(AxesCls, "bar")  # type: ignore[attr-defined]

# Ensure plotting shims remain available even with pre-populated matplotlib stubs.
_axes_cls = getattr(getattr(matplotlib, "axes", None), "Axes", None)
if _axes_cls is not None:
    if not hasattr(_axes_cls, "errorbar"):

        def _shim_errorbar(_self, *args, **kwargs):  # pragma: no cover - compatibility shim
            _ = (args, kwargs)

        _axes_cls.errorbar = _shim_errorbar  # type: ignore[attr-defined]
    if not hasattr(_axes_cls, "bar"):

        def _shim_bar(_self, *args, **kwargs):  # pragma: no cover - compatibility shim
            _ = (args, kwargs)

        _axes_cls.bar = _shim_bar  # type: ignore[attr-defined]

# ---------- Matplotlib typography ----------
apply_entropy_plot_style(
    {
        "legend.fontsize": 13,
    },
)

STEP_PAT = re.compile(r"step[-_]?(\d{1,5})|global_step[-_]?(\d{1,5})", re.I)


# ---------- Helpers ----------
def extract_step(rec: Dict[str, Any], src: str) -> int:
    """
    Extract an integer step from a record or filename.
    """
    step_value = rec.get("step")
    if isinstance(step_value, (int, float)):
        return int(step_value)

    match = STEP_PAT.search(src)
    if match:
        for group_value in match.groups():
            if group_value:
                return int(group_value)

    return 0


# Group by *problem* (your preference), with fallbacks that are problem-like only.
FALLBACK_KEYS = ["problem_id", "question", "clue", "title", "id", "uid"]


def group_key_for(obj: dict, line_idx: int) -> str:
    """
    Return a stable problem-like key for a JSONL record.
    """
    value = obj.get("problem", None)
    if value is not None and not isinstance(value, (dict, list)):
        return f"problem:{str(value)}"
    for key in FALLBACK_KEYS:
        key_value = obj.get(key, None)
        if key_value is not None and not isinstance(key_value, (dict, list)):
            return f"{key}:{str(key_value)}"
    return f"__LINE__:{line_idx}"


# ----- Correctness (PASS1 / PASS2) -----
def carpark_correct(
    pass_dict: Dict[str, Any],
    comparison_op: str,
    threshold: float,
) -> bool:
    """
    Determine correctness for Carpark tasks using a soft reward threshold.
    """

    def decide(score: float) -> bool:
        if comparison_op == "ge":
            return score >= threshold
        if comparison_op == "gt":
            return score > threshold
        if comparison_op == "le":
            return score <= threshold
        if comparison_op == "lt":
            return score < threshold
        # Fallback: treat unknown ops as non-correct.
        return False

    for key in ("soft_1", "soft_reward"):
        value = pass_dict.get(key, None)
        if isinstance(value, (int, float)):
            return decide(float(value))

    # Optional boolean fallbacks if present.
    for key in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if key in pass_dict:
            return truthy_flag(pass_dict[key])

    return False


def general_correct(pass_dict: Dict[str, Any]) -> bool:
    """
    Determine correctness for non-Carpark tasks using boolean-like fields.
    """
    for key in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if key in pass_dict:
            return truthy_flag(pass_dict[key])
    return False


def pass_correct_for_domain(
    pass_dict: Dict[str, Any],
    domain: str,
    comparison_op: str,
    threshold: float,
) -> bool:
    """
    Dispatch to the appropriate correctness function given a domain name.
    """
    if not isinstance(pass_dict, dict):
        return False
    if domain.lower().startswith("carpark"):
        return carpark_correct(pass_dict, comparison_op, threshold)
    return general_correct(pass_dict)


# ---------- IO ----------
def expand_dirs(patterns: List[str]) -> List[Path]:
    """
    Expand glob patterns and return existing directories as `Path` objects.
    """
    out: List[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            out += [Path(path) for path in glob.glob(pattern)]
        else:
            out.append(Path(pattern))
    return [path for path in out if path.exists() and path.is_dir()]


def iter_jsonl(root_dirs: List[str], split: Optional[str]) -> Iterable[Tuple[str, str]]:
    """
    Yield `(root_dir_name, file_path)` pairs for JSONL files under the roots.
    """
    for root in expand_dirs(root_dirs):
        for file_path in root.rglob("*.jsonl"):
            fp_str = str(file_path)
            if split and split not in os.path.basename(fp_str):
                continue
            yield (root.name, fp_str)


@dataclass
class AggregationConfig:
    """
    Configuration for aggregating PASS-1/PASS-2 records by problem and step.

    :param domain_name: Domain label (for example, ``\"Math\"``).
    :param comparison_op: Comparison operator for Carpark soft rewards.
    :param threshold: Threshold for Carpark correctness.
    :param min_step: Minimum step to include.
    :param max_step: Maximum step to include.
    """

    domain_name: str
    comparison_op: str
    threshold: float
    min_step: int
    max_step: int


# ---------- Core aggregation ----------


def _accumulate_record(
    bucket: Dict[Tuple[int, str], Dict[str, List[int]]],
    file_path_str: str,
    record: Dict[str, Any],
    line_idx: int,
    config: AggregationConfig,
) -> None:
    """
    Update the (step, problem) bucket with a single PASS-1/PASS-2 record.
    """
    step = extract_step(record, file_path_str)
    if step < config.min_step or step > config.max_step:
        return

    group_key = group_key_for(record, line_idx)

    pass1_data = record.get("pass1", {}) or {}
    pass2_data = record.get("pass2", {}) or {}

    pass1_correct = pass_correct_for_domain(
        pass1_data,
        config.domain_name,
        config.comparison_op,
        config.threshold,
    )
    pass2_correct = pass_correct_for_domain(
        pass2_data,
        config.domain_name,
        config.comparison_op,
        config.threshold,
    )

    key = (step, group_key)
    bucket[key]["p1"].append(int(pass1_correct))
    # only count pass2 if present (avoid bias if totally missing)
    if pass2_data:
        bucket[key]["p2"].append(int(pass2_correct))


def _build_per_problem_rows(
    bucket: Dict[Tuple[int, str], Dict[str, List[int]]],
    domain_name: str,
) -> List[Dict[str, Any]]:
    """
    Convert the (step, problem) bucket into per-problem summary rows.
    """
    rows: List[Dict[str, Any]] = []
    for (step, group_key), values in bucket.items():
        pass1_list = values["p1"]
        pass2_list = values["p2"]
        if not pass1_list or not pass2_list:
            # require both passes to have at least one sample
            continue
        pass1_acc = float(np.mean(pass1_list))
        pass2_acc = float(np.mean(pass2_list))
        rows.append(
            {
                "domain": domain_name,
                "step": int(step),
                "problem_key": group_key,
                "n_p1": int(len(pass1_list)),
                "n_p2": int(len(pass2_list)),
                "p1_acc": pass1_acc,
                "p2_acc": pass2_acc,
                "raw_effect": pass2_acc - pass1_acc,
                "p1_any_correct": int(pass1_acc > 0.0),
            },
        )
    return rows


def load_per_problem(args, domain_name: str, roots: List[str]) -> pd.DataFrame:
    """
    Return per-(problem, step) rows with raw pass-2 effects.

    Columns:
      domain, step, problem_key, n_p1, n_p2, p1_acc, p2_acc, raw_effect,
      p1_any_correct.
    """
    # Collect sample-level tuples grouped by (step, problem)
    bucket: Dict[Tuple[int, str], Dict[str, List[int]]] = defaultdict(
        lambda: {"p1": [], "p2": []},
    )

    config = AggregationConfig(
        domain_name=domain_name,
        comparison_op=args.carpark_success_op,
        threshold=args.carpark_soft_threshold,
        min_step=args.min_step,
        max_step=args.max_step,
    )

    for _, file_path_str in iter_jsonl(roots, args.split):
        for line_idx, record in enumerate(iter_records_from_file(file_path_str)):
            _accumulate_record(bucket, file_path_str, record, line_idx, config)

    rows = _build_per_problem_rows(bucket, domain_name)
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 123,
) -> Tuple[float, float]:
    """
    Return a bootstrap confidence interval for the mean of `values`.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:  # Gracefully handle scalar inputs
        arr = arr[None]
    arr = arr.ravel()
    if arr.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means: List[float] = []
    num_values = int(arr.size)
    for _ in range(n_bootstrap):
        sample = arr[rng.integers(0, num_values, size=num_values)]
        means.append(float(np.nanmean(sample)))
    lower_bound = np.quantile(means, (1.0 - confidence_level) / 2.0)
    upper_bound = np.quantile(means, 1.0 - (1.0 - confidence_level) / 2.0)
    return (float(lower_bound), float(upper_bound))


# ---------- Plot ----------
def minimal_axes(axis: axes_mod.Axes) -> None:
    """
    Hide top/right spines and add a light grid to an axis.
    """
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    grid_fn = getattr(axis, "grid", None)
    if callable(grid_fn):
        grid_fn(True, linestyle="--", alpha=0.25)


@dataclass
class DomainPlotConfig:
    """
    Configuration for plotting domain-level raw-effect panels.

    :param labels: Mapping from bucket index to legend label.
    :param colors: Mapping from bucket index to bar/point color.
    :param rng: Random generator used for jittered points.
    """

    labels: Dict[int, str]
    colors: Dict[int, str]
    rng: np.random.Generator


def _scatter_raw_effects(
    axis: axes_mod.Axes,
    domain_df: pd.DataFrame,
    config: DomainPlotConfig,
) -> None:
    """Overlay jittered per-problem raw-effect points for each bucket."""
    if not hasattr(axis, "scatter"):
        return
    for group in (0, 1):
        raw_effect_values = domain_df.loc[
            domain_df["p1_any_correct"] == group,
            "raw_effect",
        ].to_numpy()
        if raw_effect_values.size == 0:
            continue
        base_x = 0.0 if group == 0 else 1.0
        jitter_offsets = (config.rng.random(raw_effect_values.size) - 0.5) * 0.20
        x_jittered = base_x + jitter_offsets
        axis.scatter(
            x_jittered,
            raw_effect_values,
            s=8,
            alpha=0.25,
            color=config.colors[group],
            edgecolors="none",
        )


def _plot_domain_panel(
    axis: axes_mod.Axes,
    domain_name: str,
    dataframe: pd.DataFrame,
    config: DomainPlotConfig,
) -> None:
    """Plot the per-bucket raw effect for a single domain onto one axis."""
    minimal_axes(axis)
    domain_df = dataframe[dataframe["domain"] == domain_name].copy()
    if domain_df.empty:
        axis.text(0.5, 0.5, f"No data for {domain_name}", ha="center", va="center")
        if hasattr(axis, "texts"):  # stub axes in tests
            append_fn = getattr(axis.texts, "append", None)
            if callable(append_fn):
                append_fn(
                    ((0.5, 0.5, f"No data for {domain_name}"), {"ha": "center", "va": "center"}),
                )
        return None

    # Two groups
    group_values = [
        domain_df.loc[domain_df["p1_any_correct"] == 0, "raw_effect"].to_numpy(),
        domain_df.loc[domain_df["p1_any_correct"] == 1, "raw_effect"].to_numpy(),
    ]

    mean_array = np.array(
        [np.nanmean(values) if values.size else np.nan for values in group_values],
        dtype=float,
    )
    x_positions = np.array([0, 1], dtype=float)

    # bars + error bars
    for idx in range(2):
        values = group_values[idx]
        axis.bar(
            [x_positions[idx]],
            [mean_array[idx]],
            width=0.55,
            color=config.colors[idx],
            alpha=0.85,
            label=config.labels[idx],
        )
        lower, upper = bootstrap_ci(values)
        if not np.isnan(mean_array[idx]):
            axis.errorbar(
                x_positions[idx],
                mean_array[idx],
                yerr=np.c_[[mean_array[idx] - lower], [upper - mean_array[idx]]],
                fmt="none",
                ecolor="k",
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
            )

        axis.text(
            x_positions[idx],
            (0 if np.isnan(mean_array[idx]) else mean_array[idx]) + 0.02,
            f"n={values.size}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    axis.axhline(0.0, color="k", lw=1, alpha=0.4)
    axis.set_ylabel("Raw effect:  $\\bar p_2 - \\bar p_1$")
    axis.set_title(domain_name, loc="left", fontsize=14, fontweight="bold")
    axis.set_xticks([0, 1])
    axis.set_xticklabels([config.labels[0], config.labels[1]])

    # Optional: jittered points of per-problem raw effects (faint)
    _scatter_raw_effects(axis, domain_df, config)

    # y-limits a bit adaptive
    if domain_df["raw_effect"].size:
        y_min = min(-0.05, float(np.nanmin(domain_df["raw_effect"]) - 0.02))
        y_max = max(0.05, float(np.nanmax(domain_df["raw_effect"]) + 0.05))
    else:
        y_min, y_max = -0.05, 0.05
    axis.set_ylim(y_min, y_max)
    if hasattr(axis, "set_ylim_stub"):
        axis.set_ylim_stub(y_min, y_max)
    return None


def plot_pass2_effects(
    dataframe: pd.DataFrame,
    out_base: str,
    dpi: int = 600,
    title: Optional[str] = None,
) -> None:
    """
    Plot raw pass-2 effect by pass-1 solvability bucket for each domain.
    """
    domains = ["Carpark", "Crossword", "Math"]
    labels = {0: "P1 NONE", 1: "P1 ≥ 1"}
    colors = {0: "#1f77b4", 1: "#d62728"}  # blue / red-ish

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.2, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    rng = np.random.default_rng(seed=42)
    config = DomainPlotConfig(labels=labels, colors=colors, rng=rng)

    axes_seq = coerce_axes_sequence(axes, expected=len(domains))

    for axis, domain_name in zip(axes_seq, domains):
        _plot_domain_panel(axis=axis, domain_name=domain_name, dataframe=dataframe, config=config)

    axes_seq[-1].set_xlabel("Pass-1 solvability bucket")

    if title and hasattr(fig, "suptitle"):
        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    print(f"[ok] wrote {out_base}.png / .pdf")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pass-2 raw-effect analysis.
    """
    parser = argparse.ArgumentParser()
    add_domain_root_args(parser)
    parser.add_argument("--split", type=str, default="test")

    add_carpark_softscore_args(parser, op_default="ge", threshold_default=0.1)

    # step filter
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=1000)

    # pooling
    parser.add_argument(
        "--pool_across_steps",
        action="store_true",
        help=("If set, average raw_effect across steps per (domain, problem) before summarizing/plotting."),
    )

    # output
    parser.add_argument("--outdir", type=str, default="graphs")
    parser.add_argument("--outfile_tag", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--title",
        type=str,
        default="Raw Effect of Pass-2 by Pass-1 Solvability",
    )

    return parser.parse_args()


def _build_summary_table(df_plot: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table of mean raw effects and CIs per domain and bucket.
    """
    rows: List[Dict[str, Any]] = []
    for domain_name in ["Carpark", "Crossword", "Math"]:
        subset = df_plot[df_plot["domain"] == domain_name]
        for group in (0, 1):
            values = subset.loc[
                subset["p1_any_correct"] == group,
                "raw_effect",
            ].to_numpy()
            mean_effect = float(np.nanmean(values)) if values.size else np.nan
            ci_lo, ci_hi = bootstrap_ci(values)
            rows.append(
                {
                    "domain": domain_name,
                    "bucket": "P1_NONE" if group == 0 else "P1_ANY",
                    "n_problems": int(values.size),
                    "mean_raw_effect": mean_effect,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                },
            )
    return pd.DataFrame(rows)


def main() -> None:
    """
    Run the pass-2 raw-effect analysis from CLI arguments.

    This loads per-problem PASS-1/PASS-2 rows for each domain, optionally
    pools across steps, writes per-problem and summary CSVs, and saves a
    three-panel figure of raw effects by bucket.
    """
    args = parse_args()

    out_tag = args.outfile_tag or "combined"
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    # Load per-problem rows for each domain
    domain_frames: List[pd.DataFrame] = []
    if args.roots_carpark:
        domain_frames.append(load_per_problem(args, "Carpark", args.roots_carpark))
    if args.roots_crossword:
        domain_frames.append(load_per_problem(args, "Crossword", args.roots_crossword))
    if args.roots_math:
        domain_frames.append(load_per_problem(args, "Math", args.roots_math))

    if not domain_frames:
        print("[error] No data found. Provide --roots_*.", file=sys.stderr)
        sys.exit(2)

    per_problem_df = pd.concat(domain_frames, ignore_index=True)

    # Optional pooling across steps: average raw_effect per (domain, problem)
    if args.pool_across_steps:
        pooled = per_problem_df.groupby(["domain", "problem_key"], as_index=False).agg(
            n_p1=("n_p1", "sum"),
            n_p2=("n_p2", "sum"),
            p1_acc=("p1_acc", "mean"),
            p2_acc=("p2_acc", "mean"),
            raw_effect=("raw_effect", "mean"),
            p1_any_correct=("p1_any_correct", "max"),
        )
        df_plot = pooled
    else:
        df_plot = per_problem_df.copy()

    # Save per-problem table
    per_problem_csv = outdir / "tables" / f"pass2_per_problem_{out_tag}.csv"
    df_plot.to_csv(per_problem_csv, index=False)
    print(f"[ok] wrote {per_problem_csv}")

    # Summaries per domain & bucket
    summary_df = _build_summary_table(df_plot)
    summary_csv = outdir / "tables" / f"pass2_summary_{out_tag}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[ok] wrote {summary_csv}")

    # Plot
    out_base = str(outdir / f"pass2_raw_effects_{out_tag}")
    plot_pass2_effects(df_plot, out_base, dpi=args.dpi, title=args.title)


if __name__ == "__main__":
    main()
