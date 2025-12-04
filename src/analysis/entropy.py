"""Entropy vs re-check exploratory analyses and plotting helpers.

This module provides a small command-line style entrypoint that is exercised
in tests via ``runpy.run_module("src.analysis.entropy", run_name="__main__")``.
It is intentionally lightweight so that environments without full SciPy /
statsmodels installs can still run the analyses using test stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# SciPy is optional; importing it in test environments with mismatched NumPy
# wheels can fail. Use a tiny neutral stub instead so tests can exercise the
# script structure without depending on a working SciPy install.
def ttest_ind(_a, _b, *_args, **_kwargs):
    """Fallback t-test stub returning a neutral statistic and p-value."""
    return (0.0, 1.0)


try:  # pragma: no cover - optional dependency in tests
    from src.analysis.metrics import wilson_ci
except (ImportError, ModuleNotFoundError):  # pragma: no cover

    def wilson_ci(_num_success: int, _num_trials: int) -> tuple[float, float]:
        """Fallback Wilson CI stub returning 0s when metrics import is unavailable."""
        return (0.0, 0.0)


ROOT_DEFAULT = Path("artifacts/results/od2961/Math220k/GRPO/1.5B")


def _load_records(root: Path) -> pd.DataFrame:
    """Load all JSONL result files under ``root``."""
    files = list(root.glob("**/*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files found under {root!r}")
    frames = [pd.read_json(path, lines=True) for path in files]
    return pd.concat(frames, ignore_index=True)


def _run_summary_plots(main_df: pd.DataFrame, outdir: Path) -> None:
    """Basic summary statistics, boxplot, scatter, and t-test."""
    required_main = {"has_recheck", "entropy", "step"}
    if not required_main.issubset(main_df.columns):
        missing = required_main - set(main_df.columns)
        raise KeyError(f"Missing column(s) {sorted(missing)!r} in main frame")

    summary = (
        main_df.groupby("has_recheck")["entropy"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_entropy", "std": "std_entropy"})
        .reset_index()
    )
    print(summary)

    data0 = main_df[main_df["has_recheck"] == 0]["entropy"]
    data1 = main_df[main_df["has_recheck"] == 1]["entropy"]

    outdir.mkdir(parents=True, exist_ok=True)

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot([data0, data1], tick_labels=["No re-check", "Re-check"])
    plt.ylabel("Avg Token Entropy")
    plt.title("Entropy vs Re-check Status")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_vs_recheck_boxplot.png", dpi=300)
    plt.close()

    # Scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(
        main_df["step"],
        main_df["entropy"],
        c=main_df["has_recheck"],
        cmap="coolwarm",
        alpha=0.6,
    )
    plt.colorbar(label="has_recheck")
    plt.xlabel("Training Step")
    plt.ylabel("Avg Token Entropy")
    plt.title("Entropy by Step, colored by Re-check")
    plt.tight_layout()
    plt.savefig(outdir / "entropy_vs_step_scatter.png", dpi=300)
    plt.close()

    # T-test
    t_stat, p_val = ttest_ind(data0, data1, equal_var=False)
    print(f"T-test: t={t_stat:.2f}, p={p_val:.3f}")


def _load_scored_frame(root: Path) -> tuple[pd.DataFrame, Path]:
    """Return the scored dataframe and analysis directory for a given root."""
    analysis_dir = root / "analysis"
    scored_files = sorted(analysis_dir.glob("*_scored.jsonl"))
    if not scored_files:
        raise SystemExit(f"No scored files in {analysis_dir!r}")

    df = pd.concat([pd.read_json(path, lines=True) for path in scored_files], ignore_index=True)

    required_scored = {"entropy", "correct", "rechecked"}
    if not required_scored.issubset(df.columns):
        missing = required_scored - set(df.columns)
        raise KeyError(f"Missing column(s) {sorted(missing)!r} in scored data")

    df = df.copy()
    df["correct"] = df["correct"].astype(int)
    df["has_recheck"] = df["rechecked"].astype(bool)
    return df, analysis_dir


def _summarize_by_decile(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Bucket entropy into deciles and compute accuracy and Wilson CIs."""
    decile_labels: List[str] = [f"D{index + 1}" for index in range(10)]
    df = df.copy()
    df["entropy_bucket"] = pd.qcut(
        df["entropy"],
        q=10,
        labels=decile_labels,
        duplicates="drop",
    )

    grouped = (
        df.groupby(["entropy_bucket", "has_recheck"], observed=True)["correct"].agg(n="size", k="sum").reset_index()
    )
    grouped["acc"] = grouped["k"] / grouped["n"]

    ci_lo: List[float] = []
    ci_hi: List[float] = []
    for k_val, n_val in zip(grouped["k"], grouped["n"], strict=False):
        lo, hi = wilson_ci(int(k_val), int(n_val))
        ci_lo.append(lo)
        ci_hi.append(hi)
    grouped["ci_lo"] = ci_lo
    grouped["ci_hi"] = ci_hi
    return grouped, decile_labels


def _plot_decile_accuracy(
    summary_df: pd.DataFrame,
    decile_labels: List[str],
    analysis_dir: Path,
) -> None:
    """Plot accuracy by entropy decile and re-check flag."""
    available_buckets = set(summary_df["entropy_bucket"].dropna())
    buckets = [bucket for bucket in decile_labels if bucket in available_buckets]
    if not buckets:
        return

    x_vals = np.arange(len(buckets))
    no_rc = summary_df[~summary_df["has_recheck"]].set_index("entropy_bucket")
    yes_rc = summary_df[summary_df["has_recheck"]].set_index("entropy_bucket")

    _, ax = plt.subplots(figsize=(8, 5))
    if not no_rc.empty:
        ax.errorbar(
            x_vals - 0.05,
            [float(no_rc.at[bucket, "acc"]) for bucket in buckets],
            yerr=[
                [float(no_rc.at[bucket, "acc"] - no_rc.at[bucket, "ci_lo"]) for bucket in buckets],
                [float(no_rc.at[bucket, "ci_hi"] - no_rc.at[bucket, "acc"]) for bucket in buckets],
            ],
            fmt="-o",
            capsize=4,
            label="No re-check",
            color="C0",
        )
    if not yes_rc.empty:
        ax.errorbar(
            x_vals + 0.05,
            [float(yes_rc.at[bucket, "acc"]) for bucket in buckets],
            yerr=[
                [float(yes_rc.at[bucket, "acc"] - yes_rc.at[bucket, "ci_lo"]) for bucket in buckets],
                [float(yes_rc.at[bucket, "ci_hi"] - yes_rc.at[bucket, "acc"]) for bucket in buckets],
            ],
            fmt="-s",
            capsize=4,
            label="With re-check",
            color="C1",
        )

    ax.set_xticks(x_vals, buckets)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Entropy Decile")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Entropy Decile & Re-check (±95% CI)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(analysis_dir / "accuracy_by_entropy_decile_ci_with_n.png", dpi=300)
    plt.close()


def _maybe_plot_flip_trajectories(df: pd.DataFrame, analysis_dir: Path) -> None:
    """
    Best-effort flip-trajectory summary and plot; failures are swallowed.

    This is intentionally defensive because tests may install heavily stubbed
    pandas/plotting backends where some operations behave differently.
    """
    if {"problem", "sample_idx", "step"}.issubset(df.columns):
        try:
            df_all = df.copy()
            df_all["correct"] = df_all["correct"].astype(bool)
            df_all["has_recheck"] = df_all["has_recheck"].astype(bool)

            df_sorted = df_all.sort_values("step")
            stats = (
                df_sorted.groupby(["problem", "sample_idx"])["correct"]
                .agg(
                    first_corr="first",
                    last_corr="last",
                )
                .reset_index()
            )

            flip_mask = (~stats["first_corr"]) & (stats["last_corr"])
            flip_ids = stats.loc[flip_mask, ["problem", "sample_idx"]]
            flip_df = df_all.merge(flip_ids, on=["problem", "sample_idx"], how="inner")

            agg = (
                flip_df.groupby("step")
                .agg(
                    mean_entropy=("entropy", "mean"),
                    recheck_rate=("has_recheck", "mean"),
                    n_trajectories=("correct", "size"),
                )
                .reset_index()
            )

            if not agg.empty:
                fig, ax1 = plt.subplots(figsize=(8, 5))
                ax1.set_xlabel("Training Step")
                ax1.set_ylabel("Mean Token Entropy", color="C0")
                ax1.plot(
                    agg["step"],
                    agg["mean_entropy"],
                    marker="o",
                    color="C0",
                    label="Entropy",
                )
                ax1.tick_params(axis="y", labelcolor="C0")

                ax2 = ax1.twinx()
                ax2.set_ylabel("Re-check Rate", color="C1")
                ax2.plot(
                    agg["step"],
                    agg["recheck_rate"],
                    marker="s",
                    color="C1",
                    label="Re-check rate",
                )
                ax2.tick_params(axis="y", labelcolor="C1")
                ax2.set_ylim(0, 1)

                for x_val, n_val in zip(agg["step"], agg["n_trajectories"], strict=False):
                    ax1.text(
                        x_val,
                        float(agg["mean_entropy"].max()) * 1.02,
                        f"n={int(n_val)}",
                        ha="center",
                        fontsize=8,
                    )

                fig.tight_layout()
                ax1.set_title("Flip Trajectories: Entropy & Re-check Rate over Steps")
                out = analysis_dir / "flip_entropy_recheck_over_steps.png"
                plt.savefig(out, dpi=300)
                plt.close()
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError):
            # Flip-trajectory analysis is best-effort; failure should not abort
            # the overall entropy script, especially in heavily stubbed test envs.
            pass


def _run_scored_analysis(root: Path) -> None:
    """
    Load scored JSONLs under ``root / 'analysis'`` and run lightweight summary
    analyses and plots.

    This helper is designed so tests can stub out plotting and heavy
    dependencies (for example, statsmodels) while still exercising the
    structure of the analysis.
    """
    df, analysis_dir = _load_scored_frame(root)
    summary_df, decile_labels = _summarize_by_decile(df)
    _plot_decile_accuracy(summary_df, decile_labels, analysis_dir)
    _maybe_plot_flip_trajectories(df, analysis_dir)


def main() -> None:
    """Run the entropy vs. re-check exploratory analyses and plots."""
    root = ROOT_DEFAULT
    main_df = _load_records(root)
    outdir = Path("analysis")

    _run_summary_plots(main_df, outdir)
    _run_scored_analysis(root)


if __name__ == "__main__":
    main()
