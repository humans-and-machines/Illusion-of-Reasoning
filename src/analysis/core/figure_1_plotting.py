"""Plotting helpers for Figure 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .figure_1_data import bootstrap_problem_ratio, fit_trend_wls, mark_formal_pairs
from .figure_1_style import a4_size_inches, lighten_hex


def _panel_auto_ylim(
    dfs_by_domain: Dict[str, pd.DataFrame],
    pad: float = 0.05,
    clamp: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
) -> Tuple[float, float]:
    vals = []
    for data_frame in dfs_by_domain.values():
        if data_frame is None or data_frame.empty:
            continue
        pieces = []
        for column in ("ratio", "lo", "hi"):
            if column in data_frame:
                arr = data_frame[column].to_numpy(dtype=float)
                pieces.append(arr[np.isfinite(arr)])
        if pieces:
            arr = np.concatenate(pieces)
            if arr.size:
                vals.append((float(arr.min()), float(arr.max())))
    if not vals:
        return (0.0, 1.0)
    lo_val = min(value[0] for value in vals)
    hi_val = max(value[1] for value in vals)
    if hi_val == lo_val:
        hi_val = lo_val + 0.05
    rng = hi_val - lo_val
    lo_val -= pad * rng
    hi_val += pad * rng
    if clamp[0] is not None:
        lo_val = max(clamp[0], lo_val)
    if clamp[1] is not None:
        hi_val = min(clamp[1], hi_val)
    return (lo_val, hi_val)


@dataclass
class PlotSeriesOptions:
    """Configuration for series plotting."""

    alpha_ci: float
    marker_size: float = 5.0
    add_trend: bool = True
    trend_style: str = "--"
    highlight_map: Optional[Dict[str, Dict[int, bool]]] = None
    highlight_color: str = "#2ca02c"


@dataclass
class PanelSpec:
    """Axis + metadata for a single ratio panel."""

    axis: Any
    data: Dict[str, pd.DataFrame]
    title: str
    label: str
    y_limits: Tuple[float, float]
    options: PlotSeriesOptions


@dataclass
class PanelBuilderContext:
    """Inputs needed to configure the ratio panel trio."""

    axes: np.ndarray
    native_by_dom: Dict[str, pd.DataFrame]
    gpt_by_dom: Dict[str, pd.DataFrame]
    formal_by_dom: Dict[str, pd.DataFrame]
    base_options: PlotSeriesOptions
    config: Dict[str, Any]


@dataclass(frozen=True)
class FormalRatioSeries:
    """Series extracted from the formal ratio bootstrap."""

    step: pd.Series
    ratio: pd.Series
    lower_ci: pd.Series
    upper_ci: pd.Series


def _plot_series_per_domain(
    axis,
    dfs_by_domain: Dict[str, pd.DataFrame],
    domain_colors: Dict[str, str],
    options: PlotSeriesOptions,
) -> List[Dict[str, Any]]:
    """Plot per-domain dots + CIs (+ optional highlights)."""
    trend_rows: List[Dict[str, Any]] = []
    for domain, data_frame in dfs_by_domain.items():
        if data_frame is None or data_frame.empty:
            continue
        color = domain_colors.get(domain)
        ci_color = _lighten_for_ci(color) if color else None
        axis.scatter(
            data_frame["step"],
            data_frame["ratio"],
            s=(options.marker_size**2),
            marker="o",
            color=color,
            edgecolors="none",
            label=domain,
        )
        if data_frame["lo"].notna().any() and ci_color:
            fill_between = getattr(axis, "fill_between", None)
            if callable(fill_between):
                fill_between(
                    data_frame["step"],
                    data_frame["lo"],
                    data_frame["hi"],
                    alpha=options.alpha_ci,
                    color=ci_color,
                )

        trend_info = _compute_trend(data_frame)
        trend_ok = options.add_trend and np.isfinite(trend_info["slope"]) and trend_info["fit_steps"].size >= 2
        if trend_ok:
            axis.plot(
                trend_info["fit_steps"],
                trend_info["fit_ratios"],
                options.trend_style,
                lw=2.0,
                color=color,
                alpha=0.95,
            )

        _plot_highlights(axis, data_frame, options, domain)

        trend_rows.append(
            {
                "domain": domain,
                "slope_per_step": trend_info["slope"],
                "slope_per_1k": trend_info["slope_per_1k"],
                "delta_over_range": trend_info["delta_range"],
                "intercept": trend_info["intercept"],
                "weighted_R2": trend_info["r_squared"],
            },
        )
    return trend_rows


def _compute_trend(data_frame: pd.DataFrame) -> Dict[str, Any]:
    """Return slope/intercept/stats for a per-domain series."""
    (
        slope,
        intercept,
        slope_k,
        delta_range,
        r_squared_value,
        x_fit,
        y_fit,
    ) = fit_trend_wls(data_frame)
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_per_1k": slope_k,
        "delta_range": delta_range,
        "r_squared": r_squared_value,
        "fit_steps": x_fit,
        "fit_ratios": y_fit,
    }


def _plot_highlights(
    axis,
    data_frame: pd.DataFrame,
    options: PlotSeriesOptions,
    domain: str,
) -> None:
    """Overlay highlighted markers when a highlight map is provided."""
    if not options.highlight_map or str(domain) not in options.highlight_map:
        return
    steps = data_frame["step"].to_numpy(dtype=int)
    ratios = data_frame["ratio"].to_numpy(dtype=float)
    mask = np.array(
        [bool(options.highlight_map[str(domain)].get(int(step), False)) for step in steps],
        dtype=bool,
    )
    if mask.any():
        axis.scatter(
            steps[mask],
            ratios[mask],
            s=(options.marker_size**2),
            marker="o",
            color=options.highlight_color,
            edgecolors="none",
            zorder=3,
        )


def _lighten_for_ci(color: Optional[str]) -> Optional[str]:
    """Return a lighter shade for CI fills."""
    return lighten_hex(color, 0.65) if color else None


def _set_axes_box_aspect(axis, ratio: float) -> None:
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect(ratio)
    else:
        position = axis.get_position()
        fig = axis.figure
        fig_width, fig_height = fig.get_size_inches()
        new_height = (position.width * fig_width * ratio) / fig_height
        y_origin = position.y0 + 0.5 * (position.height - new_height)
        axis.set_position(
            [position.x0, max(0.0, y_origin), position.width, max(0.0, new_height)],
        )


def _setup_ratio_figure(config: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
    """Create the 1x3 figure and enforce consistent box aspect."""
    if config["a4_pdf"]:
        fig_size = a4_size_inches(config["a4_orientation"])
    else:
        fig_size = (16.5, 4.8)
    fig, axes = plt.subplots(1, 3, figsize=fig_size, dpi=150, sharex=True, sharey=False)
    axes_arr = np.array(axes).reshape(-1)
    panel_aspect = config.get("panel_box_aspect", 0.8)
    for axis in axes_arr:
        _set_axes_box_aspect(axis, panel_aspect)
    return fig, axes_arr


def _build_panel_specs(context: PanelBuilderContext) -> List[PanelSpec]:
    """Bundle axis/data metadata for each of the three ratio panels."""
    model_name = context.config["model"]
    yl0 = _panel_auto_ylim(context.native_by_dom, pad=0.06, clamp=(0.0, 1.0))
    yl1 = _panel_auto_ylim(context.gpt_by_dom, pad=0.06, clamp=(0.0, 1.0))
    yl2 = _panel_auto_ylim(context.formal_by_dom, pad=0.06, clamp=(0.0, 1.0))
    highlight_map = context.config.get("highlight_formal_by_dom")
    highlight_color = context.config.get("highlight_color", "#2ca02c")

    return [
        PanelSpec(
            axis=context.axes[0],
            data=context.native_by_dom,
            title=f'"Cue Phrases" Detection, {model_name}',
            label="Words/Cue Phrases",
            y_limits=yl0,
            options=context.base_options,
        ),
        PanelSpec(
            axis=context.axes[1],
            data=context.gpt_by_dom,
            title=f"LLM-Detected reasoning shifts, {model_name}",
            label="LLM-Detected Shifts",
            y_limits=yl1,
            options=context.base_options,
        ),
        PanelSpec(
            axis=context.axes[2],
            data=context.formal_by_dom,
            title=f"Formal Reasoning Shifts ($\\delta_3 = 0$), {model_name}",
            label="Formal Shifts",
            y_limits=yl2,
            options=replace(
                context.base_options,
                highlight_map=highlight_map,
                highlight_color=highlight_color,
            ),
        ),
    ]


def _style_ratio_axis(axis, y_limits: Tuple[float, float], title_text: str) -> None:
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Ratio")
    axis.set_title(title_text, pad=8)
    axis.grid(True, alpha=0.35)
    axis.tick_params(axis="y", labelleft=True)


def _render_ratio_panels(
    panel_specs: List[PanelSpec],
    domain_colors: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Draw dots/CIs for each axis and return aggregated trend rows."""
    trend_rows_all: List[Dict[str, Any]] = []
    for spec in panel_specs:
        rows = _plot_series_per_domain(
            spec.axis,
            spec.data,
            domain_colors,
            spec.options,
        )
        trend_rows_all += [{"series": spec.label, **row} for row in rows]
        _style_ratio_axis(spec.axis, spec.y_limits, spec.title)
    return trend_rows_all


def _add_ratio_legend(
    fig,
    domain_colors: Dict[str, str],
    marker_size: float,
    highlight_map: Optional[Dict[str, Dict[int, bool]]],
    highlight_color: str,
) -> None:
    handles: List[Line2D] = []
    labels: List[str] = []
    for domain, color in domain_colors.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=color,
                markersize=marker_size,
                label=domain,
            ),
        )
        if not hasattr(handles[-1], "get_label"):
            handles[-1].get_label = lambda lbl=domain: lbl  # type: ignore[attr-defined]
        labels.append(domain)
    any_green = bool(
        highlight_map and any(any(flag for flag in stepmap.values()) for stepmap in highlight_map.values())
    )
    if any_green:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=highlight_color,
                markersize=marker_size,
                label="Δ > 0 at shift",
            ),
        )
        if not hasattr(handles[-1], "get_label"):
            handles[-1].get_label = lambda lbl="Δ > 0 at shift": lbl  # type: ignore[attr-defined]
        labels.append("Δ > 0 at shift")

    legend_columns = min(
        4,
        max(2, len(domain_colors) + (1 if any_green else 0)),
    )
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=legend_columns,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )


def _finalize_ratio_figure(fig, config: Dict[str, Any]) -> None:
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.suptitle(
        f"{config['dataset']} — {config['model']}",
        y=1.02,
        fontsize=12,
    )
    fig.savefig(config["out_png"])
    if config.get("a4_pdf"):
        fig.set_size_inches(*a4_size_inches(config["a4_orientation"]))
    fig.savefig(config["out_pdf"])
    plt.close(fig)


def plot_three_ratios_shared_axes_multi(
    native_by_dom: Dict[str, pd.DataFrame],
    gpt_by_dom: Dict[str, pd.DataFrame],
    formal_by_dom: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Render the three-panel ratio plot with optional green highlights."""
    fig, axes = _setup_ratio_figure(config)
    base_options = PlotSeriesOptions(
        alpha_ci=config["alpha_ci"],
        marker_size=config.get("marker_size", 5.0),
    )
    highlight_map = config.get("highlight_formal_by_dom")
    highlight_color = config.get("highlight_color", "#2ca02c")

    panel_context = PanelBuilderContext(
        axes=axes,
        native_by_dom=native_by_dom,
        gpt_by_dom=gpt_by_dom,
        formal_by_dom=formal_by_dom,
        base_options=base_options,
        config=config,
    )
    panel_specs = _build_panel_specs(panel_context)
    trend_rows_all = _render_ratio_panels(panel_specs, config["domain_colors"])
    _add_ratio_legend(
        fig,
        config["domain_colors"],
        base_options.marker_size,
        highlight_map,
        highlight_color,
    )
    _finalize_ratio_figure(fig, config)
    return trend_rows_all


@dataclass(frozen=True)
class FormalSweepPlotConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for the formal sweep grid visualization."""

    min_prior_steps: int
    n_bootstrap: int
    seed: int
    out_png: str
    out_pdf: str
    dataset: str
    model: str
    primary_color: str
    ci_color: str
    ymax: float
    alpha_ci: float
    a4_pdf: bool
    orientation: str
    line_width: float = 2.0
    marker_size: float = 4.0
    delta3: Optional[float] = None


def _init_formal_grid(
    delta1_list: List[float],
    delta2_list: List[float],
    config: FormalSweepPlotConfig,
) -> Tuple[Any, np.ndarray]:
    """Create the figure/axes grid for the δ1/δ2 sweep."""
    if config.a4_pdf:
        fig_size = a4_size_inches(config.orientation)
    else:
        fig_size = (4.8 * max(1, len(delta2_list)), 3.2 * max(1, len(delta1_list)))
    fig, axes = plt.subplots(
        len(delta1_list),
        len(delta2_list),
        figsize=fig_size,
        dpi=140,
        sharex=True,
        sharey=True,
    )
    axes_arr = np.array(axes).reshape(len(delta1_list), len(delta2_list))
    return fig, axes_arr


def _compute_formal_ratio_series(
    pair_stats_base: pd.DataFrame,
    delta_pair: Tuple[float, float],
    config: FormalSweepPlotConfig,
) -> FormalRatioSeries:
    """Compute the bootstrapped formal ratio and return its key series."""
    pair_stats = mark_formal_pairs(
        pair_stats_base.copy(),
        delta1=float(delta_pair[0]),
        delta2=float(delta_pair[1]),
        min_prior_steps=config.min_prior_steps,
        delta3=config.delta3,
    )
    formal_df = bootstrap_problem_ratio(
        pair_stats,
        "aha_formal",
        num_bootstrap=config.n_bootstrap,
        seed=config.seed,
    )
    if not isinstance(formal_df, pd.DataFrame):
        raise TypeError("bootstrap_problem_ratio must return a pandas DataFrame.")
    ratio_df: pd.DataFrame = pd.DataFrame(formal_df)
    return FormalRatioSeries(
        step=ratio_df["step"],
        ratio=ratio_df["ratio"],
        lower_ci=ratio_df["lo"],
        upper_ci=ratio_df["hi"],
    )


def _plot_formal_ratio(
    axis,
    ratio_series: FormalRatioSeries,
    config: FormalSweepPlotConfig,
) -> None:
    """Plot the ratio series and CI shading for a single cell."""
    axis.plot(
        ratio_series.step,
        ratio_series.ratio,
        marker="o",
        ms=config.marker_size,
        lw=config.line_width,
        color=config.primary_color,
    )
    if ratio_series.lower_ci.notna().any():
        axis.fill_between(
            ratio_series.step,
            ratio_series.lower_ci,
            ratio_series.upper_ci,
            alpha=config.alpha_ci,
            color=config.ci_color,
        )


def _format_delta_title(delta_pair: Tuple[float, float], delta3: Optional[float]) -> str:
    delta1_value, delta2_value = delta_pair
    delta3_str = f", δ3={delta3:.2f}" if delta3 is not None else ""
    return f"δ1={delta1_value:.2f}, δ2={delta2_value:.2f}{delta3_str}"


def _plot_formal_cell(
    axis,
    pair_stats_base: pd.DataFrame,
    delta_pair: Tuple[float, float],
    config: FormalSweepPlotConfig,
    axis_flags: Tuple[bool, bool],
) -> None:
    """Render a single δ1/δ2 cell."""
    ratio_series = _compute_formal_ratio_series(pair_stats_base, delta_pair, config)
    _plot_formal_ratio(axis, ratio_series, config)
    axis.set_ylim(0.0, config.ymax)
    axis.grid(True, alpha=0.35)
    if axis_flags[0]:
        axis.set_xlabel("Training step")
    if axis_flags[1]:
        axis.set_ylabel('Formal "Aha!" ratio')
    axis.set_title(_format_delta_title(delta_pair, config.delta3), fontsize=12, pad=6)


def _formal_grid_handles(config: FormalSweepPlotConfig) -> Tuple[Line2D, Patch]:
    line = Line2D(
        [0],
        [0],
        color=config.primary_color,
        lw=config.line_width,
        marker="o",
        ms=config.marker_size,
        label="Ratio",
    )
    patch = Patch(
        facecolor=config.ci_color,
        alpha=config.alpha_ci,
        label="95% CI (bootstrap)",
    )
    return line, patch


def _finalize_formal_grid(fig, config: FormalSweepPlotConfig) -> None:
    line_handle, patch_handle = _formal_grid_handles(config)
    fig.legend(
        handles=[line_handle, patch_handle],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.15),
    )
    suptitle_text = f'Formal "Aha!" ratio sweep\n{config.dataset}, {config.model}'
    fig.suptitle(suptitle_text, y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(config.out_png)
    if config.a4_pdf:
        fig.set_size_inches(*a4_size_inches(config.orientation))
    fig.savefig(config.out_pdf)
    plt.close(fig)


def plot_formal_sweep_grid(
    ps_base: pd.DataFrame,
    delta1_list: List[float],
    delta2_list: List[float],
    config: FormalSweepPlotConfig,
) -> None:
    """Plot formal Aha ratios across a δ1/δ2 sweep grid."""
    fig, axes = _init_formal_grid(delta1_list, delta2_list, config)
    for i, delta1_value in enumerate(delta1_list):
        for j, delta2_value in enumerate(delta2_list):
            _plot_formal_cell(
                axis=axes[i, j],
                pair_stats_base=ps_base,
                delta_pair=(delta1_value, delta2_value),
                config=config,
                axis_flags=(i == len(delta1_list) - 1, j == 0),
            )

    _finalize_formal_grid(fig, config)


__all__ = [
    "plot_three_ratios_shared_axes_multi",
    "plot_formal_sweep_grid",
    "FormalSweepPlotConfig",
]
