"""Figure and table scripts for the paper.

This subpackage groups the top-level plotting / table scripts; the underlying
implementations live in modules such as ``figure_1``, ``figure_2_uncertainty``,
``final_plot_uncertainty_gate``, etc.
"""

from __future__ import annotations

from .. import (  # type: ignore F401
    figure_1,
    figure_2_uncertainty,
    final_plot_uncertainty_gate,
    flips,
    forced_aha_effect,
    graph_1,
    graph_2,
    graph_3,
    graph_3_stacked,
    graph_4,
    h2_temp_aha_eval,
    h3_uncertainty_buckets,
    h4_analysis,
    heatmap_1,
    math_plots,
    table_1,
    temp_graph,
)


__all__ = [
    "figure_1",
    "figure_2_uncertainty",
    "final_plot_uncertainty_gate",
    "math_plots",
    "table_1",
    "graph_1",
    "graph_2",
    "graph_3",
    "graph_3_stacked",
    "graph_4",
    "heatmap_1",
    "temp_graph",
    "flips",
    "forced_aha_effect",
    "h2_temp_aha_eval",
    "h3_uncertainty_buckets",
    "h4_analysis",
]
