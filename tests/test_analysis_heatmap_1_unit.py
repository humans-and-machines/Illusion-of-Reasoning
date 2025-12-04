import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.heatmap_1 as hm


def test_count_ahas_and_sweep_grid_basic():
    step_df = pd.DataFrame(
        {
            "domain_key": ["D", "D", "D"],
            "problem_id": ["p1", "p1", "p2"],
            "step": [1, 2, 1],
            "acc_frac": [0.0, 0.6, 0.2],
            "shift_frac": [0.0, 0.1, 0.5],
        }
    )
    events, pairs = hm.count_ahas(step_df, delta1=0.1, delta2=0.2)
    assert (events, pairs) == (1, 1)  # only p1 step2 qualifies

    grid = hm.sweep_grid(step_df, deltas=[0.0, 0.25])
    levels, values = hm._build_values_matrix(grid)
    assert levels == [0.0, 0.25]
    assert np.isfinite(values).any()


def test_foreground_and_annotate_heatmap():
    df_grid = pd.DataFrame({"delta1": [0.0], "delta2": [0.0], "n_events": [1], "n_pairs": [2], "pct": [50.0]})
    levels, values = hm._build_values_matrix(df_grid)
    cmap = hm.plt.get_cmap("YlGnBu")
    norm = hm.mcolors.Normalize(vmin=0, vmax=100)
    fig, ax = hm.plt.subplots()
    ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    hm._annotate_heatmap(ax, df_grid, (levels, values), cmap, norm)
    assert len(ax.texts) == 1
    hm.plt.close(fig)


def test_plot_heatmap_outputs_files(tmp_path):
    df_grid = pd.DataFrame(
        {
            "delta1": [0.0, 0.5],
            "delta2": [0.0, 0.5],
            "n_events": [1, 2],
            "n_pairs": [2, 4],
            "pct": [50.0, 50.0],
        }
    )
    out_png = tmp_path / "heat.png"
    hm.plot_heatmap(df_grid, "t", str(out_png), cmap_name="viridis")
    assert out_png.exists() and out_png.with_suffix(".pdf").exists()


def test_add_per_domain_and_group_plots(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        hm,
        "plot_heatmap",
        lambda grid, title, path, cmap_name=None: calls.append((title, Path(path).name)),
    )
    step_df = pd.DataFrame(
        {
            "domain_key": ["A", "B"],
            "problem_id": ["p1", "p2"],
            "step": [1, 1],
            "acc_frac": [0.1, 0.2],
            "shift_frac": [0.0, 0.0],
            "n_samples": [1, 1],
        }
    )
    long_rows: list[pd.DataFrame] = []
    per_config = hm.PerDomainPlotConfig(
        label_map={"A": "DomainA", "B": "DomainB"},
        delta_values=[0.0],
        long_rows=long_rows,
        out_dir=str(tmp_path),
        cmap_name="viridis",
    )
    hm._add_per_domain_grids_and_plots(step_df, per_config)
    assert {title for title, _ in calls} == {"Aha! Moment Prevalence (DomainA)", "Aha! Moment Prevalence (DomainB)"}
    assert len(long_rows) == 2

    args = types.SimpleNamespace(
        make_15b_overall=True,
        domains_15b="A,B",
        delta_values=[0.0],
        title_15b="15b",
        cmap="viridis",
    )
    hm._add_group_15b_grid_and_plot(step_df, args, long_rows, str(tmp_path))
    assert any("overall_1p5b" in name for _, name in calls) or len(calls) >= 3


def test_load_step_level_data_handles_empty(monkeypatch):
    monkeypatch.setattr(hm, "load_rows", lambda *_: pd.DataFrame())
    with pytest.raises(SystemExit):
        hm._load_step_level_data(files_by_domain={}, load_config=None)

    monkeypatch.setattr(
        hm,
        "load_rows",
        lambda *_: pd.DataFrame({"domain_key": ["X"], "problem_id": ["p"], "step": [1], "correct": [1], "shift": [0]}),
    )
    out = hm._load_step_level_data({}, None)
    assert {"acc_frac", "shift_frac", "n_samples"}.issubset(out.columns)
