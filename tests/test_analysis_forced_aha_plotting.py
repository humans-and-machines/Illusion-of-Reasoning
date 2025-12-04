from pathlib import Path

import numpy as np
import pandas as pd

import src.analysis.forced_aha_plotting as fplot
from src.analysis.plotting_styles import DEFAULT_COLORS


def test_savefig_creates_png_and_pdf(tmp_path):
    fig = fplot.plt.figure()
    out_base = tmp_path / "figs" / "test"
    fplot.savefig(fig, str(out_base))
    assert (out_base.with_suffix(".png")).exists()
    assert (out_base.with_suffix(".pdf")).exists()


def test_build_overall_delta_bars_uses_ci_and_pvalues(monkeypatch):
    inputs = fplot.OverallDeltaInputs(
        summary_rows=[
            {"metric": "sample", "delta_pp": 2.0, "p_mcnemar": 0.01},
            {"metric": "cluster_mean", "delta_pp": 1.0, "p_ttest": 0.2},
            {"metric": "cluster_any", "delta_pp": -1.0, "p_mcnemar": 0.05},
        ],
        pairs_df=pd.DataFrame({"correct1": [0, 1], "correct2": [1, 1]}),
        clusters_df=pd.DataFrame({"any_p1": [0, 1], "any_p2": [1, 1]}),
    )
    monkeypatch.setattr(fplot, "_bootstrap_delta", lambda *_args, **_kwargs: (0.0, 1.0))
    config = fplot.OverallDeltaConfig(
        n_boot=10,
        seed=0,
        palette={
            "bar_primary": "#111",
            "bar_secondary": "#222",
            "bar_tertiary": "#333",
        },
    )
    bars = fplot._build_overall_delta_bars(inputs, config)
    assert len(bars) == 3
    assert any(bar.p_value == 0.01 for bar in bars)


def test_build_overall_delta_bars_skips_missing_metric(monkeypatch):
    inputs = fplot.OverallDeltaInputs(
        summary_rows=[{"metric": "cluster_any", "delta_pp": 1.0, "p_mcnemar": 0.1}],
        pairs_df=pd.DataFrame(),
        clusters_df=pd.DataFrame(),
    )
    monkeypatch.setattr(fplot, "_bootstrap_delta", lambda *_args, **_kwargs: (1.0, 2.0))
    config = fplot.OverallDeltaConfig(
        n_boot=5,
        seed=0,
        palette={"bar_primary": "#1", "bar_secondary": "#2", "bar_tertiary": "#3"},
    )
    bars = fplot._build_overall_delta_bars(inputs, config)
    assert len(bars) == 1


def test_plot_overall_deltas_calls_draw(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(
        fplot,
        "_build_overall_delta_bars",
        lambda inputs, config: [
            fplot.OverallDeltaBar(
                label="Sample",
                height=1.0,
                error_lower=0.1,
                error_upper=0.2,
                color="#000",
                p_value=None,
            )
        ],
    )
    monkeypatch.setattr(fplot, "savefig", lambda fig, out_base: called.setdefault("out", out_base))
    fplot.plot_overall_deltas(str(tmp_path), None, None)
    assert "overall_deltas" in called["out"]


def test_plot_overall_deltas_returns_when_no_bars(monkeypatch):
    monkeypatch.setattr(fplot, "_build_overall_delta_bars", lambda inputs, config: [])
    called = {}
    monkeypatch.setattr(fplot, "_draw_overall_delta_bars", lambda *a, **k: called.setdefault("drawn", True))
    fplot.plot_overall_deltas("out", None, None)
    assert called == {}


def test_draw_overall_delta_bars_labels_pzero(monkeypatch, tmp_path):
    texts = []

    class AxisStub:
        def bar(self, *a, **k):
            return None

        def axhline(self, *a, **k):
            return None

        def set_xticks(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def text(self, x, y, label_text, **kwargs):
            texts.append(label_text)

    class FigStub:
        def add_axes(self, *_):
            return AxisStub()

        def savefig(self, *_, **__):
            return None

    monkeypatch.setattr(fplot.plt, "figure", lambda *a, **k: FigStub())
    monkeypatch.setattr(fplot, "savefig", lambda fig, out_base: None)
    bars = [
        fplot.OverallDeltaBar(
            label="A",
            height=1.0,
            error_lower=0.1,
            error_upper=0.2,
            color="#000",
            p_value=0.0,
        )
    ]
    fplot._draw_overall_delta_bars("out", bars)
    assert any("p≈0" in t for t in texts)


def test_plot_conversion_waterfall_writes_files(tmp_path):
    clusters = pd.DataFrame({"any_p1": [0, 1, 0, 1], "any_p2": [1, 0, 1, 1]})
    fplot.plot_conversion_waterfall(
        str(tmp_path),
        clusters_df=clusters,
        palette=DEFAULT_COLORS,
    )
    out_base = Path(tmp_path) / "figures" / "any_conversion_waterfall"
    assert out_base.with_suffix(".png").exists()
    assert out_base.with_suffix(".pdf").exists()


def test_plot_headroom_scatter_draws_and_saves(tmp_path):
    step_df = pd.DataFrame(
        {"metric": ["cluster_any", "cluster_any"], "acc_pass1": [0.2, 0.6], "delta_pp": [5.0, -2.0], "step": [1, 2]}
    )
    fplot.plot_headroom_scatter(str(tmp_path), step_df)
    out_base = Path(tmp_path) / "figures" / "any_headroom_scatter"
    assert out_base.with_suffix(".png").exists()
    assert out_base.with_suffix(".pdf").exists()


def test_uncertainty_bucket_computation_and_plot(tmp_path):
    pairs_df = pd.DataFrame(
        {
            "entropy_p1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "correct1": [0, 1, 0, 1, 1],
            "correct2": [1, 1, 0, 1, 1],
            "step": [1, 1, 2, 2, 3],
        }
    )
    clusters_df = pd.DataFrame(
        {
            "entropy_p1_cluster": [0.1, 0.2, 0.3, 0.4, 0.5],
            "acc_p1": [0.2, 0.4, 0.6, 0.5, 0.3],
            "acc_p2": [0.3, 0.5, 0.7, 0.55, 0.4],
            "any_p1": [0, 1, 0, 1, 1],
            "any_p2": [1, 1, 0, 1, 1],
            "step": [1, 1, 2, 2, 3],
        }
    )
    data = fplot._compute_uncertainty_bucket_data(pairs_df, clusters_df, n_boot=10, seed=0)
    assert data is not None
    assert len(data["x"]) == 5
    config = fplot.SeriesPlotConfig(out_dir=str(tmp_path), n_boot=10, seed=0)
    fplot.plot_uncertainty_buckets(pairs_df, clusters_df, config)
    out_base = Path(tmp_path) / "figures" / "uncertainty_buckets"
    assert out_base.with_suffix(".png").exists()


def test_compute_uncertainty_bucket_data_handles_empty():
    pairs_df = pd.DataFrame(columns=["entropy_p1"])
    clusters_df = pd.DataFrame(columns=["entropy_p1_cluster"])
    assert fplot._compute_uncertainty_bucket_data(pairs_df, clusters_df, n_boot=5, seed=1) is None


def test_bootstrap_bucket_groups_handles_empty_group():
    empty_group = pd.DataFrame()
    series = fplot._bootstrap_bucket_groups([empty_group], lambda df: 1.0, n_boot=2, seed=0)
    assert np.isnan(series.delta[0])
    assert np.isnan(series.lower[0]) and np.isnan(series.upper[0])


def test_stepwise_series_and_overlay(tmp_path):
    pairs_df = pd.DataFrame(
        {
            "entropy_p1": [0.1, 0.2],
            "correct1": [0, 1],
            "correct2": [1, 1],
            "step": [1, 2],
        }
    )
    clusters_df = pd.DataFrame(
        {
            "entropy_p1_cluster": [0.1, 0.2],
            "acc_p1": [0.5, 0.6],
            "acc_p2": [0.6, 0.8],
            "any_p1": [0, 1],
            "any_p2": [1, 1],
            "step": [1, 2],
        }
    )
    step_df = pd.DataFrame(
        {
            "metric": ["sample", "cluster_mean", "cluster_any"],
            "step": [1, 1, 1],
            "delta_pp": [10.0, 5.0, 7.0],
        }
    )
    ci = fplot._compute_stepwise_ci_by_metric(
        pairs_df, clusters_df, fplot.SeriesPlotConfig(out_dir="", n_boot=5, seed=1)
    )
    assert set(ci.keys()) == {"sample", "cluster_any", "cluster_mean"}
    series = fplot._build_stepwise_series(step_df, ci)
    assert set(series.keys()) == {"sample", "cluster_mean", "cluster_any"}
    empty_series = fplot._build_stepwise_series(
        pd.DataFrame(columns=["metric", "step", "delta_pp"]), {"sample": {}, "cluster_mean": {}, "cluster_any": {}}
    )
    assert empty_series["sample"].lower.size == 0
    config = fplot.SeriesPlotConfig(out_dir=str(tmp_path), n_boot=5, seed=1)
    fplot.plot_stepwise_overlay(step_df, pairs_df, clusters_df, config)
    out_base = Path(tmp_path) / "figures" / "stepwise_overlay"
    assert out_base.with_suffix(".png").exists()


def test_uncertainty_and_stepwise_combined(tmp_path):
    pairs_df = pd.DataFrame(
        {
            "entropy_p1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "correct1": [0, 1, 0, 1, 1],
            "correct2": [1, 1, 0, 1, 1],
            "step": [1, 1, 2, 2, 3],
        }
    )
    clusters_df = pd.DataFrame(
        {
            "entropy_p1_cluster": [0.1, 0.2, 0.3, 0.4, 0.5],
            "acc_p1": [0.2, 0.3, 0.4, 0.5, 0.6],
            "acc_p2": [0.3, 0.4, 0.5, 0.55, 0.65],
            "any_p1": [0, 1, 0, 1, 1],
            "any_p2": [1, 1, 0, 1, 1],
            "step": [1, 1, 2, 2, 3],
        }
    )
    step_df = pd.DataFrame(
        {"metric": ["sample", "cluster_mean", "cluster_any"], "step": [1, 1, 1], "delta_pp": [5.0, 4.0, 3.0]}
    )
    config = fplot.SeriesPlotConfig(out_dir=str(tmp_path), n_boot=5, seed=2)
    fplot.plot_uncertainty_and_stepwise(step_df, pairs_df, clusters_df, config)
    out_base = Path(tmp_path) / "figures" / "uncertainty_and_stepwise"
    assert out_base.with_suffix(".png").exists()


def test_uncertainty_and_stepwise_skips_when_no_entropy(monkeypatch, capsys):
    monkeypatch.setattr(fplot, "_compute_uncertainty_bucket_data", lambda *a, **k: None)
    fplot.plot_uncertainty_and_stepwise(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), fplot.SeriesPlotConfig(out_dir="", n_boot=1, seed=1)
    )
    out = capsys.readouterr().out
    assert "skipping combined" in out


def test_uncertainty_buckets_warn_when_empty(monkeypatch, capsys):
    monkeypatch.setattr(fplot, "_compute_uncertainty_bucket_data", lambda *a, **k: None)
    fplot.plot_uncertainty_buckets(
        pd.DataFrame(),
        pd.DataFrame(),
        fplot.SeriesPlotConfig(out_dir="", n_boot=1, seed=1),
    )
    out = capsys.readouterr().out
    assert "skipping uncertainty_buckets" in out


def test_overview_bar_data_and_waterfall(monkeypatch):
    monkeypatch.setattr(
        fplot,
        "_bar_specs_with_stats",
        lambda pairs_df, clusters_df: [
            ("sample", "Per-draw", {"delta_pp": 1.0}),
            ("cluster_mean", "Mean", {"delta_pp": 2.0}),
        ],
    )
    monkeypatch.setattr(fplot, "_bootstrap_delta", lambda *_args, **_kwargs: (0.0, 2.0))
    style = fplot.SeriesPlotConfig(out_dir="", n_boot=5, seed=1)
    data = fplot._build_overview_bar_data(pd.DataFrame(), pd.DataFrame(), style)
    assert data["labels"] == ["Per-draw", "Mean"]
    clusters = pd.DataFrame({"any_p1": [0, 1, 1], "any_p2": [1, 0, 1]})
    waterfall = fplot._compute_waterfall_data(clusters)
    assert sum(waterfall["values"]) == 3


def test_bar_specs_with_stats_pairs_empty(monkeypatch):
    monkeypatch.setattr(fplot, "summarize_cluster_mean", lambda df: {"delta_pp": 1})
    monkeypatch.setattr(fplot, "summarize_cluster_any", lambda df: {"delta_pp": 2})
    specs = fplot._bar_specs_with_stats(pd.DataFrame(), pd.DataFrame({"x": [1]}))
    assert specs[0][0] == "cluster_mean"
    assert all(spec[0] != "sample" for spec in specs)


def test_bar_specs_with_stats_includes_sample(monkeypatch):
    monkeypatch.setattr(fplot, "summarize_cluster_mean", lambda df: {"delta_pp": 2})
    monkeypatch.setattr(fplot, "summarize_cluster_any", lambda df: {"delta_pp": 3})
    monkeypatch.setattr(fplot, "summarize_sample_level", lambda df: {"delta_pp": 1})
    pairs_df = pd.DataFrame({"any": [1]})
    clusters_df = pd.DataFrame({"any": [1]})
    specs = fplot._bar_specs_with_stats(pairs_df, clusters_df)
    assert specs[0][0] == "sample"
    assert len(specs) == 3


def test_build_overview_bar_data_handles_empty_specs(monkeypatch):
    monkeypatch.setattr(fplot, "_bar_specs_with_stats", lambda pairs_df, clusters_df: [])
    data = fplot._build_overview_bar_data(
        pd.DataFrame(), pd.DataFrame(), fplot.SeriesPlotConfig(out_dir="", n_boot=1, seed=1)
    )
    assert data["ci_lowers"] == [] and data["ci_uppers"] == []


def test_plot_overview_side_by_side(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fplot,
        "_build_overview_bar_data",
        lambda **kwargs: {
            "labels": ["A", "B"],
            "heights": [1.0, -1.0],
            "ci_lowers": [0.1, 0.2],
            "ci_uppers": [0.3, 0.4],
            "colors": ["#111", "#222"],
        },
    )
    monkeypatch.setattr(
        fplot,
        "_compute_waterfall_data",
        lambda clusters_df: {
            "values": [1, 2],
            "labels": ["X", "Y"],
            "colors": ["#333", "#444"],
            "percentages": [33.3, 66.7],
            "total": 3,
        },
    )

    def fake_save(fig, out_base):
        path = Path(out_base + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    monkeypatch.setattr(fplot, "savefig", fake_save)
    config = {
        "out_dir": str(tmp_path),
        "n_boot": 5,
        "seed": 1,
        "pairs_df": pd.DataFrame(),
        "clusters_df": pd.DataFrame(),
    }
    fplot.plot_overview_side_by_side(config)
    assert (Path(tmp_path) / "figures" / "overview_deltas_and_waterfall.png").exists()


def test_bootstrap_delta_and_by_step():
    pairs = pd.DataFrame(
        {
            "correct1": [0, 1],
            "correct2": [1, 1],
            "any_p1": [0, 1],
            "any_p2": [1, 1],
            "acc_p1": [0.2, 0.3],
            "acc_p2": [0.4, 0.5],
        }
    )
    low, high = fplot._bootstrap_delta(pairs, "sample", n_boot=20, seed=0)
    assert low <= high
    step_map = {1: pairs}
    ci_map = fplot._bootstrap_delta_by_step(step_map, "sample", n_boot=5, seed=0)
    assert 1 in ci_map
    nan_low, nan_high = fplot._bootstrap_delta(pd.DataFrame(), "missing", n_boot=1, seed=0)
    assert np.isnan(nan_low) and np.isnan(nan_high)
    ci_empty = fplot._bootstrap_delta_by_step({5: pd.DataFrame()}, "sample", n_boot=1, seed=0)
    assert np.isnan(ci_empty[5][0]) and np.isnan(ci_empty[5][1])
