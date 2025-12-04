import os

import numpy as np
import pandas as pd

from src.analysis.figure_2_accuracy import (
    AccuracyPlotConfig,
    AccuracySeriesSpec,
    _overlay_accuracy_series,
    per_bin_accuracy,
    plot_accuracy_by_bin_overlay,
)
from src.analysis.figure_2_data import (
    _any_keys_true,
    _build_pass1_row,
    compute_correct_hist,
    density_from_hist,
    load_pass1_rows,
    make_edges_from_std,
    wilson_ci,
)
from src.analysis.figure_2_density import (
    DensityPlotConfig,
    FourHistConfig,
    _correct_incorrect_density_for_mask,
    _density_curves_for_correct,
    plot_correct_incorrect_by_type,
    plot_four_correct_hists,
    plot_overlaid_densities,
)


def test_build_pass1_row_and_loading(monkeypatch):
    base_rec = {
        "problem": "p1",
        "step": 1,
        "pass1": {
            "is_correct_pred": 1,
            "entropy": 0.5,
            "has_reconsider_cue": 1,
            "shift_llm": 1,
        },
    }
    row = _build_pass1_row(base_rec, None, "answer", ["shift_llm"], gate_gpt_by_words=True)
    assert row and row["aha_gpt"] == 1 and row["correct"] == 1

    missing_unc = _build_pass1_row({"pass1": {"is_correct_pred": 1}}, None, "answer", [], False)
    assert missing_unc is None

    monkeypatch.setattr(
        "src.analysis.figure_2_data.iter_pass1_records",
        lambda files: [(files[0], 1, base_rec)],
    )
    df = load_pass1_rows(["dummy"], "answer", ["shift_llm"], gate_gpt_by_words=False)
    assert len(df) == 1 and set(df.columns) >= {"uncertainty", "aha_words", "aha_gpt"}


def test_stat_helpers_and_edges():
    low, high = wilson_ci(0, 0)
    assert np.isnan(low) and np.isnan(high)

    vals = np.array([0.0, 1.0, 2.0, 3.0])
    edges = make_edges_from_std(vals, bins=3, xlim=(-1, 1))
    assert edges[0] == -1 and edges[-1] == 1

    centers, density = density_from_hist(np.array([]), np.array([0.0, 1.0]), smooth_k=4)
    assert np.all(density == 0) and np.all(np.isfinite(centers))

    hist = compute_correct_hist(np.array([0.1, 0.2, 0.9]), np.array([1, 0, 1]), np.array([0.0, 0.5, 1.0]))
    assert hist.tolist() == [1, 1]


def test_any_keys_true_gates():
    pass1 = {"a": True}
    record = {"b": 1}
    assert _any_keys_true(pass1, record, ["a", "b"]) == 1
    assert _any_keys_true({}, {}, ["missing"]) == 0


def test_accuracy_overlay_and_plotting(tmp_path):
    samples = pd.DataFrame(
        {
            "uncertainty_std": [-1.0, 0.0, 1.0],
            "correct": [1, 0, 1],
            "aha_words": [1, 0, 1],
            "aha_gpt": [0, 1, 0],
            "aha_formal": [1, 0, 0],
        }
    )
    edges = np.array([-1.5, 0.0, 1.5])
    centers, acc, lo, hi, k, n = per_bin_accuracy(
        samples["uncertainty_std"].to_numpy(), samples["correct"].to_numpy(), edges
    )
    assert centers.size == 2 and np.isfinite(acc).any()
    spec = AccuracySeriesSpec(label="All", mask=np.ones(len(samples), dtype=bool), color="#000000")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    axis = fig.add_subplot(111)
    rows = _overlay_accuracy_series(
        samples=samples,
        edges=edges,
        spec=spec,
        axis=axis,
        half_width=0.75,
    )
    assert len(rows) == 2 and all("bin_left" in r for r in rows)

    cfg = AccuracyPlotConfig(
        out_png=str(tmp_path / "acc.png"),
        out_pdf=str(tmp_path / "acc.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
    )
    out_csv = plot_accuracy_by_bin_overlay(samples, edges, cfg)
    assert os.path.exists(out_csv)
    csv_rows = pd.read_csv(out_csv)
    assert set(csv_rows["variant"]) == {"All", "Words", "LLM", "Formal"}


def test_density_curves_and_plots(tmp_path):
    d_all = pd.DataFrame(
        {
            "uncertainty": [0.1, 0.2, 0.3, 0.4],
            "correct": [1, 0, 1, 0],
            "aha_words": [1, 1, 0, 1],
            "aha_gpt": [0, 1, 1, 0],
            "aha_formal": [0, 1, 0, 0],
        }
    )
    edges = np.linspace(-1, 1, 5)
    curves = _density_curves_for_correct(d_all, edges, smooth_bins=0)
    assert len(curves) == 4 and all(curve["n"] >= 0 for curve in curves)

    density_cfg = DensityPlotConfig(
        out_png=str(tmp_path / "dens.png"),
        out_pdf=str(tmp_path / "dens.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        smooth_bins=3,
    )
    out_csv = plot_overlaid_densities(d_all, edges, density_cfg)
    assert os.path.exists(out_csv)
    written = pd.read_csv(out_csv)
    assert "bin_center" in written.columns

    hist_cfg = FourHistConfig(
        out_png=str(tmp_path / "hist.png"),
        out_pdf=str(tmp_path / "hist.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        edges=edges,
    )
    plot_four_correct_hists(d_all, hist_cfg)
    assert os.path.exists(hist_cfg.out_png) and os.path.exists(hist_cfg.out_pdf)

    mask = d_all["aha_words"] == 1
    panel_density = _correct_incorrect_density_for_mask(d_all, edges, mask, smooth_bins=3)
    assert panel_density["n_correct"] + panel_density["n_incorrect"] == int(mask.sum())

    density_cfg2 = DensityPlotConfig(
        out_png=str(tmp_path / "corr_inc.png"),
        out_pdf=str(tmp_path / "corr_inc.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        smooth_bins=3,
    )
    out_csv2 = plot_correct_incorrect_by_type(d_all, edges, density_cfg2)
    assert os.path.exists(out_csv2)
    csv_rows = pd.read_csv(out_csv2)
    assert set(csv_rows["panel"]) == {"All samples", 'Words of "Aha!"', 'LLM-Detected "Aha!"', 'Formal "Aha!"'}
