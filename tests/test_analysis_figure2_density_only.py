import os

import matplotlib


matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.analysis.figure_2_density import (
    DensityPlotConfig,
    FourHistConfig,
    _correct_incorrect_density_for_mask,
    _density_curves_for_correct,
    plot_correct_incorrect_by_type,
    plot_four_correct_hists,
)


def _sample_frame(include_formal: bool = True) -> pd.DataFrame:
    data = {
        "uncertainty": [0.1, 0.2, 0.8, 0.5],
        "correct": [1, 0, 1, 0],
        "aha_words": [1, 1, 0, 1],
        "aha_gpt": [0, 1, 1, 0],
    }
    if include_formal:
        data["aha_formal"] = [0, 1, 0, 0]
    return pd.DataFrame(data)


def test_density_curves_handle_missing_formal_column():
    d_all = _sample_frame(include_formal=False)
    edges = np.linspace(-1, 1, 6)
    curves = _density_curves_for_correct(d_all, edges, smooth_bins=3)
    assert len(curves) == 4
    formal_curve = next(c for c in curves if "Formal" in c["label"])
    assert formal_curve["n"] == 0  # no formal column → zero counts


def test_correct_incorrect_density_empty_mask():
    d_all = _sample_frame()
    edges = np.linspace(-1, 1, 4)
    mask = np.zeros(len(d_all), dtype=bool)
    density = _correct_incorrect_density_for_mask(d_all, edges, mask, smooth_bins=2)
    assert density["n_correct"] == 0 and density["n_incorrect"] == 0
    assert np.all(density["y_correct"] == 0) and np.all(density["y_incorrect"] == 0)


def test_plot_four_correct_hists_and_correct_incorrect_outputs(tmp_path):
    d_all = _sample_frame()
    edges = np.linspace(-1, 1, 5)

    hist_cfg = FourHistConfig(
        out_png=str(tmp_path / "four.png"),
        out_pdf=str(tmp_path / "four.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        edges=edges,
    )
    plot_four_correct_hists(d_all, hist_cfg)
    assert os.path.exists(hist_cfg.out_png)
    assert os.path.exists(hist_cfg.out_pdf)

    density_cfg = DensityPlotConfig(
        out_png=str(tmp_path / "corr_inc.png"),
        out_pdf=str(tmp_path / "corr_inc.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        smooth_bins=2,
    )
    out_csv = plot_correct_incorrect_by_type(d_all, edges, density_cfg)
    assert os.path.exists(out_csv)
    csv_data = pd.read_csv(out_csv)
    assert set(csv_data["panel"]) == {
        "All samples",
        'Words of "Aha!"',
        'LLM-Detected "Aha!"',
        'Formal "Aha!"',
    }
