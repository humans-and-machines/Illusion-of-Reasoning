from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.figure_2_accuracy import (
    AccuracyPlotConfig,
    AccuracySeriesSpec,
    _build_accuracy_rows,
    _overlay_accuracy_series,
    per_bin_accuracy,
    plot_accuracy_by_bin_overlay,
    plt,
)
from src.analysis.figure_2_data import wilson_ci


def test_build_accuracy_rows_sets_bin_edges_and_types():
    rows = _build_accuracy_rows(
        label="All",
        centers=np.array([0.0, 1.0]),
        accuracy=np.array([0.25, 0.5]),
        lower_bounds=np.array([0.1, 0.2]),
        upper_bounds=np.array([0.4, 0.7]),
        num_total=np.array([4, 2]),
        num_correct=np.array([1, 1]),
        half_width=0.5,
    )
    assert len(rows) == 2
    first = rows[0]
    assert first["variant"] == "All"
    assert first["bin_left"] == pytest.approx(-0.5)
    assert first["bin_right"] == pytest.approx(0.5)
    assert first["n"] == 4 and first["k"] == 1
    assert first["acc"] == pytest.approx(0.25)
    assert first["lo"] == pytest.approx(0.1)
    assert first["hi"] == pytest.approx(0.4)


def test_per_bin_accuracy_computes_counts_and_ci():
    values = np.array([0.1, 0.2, 1.2, 1.3])
    correct = np.array([1, 0, 1, 0])
    edges = np.array([0.0, 1.0, 2.0])
    centers, acc, lower, upper, k, n = per_bin_accuracy(values, correct, edges)
    assert np.allclose(centers, np.array([0.5, 1.5]))
    assert k.tolist() == [1, 1]
    assert n.tolist() == [2, 2]
    assert acc.tolist() == [0.5, 0.5]
    expected_ci = wilson_ci(1, 2)
    assert lower[0] == pytest.approx(expected_ci[0])
    assert upper[0] == pytest.approx(expected_ci[1])


def test_overlay_accuracy_series_plots_and_returns_rows():
    samples = pd.DataFrame(
        {
            "uncertainty_std": [0.1, 0.8, 1.1],
            "correct": [1, 0, 1],
        }
    )
    edges = np.array([0.0, 1.0, 2.0])
    fig, ax = plt.subplots()
    spec = AccuracySeriesSpec(label="Masked", mask=np.array([True, True, False]), color="#000000")
    rows = _overlay_accuracy_series(samples=samples, edges=edges, spec=spec, axis=ax, half_width=0.5)
    assert len(rows) == 2
    assert ax.lines[0].get_label() == "Masked"
    assert rows[0]["variant"] == "Masked"
    plt.close(fig)


def test_plot_accuracy_by_bin_overlay_writes_outputs(tmp_path):
    all_samples = pd.DataFrame(
        {
            "uncertainty_std": [0.1, 0.9, 1.1, 1.8],
            "correct": [1, 0, 1, 0],
            "aha_words": [1, 0, 1, 0],
            "aha_gpt": [0, 1, 1, 0],
            "aha_formal": [0, 0, 1, 0],
        }
    )
    edges = np.array([0.0, 1.0, 2.0])
    config = AccuracyPlotConfig(
        out_png=str(tmp_path / "acc.png"),
        out_pdf=str(tmp_path / "acc.pdf"),
        title_suffix="Demo",
        a4_pdf=False,
        a4_orientation="landscape",
    )
    csv_path = plot_accuracy_by_bin_overlay(all_samples, edges, config)
    csv_rows = pd.read_csv(csv_path)
    assert set(csv_rows["variant"]) == {"All", "Words", "LLM", "Formal"}
    assert len(csv_rows) == (len(edges) - 1) * 4  # rows per series per bin
    assert Path(config.out_png).exists()
    assert Path(config.out_pdf).exists()
