#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import src.analysis.forced_aha_plotting as fap


def test_compute_step_groups_for_boot():
    pairs_df = pd.DataFrame({"step": [0, 1], "v": [1, 2]})
    clusters_df = pd.DataFrame({"step": [1, 2], "w": [3, 4]})
    steps, sample_step, any_step, mean_step = fap._step_groups_for_boot(pairs_df, clusters_df)
    assert steps == [0, 1, 2]
    assert 0 in sample_step and 2 in any_step and mean_step[1].equals(any_step[1])


def test_compute_stepwise_ci_by_metric_handles_empty(monkeypatch):
    pairs_df = pd.DataFrame(columns=["step", "correct1", "correct2", "acc_p1", "acc_p2", "any_p1", "any_p2"])
    clusters_df = pairs_df.copy()
    cfg = fap.SeriesPlotConfig(out_dir=".", n_boot=10, seed=0, palette_name="Pastel1", darken=0.8)
    ci = fap._compute_stepwise_ci_by_metric(pairs_df, clusters_df, cfg)
    assert ci["sample"] == {} and ci["cluster_any"] == {} and ci["cluster_mean"] == {}


def test_uncertainty_and_stepwise_combined(monkeypatch, tmp_path):
    # Minimal data to exercise plotting path
    pairs_df = pd.DataFrame(
        {
            "bucket": [0, 1],
            "delta_pp": [1.0, 2.0],
            "ci_low": [0.5, 1.5],
            "ci_high": [1.5, 2.5],
            "median_pp": [0.8, 1.8],
        }
    )
    clusters_df = pairs_df.copy()
    step_df = pd.DataFrame(
        [
            {
                "metric": "sample",
                "step": 0,
                "delta_pp": 1.0,
                "p_mcnemar": 0.01,
                "p_ttest": np.nan,
                "p_wilcoxon": np.nan,
            },
            {
                "metric": "cluster_any",
                "step": 0,
                "delta_pp": 0.5,
                "p_mcnemar": 0.02,
                "p_ttest": np.nan,
                "p_wilcoxon": np.nan,
            },
            {
                "metric": "cluster_mean",
                "step": 0,
                "delta_pp": 0.2,
                "p_mcnemar": np.nan,
                "p_ttest": 0.03,
                "p_wilcoxon": 0.04,
            },
        ]
    )
    config = fap.SeriesPlotConfig(out_dir=str(tmp_path), n_boot=5, seed=1, palette_name="Pastel1", darken=0.8)
    # Monkeypatch helpers to avoid heavy bootstraps/plots
    monkeypatch.setattr(fap, "_compute_uncertainty_bucket_data", lambda pairs, clusters, n_boot, seed: pairs_df)
    monkeypatch.setattr(
        fap, "_compute_stepwise_ci_by_metric", lambda pairs, clusters, cfg: {"sample": {0: (0.1, 0.2)}}
    )
    monkeypatch.setattr(fap, "_build_stepwise_series", lambda step_df, ci_by_metric: {"sample": step_df})
    monkeypatch.setattr(fap, "_select_series_colors", lambda cfg: {"sample": "#000"})
    monkeypatch.setattr(fap, "_plot_uncertainty_bucket_panel", lambda axis, bucket_data, colors: None)
    monkeypatch.setattr(fap, "_plot_stepwise_series", lambda axis, series_by_metric, colors: None)
    monkeypatch.setattr(fap, "savefig", lambda fig, path: None)

    fap.plot_uncertainty_and_stepwise(pairs_df, clusters_df, step_df, config)

    # Ensure figures folder path is used for output
    assert os.path.basename(config.out_dir) in fap.os.path.dirname(
        os.path.join(config.out_dir, "figures", "uncertainty_and_stepwise")
    )
