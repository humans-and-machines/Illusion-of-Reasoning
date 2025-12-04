import argparse

import numpy as np
import pandas as pd

import src.analysis.figure_2_helpers as f2h


def test_build_arg_parser_has_expected_options():
    parser = f2h.build_arg_parser()
    args = parser.parse_args(
        [
            "root",
            "--unc_field",
            "think",
            "--gpt_mode",
            "canonical",
            "--hist_bins",
            "5",
            "--density_bins",
            "6",
            "--acc_bins",
            "7",
            "--ppx_buckets",
            "9",
            "--smooth_bins",
            "3",
            "--xlim_std",
            "-1.0",
            "2.0",
            "--B_ci",
            "10",
            "--rq2_dir",
            "rq2",
        ],
    )
    assert args.results_root == "root"
    assert args.hist_bins == 5
    assert args.density_bins == 6
    assert args.acc_bins == 7
    assert args.ppx_buckets == 9
    assert args.B_ci == 10
    assert args.xlim_std == [-1.0, 2.0]
    assert args.rq2_dir == "rq2"


def test_run_uncertainty_figures_invokes_all_components(monkeypatch, tmp_path):
    calls = {}

    # Avoid font/dirs work
    monkeypatch.setattr(f2h, "set_global_fonts", lambda *a, **k: calls.setdefault("fonts", True))
    monkeypatch.setattr(f2h.os, "makedirs", lambda path, exist_ok=True: calls.setdefault("mkdir", path))

    # Dataframe plumbing
    sample_df = pd.DataFrame(
        {
            "uncertainty": [0.1, 0.2, 0.3, 0.4],
            "correct": [1, 0, 1, 0],
            "problem": ["p1", "p2", "p3", "p4"],
            "step": [1, 1, 2, 2],
            "aha_gpt": [1, 0, 1, 0],
            "aha_words": [0, 1, 0, 1],
        },
    )
    monkeypatch.setattr(f2h, "_load_all_samples", lambda args: sample_df.copy())
    monkeypatch.setattr(f2h, "_standardize_uncertainty", lambda df: df.assign(uncertainty_std=df["uncertainty"]))
    monkeypatch.setattr(f2h, "make_edges_from_std", lambda vals, bins, xlim=None: np.array([0.0, 0.5, 1.0]))

    def _stub_plot(name):
        def _fn(*args, **kwargs):
            calls.setdefault(name, 0)
            calls[name] += 1
            return "csv_path" if "density" in name or "accuracy" in name or "regression" in name else None

        return _fn

    monkeypatch.setattr(f2h, "plot_four_correct_hists", _stub_plot("four"))
    monkeypatch.setattr(f2h, "plot_overlaid_densities", _stub_plot("density"))
    monkeypatch.setattr(f2h, "plot_correct_incorrect_by_type", _stub_plot("correct_incorrect"))
    monkeypatch.setattr(f2h, "plot_accuracy_by_bin_overlay", _stub_plot("accuracy"))
    monkeypatch.setattr(f2h, "plot_regression_curves", _stub_plot("regression"))

    args = argparse.Namespace(
        results_root=str(tmp_path),
        out_dir=str(tmp_path),
        dataset_name="DS",
        model_name="Model",
        unc_field="entropy",
        gpt_mode="canonical",
        gpt_gate_by_words=False,
        delta1=0.1,
        delta2=0.2,
        delta3=None,
        min_prior_steps=1,
        hist_bins=2,
        density_bins=2,
        acc_bins=2,
        ppx_buckets=3,
        smooth_bins=0,
        xlim_std=None,
        font_family="Serif",
        font_size=11,
        a4_pdf=False,
        a4_orientation="landscape",
        B_ci=5,
    )

    f2h.run_uncertainty_figures(args)

    assert calls.get("fonts")
    assert calls.get("mkdir") == str(tmp_path)
    assert calls.get("four") == 1
    assert calls.get("density") == 1
    assert calls.get("correct_incorrect") == 1
    assert calls.get("accuracy") == 1
    assert calls.get("regression") == 1
