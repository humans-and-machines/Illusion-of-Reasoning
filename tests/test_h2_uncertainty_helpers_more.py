import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.core.h2_uncertainty_helpers as h2


def test_aha_gpt_eff_and_build_bucket_row(monkeypatch):
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda p1, rec: 1)
    monkeypatch.setattr(h2, "aha_words", lambda p1: 0)
    rec = {"pass1": {"is_correct_pred": True, "entropy": 0.5}, "step": 3, "problem": "p"}
    gpt_eff, words = h2._aha_gpt_eff(rec["pass1"], rec, gate_by_words=True)
    assert gpt_eff == 0 and words == 0

    monkeypatch.setattr(h2, "resolve_problem_identifier", lambda rec: "probX")
    monkeypatch.setattr(h2, "choose_uncertainty", lambda p1, f: 0.9)
    row = h2._build_bucket_row(5, rec, "entropy", gate_gpt_by_words=False)
    assert row["problem"] == "probX"
    assert row["step"] == 3
    assert row["correct"] == 1

    assert h2._build_bucket_row(None, {"pass1": {}}, "entropy", False) is None
    assert h2._build_bucket_row(None, {"pass1": {"is_correct_pred": None}}, "entropy", False) is None


def test_load_all_defs_for_buckets(monkeypatch):
    records = [
        {"pass1": {"is_correct_pred": True, "entropy": 0.1}, "step": 1},
        {"pass1": {"is_correct_pred": False, "entropy": 0.2}, "step": 2},
    ]
    monkeypatch.setattr(h2, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(h2, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(h2, "resolve_problem_identifier", lambda rec: rec.get("problem", "p"))
    monkeypatch.setattr(h2, "choose_uncertainty", lambda p1, f: p1.get("entropy"))
    monkeypatch.setattr(h2, "aha_words", lambda p1: 0)
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda p1, rec: 1)
    df = h2.load_all_defs_for_buckets(["f1.jsonl"], "entropy", gate_gpt_by_words=False)
    assert len(df) == 2

    # Empty results raise
    monkeypatch.setattr(h2, "iter_records_from_file", lambda path: [])
    with pytest.raises(RuntimeError):
        h2.load_all_defs_for_buckets(["f1.jsonl"], "entropy", gate_gpt_by_words=False)


def test_bucket_helpers_and_aggregate():
    samples = pd.DataFrame(
        {
            "problem": ["p1", "p2", "p3"],
            "step": [1, 1, 1],
            "uncertainty": [0.1, 0.2, 0.3],
            "aha_gpt": [0, 1, 1],
            "aha_words": [1, 0, 1],
            "aha_formal": [1, 0, 1],
        },
    )
    bucketed = h2._make_uncertainty_buckets(samples, n_buckets=3)
    assert "bucket_id" in bucketed.columns

    agg = h2._aggregate_buckets(bucketed, "aha_formal")
    assert set(agg.columns) == {"bucket_id", "bucket_label", "n", "k_aha", "aha_ratio", "lo", "hi"}
    assert len(agg) >= 1

    table = h2._build_bucket_series_table(bucketed, bucketed, bucketed)
    assert set(table["series"]) == {"words", "gpt", "formal"}


def test_compute_uncertainty_hist_counts(monkeypatch):
    standardized = pd.DataFrame(
        {
            "problem": ["p1", "p2"],
            "step": [1, 2],
            "uncertainty_std": [0.0, 1.0],
            "aha_words": [1, 0],
            "aha_gpt": [1, 0],
        },
    )
    ps_df = pd.DataFrame(
        {
            "problem": ["p1"],
            "step": [1],
            "aha_formal_pair": [1],
        },
    )
    counts = h2._compute_uncertainty_hist_counts(standardized.copy(), ps_df, num_bins=2)
    assert counts.n_total.tolist() == [1, 1]
    assert counts.n_formal.sum() == 1

    # when aha_formal already present, skip recompute
    standardized["aha_formal"] = [1, 0]
    counts2 = h2._compute_uncertainty_hist_counts(standardized, ps_df, num_bins=2)
    assert counts2.n_formal.sum() == 1


def test_write_uncertainty_hist_csv_and_plot(tmp_path):
    counts = h2.UncertaintyHistCounts(
        edges=np.array([0.0, 0.5, 1.0]),
        centers=np.array([0.25, 0.75]),
        n_total=np.array([1, 2]),
        n_words=np.array([0, 1]),
        n_gpt=np.array([1, 0]),
        n_formal=np.array([1, 1]),
    )
    out_csv = tmp_path / "hist.csv"
    h2._write_uncertainty_hist_csv(str(out_csv), counts)
    assert out_csv.exists()

    out_dir = tmp_path
    slug = "slug"
    out_png = h2._plot_uncertainty_hist_figure(
        str(out_dir),
        slug,
        counts,
        num_bins=2,
        title_suffix="title",
    )
    assert Path(out_png).exists()
    assert Path(out_png.replace(".png", ".pdf")).exists()


def test_plot_uncertainty_hist_100bins(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(h2, "standardize_uncertainty", lambda df: df.assign(uncertainty_std=df["uncertainty"]))
    monkeypatch.setattr(h2, "_write_uncertainty_hist_csv", lambda path, counts: called.setdefault("csv", path))
    monkeypatch.setattr(
        h2,
        "_plot_uncertainty_hist_figure",
        lambda out_dir, slug, counts, num_bins, title_suffix: str(tmp_path / "hist.png"),
    )
    d_all = pd.DataFrame(
        {"uncertainty": [0.1, 0.2], "aha_gpt": [1, 0], "aha_words": [0, 1], "problem": ["p1", "p2"], "step": [1, 2]}
    )
    ps_df = pd.DataFrame({"problem": ["p1"], "step": [1], "aha_formal_pair": [1]})
    args = argparse.Namespace(hist_bins=5, dataset_name="DS", model_name="Model")
    out_png, out_csv = h2.plot_uncertainty_hist_100bins(d_all, ps_df, str(tmp_path), args)
    assert out_png.endswith("hist.png")
    assert out_csv.endswith("h2_uncertainty_hist_100bins.csv")
    assert "csv" in called


def test_make_all3_uncertainty_buckets_to_csv_errors(monkeypatch, tmp_path):
    records = [
        {"pass1": {"is_correct_pred": 1, "entropy": 0.1, "shift_in_reasoning_v1": 1}, "step": 1, "problem": "p"},
    ]
    monkeypatch.setattr(h2, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(h2, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(h2, "resolve_problem_identifier", lambda rec: rec.get("problem", "p"))
    monkeypatch.setattr(h2, "choose_uncertainty", lambda p1, f: p1.get("entropy"))
    monkeypatch.setattr(h2, "aha_words", lambda p1: 1)
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda p1, rec: 1)
    args = argparse.Namespace(
        unc_field="entropy",
        gpt_gate_by_words=False,
        delta1=0.1,
        delta2=0.1,
        delta3=None,
        min_prior_steps=1,
        unc_buckets=3,
        dataset_name="DS",
        model_name="Model",
    )
    # Force to_csv to raise for intermediate writes only to hit except branches
    orig_to_csv = pd.DataFrame.to_csv

    def selective_to_csv(self, path, *a, **k):
        if "h2_all3" in str(path):
            raise OSError("nope")
        return orig_to_csv(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", selective_to_csv)
    out_png, out_csv, d_all, ps_df = h2.make_all3_uncertainty_buckets_figure(
        files=[str(tmp_path / "f.jsonl")],
        out_dir=str(tmp_path),
        args=args,
    )
    assert out_png.endswith(".png") and out_csv.endswith(".csv")
    assert not d_all.empty and not ps_df.empty


def test_make_all3_uncertainty_buckets_figure(monkeypatch, tmp_path):
    # Minimal sample and problem-step data
    records = [
        {"pass1": {"is_correct_pred": True, "entropy": 0.1}, "step": 1, "problem": "p1"},
    ]
    monkeypatch.setattr(h2, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(h2, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(h2, "resolve_problem_identifier", lambda rec: rec["problem"])
    monkeypatch.setattr(h2, "choose_uncertainty", lambda p1, f: p1["entropy"])
    monkeypatch.setattr(h2, "aha_words", lambda p1: 1)
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda p1, rec: 1)
    args = argparse.Namespace(
        unc_field="entropy",
        gpt_gate_by_words=False,
        delta1=0.1,
        delta2=0.1,
        delta3=None,
        min_prior_steps=1,
        unc_buckets=3,
        dataset_name="DS",
        model_name="Model",
    )
    out_png, out_csv, d_all, ps_df = h2.make_all3_uncertainty_buckets_figure(
        files=["f1.jsonl"],
        out_dir=str(tmp_path),
        args=args,
    )
    assert Path(out_png).exists()
    assert Path(out_csv).exists()
    assert not d_all.empty
    assert not ps_df.empty
