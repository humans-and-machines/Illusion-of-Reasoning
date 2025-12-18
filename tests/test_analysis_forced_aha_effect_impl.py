#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.forced_aha_effect_impl as impl


def test_maybe_int_and_common_fields_and_sample_row(monkeypatch):
    assert impl._maybe_int("5") == 5
    assert impl._maybe_int(None, default=7) == 7

    rec = {"dataset": "d", "model": "m", "row_key": 123, "split": "train"}
    ctx = impl.SampleRowContext(variant="pass1", entropy_field="entropy_answer", step_from_name=42)

    # Force deterministic helpers
    monkeypatch.setattr(impl, "extract_sample_idx", lambda record, pass_obj: pass_obj.get("sample_idx", 0))
    monkeypatch.setattr(impl, "extract_entropy", lambda pass_obj, preferred=None: pass_obj.get(preferred, 0.5))

    row = impl._build_sample_row(rec, {"sample_idx": 3, "entropy_answer": 0.8}, 1, ctx)
    assert row["problem"] == "123"  # fallback to row_key and coerced to str
    assert row["step"] == 42
    assert row["entropy_p1"] == 0.8

    # PASS-2 rows should not include entropy
    ctx2 = impl.SampleRowContext(variant="pass2", entropy_field="entropy", step_from_name=None)
    row2 = impl._build_sample_row(rec, {"sample_idx": 1, "entropy": 0.1}, 0, ctx2)
    assert "entropy_p1" not in row2


def test_load_samples_from_root_filters_and_builds(monkeypatch):
    monkeypatch.setattr(impl, "scan_jsonl_files", lambda root, split: ["path1.jsonl"])
    monkeypatch.setattr(impl, "nat_step_from_path", lambda path: 5)
    monkeypatch.setattr(
        impl,
        "iter_records_from_file",
        lambda path: [
            {"split": "train", "pass1": {"correct": 1, "sample_idx": 7, "entropy_answer": 0.2}},
            {"split": "val", "pass1": {"correct": 1, "sample_idx": 8, "entropy_answer": 0.3}},
        ],
    )
    monkeypatch.setattr(
        impl,
        "pass_with_correctness",
        lambda record, variant, pass2_key=None: (record["pass1"], record["pass1"]["correct"]),
    )
    monkeypatch.setattr(impl, "extract_sample_idx", lambda record, pass_obj: pass_obj["sample_idx"])
    monkeypatch.setattr(impl, "extract_entropy", lambda pass_obj, preferred=None: pass_obj.get(preferred))

    df = impl.load_samples_from_root("root", split_value="train", variant="pass1", entropy_field="entropy_answer")
    assert len(df) == 1
    assert df.iloc[0]["sample_idx"] == 7
    assert df.iloc[0]["entropy_p1"] == 0.2
    assert df.iloc[0]["step"] == 5


def test_load_samples_from_root_skips_missing(monkeypatch):
    monkeypatch.setattr(impl, "scan_jsonl_files", lambda root, split: ["path1.jsonl"])
    monkeypatch.setattr(impl, "nat_step_from_path", lambda path: 1)
    records = [
        {"split": "train", "pass1": {"correct": 1}},
        {"split": "train", "pass1": {"correct": 0, "sample_idx": 2, "entropy_answer": 0.4}},
    ]
    monkeypatch.setattr(impl, "iter_records_from_file", lambda path: records)
    # First record returns None to trigger continue at line ~180
    monkeypatch.setattr(
        impl,
        "pass_with_correctness",
        lambda record, variant, pass2_key=None: None
        if record is records[0]
        else (record["pass1"], record["pass1"]["correct"]),
    )
    monkeypatch.setattr(impl, "extract_sample_idx", lambda record, pass_obj: pass_obj.get("sample_idx", -1))
    monkeypatch.setattr(impl, "extract_entropy", lambda pass_obj, preferred=None: pass_obj.get(preferred))
    df = impl.load_samples_from_root("root", split_value="train", variant="pass1", entropy_field="entropy_answer")
    assert len(df) == 1
    assert df.iloc[0]["sample_idx"] == 2


def test_load_samples_from_single_root_with_both(monkeypatch):
    monkeypatch.setattr(impl, "scan_jsonl_files", lambda root, split: ["r1"])
    monkeypatch.setattr(impl, "nat_step_from_path", lambda path: 9)
    monkeypatch.setattr(
        impl,
        "iter_records_from_file",
        lambda path: [
            {
                "split": "eval",
                "pass1": {"correct": 1, "sample_idx": 1, "entropy_answer": 0.4},
                "alt_pass2": {"correct": 0, "sample_idx": 1},
            },
            {"split": "train", "pass1": {"correct": 1}},  # filtered out
        ],
    )
    monkeypatch.setattr(impl, "first_nonempty", lambda rec, keys: rec.get("pass1"))
    monkeypatch.setattr(impl, "extract_correct_flag", lambda pass_obj: pass_obj.get("correct"))
    monkeypatch.setattr(impl, "extract_sample_idx", lambda record, pass_obj: pass_obj.get("sample_idx", -1))
    monkeypatch.setattr(impl, "extract_entropy", lambda pass_obj, preferred=None: pass_obj.get("entropy_answer", -1))

    df1, df2 = impl.load_samples_from_single_root_with_both(
        "root",
        split_value="eval",
        entropy_field="entropy_answer",
        pass2_key="alt_pass2",
    )
    assert list(df1.columns) == ["dataset", "model", "problem", "step", "split", "sample_idx", "correct", "entropy_p1"]
    assert list(df2.columns) == ["dataset", "model", "problem", "step", "split", "sample_idx", "correct"]
    assert df1.iloc[0]["step"] == 9
    assert df2.iloc[0]["correct"] == 0


def test_load_samples_from_single_root_with_both_fallback_pass2(monkeypatch):
    monkeypatch.setattr(impl, "scan_jsonl_files", lambda root, split: ["r1"])
    monkeypatch.setattr(impl, "nat_step_from_path", lambda path: 3)
    monkeypatch.setattr(
        impl,
        "iter_records_from_file",
        lambda path: [
            {
                "split": "train",
                "pass1": {"correct": 1, "sample_idx": 1, "entropy_answer": 0.2},
                "pass2": {"correct": 0, "sample_idx": 1},
            }
        ],
    )
    monkeypatch.setattr(impl, "first_nonempty", lambda rec, keys: rec.get("pass2") if "pass2" in rec else None)
    monkeypatch.setattr(impl, "extract_correct_flag", lambda pass_obj: pass_obj.get("correct"))
    monkeypatch.setattr(impl, "extract_sample_idx", lambda record, pass_obj: pass_obj.get("sample_idx", -1))
    monkeypatch.setattr(impl, "extract_entropy", lambda pass_obj, preferred=None: pass_obj.get("entropy_answer", -1))

    df1, df2 = impl.load_samples_from_single_root_with_both(
        "root",
        split_value="train",
        entropy_field="entropy_answer",
        pass2_key=None,  # forces fallback branch at line ~230
    )
    assert len(df1) == 1 and len(df2) == 1


def test_filter_by_step_filters_rows():
    frame = pd.DataFrame(
        [
            {"step": 0, "value": 1},
            {"step": 500, "value": 2},
            {"step": 1000, "value": 3},
        ],
    )
    filtered = impl._filter_by_step(frame, min_step=0, max_step=950, label="test")
    assert list(filtered["value"]) == [1, 2]
    # No filtering when column absent
    frame2 = pd.DataFrame([{"other": 1}])
    assert impl._filter_by_step(frame2, 0, 10, label="noop").equals(frame2)


def test_choose_merge_keys_and_pair_samples(monkeypatch):
    left = pd.DataFrame(
        [{"problem": "p", "dataset": None, "split": "test", "sample_idx": 0, "correct": 1}],
    )
    right = pd.DataFrame(
        [{"problem": "p", "dataset": None, "split": "test", "sample_idx": 0, "correct": 0}],
    )
    pairs, keys = impl.pair_samples(left, right)
    assert "sample_idx" in keys
    assert "(missing)" in pairs["dataset"].iloc[0]

    df_a = pd.DataFrame([{"a": 1}])
    df_b = pd.DataFrame([{"b": 2}])
    with pytest.raises(SystemExit):
        impl._choose_merge_keys(df_a, df_b)


def test_build_and_pair_clusters(monkeypatch):
    samples = pd.DataFrame(
        [
            {"problem": "p1", "correct": 1, "entropy_p1": 0.2},
            {"problem": "p1", "correct": 0, "entropy_p1": 0.4},
            {"problem": "p2", "correct": 1, "entropy_p1": 0.1},
        ],
    )
    clusters = impl.build_clusters(samples, "p1")
    assert set(clusters.columns) >= {"problem", "n_p1", "k_p1", "acc_p1", "any_p1", "entropy_p1_cluster"}

    samples2 = pd.DataFrame([{"problem": "p1", "correct": 0}, {"problem": "p2", "correct": 1}])
    merged = impl.pair_clusters(samples, samples2)
    assert set(merged["problem"]) == {"p1", "p2"}

    with pytest.raises(SystemExit):
        impl.build_clusters(pd.DataFrame([{"correct": 1}]), "p1")


def test_rng_sets_both_random_streams():
    impl.rng(123)
    assert random.random() == pytest.approx(0.05236359885)
    assert np.random.rand() == pytest.approx(0.6964691856)


def test_rng_handles_bad_seed():
    impl.rng("not-int")
    first_rand = random.random()
    first_np = np.random.rand()
    impl.rng(None)  # also coerces to 0
    assert random.random() == pytest.approx(first_rand)
    assert np.random.rand() == pytest.approx(first_np)


def test_build_stepwise_effect_table_and_verdict(monkeypatch):
    pairs = pd.DataFrame([{"step": 0, "correct1": 0, "correct2": 1}])
    clusters = pd.DataFrame(
        [
            {"step": 0, "any_p1": 0, "any_p2": 1, "acc_p1": 0.2, "acc_p2": 0.6},
            {"step": 0, "any_p1": 1, "any_p2": 1, "acc_p1": 0.5, "acc_p2": 0.7},
        ],
    )
    monkeypatch.setattr(
        impl,
        "_compute_mcnemar_counts",
        lambda df, c1, c2: impl.McNemarCounts(both_wrong=1, wins_pass2=2, wins_pass1=0, both_correct=3, p_value=0.01),
    )
    monkeypatch.setattr(impl, "paired_t_and_wilcoxon", lambda diffs: (0.02, 0.03))

    step_df = impl._build_stepwise_effect_table(pairs, clusters)
    assert {"sample", "cluster_any", "cluster_mean"} == set(step_df["metric"])

    sample_row = step_df[step_df["metric"] == "sample"].iloc[0].to_dict()
    mean_row = step_df[step_df["metric"] == "cluster_mean"].iloc[0].to_dict()

    sample_line = impl.verdict_line(sample_row)
    mean_line = impl.verdict_line(mean_row)
    assert "Δ=" in sample_line
    assert "->" in mean_line


def test_run_plots_if_requested(monkeypatch, tmp_path):
    called = []
    for name in [
        "plot_overall_deltas",
        "plot_conversion_waterfall",
        "plot_headroom_scatter",
        "plot_uncertainty_buckets",
        "plot_stepwise_overlay",
        "plot_uncertainty_and_stepwise",
        "plot_overview_side_by_side",
    ]:
        monkeypatch.setattr(impl, name, lambda *args, name=name, **kwargs: called.append(name))

    monkeypatch.setattr(impl, "parse_color_overrides", lambda colors: {"custom": "#fff"})
    args = SimpleNamespace(
        make_plots=True,
        n_boot=10,
        seed=1,
        colors=None,
        series_palette="Pastel1",
        darken=0.8,
    )
    plots = impl.PlotArtifacts(
        out_dir=str(tmp_path),
        summary_rows=[],
        pairs=pd.DataFrame(),
        clusters=pd.DataFrame(),
        step_df=pd.DataFrame(),
    )
    impl._run_plots_if_requested(args, plots)
    # All plotting helpers should have been invoked
    assert set(called) == {
        "plot_overall_deltas",
        "plot_conversion_waterfall",
        "plot_headroom_scatter",
        "plot_uncertainty_buckets",
        "plot_stepwise_overlay",
        "plot_uncertainty_and_stepwise",
        "plot_overview_side_by_side",
    }


def test_run_plots_if_requested_noop(monkeypatch, tmp_path):
    # Install sentinels that would raise if called
    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("plot function should not run")

    for name in [
        "plot_overall_deltas",
        "plot_conversion_waterfall",
        "plot_headroom_scatter",
        "plot_uncertainty_buckets",
        "plot_stepwise_overlay",
        "plot_uncertainty_and_stepwise",
        "plot_overview_side_by_side",
    ]:
        monkeypatch.setattr(impl, name, _should_not_run)

    args = SimpleNamespace(make_plots=False, n_boot=1, seed=0, colors=None, series_palette="Pastel1", darken=0.8)
    plots = impl.PlotArtifacts(
        out_dir=str(tmp_path),
        summary_rows=[],
        pairs=pd.DataFrame(),
        clusters=pd.DataFrame(),
        step_df=pd.DataFrame(),
    )
    impl._run_plots_if_requested(args, plots)  # should return without calling plots


def test_main_happy_path(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    df1 = pd.DataFrame(
        [
            {
                "dataset": "d",
                "model": "m",
                "split": "s",
                "problem": "p1",
                "step": 0,
                "sample_idx": 0,
                "correct": 0,
                "entropy_p1": 0.1,
            }
        ],
    )
    df2 = pd.DataFrame(
        [{"dataset": "d", "model": "m", "split": "s", "problem": "p1", "step": 0, "sample_idx": 0, "correct": 1}],
    )

    monkeypatch.setattr(
        impl,
        "prepare_forced_aha_samples",
        lambda args, load1, load2: (str(out_dir), df1, df2),
    )
    monkeypatch.setattr(
        impl.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            root1="r1",
            root2=None,
            split=None,
            out_dir=str(out_dir),
            pass2_key=None,
            make_plots=False,
            n_boot=2,
            seed=0,
            colors=None,
            series_palette="Pastel1",
            darken=0.8,
        ),
    )

    # Avoid real plotting; _run_plots_if_requested covered elsewhere.
    monkeypatch.setattr(impl, "_run_plots_if_requested", lambda args, plots: None)

    impl.main()

    assert (out_dir / "forced_aha_summary.csv").exists()
    assert (out_dir / "forced_aha_by_step.csv").exists()


def test_main_raises_on_empty_inputs(monkeypatch, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    empty = pd.DataFrame()
    not_empty = pd.DataFrame([{"problem": "p", "correct": 1, "sample_idx": 0}])
    monkeypatch.setattr(
        impl,
        "prepare_forced_aha_samples",
        lambda args, load1, load2: (str(out_dir), empty, not_empty),
    )
    monkeypatch.setattr(
        impl.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            root1="r1",
            root2=None,
            split=None,
            out_dir=str(out_dir),
            pass2_key=None,
            make_plots=False,
            n_boot=2,
            seed=0,
            colors=None,
            series_palette="Pastel1",
            darken=0.8,
        ),
    )
    with pytest.raises(SystemExit):
        impl.main()

    # Empty pass2 should also raise
    monkeypatch.setattr(
        impl,
        "prepare_forced_aha_samples",
        lambda args, load1, load2: (str(out_dir), not_empty, empty),
    )
    with pytest.raises(SystemExit):
        impl.main()


def test_main_raises_on_empty_clusters(monkeypatch, tmp_path):
    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    df1 = pd.DataFrame([{"problem": "p1", "step": 0, "sample_idx": 0, "correct": 0, "entropy_p1": 0.1}])
    df2 = pd.DataFrame([{"problem": "p1", "step": 0, "sample_idx": 0, "correct": 1}])

    monkeypatch.setattr(impl, "prepare_forced_aha_samples", lambda args, l1, l2: (str(out_dir), df1, df2))
    monkeypatch.setattr(impl, "pair_samples", lambda a, b: (pd.DataFrame(), ["problem"]))
    monkeypatch.setattr(impl, "pair_clusters", lambda a, b: pd.DataFrame())  # triggers empty clusters path
    monkeypatch.setattr(impl, "_run_plots_if_requested", lambda args, plots: None)
    monkeypatch.setattr(
        impl.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            root1="r1",
            root2=None,
            split=None,
            out_dir=str(out_dir),
            pass2_key=None,
            make_plots=False,
            n_boot=2,
            seed=0,
            colors=None,
            series_palette="Pastel1",
            darken=0.8,
        ),
    )
    with pytest.raises(SystemExit):
        impl.main()


def test_main_when_pairs_empty(monkeypatch, tmp_path):
    out_dir = tmp_path / "out3"
    out_dir.mkdir()
    df1 = pd.DataFrame([{"problem": "p1", "step": 0, "sample_idx": 0, "correct": 0, "entropy_p1": 0.1}])
    df2 = pd.DataFrame([{"problem": "p1", "step": 0, "sample_idx": 0, "correct": 1}])
    clusters = pd.DataFrame([{"problem": "p1", "step": 0, "any_p1": 0, "any_p2": 1, "acc_p1": 0.0, "acc_p2": 1.0}])

    monkeypatch.setattr(impl, "prepare_forced_aha_samples", lambda args, l1, l2: (str(out_dir), df1, df2))
    monkeypatch.setattr(impl, "pair_samples", lambda a, b: (pd.DataFrame(), ["problem"]))
    monkeypatch.setattr(impl, "pair_clusters", lambda a, b: clusters)
    monkeypatch.setattr(impl, "_run_plots_if_requested", lambda args, plots: None)
    monkeypatch.setattr(
        impl.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            root1="r1",
            root2=None,
            split=None,
            out_dir=str(out_dir),
            pass2_key=None,
            make_plots=False,
            n_boot=2,
            seed=0,
            colors=None,
            series_palette="Pastel1",
            darken=0.8,
        ),
    )

    # Stub summaries to keep stepwise outputs predictable
    def _summary(metric):
        return {
            "metric": metric,
            "acc_pass1": 0.0,
            "acc_pass2": 1.0,
            "delta_pp": 100.0,
            "n_units": 1,
            "p_mcnemar": 0.01,
            "p_ttest": 0.02,
            "p_wilcoxon": 0.03,
            "wins_pass1": 0,
            "wins_pass2": 1,
            "both_correct": 0,
            "both_wrong": 0,
        }

    monkeypatch.setattr(impl, "summarize_sample_level", lambda pairs: _summary("sample"))
    monkeypatch.setattr(impl, "summarize_cluster_any", lambda clusters: _summary("cluster_any"))
    monkeypatch.setattr(impl, "summarize_cluster_mean", lambda clusters: _summary("cluster_mean"))
    impl.main()
    # Summary CSV should contain only cluster rows
    summary = pd.read_csv(out_dir / "forced_aha_summary.csv")
    assert set(summary["metric"]) == {"cluster_any", "cluster_mean"}
