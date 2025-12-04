#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

import src.analysis.graph_2 as g2


def test_extract_step_success_and_entropies(monkeypatch):
    ctx = g2.RecordContext(domain="Math", path="p", step_from_name=5)
    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=True, min_step=1, max_step=10)

    rec = {"step": "7"}
    assert g2._extract_step_for_record(rec, ctx, cfg) == 7
    rec["step"] = "bad"
    assert g2._extract_step_for_record(rec, ctx, cfg) is None

    pass1 = {"is_correct_pred": "1", "entropy_think": "0.2", "entropy_answer": "0.4"}
    success = g2._compute_success_for_record(ctx, rec, pass1, lambda v: None)
    ent_think, ent_answer, ent_joint = g2._compute_entropies(pass1)
    assert success == 1
    assert ent_joint == pytest.approx(0.3)

    ctx_cp = g2.RecordContext(domain="Carpark", path="p", step_from_name=None)
    success_cp = g2._compute_success_for_record(ctx_cp, {}, {}, lambda v: 0)
    assert success_cp == 0


def test_process_single_record_filters(monkeypatch):
    cfg = g2.RowLoadConfig(gpt_keys=["k"], gpt_subset_native=True, min_step=None, max_step=None)
    ctx = g2.RecordContext(domain="Math", path="p", step_from_name=1)

    monkeypatch.setattr(g2, "aha_gpt_for_rec", lambda pass1, rec, subset, keys, dom: True)
    rec = {"pass1": {"is_correct_pred": True, "entropy": 0.1}, "step": 2, "problem": "prob"}
    row = g2._process_single_record(ctx, rec, cfg, carpark_success_fn=lambda v: 1)
    assert row["aha"] == 1
    assert row["entropy_joint"] == 0.1

    # Missing entropies -> None
    rec2 = {"pass1": {"is_correct_pred": True}, "step": 2}
    assert g2._process_single_record(ctx, rec2, cfg, carpark_success_fn=lambda v: 1) is None

    # Invalid pass1 type
    rec3 = {"pass1": "x"}
    assert g2._process_single_record(ctx, rec3, cfg, carpark_success_fn=lambda v: 1) is None


def test_load_rows(monkeypatch):
    recs = [
        {"pass1": {"is_correct_pred": True, "entropy": 0.1}, "problem": "p1", "step": 1},
        {"pass1": {"is_correct_pred": False, "entropy": 0.2}, "problem": "p2", "step": 2},
    ]
    monkeypatch.setattr(g2, "iter_records_from_file", lambda path: recs)
    monkeypatch.setattr(g2, "nat_step_from_path", lambda path: 0)
    monkeypatch.setattr(g2, "aha_gpt_for_rec", lambda *a, **k: False)

    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=True, min_step=None, max_step=None)
    df = g2.load_rows({"Math": ["f1.jsonl"]}, cfg, carpark_success_fn=lambda v: 1)
    assert len(df) == 2
    assert set(df["problem_id"])  # should be populated


def test_make_bins_uniform_and_quantile():
    series = pd.Series([0.0, 1.0, 2.0, 3.0])
    edges_u, centers_u = g2.make_bins(series, n_bins=2, mode="uniform")
    assert len(edges_u) == 3 and len(centers_u) == 2

    edges_q, centers_q = g2.make_bins(series, n_bins=3, mode="quantile")
    assert edges_q[0] <= centers_q[0] <= edges_q[-1]

    empty_edges, empty_centers = g2.make_bins(pd.Series([], dtype=float), 2, "uniform")
    assert empty_edges.size == 0 and empty_centers.size == 0


def test_aggregate_bins_and_rows_from_pivots():
    df = pd.DataFrame(
        {
            "entropy_think": [0.1, 0.2, 0.3, 0.4],
            "aha": [0, 1, 0, 1],
            "correct": [1, 0, 1, 1],
        },
    )
    stat = g2.aggregate_bins(df, "entropy_think", n_bins=2, mode="uniform", min_per_bar=1)
    assert not stat.empty
    assert {"n_aha", "n_noaha", "acc_aha", "acc_noaha"}.issubset(stat.columns)

    # Insufficient support -> empty
    stat2 = g2.aggregate_bins(df, "entropy_think", n_bins=2, mode="uniform", min_per_bar=3)
    assert stat2.empty


def test_plot_domain_panels_writes_figs(tmp_path, monkeypatch):
    # Use real matplotlib but redirect output files to tmp_path
    stat = pd.DataFrame(
        {
            "bin_left": [0.0, 0.5],
            "bin_right": [0.5, 1.0],
            "acc_noaha": [0.2, 0.8],
            "acc_aha": [0.3, 0.9],
        },
    )
    stats_by_metric = {name: stat for name, _ in g2.METRIC_CONFIGS}
    out_png = tmp_path / "fig.png"
    g2.plot_domain_panels("Math", stats_by_metric, str(out_png), dpi=50, title_prefix="T")
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()


def test_compute_domain_stats_and_plots(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "domain": ["A", "A"],
            "entropy_think": [0.1, 0.2],
            "entropy_answer": [0.1, 0.2],
            "entropy_joint": [0.1, 0.2],
            "aha": [0, 1],
            "correct": [1, 0],
        },
    )
    args = SimpleNamespace(
        dataset_name="D",
        model_name="M",
        n_bins=2,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=50,
    )
    captured = {}
    monkeypatch.setattr(
        g2,
        "plot_domain_panels",
        lambda domain, stats_by_metric, out_path_png, dpi, title_prefix=None: captured.update(
            {"domain": domain, "out": out_path_png}
        ),
    )
    g2._compute_domain_stats_and_plots(df, args, str(tmp_path))
    assert captured["domain"] == "A"
    assert os.path.exists(tmp_path / "entropy_hist__D__M.csv")


def test_main_happy_path(monkeypatch, tmp_path):
    args = SimpleNamespace(
        root_crossword=None,
        root_math="root",
        root_math2=None,
        root_math3=None,
        root_carpark=None,
        split=None,
        out_dir=str(tmp_path),
        dataset_name="D",
        model_name="M",
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=1000,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        n_bins=2,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=10,
    )

    monkeypatch.setattr(g2, "_build_arg_parser", lambda: g2.argparse.ArgumentParser())
    monkeypatch.setattr(g2.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(g2, "build_files_by_domain_for_args", lambda a: ({"Math": ["f1"]}, "root"))
    monkeypatch.setattr(
        g2,
        "load_rows",
        lambda files_by_domain, cfg, carpark_success_fn: pd.DataFrame(
            {
                "domain": ["Math"],
                "entropy_think": [0.1],
                "entropy_answer": [0.1],
                "entropy_joint": [0.1],
                "aha": [0],
                "correct": [1],
            },
        ),
    )
    monkeypatch.setattr(g2, "_compute_domain_stats_and_plots", lambda df, args, out_dir: None)

    g2.main()
    assert os.path.isdir(args.out_dir)
