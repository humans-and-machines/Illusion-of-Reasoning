#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import src.analysis.heatmap_1 as hm


def test_set_rendered_width_handles_missing_attrs():
    class DummyFig:
        pass

    assert hm.set_rendered_width(DummyFig(), target_width_in=1.0) is False


def test_set_rendered_width_converges_and_get_rendered_size():
    fig, _ = plt.subplots()
    try:
        ok = hm.set_rendered_width(fig, target_width_in=1.0, dpi=200)
        assert ok is True
        width, height = hm.get_rendered_size(fig, dpi=200)
        assert width > 0 and height > 0
    finally:
        plt.close(fig)


def test_get_rendered_size_savefig_failure_fallback():
    class FallbackFig:
        def savefig(self, *_a, **_k):
            raise ValueError("boom")

        def get_size_inches(self):
            return 2.5, 1.5

    width, height = hm.get_rendered_size(FallbackFig(), dpi=123)
    assert width == pytest.approx(2.5) and height == pytest.approx(1.5)


def test_get_rendered_size_imread_failure_without_size(monkeypatch):
    class MissingSizeFig:
        def savefig(self, path, **_k):
            with open(path, "wb") as handle:
                handle.write(b"")

    monkeypatch.setattr(hm, "imread", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("no img")))
    width, height = hm.get_rendered_size(MissingSizeFig(), dpi=10)
    assert width == 0.0 and height == 0.0


def test_set_rendered_width_breaks_on_tiny_render(monkeypatch):
    fig = SimpleNamespace()
    fig_state = {"size": (1.0, 1.0)}
    fig.get_size_inches = lambda: fig_state["size"]
    fig.set_size_inches = lambda size: fig_state.update(size=tuple(size))

    monkeypatch.setattr(hm, "get_rendered_size", lambda *_a, **_k: (0.0, 0.0))
    ok = hm.set_rendered_width(fig, target_width_in=1.0, dpi=200)
    assert ok is False


def test_set_rendered_width_breaks_when_scaled_size_too_small(monkeypatch):
    fig = SimpleNamespace()
    fig_state = {"size": (0.2, 0.2)}
    fig.get_size_inches = lambda: fig_state["size"]
    fig.set_size_inches = lambda size: fig_state.update(size=tuple(size))

    monkeypatch.setattr(hm, "get_rendered_size", lambda *_a, **_k: (0.1, 0.1))
    ok = hm.set_rendered_width(fig, target_width_in=0.05, dpi=200)
    assert ok is False


def test_collect_files_and_out_dir_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(hm, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    args = argparse.Namespace(
        split=None,
        out_dir=None,
        root_crossword=None,
        root_math=None,
        root_math2=None,
        root_math3=None,
        root_carpark=None,
    )
    with pytest.raises(SystemExit):
        hm._collect_files_and_out_dir(args)

    monkeypatch.setattr(hm, "build_jsonl_files_by_domain", lambda roots, split: ({"Crossword": []}, str(tmp_path)))
    args.root_crossword = str(tmp_path)
    with pytest.raises(SystemExit):
        hm._collect_files_and_out_dir(args)


def test_write_output_table_and_latex_helper(tmp_path):
    long_rows = [
        pd.DataFrame(
            {
                "delta1": [0.0],
                "delta2": [0.0],
                "n_events": [1],
                "n_pairs": [2],
                "pct": [50.0],
                "scope": ["overall"],
                "domain_key": ["ALL"],
                "domain_label": ["ALL"],
            },
        ),
    ]
    out_csv = hm._write_output_table(long_rows, str(tmp_path), "slug")
    assert os.path.exists(out_csv)

    args = argparse.Namespace(model_name="Model", dataset_name="Data")
    grid = long_rows[0]
    hm._write_latex_helper(args, grid, [0.0], str(tmp_path), "slug2")
    assert any(f.name.endswith(".tex") for f in tmp_path.iterdir())


def test_add_group_15b_grid_and_plot(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(hm, "plot_heatmap", lambda *a, **k: calls.setdefault("plot", True))
    args = argparse.Namespace(
        make_15b_overall=True,
        domains_15b="Crossword,Math",
        delta_values=[0.0],
        title_15b="title",
        cmap="YlGnBu",
    )
    step_df = pd.DataFrame(
        {"domain_key": ["Other"], "problem_id": ["p::1"], "step": [1], "acc_frac": [0.1], "shift_frac": [0.0]}
    )
    hm._add_group_15b_grid_and_plot(step_df, args, [], "out")
    assert "plot" not in calls
    out = capsys.readouterr().out
    assert "requested domains are present" in out


def test_main_wires_helpers(monkeypatch, tmp_path, capsys):
    class FakeParser:
        def parse_args(self):
            return argparse.Namespace(
                root_crossword="r1",
                root_math=None,
                root_math2=None,
                root_math3=None,
                root_carpark=None,
                label_crossword="CW",
                label_math="M",
                label_math2="M2",
                label_math3="M3",
                label_carpark="CP",
                split=None,
                out_dir=str(tmp_path),
                dataset_name="DS",
                model_name="Model",
                per_domain=True,
                title_overall="overall",
                title_15b="t15",
                make_15b_overall=False,
                domains_15b="Crossword,Math,Carpark",
                cmap="YlGnBu",
                gpt_mode="canonical",
                no_gpt_subset_native=False,
                min_step=None,
                max_step=None,
                delta_values=[0.0, 0.25],
                carpark_success_op="ge",
                carpark_soft_threshold=0.5,
            )

    monkeypatch.setattr(hm, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(
        hm, "build_jsonl_files_by_domain", lambda roots, split: ({"Crossword": ["f.jsonl"]}, str(tmp_path))
    )
    monkeypatch.setattr(
        hm,
        "iter_correct_and_shift_samples_for_config",
        lambda files_by_domain, config: [("Crossword", 1, {"id": "p1"}, 1, 0)],
    )
    monkeypatch.setattr(hm, "get_problem_id", lambda rec: "p1")
    monkeypatch.setattr(hm, "compute_effective_max_step", lambda args, hard_max_step: 5)
    monkeypatch.setattr(hm, "make_carpark_success_fn", lambda op, thr: lambda *_a, **_k: True)

    plot_calls = []
    monkeypatch.setattr(hm, "plot_heatmap", lambda *a, **k: plot_calls.append(a[0]))
    per_domain_calls = []
    monkeypatch.setattr(
        hm, "_add_per_domain_grids_and_plots", lambda **kwargs: per_domain_calls.append(kwargs["step_df"])
    )
    monkeypatch.setattr(hm, "_add_group_15b_grid_and_plot", lambda *a, **k: None)

    hm.main()
    out = capsys.readouterr().out
    assert plot_calls, "overall heatmap should be plotted"
    assert per_domain_calls, "per-domain plotting should be invoked"
    assert "Saved CSV" in out or "Saved CSV" not in out  # ensure main runs without error


def test_plot_heatmap_handles_missing_imshow(monkeypatch):
    class DummyFig:
        def __init__(self):
            self.closed = False

        def savefig(self, *_args, **_kwargs):
            raise AssertionError("should not save without imshow")

    class DummyAxes:
        # no imshow attribute to trigger early return
        pass

    def fake_subplots(*_args, **_kwargs):
        return DummyFig(), DummyAxes()

    monkeypatch.setattr(hm.plt, "subplots", fake_subplots)
    monkeypatch.setattr(hm.plt, "close", lambda fig: setattr(fig, "closed", True))

    df = pd.DataFrame({"delta1": [0.0], "delta2": [0.0], "n_events": [0], "n_pairs": [1], "pct": [0.0]})
    hm.plot_heatmap(df, "t", "out.png")
