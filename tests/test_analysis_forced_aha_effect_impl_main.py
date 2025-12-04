import runpy
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import src.analysis.forced_aha_effect_impl as impl


def test_main_runs_with_stubbed_io(monkeypatch, tmp_path):
    # Stub arg parser to avoid CLI parsing.
    args = SimpleNamespace(
        root_pass1="r1",
        root_pass2=None,
        split=None,
        entropy_field="entropy_answer",
        pass2_key=None,
        out_dir=str(tmp_path),
        n_boot=10,
        seed=0,
        series_palette="deep",
        darken=0.1,
        plot_figures=False,
    )

    class FakeParser:
        def parse_args(self_inner):
            return args

    # Minimal sample-level data to flow through pairing and summaries.
    df1 = pd.DataFrame(
        {
            "problem": ["p1"],
            "dataset": ["d"],
            "model": ["m"],
            "step": [1],
            "split": ["s"],
            "sample_idx": [0],
            "correct": [1],
            "entropy_p1": [0.1],
        }
    )
    df2 = pd.DataFrame(
        {
            "problem": ["p1"],
            "dataset": ["d"],
            "model": ["m"],
            "step": [1],
            "split": ["s"],
            "sample_idx": [0],
            "correct": [0],
        }
    )

    monkeypatch.setattr(impl, "build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(impl, "prepare_forced_aha_samples", lambda *a, **k: (str(tmp_path), df1, df2))
    monkeypatch.setattr(
        impl,
        "pair_samples",
        lambda a, b: (
            pd.DataFrame({"step": [1], "correct1": [1], "correct2": [0]}),
            ["problem", "step", "sample_idx"],
        ),
    )
    monkeypatch.setattr(
        impl,
        "pair_clusters",
        lambda a, b: pd.DataFrame(
            {"problem": ["p1"], "step": [1], "any_p1": [1], "any_p2": [0], "acc_p1": [1.0], "acc_p2": [0.0]}
        ),
    )
    monkeypatch.setattr(
        impl, "_build_stepwise_effect_table", lambda pairs, clusters: pd.DataFrame({"step": [1], "delta_any": [-1.0]})
    )
    monkeypatch.setattr(impl, "_run_plots_if_requested", lambda args, artifacts: None)

    impl.main()
    # Verify outputs were written
    assert (tmp_path / "forced_aha_summary.csv").exists()
    assert (tmp_path / "forced_aha_by_step.csv").exists()


def test_module_entrypoint(monkeypatch, tmp_path):
    # Reuse the stubbed main via monkeypatch to avoid heavy IO in runpy.
    called = {}
    monkeypatch.setattr(impl, "main", lambda: called.setdefault("ran", True))
    # Ensure argparse won't exit due to missing roots when the module re-executes.
    monkeypatch.setattr(
        __import__("argparse").ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            root1="r1",
            root2=None,
            split=None,
            out_dir=str(tmp_path),
            pass2_key=None,
            make_plots=False,
            n_boot=1,
            seed=0,
            colors=None,
            entropy_field="entropy_answer",
            series_palette="deep",
            darken=0.0,
        ),
    )
    monkeypatch.setattr(__import__("sys"), "argv", ["forced_aha_effect_impl"])
    with pytest.raises(SystemExit):
        sys.modules.pop("src.analysis.forced_aha_effect_impl", None)
        runpy.run_module("src.analysis.forced_aha_effect_impl", run_name="__main__")
