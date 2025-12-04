import importlib
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _stub_statsmodels(monkeypatch):
    stats_pkg = ModuleType("statsmodels")
    stats_pkg.__path__ = []
    api = ModuleType("statsmodels.api")
    api.families = SimpleNamespace(Binomial=lambda: None)
    smf_parent = ModuleType("statsmodels.formula")
    smf_parent.__path__ = []
    smf = ModuleType("statsmodels.formula.api")
    smf.glm = lambda **kwargs: None
    exc_mod = ModuleType("statsmodels.tools.sm_exceptions")
    exc_mod.PerfectSeparationError = RuntimeError
    monkeypatch.setitem(sys.modules, "statsmodels", stats_pkg)
    monkeypatch.setitem(sys.modules, "statsmodels.api", api)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", smf_parent)
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", smf)
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", exc_mod)


@pytest.fixture()
def ebr(monkeypatch):
    _stub_statsmodels(monkeypatch)
    import src.analysis.entropy_bin_regression as ebr_mod

    return importlib.reload(ebr_mod)


def test_path_helpers_and_parse_list(ebr):
    assert ebr.scan_files("root", "split", {"x"}) == []
    assert ebr.parse_comma_list(None) == []
    assert ebr.parse_comma_list("a, b , ,c") == ["a", "b", "c"]

    rec = {"sample_idx": {"k": "v"}}
    assert ebr.get_problem(rec) == "sample_{'k': 'v'}"
    assert ebr.get_sample({"sample_idx": "xyz"}) == "sxyz"
    assert ebr.get_sample(rec) == "s{'k': 'v'}"


def test_entropy_from_pass1_combined_branches(monkeypatch, ebr):
    pass1 = {"entropy": None, "entropy_think": 1.0, "entropy_answer": 3.0}
    assert ebr.entropy_from_pass1(pass1, mode="combined") == 2.0
    assert ebr.entropy_from_pass1(pass1, mode="think") == 1.0

    monkeypatch.setattr(ebr, "gpt_keys_for_mode", lambda mode: ["k"])
    monkeypatch.setattr(ebr, "coerce_bool", lambda v: v)
    assert ebr.compute_shift_at_1({"k": 1}, {}, gpt_mode="canonical") == 1
    assert ebr.compute_shift_at_1({}, {}, gpt_mode="canonical") is None


def test_build_row_and_rows_filters(monkeypatch, ebr):
    monkeypatch.setattr(ebr, "extract_correct", lambda obj, rec: None)
    monkeypatch.setattr(ebr, "carpark_success_from_soft_reward", lambda rec, pass1, op, thr: 1)
    monkeypatch.setattr(ebr, "compute_shift_at_1", lambda *a, **k: 0)
    monkeypatch.setattr(ebr, "coerce_float", lambda v: float(v) if v is not None else None)

    cfg = ebr.RowBuildConfig("gt", 0.0, "canonical", min_step=1, max_step=2)
    rec = {"pass1": {"entropy": 0.5}, "is_correct_after_reconsideration": True, "sample_idx": 1}
    row = ebr._build_row_for_record(rec, "Carpark", step=2, entropy_mode="combined", config=cfg)
    assert row["correct_at_1"] == 1 and row["correct_at_2"] == 1 and row["shift_at_1"] == 0

    records = [
        {"pass1": {"entropy": 0.5}, "sample_idx": 0, "step": 0},  # filtered by min_step
        {"pass1": {"entropy": 0.6}, "sample_idx": 1, "step": 2, "is_correct_after_reconsideration": True},
    ]
    monkeypatch.setattr(ebr, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(ebr, "step_from_rec_or_path", lambda rec, path: rec.get("step", 0))
    df = ebr.build_rows(["a/math_step2.jsonl"], entropy_mode="combined", config=cfg)
    assert len(df) == 1
    assert df.iloc[0]["entropy_at_1"] == 0.6


def test_binning_and_edges_branches(ebr):
    edges = ebr.parse_fixed_bins("0,-inf,inf")
    assert edges[1] == float("-inf")
    with pytest.raises(SystemExit):
        ebr.parse_fixed_bins("1")

    with pytest.raises(SystemExit):
        ebr.compute_edges(np.array([np.nan]), "uniform", 2, None)
    adjusted = ebr.compute_edges(np.array([1.0, 1.0]), "uniform", 2, None)
    assert adjusted[-1] > adjusted[0]

    quant = ebr.compute_edges(np.array([0.0, 0.0, 0.0]), "quantile", 2, None)
    assert quant[1] > quant[0]
    with pytest.raises(SystemExit):
        ebr.compute_edges(np.array([1.0]), "oops", 2, None)

    assert pd.isna(ebr.label_interval(object()))

    df = pd.DataFrame(
        {"domain": ["A", "A", "B"], "entropy_at_1": [0.0, 0.5, 0.2]},
    )
    binned = ebr.apply_binning_cut(df, [0.0, 0.25, 1.0], scope="domain")
    assert "entropy_bin_label" in binned.columns


def test_rank_bins_and_equal_n(monkeypatch, ebr):
    series = pd.Series([np.nan, 0.1])
    bins = ebr._rank_based_bins(series, bins=2, tie_break="random", seed=0)
    assert len(bins) == 2

    df = pd.DataFrame({"entropy_at_1": [0.1, 0.2], "domain": ["A", "B"]})
    equal = ebr.apply_equal_n_binning(df, bins=2, scope="global", tie_break="stable", seed=0)
    assert "entropy_bin_label" in equal.columns


def test_compute_bin_ame_and_plot(monkeypatch, tmp_path, ebr):
    df = pd.DataFrame({"entropy_bin_label": ["a", "b"], "x": [1, 1]})

    class DummyRes:
        def predict(self, frame):
            return np.ones(len(frame))

    ame_df = ebr.compute_bin_ame(DummyRes(), df, "entropy_bin_label", baseline_label="a")
    assert set(ame_df["bin"]) == {"a", "b"}

    plot_inputs = ebr.BinContrastPlotInputs(
        ame_none=pd.DataFrame({"bin": ["a"], "ame": [0.1], "n_rows": [1]}),
        ame_false=pd.DataFrame({"bin": ["a"], "ame": [0.2], "n_rows": [1]}),
        out_png=str(tmp_path / "out.png"),
        out_pdf=str(tmp_path / "out.pdf"),
        title="t",
    )

    class _AxisStub:
        def __init__(self):
            self.calls = []

        def bar(self, *a, **k):
            self.calls.append("bar")

        def set_xticks(self, *a, **k):
            self.calls.append("xticks")

        def set_ylabel(self, *a, **k):
            return type("L", (), {"set_multialignment": lambda *a, **k: None})()

        def set_title(self, *a, **k):
            self.calls.append("title")

        def grid(self, *a, **k):
            self.calls.append("grid")

        def axhline(self, *a, **k):
            self.calls.append("axh")

        def legend(self, *a, **k):
            self.calls.append("legend")

    class _FigStub:
        def __init__(self):
            self.saved = []

        def tight_layout(self):
            return None

        def savefig(self, path, **kwargs):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("x")

        def close(self):
            return None

    monkeypatch.setattr(ebr.plt, "subplots", lambda figsize=None: (_FigStub(), _AxisStub()))
    monkeypatch.setattr(ebr.plt, "close", lambda fig: None)
    ebr.plot_bin_contrasts(plot_inputs, dpi=10)
    assert os.path.exists(plot_inputs.out_png)
    assert os.path.exists(plot_inputs.out_pdf)


def test_resolve_modes_and_scan_filters(monkeypatch, ebr):
    args = SimpleNamespace(entropy_modes=["sum"], entropy_mode=None)
    assert ebr._resolve_entropy_modes(args) == ["sum"]
    args2 = SimpleNamespace(entropy_modes=None, entropy_mode="think")
    assert ebr._resolve_entropy_modes(args2) == ["think"]
    args3 = SimpleNamespace(entropy_modes=None, entropy_mode=None)
    assert "answer" in ebr._resolve_entropy_modes(args3)

    monkeypatch.setattr(ebr, "scan_files_step_only", lambda root, split, skip: ["a.jsonl", "b.jsonl"])
    args_scan = SimpleNamespace(
        scan_root="root",
        split=None,
        path_include="a",
        path_exclude="skip",
    )
    filtered = ebr._scan_and_filter_files(args_scan)
    assert filtered == ["a.jsonl"]
    monkeypatch.setattr(ebr, "scan_files_step_only", lambda *a, **k: [])
    with pytest.raises(SystemExit):
        ebr._scan_and_filter_files(args_scan)


def test_apply_binning_strategy_branches(ebr):
    df = pd.DataFrame({"entropy_at_1": [0.1, 0.2], "domain": ["A", "A"]})
    args_fixed = SimpleNamespace(
        fixed_bins="0,1",
        bin_scope="global",
        equal_n_bins=False,
        binning="uniform",
        bins=2,
        tie_break="stable",
        random_seed=0,
    )
    out_df, info = ebr._apply_binning_strategy(df, args_fixed)
    assert "fixed edges" in info and not out_df.empty

    args_equal = SimpleNamespace(
        fixed_bins=None,
        bin_scope="global",
        equal_n_bins=True,
        binning="uniform",
        bins=2,
        tie_break="stable",
        random_seed=0,
    )
    out_df2, info2 = ebr._apply_binning_strategy(df, args_equal)
    assert "equal_n_bins" in info2 and not out_df2.empty

    args_bad = SimpleNamespace(
        fixed_bins=None,
        bin_scope="global",
        equal_n_bins=False,
        binning="fixed",
        bins=2,
        tie_break="stable",
        random_seed=0,
    )
    with pytest.raises(SystemExit):
        ebr._apply_binning_strategy(df, args_bad)


def test_process_domain_skip_plot(monkeypatch, tmp_path, ebr):
    df_dom = pd.DataFrame(
        {
            "problem": ["p1", "p2"],
            "entropy_bin_label": pd.Categorical(["b1", "b1"]),
            "shift_at_1": [np.nan, 0],
            "correct_at_1": [1, 0],
            "correct_at_2": [1, 1],
            "entropy_at_1": [0.1, 0.2],
        },
    )
    args = SimpleNamespace(
        min_rows_per_problem=1,
        debug=True,
        make_plot=True,
        dpi=10,
        model_name="m",
    )
    context = ebr.DomainRunContext(domain="Math", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))
    monkeypatch.setattr(ebr, "_fit_subset_and_save", lambda **kwargs: kwargs["subset_df"])
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    ebr._process_domain(context, args, df_dom)
