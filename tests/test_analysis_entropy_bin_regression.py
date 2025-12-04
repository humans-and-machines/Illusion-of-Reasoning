import importlib
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ebr():
    """Import entropy_bin_regression with statsmodels stubbed out."""

    # Provide lightweight stubs to satisfy imports without real statsmodels
    class _PerfectSeparationError(Exception):
        pass

    fake_api = types.ModuleType("statsmodels.api")
    fake_api.families = types.SimpleNamespace(Binomial=lambda: None)

    fake_formula = types.ModuleType("statsmodels.formula")
    fake_smf = types.ModuleType("statsmodels.formula.api")
    fake_smf.glm = lambda **kwargs: None
    fake_formula.api = fake_smf

    fake_tools = types.ModuleType("statsmodels.tools")
    fake_tools_ex = types.ModuleType("statsmodels.tools.sm_exceptions")
    fake_tools_ex.PerfectSeparationError = _PerfectSeparationError

    sys.modules["statsmodels"] = types.ModuleType("statsmodels")
    sys.modules["statsmodels.api"] = fake_api
    sys.modules["statsmodels.formula"] = fake_formula
    sys.modules["statsmodels.formula.api"] = fake_smf
    sys.modules["statsmodels.tools"] = fake_tools
    sys.modules["statsmodels.tools.sm_exceptions"] = fake_tools_ex

    return importlib.import_module("src.analysis.entropy_bin_regression")


def test_parse_fixed_bins_accepts_edges_and_inf(ebr):
    assert ebr.parse_fixed_bins("0,1,inf") == [0.0, 1.0, float("inf")]
    with pytest.raises(SystemExit):
        ebr.parse_fixed_bins("1")  # needs at least two edges


def test_compute_edges_uniform_and_quantile(ebr):
    values = np.array([0.0, 1.0, 2.0])
    uniform_edges = ebr.compute_edges(values, binning="uniform", bins=2, fixed=None)
    assert len(uniform_edges) == 3 and uniform_edges[0] == 0.0 and uniform_edges[-1] == 2.0

    quant_edges = ebr.compute_edges(values, binning="quantile", bins=2, fixed=None)
    assert len(quant_edges) == 3
    assert quant_edges[0] <= quant_edges[1] <= quant_edges[2]

    with pytest.raises(SystemExit):
        ebr.compute_edges(values, binning="unknown", bins=2, fixed=None)


def test_apply_binning_cut_sets_labels(ebr):
    df = pd.DataFrame({"entropy_at_1": [0.1, 0.4, 0.9], "domain": ["A", "A", "A"]})
    edges = [0.0, 0.5, 1.0]
    binned = ebr.apply_binning_cut(df, edges, scope="global")
    assert "entropy_bin_label" in binned.columns
    assert binned["entropy_bin_at_1"].cat.ordered


def test_rank_based_bins_and_equal_n_binning(ebr):
    series = pd.Series([0.1, 0.2, 0.3, 0.4])
    bins = ebr._rank_based_bins(series, bins=2, tie_break="stable", seed=0)
    assert set(bins) <= {0, 1}
    bins_rand = ebr._rank_based_bins(series, bins=2, tie_break="random", seed=42)
    assert len(bins_rand) == len(series)

    df = pd.DataFrame({"entropy_at_1": [0.1, 0.2, 0.3, 0.4], "domain": ["X", "X", "Y", "Y"]})
    equal_df = ebr.apply_equal_n_binning(df, bins=2, scope="domain", tie_break="stable", seed=0)
    assert "entropy_bin_label" in equal_df.columns
    assert equal_df["entropy_bin_at_1"].cat.categories.tolist() == [0, 1]


def test_prune_subset_drops_insufficient_variation(ebr, capsys):
    df = pd.DataFrame(
        {
            "problem": ["p1", "p1", "p2"],
            "correct_at_2": [1, 0, 0],
            "entropy_bin_label": pd.Categorical(["a", "a", "b"]),
        }
    )
    pruned = ebr.prune_subset(df, min_rows_per_problem=2)
    assert pruned["problem"].tolist() == ["p1", "p1"]
    assert "a" in pruned["entropy_bin_label"].cat.categories


def test_entropy_from_pass1_modes_and_shift(ebr):
    pass1 = {"entropy_think": 0.3, "entropy_answer": 0.7, "entropy": 1.0}
    assert ebr.entropy_from_pass1(pass1, mode="sum") == pytest.approx(1.0)
    assert ebr.entropy_from_pass1(pass1, mode="think") == pytest.approx(0.3)
    assert ebr.entropy_from_pass1(pass1, mode="answer") == pytest.approx(0.7)
    assert ebr.entropy_from_pass1(pass1, mode="combined") == pytest.approx(1.0)

    rec = {"shift_in_reasoning_v1": 1}
    gpt = ebr.compute_shift_at_1({"shift_in_reasoning_v1": 1}, rec, gpt_mode="canonical")
    assert gpt == 1


def test_parse_comma_list_and_helpers(ebr):
    assert ebr.parse_comma_list("a, b , ,c") == ["a", "b", "c"]
    assert ebr.domain_from_path("path/to/xword/file.jsonl") == "Crossword"
    assert ebr.domain_from_path("carpark-file.jsonl") == "Carpark"
    assert ebr.domain_from_path("math-file.jsonl") == "Math"

    assert ebr.get_problem({"problem_id": "123"}) == "problem_id:123"
    assert ebr.get_sample({"sample_idx": 5}) == "s5"


def test_import_uses_stubs_when_io_helpers_missing(monkeypatch):
    """Importing with missing IO helpers should fall back to local stubs."""
    dummy_io = types.ModuleType("src.analysis.io")
    dummy_io.scan_files_step_only = None
    dummy_io.iter_records_from_file = None
    monkeypatch.setitem(sys.modules, "src.analysis.io", dummy_io)
    if "src.analysis" in sys.modules:
        monkeypatch.setattr(sys.modules["src.analysis"], "io", dummy_io, raising=False)

    perfect_sep = type("PerfectSeparationError", (Exception,), {})
    fake_api = types.ModuleType("statsmodels.api")
    fake_api.families = types.SimpleNamespace(Binomial=lambda: None)
    fake_smf = types.ModuleType("statsmodels.formula.api")
    fake_smf.glm = lambda **kwargs: None
    fake_tools = types.ModuleType("statsmodels.tools")
    fake_tools_ex = types.ModuleType("statsmodels.tools.sm_exceptions")
    fake_tools_ex.PerfectSeparationError = perfect_sep
    monkeypatch.setitem(sys.modules, "statsmodels", types.ModuleType("statsmodels"))
    monkeypatch.setitem(sys.modules, "statsmodels.api", fake_api)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", types.ModuleType("statsmodels.formula"))
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", fake_smf)
    monkeypatch.setitem(sys.modules, "statsmodels.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", fake_tools_ex)

    module_name = "entropy_bin_regression_missing_io"
    script_path = Path(__file__).resolve().parents[1] / "src/analysis/entropy_bin_regression.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ImportError):
        module.scan_files_step_only("root", None, set())
    assert module.iter_records_from_file() == []
    sys.modules.pop(module_name, None)


def test_build_rows_respects_max_step(monkeypatch, ebr):
    records = [
        {
            "sample_idx": 0,
            "problem_id": "p1",
            "step": 1,
            "pass1": {"is_correct": 1, "entropy": 0.2},
            "pass2": {"is_correct": 0},
        },
        {
            "sample_idx": 1,
            "problem_id": "p2",
            "step": 5,
            "pass1": {"is_correct": 0, "entropy": 0.4},
            "pass2": {"is_correct": 1},
        },
    ]
    monkeypatch.setattr(ebr, "iter_records_from_file", lambda _path: records)
    monkeypatch.setattr(ebr, "step_from_rec_or_path", lambda rec, _path: rec["step"])
    monkeypatch.setattr(ebr, "extract_correct", lambda payload, _rec: payload.get("is_correct"))
    monkeypatch.setattr(ebr, "carpark_success_from_soft_reward", lambda *a, **k: None)
    config = ebr.RowBuildConfig(
        carpark_op="gt",
        carpark_thr=0.0,
        gpt_mode="canonical",
        min_step=None,
        max_step=2,
    )
    df = ebr.build_rows(files=["math.jsonl"], entropy_mode="combined", config=config)
    assert df["step"].tolist() == [1]


def test_compute_edges_returns_fixed(ebr):
    fixed = [0.0, 1.0, 2.0]
    edges = ebr.compute_edges(np.array([0.1, 0.9]), binning="uniform", bins=3, fixed=fixed)
    assert edges == fixed


def test_label_interval_handles_nan(ebr):
    assert np.isnan(ebr.label_interval(np.nan))


def test_rank_based_bins_empty_series(ebr):
    empty_bins = ebr._rank_based_bins(pd.Series([np.nan, np.nan]), bins=3, tie_break="stable", seed=0)
    assert (empty_bins == -1).all()


def test_fit_clustered_glm_handles_exceptions(monkeypatch, ebr):
    class FailingThenSucceeding:
        def __init__(self):
            self.calls = 0

        def fit(self, cov_type=None, cov_kwds=None):
            self.calls += 1
            if self.calls == 1:
                raise np.linalg.LinAlgError("bad cov")
            return types.SimpleNamespace(
                bse=np.array([0.1]),
                cov_type=cov_type,
                predict=lambda *_a, **_k: np.array([0.2, 0.8]),
                summary=lambda: types.SimpleNamespace(as_text=lambda: "ok"),
            )

    monkeypatch.setattr(
        ebr,
        "smf",
        types.SimpleNamespace(glm=lambda *args, **kwargs: FailingThenSucceeding()),
    )
    monkeypatch.setattr(
        ebr,
        "sm",
        types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)),
    )
    df = pd.DataFrame({"y": [0, 1], "x": [0, 1], "cluster": [0, 0]})
    res, _, cov = ebr.fit_clustered_glm(df, "y ~ x", "cluster")
    assert cov == "HC1"
    assert res.cov_type == "HC1"


def test_scan_and_filter_files_respects_exclude(monkeypatch, ebr):
    monkeypatch.setattr(ebr, "scan_files_step_only", lambda *_a, **_k: ["keep/a.jsonl"])
    args = types.SimpleNamespace(
        scan_root=".",
        split=None,
        path_include=None,
        path_exclude="a",
    )
    with pytest.raises(SystemExit):
        ebr._scan_and_filter_files(args)


def test_process_domain_plotting_with_missing_none(monkeypatch, tmp_path, ebr):
    def fake_fit(subset_tag, subset_df, domain_df, args, context):
        out_csv = tmp_path / f"bin_contrasts__{subset_tag}__{context.slug_mode}__{context.domain}.csv"
        if subset_tag == "false":
            pd.DataFrame({"bin": ["b1"], "ame": [0.1], "n_rows": [1]}).to_csv(out_csv, index=False)
        return subset_df

    called = {}
    monkeypatch.setattr(ebr, "_fit_subset_and_save", fake_fit)
    monkeypatch.setattr(
        ebr,
        "plot_bin_contrasts",
        lambda *_a, **_k: called.setdefault("plotted", True),
    )

    df_dom = pd.DataFrame(
        {
            "entropy_bin_label": ["x", "y"],
            "shift_at_1": [np.nan, 0.0],
            "problem": ["p1", "p2"],
            "correct_at_2": [1, 0],
        }
    )
    args = types.SimpleNamespace(
        make_plot=True,
        min_rows_per_problem=1,
        model_name="Model",
        debug=False,
        dpi=72,
    )
    context = ebr.DomainRunContext(
        domain="Math",
        entropy_mode="sum",
        slug_mode="slug",
        out_dir=str(tmp_path),
    )
    ebr._process_domain(context, args, df_dom)
    assert called.get("plotted") is True


def test_main_entrypoint_runs_under_runpy(monkeypatch, tmp_path):
    perfect_sep = type("PerfectSeparationError", (Exception,), {})
    fake_api = types.ModuleType("statsmodels.api")
    fake_api.families = types.SimpleNamespace(Binomial=lambda: None)
    fake_smf = types.ModuleType("statsmodels.formula.api")
    fake_smf.glm = lambda **kwargs: None
    fake_tools = types.ModuleType("statsmodels.tools")
    fake_tools_ex = types.ModuleType("statsmodels.tools.sm_exceptions")
    fake_tools_ex.PerfectSeparationError = perfect_sep
    monkeypatch.setitem(sys.modules, "statsmodels", types.ModuleType("statsmodels"))
    monkeypatch.setitem(sys.modules, "statsmodels.api", fake_api)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", types.ModuleType("statsmodels.formula"))
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", fake_smf)
    monkeypatch.setitem(sys.modules, "statsmodels.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", fake_tools_ex)

    script_path = Path(__file__).resolve().parents[1] / "src/analysis/entropy_bin_regression.py"
    argv_before = sys.argv
    sys.argv = ["entropy_bin_regression.py", "--help"]
    try:
        with pytest.raises(SystemExit):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = argv_before
