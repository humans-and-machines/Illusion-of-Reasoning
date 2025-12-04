import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def _import_with_stubbed_statsmodels(monkeypatch):
    fake_sm = ModuleType("statsmodels")
    fake_sm.__path__ = []
    fake_sm_api = ModuleType("statsmodels.api")
    fake_sm_api.families = ModuleType("families")
    fake_sm_formula = ModuleType("statsmodels.formula.api")
    fake_sm_formula_pkg = ModuleType("statsmodels.formula")
    fake_sm_formula_pkg.api = fake_sm_formula
    fake_sm.tools = ModuleType("statsmodels.tools")
    fake_sm.tools.sm_exceptions = ModuleType("statsmodels.tools.sm_exceptions")
    fake_sm.tools.sm_exceptions.PerfectSeparationError = RuntimeError

    class _DummyFitResult:
        bse = []
        cov_type = "HC1"

        def summary(self):
            return "summary"

    class _DummyModel:
        def fit(self, cov_type=None, cov_kwds=None):
            return _DummyFitResult()

    fake_sm_formula.glm = lambda *args, **kwargs: _DummyModel()

    monkeypatch.setitem(sys.modules, "statsmodels", fake_sm)
    monkeypatch.setitem(sys.modules, "statsmodels.api", fake_sm_api)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", fake_sm_formula_pkg)
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", fake_sm_formula)
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", fake_sm.tools.sm_exceptions)
    return importlib.import_module("src.analysis.entropy_bin_regression")


def test_apply_binning_strategy_fixed_and_equal(monkeypatch):
    eb = _import_with_stubbed_statsmodels(monkeypatch)
    df = pd.DataFrame({"entropy_at_1": [0.1, 0.2], "entropy_bin_label": [None, None]})
    args = SimpleNamespace(
        fixed_bins="0,0.5,1",
        equal_n_bins=False,
        bin_scope="global",
        bins=2,
        tie_break="stable",
        random_seed=0,
        binning="uniform",
    )
    fixed_df, info = eb._apply_binning_strategy(df.copy(), args)
    assert "fixed edges" in info and not fixed_df["entropy_bin_label"].isna().any()

    args.fixed_bins = None
    args.equal_n_bins = True
    equal_df, info = eb._apply_binning_strategy(df.copy(), args)
    assert "equal_n_bins" in info and not equal_df["entropy_bin_label"].isna().any()


def test_apply_binning_strategy_fixed_binning_requires_edges():
    eb = _import_with_stubbed_statsmodels(pytest.MonkeyPatch())
    df = pd.DataFrame({"entropy_at_1": [0.1], "entropy_bin_label": [None]})
    args = SimpleNamespace(
        fixed_bins=None,
        equal_n_bins=False,
        bin_scope="global",
        bins=2,
        tie_break="stable",
        random_seed=0,
        binning="fixed",
    )
    with pytest.raises(SystemExit):
        eb._apply_binning_strategy(df, args)


def test_process_domain_and_run_entropy_mode(monkeypatch, tmp_path, capsys):
    eb = _import_with_stubbed_statsmodels(monkeypatch)
    # Build a tiny dataframe with required columns.
    df_dom = pd.DataFrame(
        {
            "problem": ["p1", "p2"],
            "entropy_bin_label": pd.Categorical([0, 1]),
            "shift_at_1": [0, 1],
            "correct_at_1": [1, 0],
            "correct_at_2": [1, 1],
            "domain": ["Math", "Math"],
            "entropy_at_1": [0.1, 0.2],
            "entropy_think_at_1": [0.1, 0.2],
            "entropy_answer_at_1": [0.1, 0.2],
            "sample": ["s1", "s2"],
        }
    )
    args = SimpleNamespace(
        min_rows_per_problem=1,
        make_plot=False,
        debug=True,
        bins=2,
        bin_scope="global",
        tie_break="stable",
        random_seed=0,
        binning="uniform",
        fixed_bins=None,
        equal_n_bins=False,
        carpark_success_op=">=",
        carpark_soft_threshold=0.5,
        gpt_mode="canonical",
        min_step=None,
        max_step=None,
        dataset_name="ds",
        model_name="m",
        out_dir=str(tmp_path),
    )
    context = eb.DomainRunContext(domain="Math", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))

    # Stub heavy functions to avoid statsmodels/plots.
    monkeypatch.setattr(
        eb, "fit_clustered_glm", lambda df, formula, cluster_col: (SimpleNamespace(), "summary", "HC1")
    )
    monkeypatch.setattr(eb, "compute_bin_ame", lambda *a, **k: pd.DataFrame({"bin": [0], "ame": [0], "n_rows": [1]}))
    monkeypatch.setattr(eb, "plot_bin_contrasts", lambda *a, **k: None)

    eb._process_domain(context, args, df_dom.copy())
    out = capsys.readouterr().out
    assert "rows:" in out

    # Now run _run_entropy_mode with stubs to avoid filesystem scanning.
    files = ["f1"]
    monkeypatch.setattr(eb, "build_rows", lambda files, entropy_mode, config: df_dom.copy())
    eb._run_entropy_mode("sum", args, files, keep_domains=None, out_root_base=str(tmp_path))


def test_main_runs_with_minimal_args(monkeypatch, tmp_path):
    eb = _import_with_stubbed_statsmodels(monkeypatch)
    # Provide fake CLI arguments and stub scanning/parsing to short-circuit.
    monkeypatch.setattr(
        eb,
        "build_arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                scan_root=str(tmp_path),
                split=None,
                path_include=None,
                path_exclude=None,
                min_step=None,
                max_step=None,
                domains=None,
                out_dir=str(tmp_path),
                dataset_name="ds",
                model_name="m",
                gpt_mode="canonical",
                carpark_success_op=">=",
                carpark_soft_threshold=0.5,
                entropy_modes="sum",
                make_plot=False,
                debug=False,
                bins=2,
                bin_scope="global",
                tie_break="stable",
                random_seed=0,
                binning="uniform",
                fixed_bins=None,
                equal_n_bins=False,
            )
        ),
    )
    monkeypatch.setattr(eb, "_scan_and_filter_files", lambda args: ["f1"])
    monkeypatch.setattr(eb, "_run_entropy_mode", lambda **kwargs: None)
    eb.main()
