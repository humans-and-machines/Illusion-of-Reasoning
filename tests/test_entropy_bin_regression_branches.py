import builtins
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _stub_statsmodels(monkeypatch):
    """Install lightweight statsmodels stubs so the module can import safely."""
    fake_sm = ModuleType("statsmodels")
    fake_sm.__path__ = []
    fake_sm_api = ModuleType("statsmodels.api")
    fake_sm_api.families = SimpleNamespace(Binomial=lambda: None)
    fake_sm_formula = ModuleType("statsmodels.formula.api")
    fake_sm_formula_pkg = ModuleType("statsmodels.formula")
    fake_sm_formula_pkg.__path__ = []
    fake_sm_formula_pkg.api = fake_sm_formula

    class _DummyResult:
        def __init__(self, cov_type="cluster", bse=None):
            self.cov_type = cov_type
            self.bse = np.array([1.0]) if bse is None else np.array(bse)

        def summary(self):
            return SimpleNamespace(as_text=lambda: "summary")

    class _DummyModel:
        def __init__(self):
            self.calls = 0

        def fit(self, cov_type=None, cov_kwds=None):
            self.calls += 1
            return _DummyResult(cov_type=cov_type or "cluster")

    fake_sm_formula.glm = lambda **kwargs: _DummyModel()
    fake_exc_module = ModuleType("statsmodels.tools.sm_exceptions")
    fake_exc_module.PerfectSeparationError = RuntimeError

    monkeypatch.setitem(sys.modules, "statsmodels", fake_sm)
    monkeypatch.setitem(sys.modules, "statsmodels.api", fake_sm_api)
    monkeypatch.setitem(sys.modules, "statsmodels.formula", fake_sm_formula_pkg)
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", fake_sm_formula)
    monkeypatch.setitem(sys.modules, "statsmodels.tools.sm_exceptions", fake_exc_module)


def test_import_requires_statsmodels(monkeypatch):
    # Simulate ImportError and verify heavy deps are deferred until needed.
    for name in list(sys.modules):
        if name.startswith("statsmodels"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("missing statsmodels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec = importlib.util.spec_from_file_location(
        "entropy_bin_regression_missing",
        os.path.join(os.path.dirname(__file__), "..", "src", "analysis", "entropy_bin_regression.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    assert module.sm is None and module.smf is None
    df = pd.DataFrame({"y": [0, 1], "x": [0, 1], "cluster": [0, 0]})
    with pytest.raises(ImportError):
        module.fit_clustered_glm(df, "y ~ x", "cluster")


@pytest.fixture()
def ebin(monkeypatch):
    _stub_statsmodels(monkeypatch)
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "entropy_bin_regression_fixture",
            os.path.join(os.path.dirname(__file__), "..", "src", "analysis", "entropy_bin_regression.py"),
        )
    )
    spec = mod.__spec__
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def test_scan_files_proxy(monkeypatch, ebin):
    called = {}
    monkeypatch.setattr(
        ebin,
        "scan_files_step_only",
        lambda root, split_substr, skip_substrings: called.setdefault(
            "args",
            (root, split_substr, skip_substrings),
        )
        or [],
    )
    ebin.scan_files("root", "split", {"x"})
    assert called["args"] == ("root", "split", {"x"})


def test_entropy_combined_fallback_single_part(ebin):
    val = ebin.entropy_from_pass1({"entropy_think": 0.9}, mode="combined")
    assert val == pytest.approx(0.9)


def test_build_row_carpark_uses_pass2_carpark_success(monkeypatch, ebin):
    monkeypatch.setattr(ebin, "extract_correct", lambda *a, **k: None)
    monkeypatch.setattr(ebin, "carpark_success_from_soft_reward", lambda *a, **k: 1)
    rec = {
        "pass1": {"entropy": 0.1},
        "pass2": {"foo": "bar"},
        "is_correct_after_reconsideration": None,
    }
    cfg = ebin.RowBuildConfig("gt", 0.0, "canonical", min_step=None, max_step=None)
    row = ebin._build_row_for_record(rec, domain="Carpark", step=1, entropy_mode="combined", config=cfg)
    assert row["correct_at_2"] == 1


def test_label_interval_handles_string_and_missing_attrs(ebin):
    assert ebin.label_interval("bin-label") == "bin-label"
    assert pd.isna(ebin.label_interval(object()))


def test_equal_n_binning_marks_empty_bins(ebin):
    df = pd.DataFrame({"entropy_at_1": [0.5], "domain": ["Math"]})
    out = ebin.apply_equal_n_binning(df, bins=2, scope="global", tie_break="stable")
    assert any("∅" in cat for cat in out["entropy_bin_label"].dtype.categories)


def test_prune_subset_returns_empty_immediately(ebin):
    empty = pd.DataFrame(columns=["problem", "correct_at_2"])
    assert ebin.prune_subset(empty).empty


def test_build_arg_parser_defaults(ebin):
    parser = ebin.build_arg_parser()
    args = parser.parse_args(["--scan_root", "root"])
    assert args.binning == "uniform"
    assert args.bins == 4
    assert args.bin_scope == "global"
    assert args.entropy_mode is None


def test_fit_subset_and_save_writes_outputs(tmp_path, monkeypatch, ebin):
    # Force fit_clustered_glm to return predictable values.
    class DummyResult:
        cov_type = "cluster"

        def summary(self):
            return SimpleNamespace(as_text=lambda: "summary")

        def predict(self, df):
            return np.linspace(0.1, 0.3, len(df))

    monkeypatch.setattr(
        ebin,
        "fit_clustered_glm",
        lambda subset_df, formula, cluster_col: (DummyResult(), "summary", "cluster"),
    )
    subset_df = pd.DataFrame(
        {
            "entropy_bin_label": pd.Categorical(["b1", "b1", "b2", "b2"]),
            "correct_at_2": [1, 0, 1, 0],
            "correct_at_1": [1, 1, 0, 0],
            "problem": ["p1", "p1", "p2", "p2"],
        }
    )
    domain_df = subset_df.copy()
    args = SimpleNamespace(min_rows_per_problem=1, debug=False, make_plot=False, dpi=50, model_name="m")
    context = ebin.DomainRunContext(domain="Math", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))
    out = ebin._fit_subset_and_save("none", subset_df, domain_df, args, context)
    assert not out.empty
    assert (tmp_path / "model_none__slug__Math.txt").exists()
    assert (tmp_path / "bin_contrasts__none__slug__Math.csv").exists()


def test_process_domain_make_plot_no_contrasts_returns(monkeypatch, tmp_path, ebin):
    df = pd.DataFrame(
        {
            "domain": ["Math"],
            "problem": ["p1"],
            "entropy_bin_label": pd.Categorical(["b1"]),
            "shift_at_1": [0],
            "correct_at_1": [1],
            "correct_at_2": [1],
        }
    )
    args = SimpleNamespace(min_rows_per_problem=1, debug=False, make_plot=True, dpi=10, model_name="m")
    context = ebin.DomainRunContext(domain="Math", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))
    monkeypatch.setattr(ebin, "_fit_subset_and_save", lambda *a, **k: df)
    called = {}
    monkeypatch.setattr(ebin, "plot_bin_contrasts", lambda *a, **k: called.setdefault("plot", True))
    ebin._process_domain(context, args, df)
    assert "plot" not in called  # should return early when contrast CSVs are missing


def test_main_invokes_run_entropy_mode(monkeypatch, tmp_path, ebin):
    parsed = SimpleNamespace(
        scan_root="root",
        split=None,
        min_step=None,
        max_step=None,
        path_include=None,
        path_exclude=None,
        bin_scope="global",
        binning="uniform",
        bins=2,
        fixed_bins=None,
        equal_n_bins=False,
        tie_break="stable",
        random_seed=0,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        gpt_mode="canonical",
        domains="Math",
        entropy_mode="sum",
        entropy_modes=None,
        min_rows_per_problem=1,
        out_dir=str(tmp_path),
        dataset_name="ds",
        model_name="m",
        debug=False,
        make_plot=False,
        dpi=10,
    )

    monkeypatch.setattr(
        ebin.argparse.ArgumentParser,
        "parse_args",
        lambda self: parsed,
    )
    monkeypatch.setattr(ebin, "_scan_and_filter_files", lambda args: ["f.jsonl"])
    monkeypatch.setattr(ebin, "_run_entropy_mode", lambda *a, **k: parsed.__dict__.setdefault("ran", True))
    ebin.main()
    assert parsed.__dict__.get("ran") is True
