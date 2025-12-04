import os
import sys
import types

import numpy as np
import pandas as pd


# Install lightweight statsmodels stubs before importing the module to avoid sys.exit.
PerfectSeparationError = type("PerfectSeparationError", (Exception,), {})
_sm_api = types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None))
_smf_api = types.SimpleNamespace(glm=lambda *args, **kwargs: None)
sys.modules.setdefault(
    "statsmodels", types.SimpleNamespace(api=_sm_api, formula=types.SimpleNamespace(api=_smf_api), tools=None)
)
sys.modules.setdefault("statsmodels.api", _sm_api)
sys.modules.setdefault("statsmodels.formula", types.SimpleNamespace(api=_smf_api))
sys.modules.setdefault("statsmodels.formula.api", _smf_api)
sys.modules.setdefault(
    "statsmodels.tools",
    types.SimpleNamespace(sm_exceptions=types.SimpleNamespace(PerfectSeparationError=PerfectSeparationError)),
)
sys.modules.setdefault(
    "statsmodels.tools.sm_exceptions", types.SimpleNamespace(PerfectSeparationError=PerfectSeparationError)
)

import src.analysis.entropy_bin_regression as ebr  # noqa: E402


def test_fit_clustered_glm_falls_back_on_nan_bse(monkeypatch):
    class DummyRes:
        def __init__(self):
            self.bse = np.array([np.nan])
            self.cov_type = "cluster"

        def predict(self, *_args, **_kwargs):
            return np.array([0.5, 0.5])

        def summary(self):
            return types.SimpleNamespace(as_text=lambda: "summary")

    class DummyModel:
        def __init__(self):
            self.calls = []

        def fit(self, cov_type=None, cov_kwds=None):
            self.calls.append(cov_type)
            if cov_type == "cluster":
                return DummyRes()
            return types.SimpleNamespace(
                bse=np.array([0.1]),
                cov_type=cov_type,
                predict=lambda *_a, **_k: np.array([0.4, 0.6]),
                summary=lambda: types.SimpleNamespace(as_text=lambda: "hc1"),
            )

    monkeypatch.setattr(ebr, "smf", types.SimpleNamespace(glm=lambda *args, **kwargs: DummyModel()))
    monkeypatch.setattr(ebr, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)))

    df = pd.DataFrame({"y": [0, 1], "x": [0, 1], "cluster": [0, 0]})
    res, summary, cov = ebr.fit_clustered_glm(df, "y ~ x", "cluster")
    assert cov == "HC1"
    assert "hc1" in summary


def test_apply_binning_strategy_fixed_edges(monkeypatch):
    args = types.SimpleNamespace(
        fixed_bins="0,1,inf",
        bin_scope="global",
        equal_n_bins=False,
        binning="uniform",
        bins=3,
        tie_break="stable",
        random_seed=0,
    )
    df = pd.DataFrame({"entropy_at_1": [0.5, 1.5, 2.5]})
    out_df, info = ebr._apply_binning_strategy(df, args)
    assert "fixed" in info
    labels = out_df["entropy_bin_label"].cat.categories
    assert labels[0].startswith("[0")


def test__fit_subset_and_save_prunes_and_skips(monkeypatch, tmp_path):
    # Avoid writing files by monkeypatching fit_clustered_glm and compute_bin_ame
    monkeypatch.setattr(ebr, "fit_clustered_glm", lambda *args, **kwargs: (None, "s", "cov"))
    monkeypatch.setattr(ebr, "compute_bin_ame", lambda *args, **kwargs: pd.DataFrame())

    args = types.SimpleNamespace(min_rows_per_problem=3)
    context = ebr.DomainRunContext(domain="D", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))
    df_dom = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "entropy_bin_label": pd.Categorical(["a", "a"], categories=["a"], ordered=True),
            "correct_at_2": [1, 1],
        }
    )
    out = ebr._fit_subset_and_save("none", df_dom, df_dom, args, context)
    assert out.empty


def test_process_domain_loads_and_skips_plot(monkeypatch, tmp_path):
    # Stub _fit_subset_and_save to skip writing and return data unchanged.
    monkeypatch.setattr(ebr, "_fit_subset_and_save", lambda *a, **k: k.get("subset_df"))
    monkeypatch.setattr(ebr, "plot_bin_contrasts", lambda *a, **k: None)

    df_dom = pd.DataFrame(
        {
            "entropy_bin_label": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True),
            "shift_at_1": [np.nan, 0.0],
            "problem": ["p1", "p2"],
            "correct_at_2": [1, 0],
        }
    )
    args = types.SimpleNamespace(make_plot=False, min_rows_per_problem=2, model_name="M", debug=False)
    context = ebr.DomainRunContext(domain="Crossword", entropy_mode="sum", slug_mode="slug", out_dir=str(tmp_path))
    ebr._process_domain(context, args, df_dom)
    assert (tmp_path / "rows__slug__Crossword.csv").exists()


def test_run_entropy_mode_no_rows_after_domain_filter(monkeypatch):
    monkeypatch.setattr(ebr, "build_rows", lambda *a, **k: pd.DataFrame({"domain": [], "entropy_at_1": []}))
    args = types.SimpleNamespace(
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        gpt_mode="canonical",
        min_step=None,
        max_step=None,
        debug=False,
        domains=None,
        bin_scope="global",
        equal_n_bins=False,
        binning="uniform",
        bins=4,
        fixed_bins=None,
        tie_break="stable",
        random_seed=0,
        make_plot=False,
        model_name="M",
        dataset_name="D",
        out_dir=None,
    )
    # keep_domains provided but build_rows returns empty
    ebr._run_entropy_mode("sum", args, files=[], keep_domains={"Crossword"}, out_root_base="/tmp")


def test_main_executes_with_stubbed_scan(monkeypatch, tmp_path):
    # Stub scan and downstream calls to avoid IO
    monkeypatch.setattr(
        ebr,
        "build_rows",
        lambda *a, **k: pd.DataFrame(
            {
                "domain": ["Crossword"],
                "entropy_at_1": [0.1],
                "problem": ["p1"],
                "shift_at_1": [0],
                "correct_at_1": [1],
                "correct_at_2": [0],
            }
        ),
    )
    monkeypatch.setattr(
        ebr,
        "_apply_binning_strategy",
        lambda df, args: (df.assign(entropy_bin_label=pd.Categorical(["b"], categories=["b"], ordered=True)), "info"),
    )
    monkeypatch.setattr(ebr, "_process_domain", lambda *a, **k: None)
    monkeypatch.setattr(ebr, "_scan_and_filter_files", lambda args: ["f1"])
    monkeypatch.setattr(ebr, "_resolve_entropy_modes", lambda args: ["sum"])
    monkeypatch.setattr(ebr, "parse_comma_list", lambda text: [])
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    argv_before = sys.argv
    sys.argv = ["entropy_bin_regression.py", "--scan_root", str(tmp_path)]
    try:
        ebr.main()
    finally:
        sys.argv = argv_before
