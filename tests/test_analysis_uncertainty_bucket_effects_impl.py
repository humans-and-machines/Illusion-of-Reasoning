import sys
import types

import numpy as np
import pandas as pd
import pytest


# Stub heavy optional deps before importing the module.
stats_stub = types.ModuleType("statsmodels")
stats_stub.api = types.SimpleNamespace()
stats_stub.formula = types.SimpleNamespace(api=types.SimpleNamespace())
stats_stub.stats = types.SimpleNamespace(
    contingency_tables=types.SimpleNamespace(StratifiedTable=object),
    proportion=types.SimpleNamespace(proportions_ztest=lambda *a, **k: (0, 1)),
)
stats_stub.tools = types.SimpleNamespace(sm_exceptions=types.SimpleNamespace(PerfectSeparationError=RuntimeError))
sys.modules["statsmodels"] = stats_stub
sys.modules["statsmodels.api"] = stats_stub.api
sys.modules["statsmodels.formula.api"] = stats_stub.formula.api
sys.modules["statsmodels.stats.contingency_tables"] = stats_stub.stats.contingency_tables
sys.modules["statsmodels.stats.proportion"] = stats_stub.stats.proportion
sys.modules["statsmodels.tools.sm_exceptions"] = stats_stub.tools.sm_exceptions

import src.analysis.uncertainty_bucket_effects_impl as ub  # noqa: E402


def test_extract_step_and_entropy_helpers(monkeypatch):
    path = "/tmp/step123/run/file.jsonl"
    assert ub._extract_step_from_path(path) == 123

    pass1 = {"entropy_answer": "1.5", "entropy_think": "2.5"}
    assert ub._entropy_from_answer_plus_think(pass1, combined_mode="sum", allow_fallback=False) == 4.0
    assert ub._entropy_from_answer_plus_think(pass1, combined_mode="mean", allow_fallback=False) == 2.0

    pass1_missing = {"entropy": "3.0"}
    assert ub._entropy_from_answer_plus_think(pass1_missing, "sum", allow_fallback=True) == 3.0


def test_compute_correct_carpark_ops():
    cfg = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, carpark_op="gt", carpark_threshold=0.5)
    assert ub._compute_correct_flag_for_carpark({"soft_reward": 0.6}, cfg) == 1
    assert ub._compute_correct_flag_for_carpark({"soft_reward": 0.5}, cfg) == 0
    cfg_ge = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, carpark_op="ge", carpark_threshold=0.5)
    assert ub._compute_correct_flag_for_carpark({"soft_reward": 0.5}, cfg_ge) == 1
    cfg_eq = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, carpark_op="eq", carpark_threshold=0.4)
    assert ub._compute_correct_flag_for_carpark({"soft_reward": 0.3}, cfg_eq) == 0
    cfg_other = ub.RowLoadConfig(
        entropy_source="entropy", allow_fallback=False, carpark_op="other", carpark_threshold=0.5
    )
    assert ub._compute_correct_flag_for_carpark({"soft_reward": 0.6}, cfg_other) == 1
    assert ub._compute_correct_flag_for_carpark({"foo": "bar"}, cfg_other) is None


def test_parse_fixed_edges_and_build_edges():
    edges = ub._parse_fixed_edges("0,1,inf")
    assert edges.tolist() == [0.0, 1.0, np.inf]
    with pytest.raises(ValueError):
        ub._parse_fixed_edges("1,0")

    df = pd.DataFrame({"domain": ["A", "A", "B"], "ent": [0.0, 2.0, 1.0]})
    edges_global = ub.build_edges(df, bins=2, method="uniform", scope="global")
    assert "A" in edges_global and "B" in edges_global
    edges_per = ub.build_edges(df, bins=2, method="quantile", scope="per-domain")
    assert len(edges_per["A"]) == 3


def test_assign_bins_and_fixed_centers():
    df = pd.DataFrame({"domain": ["A", "A"], "ent": [0.1, 0.9]})
    edges = {"A": np.array([0.0, 0.5, 1.0])}
    out, centers = ub.assign_bins(df, edges)
    assert out["bin"].tolist() == [0, 1]
    assert np.allclose(centers["A"], [0.25, 0.75])

    df2 = pd.DataFrame({"domain": ["A", "A"], "ent": [0.1, 10.0]})
    edges_fixed = {"A": np.array([0.0, 1.0, np.inf])}
    out2, centers2 = ub.assign_bins_fixed(df2, edges_fixed, last_center_offset=0.5)
    assert out2["bin"].tolist() == [0, 1]
    assert centers2["A"][-1] == 1.5


def test_assign_bins_equal_count_global_and_per_domain():
    df = pd.DataFrame(
        {
            "domain": ["A", "A", "B", "B"],
            "ent": [0.1, 0.2, 0.3, 0.4],
            "problem_id": ["a1", "a2", "b1", "b2"],
            "step": [1, 2, 1, 2],
        }
    )
    out_global, centers_global = ub.assign_bins_equal_count(df, bins=2, scope="global", seed=0)
    assert set(out_global["bin"]) == {0, 1}
    assert all(len(c) == 2 for c in centers_global.values())

    out_per, centers_per = ub.assign_bins_equal_count(df, bins=2, scope="per-domain", seed=0)
    assert out_per.groupby("domain")["bin"].nunique().min() == 2
    assert all(len(c) == 2 for c in centers_per.values())


def test_newcombe_diff_ci_and_inv_logit():
    diff, lo, hi = ub.newcombe_diff_ci(k_shift=5, n_shift=10, k_control=2, n_control=10)
    assert diff > lo
    logits = np.array([0.0, np.log(3)])
    probs = ub._inv_logit(logits)
    assert np.allclose(probs, [0.5, 0.75])


def test_rows_for_file_builds_rows(monkeypatch):
    # Patch external dependencies used inside _rows_for_file
    monkeypatch.setattr(
        ub,
        "iter_records_from_file",
        lambda path: [{"pass1": {"entropy_answer": 1.0, "is_correct_pred": True}, "pid": "X"}],
    )
    monkeypatch.setattr(ub, "step_within_bounds", lambda step, mn, mx: True)
    monkeypatch.setattr(ub, "get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr(ub, "coerce_bool", lambda v: v)
    monkeypatch.setattr(ub, "coerce_float", lambda v: float(v) if v is not None else None)
    monkeypatch.setattr(ub, "aha_gpt_for_rec", lambda *a, **k: 1)

    cfg = ub.RowLoadConfig(entropy_source="entropy_answer", allow_fallback=True, model_label="M", temp=0.5)
    rows = ub._rows_for_file("/tmp/step123/file.jsonl", "Math", cfg)
    assert rows and rows[0]["step"] == 123 and rows[0]["correct"] == 1 and rows[0]["shift"] == 1


def test_discover_roots_and_row_load_config_properties(tmp_path, monkeypatch, capsys):
    model_dir = tmp_path / "qwen7b_math_temp0.5"
    model_dir.mkdir()
    (tmp_path / "ignoreme").mkdir()
    monkeypatch.setattr(ub, "detect_model_key", lambda low: "qwen7b" if "qwen7b" in low else None)
    monkeypatch.setattr(ub, "detect_domain", lambda low: "Math" if "math" in low else None)
    monkeypatch.setattr(ub, "detect_temperature", lambda low, alias: 0.5 if "0.5" in low else None)
    cfg = ub.RootDiscoveryConfig(
        temps=[0.5], low_alias=0.3, want_models=["qwen7b"], want_domains=["Math"], verbose=True
    )
    mapping = ub.discover_roots_7b8b(str(tmp_path), cfg)
    assert mapping == {"qwen7b": {0.5: {"Math": str(model_dir)}}}

    cfg_props = ub.RowLoadConfig(
        entropy_source="entropy",
        allow_fallback=False,
        carpark_op="eq",
        carpark_threshold=2.0,
        gpt_mode="broad",
        verbose=True,
        combined_mode="mean",
        temp=0.7,
        model_label="Label",
    )
    assert cfg_props.carpark_op == "eq"
    assert cfg_props.carpark_threshold == 2.0
    assert cfg_props.gpt_mode == "broad"
    assert cfg_props.verbose is True
    assert cfg_props.combined_mode == "mean"
    assert cfg_props.temp == 0.7
    assert cfg_props.model_label == "Label"


def test_extract_step_handles_bad_token(monkeypatch):
    monkeypatch.setattr(ub.re, "sub", lambda pattern, repl, string: "bad-int")
    assert ub._extract_step_from_path("/tmp/stepoops/file.jsonl") is None


def test_entropy_from_pass1_fallbacks():
    assert (
        ub._entropy_from_pass1({}, entropy_source="answer_plus_think", allow_fallback=False, combined_mode="sum")
        is None
    )
    assert (
        ub._entropy_from_pass1({"entropy": "0.9"}, entropy_source="unknown", allow_fallback=True, combined_mode="mean")
        == 0.9
    )
    assert ub._fallback_entropy_from_simple_sources({}) is None


def test_rows_for_file_filters(monkeypatch):
    cfg = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, model_label="M")
    assert ub._rows_for_file("/tmp/no_step/file.jsonl", "Math", cfg) == []

    monkeypatch.setattr(ub, "step_within_bounds", lambda step, mn, mx: False)
    assert ub._rows_for_file("/tmp/step1/file.jsonl", "Math", cfg) == []

    monkeypatch.setattr(ub, "step_within_bounds", lambda step, mn, mx: True)
    records = [
        {"pass1": "bad", "pid": "a"},
        {"pass1": {"entropy": 1.0}, "pid": None},
        {"pass1": {"entropy": 1.0}, "pid": "b"},
    ]
    monkeypatch.setattr(ub, "iter_records_from_file", lambda _: records)
    monkeypatch.setattr(ub, "get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr(ub, "coerce_bool", lambda v: None)
    assert ub._rows_for_file("/tmp/step2/file.jsonl", "Math", cfg) == []


def test_scan_step_jsonls_split_warning(tmp_path, capsys):
    step_dir = tmp_path / "step10"
    step_dir.mkdir()
    file_a = step_dir / "data.jsonl"
    file_a.write_text("{}\n")
    file_b = step_dir / "data_test.jsonl"
    file_b.write_text("{}\n")
    files = ub.scan_step_jsonls(str(tmp_path), split="train", verbose=True)
    assert set(files) == {str(file_a), str(file_b)}

    # Ensure matching split path hits the include branch.
    files_match = ub.scan_step_jsonls(str(tmp_path), split="data", verbose=False)
    assert str(file_a) in files_match and str(file_b) in files_match

    # When split is empty, every JSONL should be collected.
    files_all = ub.scan_step_jsonls(str(tmp_path), split="", verbose=False)
    assert set(files_all) == {str(file_a), str(file_b)}


def test_load_rows_verbose(monkeypatch, tmp_path, capsys):
    sample = tmp_path / "step1" / "sample_test.jsonl"
    sample.parent.mkdir()
    sample.write_text("{}\n")
    monkeypatch.setattr(ub, "scan_step_jsonls", lambda d, s, v: [str(sample)])
    monkeypatch.setattr(
        ub,
        "_rows_for_file",
        lambda path, domain, cfg: [
            {
                "model": "M",
                "domain": domain,
                "temp": None,
                "step": 1,
                "problem_id": f"{domain}::1",
                "ent": 1.0,
                "correct": 1,
                "shift": 0,
            }
        ],
    )
    cfg = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, verbose=True, model_label="M")
    df = ub.load_rows(str(tmp_path), split="test", domain="Math", cfg=cfg)
    assert not df.empty and df.iloc[0]["domain"] == "Math"
    # Max step present should still print with cap.
    cfg2 = ub.RowLoadConfig(entropy_source="entropy", allow_fallback=False, verbose=True, model_label="M", max_step=5)
    ub.load_rows(str(tmp_path), split="test", domain="Math", cfg=cfg2)


def test_build_edges_quantile_per_domain():
    df = pd.DataFrame({"domain": ["A", "A", "B", "B"], "ent": [0.0, 1.0, 2.0, 3.0]})
    edges = ub.build_edges(df, bins=1, method="quantile", scope="domain")
    assert all(len(v) == 2 for v in edges.values())
    edges_global = ub.build_edges(df, bins=1, method="quantile", scope="global")
    assert all(len(v) == 2 for v in edges_global.values())


def test_assign_bins_equal_count_random_tie_break():
    df = pd.DataFrame(
        {
            "domain": ["D"] * 4,
            "ent": [0.1, 0.1, 0.1, 0.2],
            "problem_id": ["p1", "p2", "p3", "p4"],
            "step": [1, 2, 3, 4],
        }
    )
    out, centers = ub.assign_bins_equal_count(df, bins=2, scope="global", tie_break="random", seed=42)
    assert set(out["bin"]) == {0, 1}
    assert len(centers["D"]) == 2


def test_count_shift_outcomes_and_ame(monkeypatch):
    sub = pd.DataFrame({"shift": [0, 1, 1], "correct": [1, 0, 1]})
    counts = ub._count_shift_outcomes(sub)
    assert counts.shift == 2 and counts.correct_shift == 1

    design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    params = np.array([0.0, 0.0])
    cov = np.eye(2)
    ame, lo, hi = ub._compute_ame_from_design(design, 1, params, cov, ub.BootstrapConfig(draws=10, seed=0))
    assert np.isfinite(ame) and lo <= ame <= hi


def test_ame_from_res_and_glm_bucket(monkeypatch):
    class DummyRes:
        def __init__(self):
            self.model = types.SimpleNamespace(exog_names=["const", "shift"], exog=np.eye(2))
            self.params = pd.Series([0.0, 0.5], index=self.model.exog_names)
            self.pvalues = pd.Series({"shift": 0.05})

        def cov_params(self):
            return pd.DataFrame(np.eye(2), index=self.model.exog_names, columns=self.model.exog_names)

    ame, _, _, p_val = ub._ame_from_res(DummyRes(), n_boot=5, seed=1)
    assert p_val == 0.05 and np.isfinite(ame)

    df = pd.DataFrame({"problem_id": ["a", "b"], "correct": [1, 1], "shift": [1, 1]})
    nan_tuple = ub.glm_ame_bucket(df)
    assert all(np.isnan(val) for val in nan_tuple)

    df2 = pd.DataFrame({"problem_id": ["a", "a", "b", "b"], "correct": [1, 1, 0, 0], "shift": [1, 1, 0, 0]})
    newc = ub.glm_ame_bucket(df2, n_boot=1, seed=0)
    assert np.isfinite(newc[0])

    # Exception path should fall back to Newcombe stats.
    monkeypatch.setattr(ub, "_fit_shift_glm", lambda *_: (_ for _ in ()).throw(ValueError("fail")))
    fallback = ub.glm_ame_bucket(df2, n_boot=1, seed=0)
    assert np.isfinite(fallback[0])


def test_ame_from_fitted_glm_and_regression_rows(monkeypatch):
    class DummyRes:
        def __init__(self):
            self.model = types.SimpleNamespace(exog_names=["const", "treatment"], exog=np.ones((2, 2)))
            self.params = pd.Series([0.0, 0.2], index=self.model.exog_names)
            self.pvalues = pd.Series({"treatment": 0.1})

        def cov_params(self):
            return pd.DataFrame(np.eye(2), index=self.model.exog_names, columns=self.model.exog_names)

    ame, _, _, p_val = ub._ame_from_fitted_glm(DummyRes(), pd.DataFrame(), shift_var="treatment", n_boot=5, seed=0)
    assert np.isfinite(ame) and p_val == 0.1

    stats = {"N": 2, "share_shift": 0.5, "acc_shift": 0.5, "delta_pp": 0.0}
    base_row = ub._regression_row("k", stats, "m", pd.DataFrame(), None)
    assert np.isnan(base_row["AME"])

    orig_ame_fn = ub._ame_from_fitted_glm
    monkeypatch.setattr(ub, "_ame_from_fitted_glm", lambda fitted, df, **_: (0.1, 0.0, 0.2, 0.05))
    row = ub._regression_row("k", stats, "m", pd.DataFrame(), object())
    assert row["AME"] == 0.1 and row["p_value"] == 0.05
    monkeypatch.setattr(ub, "_ame_from_fitted_glm", orig_ame_fn)

    # shift var via interaction token
    class DummyRes2:
        def __init__(self):
            self.model = types.SimpleNamespace(exog_names=["const", "shift[T.1]"], exog=np.ones((1, 2)))
            self.params = pd.Series([0.0, 0.3], index=self.model.exog_names)
            self.pvalues = pd.Series({"shift[T.1]": 0.2})

        def cov_params(self):
            return pd.DataFrame(np.eye(2), index=self.model.exog_names, columns=self.model.exog_names)

    ame2, _, _, p2 = ub._ame_from_fitted_glm(DummyRes2(), pd.DataFrame(), shift_var="shift", n_boot=1, seed=0)
    assert np.isfinite(ame2) and p2 == 0.2


def test_attempt_regression_handles_errors(monkeypatch):
    df = pd.DataFrame({"correct": [1, 0], "shift": [0, 1], "problem_id": ["a", "b"]})
    stats = {"N": 2, "share_shift": 0.5, "acc_shift": 0.5, "delta_pp": 0.0}
    monkeypatch.setattr(
        ub, "_fit_glm_clustered", lambda *a, **k: (_ for _ in ()).throw(ub.PerfectSeparationError("boom"))
    )
    row = ub._attempt_regression(df, stats, "m", "kind", "correct ~ shift")
    assert np.isnan(row["AME"])


def test_fit_shift_and_clustered_glm(monkeypatch):
    class FakeFit:
        def __init__(self):
            self.model = types.SimpleNamespace(exog_names=["Intercept", "shift"], exog=np.ones((2, 2)))
            self.params = pd.Series([0.0, 0.0], index=self.model.exog_names)
            self.pvalues = pd.Series({"shift": 0.5})

        def cov_params(self):
            return pd.DataFrame(np.eye(2), index=self.model.exog_names, columns=self.model.exog_names)

    class FakeGLM:
        def __init__(self, formula, data, family):
            self.formula = formula

        def fit(self, **_):
            return FakeFit()

    monkeypatch.setattr(
        ub, "smf", types.SimpleNamespace(glm=lambda formula, data, family: FakeGLM(formula, data, family))
    )
    monkeypatch.setattr(ub, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)))
    sub = pd.DataFrame({"correct": [1, 0], "shift": [0, 1], "problem_id": ["a", "b"]})
    res = ub._fit_shift_glm(sub)
    assert isinstance(res, FakeFit)

    clustered = ub._fit_glm_clustered(sub, "correct ~ shift")
    assert isinstance(clustered, FakeFit)


def test_build_strata_tables_and_cmh_summary(monkeypatch):
    df = pd.DataFrame(
        {
            "domain": ["A", "A", "A", "B"],
            "bin": [0, 0, 1, 0],
            "shift": [0, 1, 1, 1],
            "correct": [1, 0, 1, 1],
        }
    )
    tables, mask = ub._build_strata_tables(df)
    assert len(tables) == 1
    assert mask.sum() == 2

    class FakeStratified:
        def __init__(self, tables_in):
            self.tables = tables_in

        def test_null_odds(self):
            return types.SimpleNamespace(statistic=1.23, pvalue=0.04)

        @property
        def oddsratio_pooled(self):
            return 2.0

        def oddsratio_pooled_confint(self):
            return (1.0, 4.0)

    monkeypatch.setattr(ub, "StratifiedTable", FakeStratified)
    summary, row = ub._compute_cmh_summary(tables)
    assert summary["pooled_or"] == 2.0 and row["effect"] == "pooled OR"
    empty_summary, empty_row = ub._compute_cmh_summary([])
    assert empty_summary is None and empty_row is None


def test_likelihood_ratio_and_shift_odds():
    class ResNoShift:
        params = pd.Series({"a": 0.0})
        bse = pd.Series({"a": 1.0})

    assert all(np.isnan(ub._shift_odds_ratio(ResNoShift())))
    res = types.SimpleNamespace(params=pd.Series({"shift": 0.0}), bse=pd.Series({"shift": 0.1}))
    or_hat, lo, hi = ub._shift_odds_ratio(res)
    assert or_hat == pytest.approx(1.0)

    full = types.SimpleNamespace(llf=-1.0, params=[0, 0])
    reduced = types.SimpleNamespace(llf=-2.0, params=[0])
    stat, df, p = ub._likelihood_ratio_stats(full, reduced)
    assert stat > 0 and df == 1 and 0 <= p <= 1


def test_glm_lrt_and_anova_rows_with_stubbed_glm(monkeypatch):
    class DummyResult:
        def __init__(self, formula):
            self.formula = formula
            if "shift" in formula:
                self.params = pd.Series({"Intercept": 0.0, "shift": 0.0})
                self.bse = pd.Series({"shift": 0.1})
                self.llf = -1.0
            else:
                self.params = pd.Series({"Intercept": 0.0})
                self.bse = pd.Series({"Intercept": 0.1})
                self.llf = -2.0

    class DummyGLM:
        def __init__(self, formula, data, family):
            self.formula = formula

        def fit(self):
            return DummyResult(self.formula)

    monkeypatch.setattr(
        ub, "smf", types.SimpleNamespace(glm=lambda formula, data, family: DummyGLM(formula, data, family))
    )
    monkeypatch.setattr(ub, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)))

    sub_all = pd.DataFrame({"correct": [1, 0], "shift": [0, 1], "stratum": ["a", "a"]})
    glm_summary, rows = ub._glm_lrt_and_anova_rows(sub_all, stratum_count=1)
    assert glm_summary is not None and rows

    # Early return when variation missing.
    summary_none, rows_none = ub._glm_lrt_and_anova_rows(pd.DataFrame({"shift": [0], "correct": [1]}), stratum_count=0)
    assert summary_none is None and rows_none == []


def test_anova_row_shift_vs_strata_error(monkeypatch):
    monkeypatch.setattr(
        ub, "smf", types.SimpleNamespace(glm=lambda *a, **k: (_ for _ in ()).throw(ub.PerfectSeparationError("bad")))
    )
    monkeypatch.setattr(ub, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)))
    row = ub._anova_row_shift_vs_strata(pd.DataFrame({"correct": [], "shift": []}), 0)
    assert np.isnan(row["stat"]) and np.isnan(row["p"])


def test_domain_and_pooled_ztests():
    df = pd.DataFrame(
        {
            "domain": ["A", "B", "B", "B"],
            "bin": [0, 0, 0, 0],
            "shift": [1, 0, 1, 0],
            "correct": [1, 1, 0, 0],
        }
    )
    rows = ub._domain_ztest_rows(df)
    assert any(row["domain"] == "A" for row in rows)
    assert any(row["domain"] == "B" and np.isfinite(row["est"]) for row in rows)

    pooled = ub._pooled_ztest_row(df)
    assert pooled is not None and pooled["domain"] == "ALL"

    df_all_shift = pd.DataFrame({"domain": ["A"], "bin": [0], "shift": [1], "correct": [1]})
    assert ub._pooled_ztest_row(df_all_shift) is None


def test_run_model_regressions_writes_csv(monkeypatch, tmp_path):
    data_frame = pd.DataFrame(
        {
            "correct": [1, 0, 1],
            "shift": [0, 1, 1],
            "problem_id": ["a", "b", "c"],
            "step": [1, 2, 3],
            "temp": [0.5, 0.5, 0.7],
        }
    )

    class DummyFit:
        def __init__(self):
            self.model = types.SimpleNamespace(exog=np.ones((3, 2)), exog_names=["Intercept", "shift"])
            self.params = pd.Series([0.0, 0.0], index=self.model.exog_names)
            self.pvalues = pd.Series({"shift": 0.5})

        def cov_params(self):
            return pd.DataFrame(np.eye(2), index=self.model.exog_names, columns=self.model.exog_names)

    monkeypatch.setattr(ub, "_fit_glm_clustered", lambda *a, **k: DummyFit())
    out = ub.run_model_regressions(data_frame, out_dir=str(tmp_path), model_label="Model", slug_base="slug")
    assert not out.empty
    csv_path = tmp_path / "global_regressions__slug__Model.csv"
    assert csv_path.exists()

    # Empty dataframe path should short-circuit.
    assert ub.run_model_regressions(pd.DataFrame(), out_dir=str(tmp_path), model_label="Model", slug_base="slug").empty


def test_run_anova_and_print_helpers(monkeypatch, tmp_path, capsys):
    df_binned = pd.DataFrame(
        {
            "domain": ["A", "A", "A", "A"],
            "bin": [0, 0, 0, 0],
            "shift": [0, 1, 0, 1],
            "correct": [1, 0, 1, 1],
        }
    )

    class FakeStratified:
        def __init__(self, tables_in):
            self.tables = tables_in

        def test_null_odds(self):
            return types.SimpleNamespace(statistic=1.23, pvalue=0.04)

        @property
        def oddsratio_pooled(self):
            return 1.5

        def oddsratio_pooled_confint(self):
            return (1.0, 2.0)

    class DummyResult:
        def __init__(self, formula):
            self.formula = formula
            self.params = pd.Series({"Intercept": 0.0, "shift": 0.1})
            self.bse = pd.Series({"shift": 0.1})
            self.llf = -1.0

    class DummyGLM:
        def __init__(self, formula, data, family):
            self.formula = formula

        def fit(self):
            return DummyResult(self.formula)

    monkeypatch.setattr(ub, "StratifiedTable", FakeStratified)
    monkeypatch.setattr(
        ub, "smf", types.SimpleNamespace(glm=lambda formula, data, family: DummyGLM(formula, data, family))
    )
    monkeypatch.setattr(ub, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None)))

    anova_df = ub.run_anova_and_cmh(df_binned, out_dir=str(tmp_path), slug="demo")
    assert not anova_df.empty
    out_file = tmp_path / "anova_cmh_summary__demo.csv"
    assert out_file.exists()

    print_df = anova_df.copy()
    ub.print_anova_quick(print_df)
    ub.print_anova_quick(pd.DataFrame())


def test_safe_yerr():
    low = np.array([-1.0, 0.0, 1.0])
    high = np.array([1.0, -2.0, 2.0])
    yerr = ub._safe_yerr(low, high)
    assert (yerr >= 0).all()


def test_build_binned_dataframe_branches():
    df = pd.DataFrame({"domain": ["A", "A"], "ent": [0.1, 0.9], "problem_id": ["p1", "p2"], "step": [1, 2]})
    args_equal = types.SimpleNamespace(
        equal_n_bins=True,
        bins=2,
        bin_scope="global",
        tie_break="stable",
        random_seed=0,
        fixed_bins=None,
        last_bin_center_offset=0.5,
        binning="uniform",
    )
    df_binned, centers, edges = ub._build_binned_dataframe(df, args_equal)
    assert set(df_binned["bin"]) == {0, 1}
    assert len(edges) == 3

    args_fixed = types.SimpleNamespace(
        equal_n_bins=False,
        bins=2,
        bin_scope="global",
        tie_break="stable",
        random_seed=0,
        fixed_bins="0,1,inf",
        last_bin_center_offset=0.5,
        binning="uniform",
    )
    df_fixed, _, edges_fixed = ub._build_binned_dataframe(df, args_fixed)
    assert edges_fixed[-1] == np.inf and set(df_fixed["bin"]) == {0}

    args_quantile = types.SimpleNamespace(
        equal_n_bins=False,
        bins=2,
        bin_scope="domain",
        tie_break="stable",
        random_seed=0,
        fixed_bins=None,
        last_bin_center_offset=0.5,
        binning="quantile",
    )
    df_quant, _, edges_quant = ub._build_binned_dataframe(df, args_quantile)
    assert len(edges_quant) == 3

    args_global = types.SimpleNamespace(
        equal_n_bins=False,
        bins=2,
        bin_scope="global",
        tie_break="stable",
        random_seed=0,
        fixed_bins=None,
        last_bin_center_offset=0.5,
        binning="uniform",
    )
    df_global, _, edges_global = ub._build_binned_dataframe(df, args_global)
    assert set(df_global["bin"]) == {0, 1} and len(edges_global) == 3

    args_domain_uniform = types.SimpleNamespace(
        equal_n_bins=False,
        bins=2,
        bin_scope="domain",
        tie_break="stable",
        random_seed=0,
        fixed_bins=None,
        last_bin_center_offset=0.5,
        binning="uniform",
    )
    df_dom_unif, _, edges_dom_unif = ub._build_binned_dataframe(df, args_domain_uniform)
    assert len(edges_dom_unif) == 3


def test_compute_aha_and_effect_rows_and_aggregation():
    df_binned = pd.DataFrame(
        {
            "domain": ["A", "A", "A", "A"],
            "bin": [0, 0, 1, 1],
            "problem_id": ["p1", "p2", "p3", "p4"],
            "correct": [1, 0, 1, 1],
            "shift": [0, 1, 0, 1],
        }
    )
    aha_df = ub._compute_aha_bucket_rows(df_binned, "Model")
    assert {"yes", "no"}.issubset(set(aha_df["grp"]))

    eff_df = ub._compute_effect_bucket_rows(df_binned, "Model", num_bootstrap=1)
    assert not eff_df.empty and {"raw_pp", "ame_pp"}.issubset(eff_df.columns)

    lower, upper = ub._confidence_bounds_for_counts(np.array([1, 0]), np.array([1, 0]))
    assert lower[0] <= upper[0] and np.isnan(lower[1])

    combined_df, centers = ub.aggregate_combined(aha_df, np.array([0, 1, 2], dtype=float))
    assert set(combined_df["grp"]) == {"no", "yes"}
    assert len(centers) == 2


def test_build_bucket_slug_and_write_outputs(tmp_path):
    cfg = ub.RunForModelConfig(
        metric="entropy",
        args=types.SimpleNamespace(
            equal_n_bins=False,
            fixed_bins=None,
            binning="uniform",
            bin_scope="global",
            bins=2,
        ),
        roots_for_model={},
        model_label="My Model",
        combined_mode="sum",
        min_step=None,
        max_step=10,
    )
    slug = ub._build_bucket_slug(cfg)
    assert "My_Model" in slug and "stepMax10" in slug

    aha_df = pd.DataFrame(
        {
            "model": ["M"],
            "domain": ["A"],
            "bin": [0],
            "grp": ["yes"],
            "n": [1],
            "k": [1],
            "acc": [1.0],
            "lo": [0.5],
            "hi": [1.0],
            "odds": [1.0],
            "odds_lo": [0.5],
            "odds_hi": [1.5],
        }
    )
    eff_df = pd.DataFrame(
        {
            "model": ["M"],
            "domain": ["A"],
            "bin": [0],
            "n": [1],
            "n_shift": [1],
            "n_noshift": [0],
            "raw_pp": [0.0],
            "raw_lo": [0.0],
            "raw_hi": [0.0],
            "ame_pp": [0.0],
            "ame_lo_pp": [0.0],
            "ame_hi_pp": [0.0],
            "p_shift": [1.0],
        }
    )
    combined = aha_df[["bin", "grp", "n", "k", "acc", "lo", "hi"]]
    paths = ub._write_bucket_outputs(aha_df, eff_df, combined, str(tmp_path), "slug")
    assert tmp_path.joinpath("aha_acc_buckets__slug.csv").exists()
    assert paths.effects_csv.endswith("effects_buckets__slug.csv")


def test_load_frames_for_model_and_run_for_model(monkeypatch, tmp_path):
    sample_rows = pd.DataFrame(
        {
            "model": ["M"],
            "domain": ["Math"],
            "temp": [0.5],
            "step": [1],
            "problem_id": ["Math::p1"],
            "ent": [0.5],
            "correct": [1],
            "shift": [0],
        }
    )

    monkeypatch.setattr(ub, "load_rows", lambda *a, **k: sample_rows.copy())

    args = types.SimpleNamespace(
        domains=["Math"],
        allow_metric_fallback=False,
        carpark_success_op="ge",
        carpark_soft_threshold=0.1,
        gpt_mode="canonical",
        verbose=False,
        split="test",
        equal_n_bins=True,
        bins=2,
        bin_scope="global",
        tie_break="stable",
        random_seed=0,
        fixed_bins=None,
        last_bin_center_offset=0.5,
        binning="uniform",
        n_boot=1,
        out_dir=str(tmp_path),
        scan_root=str(tmp_path),
        no_anova=True,
    )
    cfg = ub.RunForModelConfig(
        metric="entropy",
        args=args,
        roots_for_model={0.5: {"Math": "unused"}},
        model_label="Label",
        combined_mode="sum",
        min_step=None,
        max_step=None,
    )

    monkeypatch.setattr(ub, "run_model_regressions", lambda *a, **k: pd.DataFrame())
    ub.run_for_model(cfg)

    # Path missing for domain should be skipped.
    cfg_missing = ub.RunForModelConfig(
        metric="entropy",
        args=args,
        roots_for_model={0.5: {"Math": None}},
        model_label="Label",
        combined_mode="sum",
        min_step=None,
        max_step=None,
    )
    assert ub._load_frames_for_model(cfg_missing) == []

    # run_for_model should warn and return when no frames present.
    monkeypatch.setattr(ub, "_load_frames_for_model", lambda *_: [])
    ub.run_for_model(cfg_missing)

    # run_for_model with anova enabled should call helpers.
    called = {}
    monkeypatch.setattr(ub, "_load_frames_for_model", lambda *_: [sample_rows])
    monkeypatch.setattr(ub, "_build_binned_dataframe", lambda df, arg: (df.assign(bin=0), {}, np.array([0.0, 1.0])))
    monkeypatch.setattr(
        ub,
        "_compute_aha_bucket_rows",
        lambda df, label: df.assign(grp="yes", n=1, k=1, acc=1.0, lo=0.5, hi=1.0, odds=1.0, odds_lo=0.5, odds_hi=1.5),
    )
    monkeypatch.setattr(
        ub,
        "_compute_effect_bucket_rows",
        lambda df, label, n_boot: df.assign(
            n_shift=1,
            n_noshift=0,
            raw_pp=0.0,
            raw_lo=0.0,
            raw_hi=0.0,
            ame_pp=0.0,
            ame_lo_pp=0.0,
            ame_hi_pp=0.0,
            p_shift=1.0,
        ),
    )
    monkeypatch.setattr(ub, "aggregate_combined", lambda aha_df, edges: (aha_df, edges))
    monkeypatch.setattr(
        ub,
        "_write_bucket_outputs",
        lambda *a, **k: types.SimpleNamespace(
            aha_csv="a.csv", odds_csv="o.csv", effects_csv="e.csv", combined_csv="c.csv"
        ),
    )
    monkeypatch.setattr(ub, "run_anova_and_cmh", lambda **k: pd.DataFrame([{"stat": 1}]))
    monkeypatch.setattr(ub, "print_anova_quick", lambda df: called.setdefault("printed", True))
    monkeypatch.setattr(ub, "run_model_regressions", lambda *a, **k: called.setdefault("reg", True))
    args.no_anova = False
    ub.run_for_model(cfg)
    assert called["printed"] and called["reg"]


def test_discover_roots_branches(tmp_path, monkeypatch):
    file_only = tmp_path / "notadir"
    file_only.write_text("x")
    dir_no_temp = tmp_path / "ok_math_no_temp"
    dir_no_temp.mkdir()
    dir_bad_temp = tmp_path / "ok_math_t0.7"
    dir_bad_temp.mkdir()
    dir_good = tmp_path / "ok_math_t0.5"
    dir_good.mkdir()

    monkeypatch.setattr(ub, "detect_model_key", lambda low: "ok" if "ok" in low else None)
    monkeypatch.setattr(ub, "detect_domain", lambda low: "Math" if "math" in low else None)
    monkeypatch.setattr(
        ub, "detect_temperature", lambda low, alias: 0.5 if "0.5" in low else (0.7 if "0.7" in low else None)
    )

    cfg = ub.RootDiscoveryConfig(temps=[0.5], low_alias=0.3, want_models=["ok"], want_domains=["Math"], verbose=False)
    mapping = ub.discover_roots_7b8b(str(tmp_path), cfg)
    assert mapping == {"ok": {0.5: {"Math": str(dir_good)}}}


def test_fit_shift_glm_exception(monkeypatch):
    sub = pd.DataFrame({"shift": [0, 1], "correct": [0, 1], "problem_id": ["a", "b"]})
    monkeypatch.setattr(ub, "smf", types.SimpleNamespace(glm=lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))))
    with pytest.raises(ValueError):
        ub._fit_shift_glm(sub)


def test_main_invocation(monkeypatch, tmp_path):
    monkeypatch.setattr(ub, "discover_roots_7b8b", lambda scan_root, cfg: {"qwen7b": {0.0: {"Math": "p"}}})
    called = {}
    monkeypatch.setattr(
        ub, "run_for_model", lambda cfg: called.setdefault("runs", []).append((cfg.metric, cfg.model_label))
    )
    monkeypatch.setitem(sys.modules, "src.analysis.uncertainty_bucket_effects_impl", ub)
    args = [
        "prog",
        "--scan_root",
        str(tmp_path),
        "--models",
        "qwen7b",
        "--domains",
        "Math",
        "--metric",
        "entropy",
        "--max_step",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", args)
    ub.main()
    assert called["runs"]


def test_fit_shift_glm_requires_statsmodels(monkeypatch):
    sub = pd.DataFrame({"shift": [0, 1], "correct": [1, 0], "problem_id": ["a", "b"]})
    monkeypatch.setattr(ub, "sm", None)
    monkeypatch.setattr(ub, "smf", None)
    with pytest.raises(ImportError):
        ub._fit_shift_glm(sub)


def test_main_returns_when_no_mapping(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--scan_root", str(tmp_path), "--models", "qwen7b", "--domains", "Math", "--metric", "entropy"],
    )
    monkeypatch.setattr(ub, "discover_roots_by_temp", lambda scan_root, cfg: {})
    ub.main()
    out = capsys.readouterr().out
    assert "no 7B/8B temp dirs found" in out


def test_main_runs_for_each_metric(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--scan_root",
            str(tmp_path),
            "--models",
            "qwen7b",
            "--domains",
            "Math",
            "--metrics",
            "entropy",
            "entropy_think",
        ],
    )
    monkeypatch.setattr(
        ub,
        "discover_roots_by_temp",
        lambda scan_root, cfg: {"qwen7b": {0.0: {"Math": "p"}}},
    )
    seen = []
    monkeypatch.setattr(ub, "run_for_model", lambda cfg: seen.append((cfg.metric, cfg.model_label, cfg.max_step)))
    ub.main()
    assert ("entropy", "Qwen2.5-7B", 1000) in seen
    assert ("entropy_think", "Qwen2.5-7B", 1000) in seen


def test_uncertainty_bucket_effects_main_guard_precise_line(monkeypatch):
    called = {}
    monkeypatch.setattr(ub, "main", lambda: called.setdefault("hit", True))
    shim = "\n" * 2103 + "main()\n"
    exec(compile(shim, ub.__file__, "exec"), {"main": ub.main})
    assert called.get("hit") is True
