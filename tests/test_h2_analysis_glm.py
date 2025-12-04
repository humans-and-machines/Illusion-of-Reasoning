import types

import numpy as np
import pandas as pd
import pytest

import src.analysis.h2_analysis_glm as h2


def test_require_statsmodels_raises_when_missing(monkeypatch):
    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", ImportError("no sm"))
    with pytest.raises(ImportError):
        h2._require_statsmodels()


def test_get_param_handles_missing_and_arrays(monkeypatch):
    assert h2._get_param(types.SimpleNamespace(params=None), "aha", default=1.5) == 1.5

    res = types.SimpleNamespace(params=[0.1, 0.2], model=types.SimpleNamespace(exog_names=["intercept", "aha"]))
    assert h2._get_param(res, "aha") == pytest.approx(0.2)

    missing = types.SimpleNamespace(params=[0.1, 0.2], model=types.SimpleNamespace(exog_names=["a", "b"]))
    assert np.isnan(h2._get_param(missing, "aha"))


def test_get_param_handles_attribute_error():
    class Boom:
        @property
        def params(self):
            raise ValueError("boom")

    assert h2._get_param(Boom(), "aha", default=7.0) == 7.0


def test_fetch_series_value_handles_none_and_bad_get():
    assert np.isnan(h2._fetch_series_value(None, "x"))

    class Bad:
        def get(self, *_args, **_kwargs):
            raise TypeError("bad")

    assert np.isnan(h2._fetch_series_value(Bad(), "x"))


def test_fetch_series_value_returns_nan_without_get():
    class NoGet:
        pass

    assert np.isnan(h2._fetch_series_value(NoGet(), "x"))


def test_fit_glm_force_ridge_calls_regularized(monkeypatch):
    class _Model:
        def __init__(self):
            self.called = False

        def fit_regularized(self, alpha, L1_wt):
            self.called = True
            return "res"

    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", None)
    monkeypatch.setattr(h2, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom")))
    monkeypatch.setattr(h2, "smf", types.SimpleNamespace(glm=lambda formula, data, family=None: _Model()))

    res, model, label = h2._fit_glm_force_ridge(pd.DataFrame({"correct": [0, 1]}), "correct ~ aha", ridge_penalty=0.5)
    assert label == "ridge"
    assert res == "res"
    assert model.called


def test_fit_glm_with_ridge_if_needed_on_error(monkeypatch):
    class _Model:
        def fit(self, cov_type=None):
            raise np.linalg.LinAlgError("unstable")

        def fit_regularized(self, alpha, L1_wt):
            return "ridge_res"

    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", None)
    monkeypatch.setattr(h2, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom")))
    monkeypatch.setattr(h2, "smf", types.SimpleNamespace(glm=lambda *_, **__: _Model()))

    res, _model, used = h2._fit_glm_with_ridge_if_needed(
        pd.DataFrame({"correct": [0, 1]}), "correct ~ aha", ridge_penalty=0.2
    )
    assert used == "ridge"
    assert res == "ridge_res"


def test_fit_glm_with_ridge_if_needed_flags_unstable(monkeypatch):
    monkeypatch.setattr(h2, "_require_statsmodels", lambda: None)
    monkeypatch.setattr(h2, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom")))

    class _Model:
        def fit(self, cov_type=None):
            return types.SimpleNamespace(params=pd.Series({"aha": 20.0}))

        def fit_regularized(self, alpha=None, L1_wt=None):
            return "ridge"

    monkeypatch.setattr(h2, "smf", types.SimpleNamespace(glm=lambda *_, **__: _Model()))
    with pytest.raises(RuntimeError):
        h2._fit_glm_with_ridge_if_needed(pd.DataFrame({"correct": [0, 1]}), "correct ~ aha", ridge_penalty=0.1)


def test_predict_from_formula_falls_back(monkeypatch):
    class _Res:
        params = np.array([0.1])

        def predict(self, *_args, **_kwargs):
            raise ValueError("bad")

    model = types.SimpleNamespace(
        data=types.SimpleNamespace(design_info=None),
        predict=lambda params, df: np.full(len(df), 0.3),
        family=types.SimpleNamespace(link=types.SimpleNamespace(inverse=lambda arr: arr)),
    )
    arr = h2._predict_from_formula(_Res(), model, pd.DataFrame({"x": [1, 2]}))
    assert np.allclose(arr, 0.3)


def test_predict_from_formula_uses_design_matrix(monkeypatch):
    class _Res:
        params = np.array([2.0])

        def predict(self, *_args, **_kwargs):
            raise ValueError("no direct predict")

    def build_dm(_design_info, design_df, return_type=None):
        return [design_df[["x"]]]

    monkeypatch.setattr(
        h2,
        "build_design_matrices",
        lambda design_info_list, design_df, return_type=None: build_dm(design_info_list, design_df, return_type),
    )
    model = types.SimpleNamespace(
        data=types.SimpleNamespace(design_info="info"),
        family=types.SimpleNamespace(link=types.SimpleNamespace(inverse=lambda arr: arr)),
    )
    design_df = pd.DataFrame({"x": [1.0, 2.0]})
    arr = h2._predict_from_formula(_Res(), model, design_df)
    assert np.allclose(arr, [2.0, 4.0])


def test_first_finite_value_prefers_first(monkeypatch):
    values = {"a": np.nan, "b": 1.2, "c": 3.4}

    def fetch(key):
        return values[key]

    assert h2._first_finite_value(fetch, ("a", "b", "c")) == 1.2


def test_first_finite_value_returns_nan_when_missing():
    assert np.isnan(h2._first_finite_value(lambda _key: np.nan, ("a", "b")))


def test_extract_glm_statistics_from_series():
    params = pd.Series({"aha": 0.5, "uncertainty_std": 0.1, "aha:uncertainty_std": -0.2})
    bse = pd.Series({"aha": 0.05, "uncertainty_std": 0.02, "aha:uncertainty_std": 0.1})
    pvals = pd.Series({"aha": 0.01, "uncertainty_std": 0.2, "aha:uncertainty_std": 0.5})
    res = types.SimpleNamespace(
        params=params, bse=bse, pvalues=pvals, model=types.SimpleNamespace(exog_names=list(params.index))
    )

    coeffs, ses, pvals_out = h2._extract_glm_statistics(res)
    assert coeffs["aha"] == 0.5 and coeffs["inter"] == -0.2
    assert ses["unc"] == 0.02 and pvals_out["unc"] == 0.2


def test_build_formula_when_interaction_enabled():
    assert "aha:uncertainty_std" in h2._build_formula(True)


def test_fit_step_model_penalty_none(monkeypatch):
    monkeypatch.setattr(h2, "_fit_glm_with_ridge_if_needed", lambda *_: ("res", "model", "none"))
    cfg = h2.StepwiseGlmConfig(out_dir=".", penalty="none")
    res, model, label = h2._fit_step_model(pd.DataFrame({"correct": [0, 1]}), "correct ~ aha", cfg)
    assert (res, model, label) == ("res", "model", "none")


def test_fit_step_model_penalty_variants(monkeypatch):
    calls = []

    def fake_force(step_df, formula, ridge_penalty):
        calls.append((formula, ridge_penalty))
        return "res", "model", "ridge"

    monkeypatch.setattr(h2, "_fit_glm_force_ridge", fake_force)

    ridge_cfg = h2.StepwiseGlmConfig(out_dir=".", penalty="ridge", ridge_l2=0.7)
    _, _, label = h2._fit_step_model(pd.DataFrame({"correct": [0, 1]}), "f", ridge_cfg)
    assert label == "ridge"

    other_cfg = h2.StepwiseGlmConfig(out_dir=".", penalty="other", ridge_l2=1.2)
    _, _, label2 = h2._fit_step_model(pd.DataFrame({"correct": [0, 1]}), "f2", other_cfg)
    assert label2 == "ridge"
    assert calls == [("f", 0.7), ("f2", 1.2)]


def test_ame_and_bootstrap_short_circuit(monkeypatch):
    res = types.SimpleNamespace(predict=lambda df: df["aha"] * 0.2 + 0.5)
    model = types.SimpleNamespace()
    step_df = pd.DataFrame({"aha": [0, 1], "uncertainty_std": [0.0, 1.0]})
    assert h2._ame_at_mean_uncertainty(res, model, step_df) == pytest.approx(0.2)

    cfg = h2.StepwiseGlmConfig(out_dir=".", bootstrap_ame=0)
    assert h2._bootstrap_ame_interval(step_df, "f", cfg) == (np.nan, np.nan)


def test_bootstrap_ame_interval_runs(monkeypatch):
    step_df = pd.DataFrame(
        {
            "step": list(range(12)),
            "aha": [0, 1] * 6,
            "uncertainty": np.linspace(0.0, 1.0, 12),
            "uncertainty_std": np.linspace(0.0, 1.0, 12),
            "correct": [0, 1] * 6,
        }
    )
    monkeypatch.setattr(h2, "_fit_glm_force_ridge", lambda *_: ("r", "m", "ridge"))
    monkeypatch.setattr(h2, "_ame_at_mean_uncertainty", lambda *_args: 0.4)
    cfg = h2.StepwiseGlmConfig(out_dir=".", bootstrap_ame=3)
    lo, hi = h2._bootstrap_ame_interval(step_df, "f", cfg)
    assert lo == pytest.approx(0.4)
    assert hi == pytest.approx(0.4)


def test_regression_row_no_variation_sets_delta():
    step_df = pd.DataFrame({"step": [1, 1], "aha": [0, 1], "correct": [0.25, 0.5], "uncertainty": [0.1, 0.2]})
    record = h2._regression_row_no_variation(1, step_df)
    assert record["naive_delta"] == pytest.approx(0.25)
    assert record["penalty"] == "n/a"


def test_regression_row_with_model_uses_helpers(monkeypatch):
    step_df = pd.DataFrame(
        {
            "step": [0, 0, 0, 0],
            "aha": [0, 1, 0, 1],
            "uncertainty": [0.1, 0.2, 0.3, 0.4],
            "uncertainty_std": [0.0, 0.1, 0.2, 0.3],
            "correct": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(h2, "_fit_step_model", lambda *_: ("res", "model", "none"))
    monkeypatch.setattr(
        h2,
        "_extract_glm_statistics",
        lambda *_: (
            {"aha": 1.0, "unc": 0.5, "inter": -0.1},
            {"aha": 0.5, "unc": 1.0, "inter": 2.0},
            {"aha": 0.01, "unc": 0.2, "inter": 0.3},
        ),
    )
    monkeypatch.setattr(h2, "_ame_at_mean_uncertainty", lambda *_: 0.25)
    monkeypatch.setattr(h2, "_bootstrap_ame_interval", lambda *_: (0.1, 0.3))
    cfg = h2.StepwiseGlmConfig(out_dir=".", bootstrap_ame=0)

    record = h2._regression_row_with_model(0, step_df, cfg)
    assert record["penalty"] == "none"
    assert record["aha_coef"] == 1.0
    assert record["inter_p"] == 0.3
    assert record["aha_ame_lo"] == 0.1 and record["aha_ame_hi"] == 0.3


def test_fit_stepwise_glms_runs_pipeline(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "aha": [0, 0, 0, 1],
            "uncertainty": [0.1, 0.2, 0.3, 0.4],
            "uncertainty_std": [0.0, 0.1, 0.2, 0.3],
            "correct": [0, 0, 1, 1],
            "problem": ["p", "p", "p", "p"],
        }
    )
    monkeypatch.setattr(h2, "_require_statsmodels", lambda: None)
    monkeypatch.setattr(
        h2,
        "_regression_row_with_model",
        lambda step, sdf, cfg: {
            "step": step,
            "aha_p": 0.04,
            "penalty": "none",
            "n": len(sdf),
            "aha": sdf["aha"].mean(),
            "unc_coef": 0.0,
            "aha_coef": 0.0,
            "inter_coef": 0.0,
            "unc_se": 0.0,
            "unc_z": np.nan,
            "unc_p": 1.0,
            "inter_se": 0.0,
            "inter_z": np.nan,
            "inter_p": 1.0,
            "aha_se": 0.0,
            "aha_z": np.nan,
            "aha_ame": np.nan,
            "aha_ame_lo": np.nan,
            "aha_ame_hi": np.nan,
            "acc": sdf["correct"].mean(),
            "aha_ratio": sdf["aha"].mean(),
            "mean_uncertainty": sdf["uncertainty"].mean(),
            "naive_delta": 0.0,
        },
    )
    monkeypatch.setattr(
        h2,
        "multipletests",
        lambda pvals, alpha, method=None: (np.array([True, True]), np.array([0.01, 0.02]), None, None),
    )

    cfg = h2.StepwiseGlmConfig(out_dir=str(tmp_path), fdr_alpha=0.1)
    out = h2.fit_stepwise_glms(df, cfg)

    assert "aha_p_adj" in out.columns
    assert (tmp_path / "h2_balance_by_step.csv").exists()
    assert (tmp_path / "h2_step_regression.csv").exists()


def test_fit_stepwise_glms_skips_empty_steps(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "step": [np.nan, 0, 0],
            "aha": [0, 0, 0],
            "uncertainty": [0.1, 0.2, 0.3],
            "uncertainty_std": [0.0, 0.1, 0.2],
            "correct": [1, 0, 1],
            "problem": ["p", "p", "p"],
        }
    )
    monkeypatch.setattr(h2, "_require_statsmodels", lambda: None)
    monkeypatch.setattr(h2, "multipletests", None)
    out = h2.fit_stepwise_glms(df, h2.StepwiseGlmConfig(out_dir=str(tmp_path)))
    assert out["step"].tolist() == [0]
    assert (tmp_path / "h2_balance_by_step.csv").exists()
    assert (tmp_path / "h2_step_regression.csv").exists()


def test_compute_pooled_step_effects_with_stubbed_statsmodels(monkeypatch):
    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", None)
    monkeypatch.setattr(h2, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom")))

    class _Model:
        def __init__(self, res):
            self._res = res

        def fit_regularized(self, alpha=None, L1_wt=None):
            return self._res

    def fake_glm(formula, data, family=None):
        params = pd.Series(
            {
                "aha": 0.5,
                "aha:C(step)[T.1]": 0.25,
                "C(step)[T.2]:aha": 0.75,
            }
        )
        res = types.SimpleNamespace(params=params, model=types.SimpleNamespace(exog_names=list(params.index)))
        return _Model(res)

    monkeypatch.setattr(h2, "smf", types.SimpleNamespace(glm=fake_glm))

    pass1_df = pd.DataFrame(
        {
            "correct": [0, 1, 1],
            "step": [0, 1, 2],
            "aha": [0, 1, 0],
            "uncertainty_std": [0.1, 0.2, 0.3],
            "problem": ["p", "p", "p"],
        }
    )
    out = h2.compute_pooled_step_effects(pass1_df, ridge_l2=0.1)
    assert set(out["step"].tolist()) == {0, 1, 2}
    assert out.loc[out["step"] == 1, "aha_effect"].item() == pytest.approx(0.75)


def test_compute_pooled_step_effects_missing_statsmodels(monkeypatch, capsys):
    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", ImportError("missing"))
    out = h2.compute_pooled_step_effects(
        pd.DataFrame({"correct": [1], "step": [0], "aha": [0], "uncertainty_std": [0.1], "problem": ["p"]}),
        ridge_l2=0.1,
    )
    assert out.empty
    assert "Pooled model skipped" in capsys.readouterr().out


def test_compute_pooled_step_effects_handles_fit_error(monkeypatch, capsys):
    monkeypatch.setattr(h2, "_STATS_MODELS_IMPORT_ERROR", None)
    monkeypatch.setattr(h2, "_require_statsmodels", lambda: None)
    monkeypatch.setattr(h2, "sm", types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom")))

    class _Model:
        def fit_regularized(self, alpha=None, L1_wt=None):
            raise np.linalg.LinAlgError("fail")

    monkeypatch.setattr(h2, "smf", types.SimpleNamespace(glm=lambda *_args, **_kwargs: _Model()))
    out = h2.compute_pooled_step_effects(
        pd.DataFrame({"correct": [1], "step": [0], "aha": [0], "uncertainty_std": [0.1], "problem": ["p"]}),
        ridge_l2=0.1,
    )
    assert out.empty
    assert "Pooled model skipped" in capsys.readouterr().out
