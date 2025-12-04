#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.metrics as metrics


def test_correctness_helpers_cover_paths():
    mapping = {"is_correct_pred": "yes"}
    assert metrics._correct_from_mapping(mapping) == 1
    mapping2 = {"accuracy": 0.7}
    assert metrics._correct_from_mapping(mapping2) == 1
    mapping2b = {"acc": 0.0}
    assert metrics._correct_from_mapping(mapping2b) == 0
    mapping3 = {"foo": "bar"}
    assert metrics._correct_from_mapping(mapping3) is None
    mapping4 = {"accuracy": None}
    assert metrics._correct_from_mapping(mapping4) is None
    mapping5 = {"score": 1.0}
    assert metrics._correct_from_mapping(mapping5) == 1

    nested = {"a": {"exact_match": 0}}
    assert metrics.find_correct_in_obj(nested) == 0
    assert metrics._children_for_traversal([1, 2]) == [1, 2]

    # extract_correct: canonical match path
    obj_like = {"pred_answer_canon": "Ans"}
    rec = {"gold_answer_canon": "Ans"}
    assert metrics.extract_correct(obj_like, rec) == 1

    # raw answer comparison path
    obj_like = {"pred_answer": "a "}
    rec = {"gold_answer": "a"}
    assert metrics.extract_correct(obj_like, rec) == 1

    # fall back to None when nothing matches
    assert metrics.extract_correct({}, {}) is None


def test_carpark_success_and_builder():
    rec = {"soft_reward": "0.4"}
    assert metrics.carpark_success_from_soft_reward(rec, {}, "gt", 0.5) == 0
    assert metrics.carpark_success_from_soft_reward(rec, {}, "ge", 0.4) == 1
    assert metrics.carpark_success_from_soft_reward({}, {"soft_reward": 0.3}, "eq", 0.3) == 1
    assert metrics.carpark_success_from_soft_reward({}, {"soft_reward": 0.9}, "weird", 0.5) == 1
    assert metrics.carpark_success_from_soft_reward({}, {}, "gt", 0.5) is None

    cmp_fn = metrics.make_carpark_success_fn("gt", 0.1)
    cmp_eq = metrics.make_carpark_success_fn("eq", 0.2)
    cmp_ge = metrics.make_carpark_success_fn("ge", 0.2)
    cmp_weird = metrics.make_carpark_success_fn("other", 0.5)
    assert cmp_fn("0.2") == 1
    assert cmp_eq(0.2) == 1
    assert cmp_ge(0.2) == 1
    assert cmp_weird(0.4) == 0
    assert cmp_fn(None) is None


def test_shift_and_cond_counts_and_step_std():
    df = pd.DataFrame({"shift": [1, 0, 1], "correct": [1, 0, 0]})
    total, share, p_shift, p_noshift = metrics.shift_conditional_counts(df)
    total2, share2, p_shift2, p_noshift2 = metrics.cond_counts(df)
    assert (total, share, p_shift, p_noshift) == (total2, share2, p_shift2, p_noshift2)
    assert total == 3 and p_shift == pytest.approx(0.5)

    df_step = pd.DataFrame({"step": [0, 1, 2]})
    with_std = metrics.add_step_std_column(df_step)
    assert "step_std" in with_std.columns
    assert np.isclose(with_std["step_std"].mean(), 0.0)


def test_wilson_ci_and_glm_header(tmp_path):
    lo, hi = metrics.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)
    lo2, hi2 = metrics.wilson_ci(5, 10)
    assert 0 <= lo2 <= hi2 <= 1

    class DummySummary:
        def as_text(self):
            return "SUMMARY"

    class DummyRes:
        def summary(self):
            return DummySummary()

    out_txt = tmp_path / "glm.txt"
    metrics.write_glm_summary_header(str(out_txt), DummyRes(), "cluster", {"groups": [0, 1]})
    contents = out_txt.read_text()
    assert "SUMMARY" in contents
    assert "Covariance" in contents


def test_glm_covariance_and_fit(monkeypatch):
    df = pd.DataFrame({"problem": ["a", "b"], "y": [0, 1]})
    cov_type, cov_kwds = metrics.glm_cov_spec(df, cluster_by="problem")
    assert cov_type == "cluster"
    assert "groups" in cov_kwds
    cov_type_hc1, cov_kwds_hc1 = metrics.glm_cov_spec(df, cluster_by="other")
    assert cov_type_hc1 == "HC1" and cov_kwds_hc1 is None

    class DummyModel:
        def __init__(self):
            self.fit_calls = []

        def fit(self, cov_type=None, cov_kwds=None):
            self.fit_calls.append((cov_type, cov_kwds))
            if cov_kwds and "use_correction" in cov_kwds:
                raise TypeError("rejecting kwds")
            return SimpleNamespace()

    dummy_model = DummyModel()
    res, cov_type_out, cov_kwds_out = metrics.glm_fit_with_covariance(dummy_model, df, cluster_by="problem")
    assert cov_type_out == "cluster"
    # Fallback should have retried with reduced cov_kwds after TypeError
    assert len(dummy_model.fit_calls) == 2


def test_lazy_import_statsmodels_handles_missing(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name.startswith("statsmodels"):
            raise ImportError("missing")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError):
        metrics.lazy_import_statsmodels()

    # success path returns both modules
    def fake_import_ok(name, package=None):
        if name.endswith("api") or name.endswith("formula.api"):
            return SimpleNamespace()
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_ok)
    sm_mod, smf_mod = metrics.lazy_import_statsmodels()
    assert sm_mod is not None and smf_mod is not None


def test_predict_formula_prefers_design_matrix(monkeypatch):
    class DummyRes:
        def __init__(self):
            self.params = np.array([1.0, 2.0])

        def predict(self, df):
            raise TypeError("force fallback")

    class DummyFamily:
        class link:
            @staticmethod
            def inverse(x):
                return x

    class DummyModel:
        def __init__(self):
            self.data = SimpleNamespace(design_info="design")
            self.family = DummyFamily()

    def build_design_matrices(design_infos, new_data, return_type=None):
        return [np.column_stack([np.ones(len(new_data)), np.array(new_data["x"])])]

    fake_patsy = SimpleNamespace(build_design_matrices=build_design_matrices)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: fake_patsy if name == "patsy" else importlib.import_module(name)
    )

    res = DummyRes()
    model = DummyModel()
    new_df = pd.DataFrame({"x": [1, 2]})
    preds = metrics.predict_formula(res, model, new_df)
    assert np.allclose(preds, np.array([3.0, 5.0]))

    # fallback to model.predict when design_info missing
    model2 = SimpleNamespace(data=None, predict=lambda params, df: np.array([42] * len(df)))
    res2 = SimpleNamespace(predict=lambda df: np.array([7] * len(df)))
    preds2 = metrics.predict_formula(res2, model2, new_df)
    assert list(preds2) == [7, 7]

    # bottom-of-function fallback to model.predict when res.predict fails and no design_info
    class ResBottom:
        def __init__(self):
            self.params = np.array([2.0])

        def predict(self, df):
            raise AttributeError("force full fallback")

    class ModelBottom:
        def __init__(self):
            self.data = SimpleNamespace(design_info=None)

        def predict(self, params, df):
            return np.array([params[0]] * len(df))

        family = SimpleNamespace(link=SimpleNamespace(inverse=lambda x: x))

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "patsy" else importlib.import_module(name),
    )
    bottom_preds = metrics.predict_formula(ResBottom(), ModelBottom(), new_df)
    assert list(bottom_preds) == [2.0, 2.0]
