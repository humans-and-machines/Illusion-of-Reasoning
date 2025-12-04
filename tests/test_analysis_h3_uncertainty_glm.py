import types

import numpy as np
import pandas as pd

import src.analysis.h3_uncertainty.glm as h3_glm


def test_cov_spec_returns_cluster_and_hc1():
    df = pd.DataFrame({"problem": ["a", "a", "b"]})
    cov_type, cov_kwargs = h3_glm._cov_spec(df, cluster_by="problem")
    assert cov_type == "cluster"
    assert cov_kwargs and "groups" in cov_kwargs
    # Non-problem cluster falls back to HC1 with no kwargs.
    cov_type2, cov_kwargs2 = h3_glm._cov_spec(df, cluster_by="other")
    assert cov_type2 == "HC1" and cov_kwargs2 is None


def test_build_glm_formula_variants():
    full = h3_glm._build_glm_formula("aha", strict_interaction_only=False)
    assert "aha" in full and "C(perplexity_bucket)" in full
    strict = h3_glm._build_glm_formula("aha", strict_interaction_only=True)
    assert "aha" in strict and "C(perplexity_bucket)" in strict
    assert "+ aha +" not in strict  # interaction-only drops standalone main effect


def test_write_glm_summary_includes_cluster_note(tmp_path):
    out_path = tmp_path / "summary.txt"

    class _FakeSummary:
        def as_text(self):
            return "SUMMARY"

    class _FakeResult:
        def summary(self):
            return _FakeSummary()

    h3_glm._write_glm_summary(
        str(out_path),
        _FakeResult(),
        cov_type="cluster",
        cov_kwargs={"groups": np.array([0, 1])},
    )

    content = out_path.read_text(encoding="utf-8")
    assert "SUMMARY" in content
    assert "Covariance: cluster" in content
    assert "clustered by problem" in content


def test_compute_bucket_rows_uses_predict_difference():
    class _FakeResult:
        def predict(self, frame):
            return frame["aha"].to_numpy()

    glm_df = pd.DataFrame(
        {
            "perplexity_bucket": [0, 0, 1, 1],
            "aha": [0, 1, 0, 1],
            "correct": [1, 0, 1, 1],
        }
    )
    rows = h3_glm._compute_bucket_rows(_FakeResult(), glm_df, "aha")
    assert len(rows) == 2
    assert all("AME_bucket" in r for r in rows)
    # With predict returning aha, alt-base should yield AME close to 1
    assert all(abs(r["AME_bucket"] - 1.0) < 1e-8 for r in rows)


def test_fit_glm_bucket_interaction_uses_fallback(monkeypatch, tmp_path):
    # Fake statsmodels modules
    fake_sm = types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: "binom"))

    class _FakeResult:
        def __init__(self):
            self.predict_calls = []

        def predict(self, frame):
            self.predict_calls.append(len(frame))
            return frame["aha_words"].to_numpy()

        def summary(self):
            class _Sum:
                def as_text(self_inner):
                    return "SUM"

            return _Sum()

    class _FakeModel:
        def __init__(self):
            self.calls = []
            self.fail_once = True

        def fit(self, cov_type=None, cov_kwds=None):
            self.calls.append((cov_type, cov_kwds))
            if self.fail_once:
                self.fail_once = False
                raise TypeError("bad cov_kwds")
            return _FakeResult()

    class _FakeSMF:
        def glm(self, *args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(h3_glm, "sm", fake_sm)
    monkeypatch.setattr(h3_glm, "smf", _FakeSMF())

    df = pd.DataFrame(
        {
            "problem": ["p1", "p1", "p2", "p2"],
            "step": [1, 2, 1, 2],
            "correct": [1, 0, 1, 1],
            "perplexity_bucket": [0, 0, 1, 1],
            "aha_words": [0, 1, 0, 1],
        }
    )
    out_txt = tmp_path / "glm.txt"
    summary, result = h3_glm.fit_glm_bucket_interaction(
        df,
        aha_col="aha_words",
        strict_interaction_only=False,
        cluster_by="problem",
        out_txt=str(out_txt),
    )
    assert summary["N"] == len(df)
    assert out_txt.exists()
    assert hasattr(result, "predict")


def test_bucket_group_accuracy():
    df = pd.DataFrame(
        {
            "perplexity_bucket": ["a", "a", "b"],
            "aha": [0, 1, 0],
            "correct": [1, 0, 1],
        }
    )
    grouped = h3_glm.bucket_group_accuracy(df, aha_col="aha")
    assert set(grouped.columns) == {"perplexity_bucket", "aha", "n", "k", "accuracy"}
    # Accuracy computed per group
    assert grouped.loc[(grouped["perplexity_bucket"] == "a") & (grouped["aha"] == 0), "accuracy"].iloc[0] == 1.0
