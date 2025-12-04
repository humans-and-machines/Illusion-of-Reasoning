import numpy as np
import pandas as pd
import pytest

import src.analysis.forced_aha_shared as fhs


def test_first_nonempty_and_select_pass_object():
    mapping = {"pass1": {}, "p1": {"a": 1}, "pass2": {"b": 2}}
    assert fhs.first_nonempty(mapping, fhs.PASS1_KEYS) == {"a": 1}

    record = {"pass1": {"is_correct_pred": True}, "pass2_custom": {"ok": True}}
    assert fhs.select_pass_object(record, "pass1")["is_correct_pred"] is True
    assert fhs.select_pass_object(record, "pass2", pass2_key="pass2_custom") == {"ok": True}
    assert fhs.select_pass_object({}, "pass2", pass2_key="pass2_custom") == {}


def test_extract_correct_entropy_and_sample_idx():
    assert fhs.extract_correct_flag({"is_correct_pred": "true"}) == 1
    assert fhs.extract_correct_flag({"sample": {"correct": 0}}) == 0

    ent_pref = {"entropy_answer": "0.5", "entropy": "0.9"}
    assert fhs.extract_entropy(ent_pref) == pytest.approx(0.5)

    ent_fallback = {"entropy_answer": "bad", "entropy": "0.8"}
    assert fhs.extract_entropy(ent_fallback) == pytest.approx(0.8)
    assert fhs.extract_entropy({"entropy_answer": "bad", "entropy": "nope"}) is None

    rec = {"sample_idx": "abc"}
    pass_obj = {"i": "3"}
    assert fhs.extract_sample_idx(rec, pass_obj) == 3


def test_pass_with_correctness_handles_missing():
    record = {"pass1": {"is_correct_pred": True}}
    pass_obj, corr = fhs.pass_with_correctness(record, "pass1")
    assert pass_obj["is_correct_pred"] is True
    assert corr == 1
    assert fhs.pass_with_correctness({}, "pass1") is None


def test_mcnemar_pvalue_table_fallback(monkeypatch):
    def fake_mcnemar(*args, **kwargs):
        raise TypeError("fail")

    monkeypatch.setattr(fhs, "_statsmodels_mcnemar", fake_mcnemar)
    assert fhs.mcnemar_pvalue_table(1, 5, 1, 0) == pytest.approx(0.2)
    assert fhs.mcnemar_pvalue_table(2, 0, 0, 3) == pytest.approx(1.0)


def test_mcnemar_from_pairs_counts(monkeypatch):
    monkeypatch.setattr(fhs, "_statsmodels_mcnemar", None)
    df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    num00, num01, num10, num11, pval = fhs.mcnemar_from_pairs(df, "a", "b")
    assert (num00, num01, num10, num11) == (1, 1, 1, 1)
    assert 0 <= pval <= 1


def test_paired_t_and_wilcoxon_branches(monkeypatch):
    monkeypatch.setattr(fhs, "_scipy_stats", None)
    assert fhs.paired_t_and_wilcoxon(np.array([1.0, 2.0])) == (None, None)

    class DummyResult:
        def __init__(self, p):
            self.pvalue = p

    class DummyStats:
        def __init__(self):
            self.calls = {}

        def ttest_rel(self, diffs, zeros, nan_policy=None):
            self.calls["ttest_rel"] = (diffs.tolist(), zeros.tolist(), nan_policy)
            return DummyResult(0.123)

        def wilcoxon(self, diffs):
            self.calls["wilcoxon"] = diffs.tolist()
            return DummyResult(0.456)

    dummy = DummyStats()
    monkeypatch.setattr(fhs, "_scipy_stats", dummy)
    t_p, w_p = fhs.paired_t_and_wilcoxon(np.array([0.4, -0.1, 0.0]))
    assert t_p == pytest.approx(0.123)
    assert w_p == pytest.approx(0.456)
    assert dummy.calls["ttest_rel"][2] == "omit"


def test_paired_t_and_wilcoxon_handles_wilcoxon_failure(monkeypatch):
    class DummyResult:
        def __init__(self, p):
            self.pvalue = p

    class DummyStats:
        def ttest_rel(self, diffs, zeros, nan_policy=None):
            return DummyResult(0.9)

        def wilcoxon(self, diffs):
            raise ValueError("bad")

    monkeypatch.setattr(fhs, "_scipy_stats", DummyStats())
    t_p, w_p = fhs.paired_t_and_wilcoxon(np.array([0.2, 0.3]))
    assert t_p == pytest.approx(0.9)
    assert w_p is None


def test_summarize_helpers(monkeypatch):
    merged_any = pd.DataFrame({"any_p1": [0, 1], "any_p2": [1, 1]})
    any_summary = fhs.summarize_cluster_any(merged_any)
    assert any_summary["metric"] == "cluster_any"
    assert any_summary["n_units"] == 2
    assert any_summary["wins_pass2"] == 1
    assert any_summary["wins_pass1"] == 0
    assert any_summary["both_correct"] == 1

    merged_mean = pd.DataFrame({"acc_p1": [0.2, 0.4], "acc_p2": [0.6, 0.4]})
    monkeypatch.setattr(fhs, "paired_t_and_wilcoxon", lambda diffs: (0.1, 0.2))
    mean_summary = fhs.summarize_cluster_mean(merged_mean)
    assert mean_summary["delta_pp"] == pytest.approx(20.0)
    assert mean_summary["p_ttest"] == 0.1
    assert mean_summary["p_wilcoxon"] == 0.2

    pairs_df = pd.DataFrame({"correct1": [1, 0, 1], "correct2": [1, 1, 0]})
    sample_summary = fhs.summarize_sample_level(pairs_df)
    assert sample_summary["metric"] == "sample"
    assert sample_summary["wins_pass2"] == 1
    assert sample_summary["wins_pass1"] == 1
    assert sample_summary["both_correct"] == 1
    assert sample_summary["n_units"] == 3
