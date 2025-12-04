import numpy as np

import src.analysis.forced_aha_shared as fah


def test_extract_correct_flag_returns_none_for_missing_fields():
    # No correctness fields at top level or nested → None
    assert fah.extract_correct_flag({"sample": {"foo": 1}}) is None


def test_extract_sample_idx_handles_bad_values_and_returns_none():
    rec = {"idx": "not-an-int"}
    pass_obj = {"i": "NaN"}
    assert fah.extract_sample_idx(rec, pass_obj) is None


def test_select_pass_object_with_pass2_key_non_dict_returns_candidate():
    record = {"custom": "fallback"}
    selected = fah.select_pass_object(record, variant="pass2", pass2_key="custom")
    assert selected == "fallback"


def test_pass_with_correctness_stops_when_flag_missing():
    record = {"pass2": {"foo": "bar"}}
    assert fah.pass_with_correctness(record, "pass2") is None


def test_mcnemar_pvalue_table_fallback_threshold(monkeypatch):
    monkeypatch.setattr(fah, "_statsmodels_mcnemar", None)
    # Off-diagonal counts make chi^2 land between thresholds → approx_p changes
    p_val = fah.mcnemar_pvalue_table(count_00=0, count_01=15, count_10=1, count_11=0)
    assert p_val == 0.01  # triggered at chi2 >= 6.63


def test_paired_t_and_wilcoxon_short_vector_returns_none(monkeypatch):
    fake_stats = type(
        "FakeStats",
        (),
        {
            "ttest_rel": lambda *a, **k: type("res", (), {"pvalue": 0.5}),
            "wilcoxon": lambda *a, **k: type("res", (), {"pvalue": 0.7}),
        },
    )
    monkeypatch.setattr(fah, "_scipy_stats", fake_stats)
    diffs = np.array([0.1])  # size < 2 triggers early return
    t_p, w_p = fah.paired_t_and_wilcoxon(diffs)
    assert t_p is None and w_p is None
