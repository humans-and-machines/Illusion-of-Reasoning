import numpy as np
import pandas as pd
import pytest

import src.analysis.core.figure_1_data as f1d


def test_problem_identifier_prefers_problem_and_fallbacks():
    assert f1d._problem_identifier({"problem": "p1"}) == "p1"
    assert f1d._problem_identifier({"clue": "c1"}) == "c1"
    assert f1d._problem_identifier({"row_key": 123}) == "123"
    assert f1d._problem_identifier({"dataset_index": 7}) == "idx:7"


def test_row_from_pass1_record_filters_missing(monkeypatch):
    monkeypatch.setattr(f1d, "aha_words", lambda p1: 1)
    monkeypatch.setattr(f1d, "aha_gpt_for_rec", lambda *args, **kwargs: 0)
    rec = {"pass1": {"is_correct_pred": True}, "step": 1, "problem": "p"}
    row = f1d._row_from_pass1_record(rec, "Math", None, ["k"], True)
    assert row["problem"] == "p" and row["aha_native"] == 1

    assert f1d._row_from_pass1_record({"pass1": {}}, "Math", None, [], True) is None
    assert f1d._row_from_pass1_record({"pass1": {"is_correct_pred": None}}, "Math", None, [], True) is None


def test_build_problem_step_and_formal_flags():
    df = pd.DataFrame(
        {
            "domain": ["Math", "Math"],
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "correct": [0, 1],
            "aha_gpt": [0, 1],
            "aha_native": [0, 0],
        },
    )
    ps = f1d.build_problem_step(df)
    assert list(ps["n_samples"]) == [1, 1]
    assert ps.loc[ps["step"] == 2, "p_correct_given_shift"].iloc[0] == 1.0

    formal = f1d.mark_formal_pairs(ps, delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=None)
    assert "aha_formal" in formal.columns
    assert formal["aha_formal"].tolist() == [0, 1]


def test_bootstrap_problem_ratio_handles_missing_and_present():
    df = pd.DataFrame({"step": [1, 2], "aha_any_gpt": [0, 1]})
    res = f1d.bootstrap_problem_ratio(df, "aha_any_gpt", num_bootstrap=0, seed=1)
    assert res.loc[res["step"] == 2, "ratio"].item() == 1.0
    assert np.isnan(res.loc[res["step"] == 1, "lo"]).all()

    with pytest.raises(KeyError):
        f1d.bootstrap_problem_ratio(df, "missing_col")


def test_weighted_trend_and_fit_trend_wls():
    df = pd.DataFrame({"step": [1, 2, 3], "ratio": [0.1, 0.2, 0.3], "n": [1, 1, 1]})
    slope, intercept, slope_k, delta_range, r2, x_fit, y_fit = f1d.fit_trend_wls(df)
    assert slope_k == pytest.approx(slope * 1000)
    assert len(x_fit) == len(y_fit) == 200
    # Degenerate path: zero weights
    df_bad = pd.DataFrame({"step": [1], "ratio": [0.1], "n": [0]})
    slope2, _, _, _, _, x2, y2 = f1d.fit_trend_wls(df_bad)
    assert np.isnan(slope2) and x2.size == 0 and y2.size == 0


def test_positive_delta_flags_and_iter_helper():
    df = pd.DataFrame(
        {
            "step": [1, 1],
            "domain": ["Math", "Math"],
            "aha_any_gpt": [1, 1],
            "p_correct_given_shift": [0.5, 0.7],
            "freq_correct": [0.2, 0.1],
        },
    )
    flags = f1d.build_positive_delta_flags(df)
    assert flags["Math"][1] is True

    df_nodom = pd.DataFrame(
        {
            "step": [2],
            "aha_any_gpt": [0],
            "p_correct_given_shift": [np.nan],
            "freq_correct": [0.0],
        },
    )
    flags2 = f1d.build_positive_delta_flags(df_nodom)
    assert flags2["All"][2] is False


def test_parse_float_list_parses_and_strips():
    assert f1d.parse_float_list("0.1, 0.2 ,") == [0.1, 0.2]


def test_scan_files_delegates(monkeypatch):
    monkeypatch.setattr(f1d, "scan_jsonl_files", lambda root, split_substr=None: [f"{root}/a.jsonl"])
    assert f1d.scan_files("rootdir", "test") == ["rootdir/a.jsonl"]


def test_bootstrap_problem_ratio_handles_empty_group():
    class DummyFrame:
        columns = {"aha_any_gpt"}

        def groupby(self, key):
            return [(1, pd.DataFrame({"aha_any_gpt": []}))]

    res = f1d.bootstrap_problem_ratio(DummyFrame(), "aha_any_gpt", num_bootstrap=1, seed=0)
    assert res.iloc[0]["n"] == 0
    assert np.isnan(res.iloc[0]["ratio"])


def test_weighted_trend_metrics_nan():
    series = f1d.TrendSeries(
        step=np.array([1.0, 2.0]),
        ratio=np.array([0.1, 0.2]),
        weights=np.array([1.0, 1.0]),
        ratio_mean=0.15,
    )
    vals = f1d._weighted_trend_metrics(series, slope=np.nan, intercept=np.nan)
    assert all(np.isnan(v) for v in vals)
