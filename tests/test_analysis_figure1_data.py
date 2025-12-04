import numpy as np
import pandas as pd
import pytest

from src.analysis.core import figure_1_data as f1


def test_problem_identifier_fallbacks():
    assert f1._problem_identifier({"problem": "p1"}) == "p1"
    assert f1._problem_identifier({"clue": 5}) == "5"
    assert f1._problem_identifier({"dataset_index": 7}) == "idx:7"
    assert f1._problem_identifier({}) == "unknown"


def test_row_from_pass1_record_validates_and_builds(monkeypatch):
    # Reject non-dict/empty cases
    assert f1._row_from_pass1_record(None, "Math", None, [], False) is None
    assert f1._row_from_pass1_record({"pass1": {}}, "Math", None, [], False) is None
    assert f1._row_from_pass1_record({"pass1": {"is_correct_pred": 1}}, "Math", None, [], False) is None
    assert f1._row_from_pass1_record({"pass1": {"is_correct_pred": None}, "step": 1}, "Math", None, [], False) is None

    monkeypatch.setattr(f1, "aha_words", lambda pass1: 1)
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *args, **kwargs: 0)
    rec = {"problem": "p1", "step": 2, "pass1": {"is_correct_pred": True}}
    row = f1._row_from_pass1_record(rec, "Math", None, [], False)
    assert row == {
        "domain": "Math",
        "step": 2,
        "problem": "p1",
        "aha_native": 1,
        "aha_gpt": 0,
        "correct": 1,
    }


def test_load_pass1_samples_multi(monkeypatch):
    # Empty yields SystemExit
    with pytest.raises(SystemExit):
        f1.load_pass1_samples_multi({"Math": []}, [], False)

    rec = {"problem": "p", "step": 1, "pass1": {"is_correct_pred": 1}}
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(f1, "iter_records_from_file", lambda path: [rec])
    monkeypatch.setattr(f1, "aha_words", lambda pass1: 0)
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *args, **kwargs: 1)
    df = f1.load_pass1_samples_multi({"Math": ["step1.jsonl"]}, [], False)
    assert df.iloc[0].to_dict() == {
        "domain": "Math",
        "step": 1,
        "problem": "p",
        "aha_native": 0,
        "aha_gpt": 1,
        "correct": 1,
    }


def test_build_problem_step_and_formal_flags():
    samples = pd.DataFrame(
        {
            "domain": ["Math", "Math"],
            "step": [1, 2],
            "problem": ["p1", "p1"],
            "aha_gpt": [0, 1],
            "aha_native": [0, 1],
            "correct": [0, 1],
        }
    )
    problem_step = f1.build_problem_step(samples)
    assert problem_step.loc[0, "n_samples"] == 1
    assert np.isnan(problem_step.loc[0, "p_correct_given_shift"])

    with pytest.raises(ValueError):
        f1.mark_formal_pairs(problem_step.drop(columns=["freq_correct"]))

    formal_ready = pd.DataFrame(
        {
            "domain": ["Math", "Math"],
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "freq_correct": [0.1, 0.15],
            "aha_rate_gpt": [0.0, 0.0],
            "aha_any_gpt": [0, 1],
            "p_correct_given_shift": [np.nan, 0.5],
        }
    )
    criteria_kwargs = dict(delta1=0.2, delta2=0.2, min_prior_steps=1, delta3=0.2)
    marked = f1.mark_formal_pairs(formal_ready, **criteria_kwargs)
    assert marked["aha_formal"].tolist() == [0, 1]


def test_bootstrap_and_trend_helpers():
    # _bootstrap_ratio_interval edge and normal cases
    rng = np.random.default_rng(0)
    lo, hi = f1._bootstrap_ratio_interval(np.array([1]), num_bootstrap=10, rng=rng)
    assert np.isnan(lo) and np.isnan(hi)
    lo2, hi2 = f1._bootstrap_ratio_interval(np.array([1, 0, 1]), num_bootstrap=50, rng=np.random.default_rng(1))
    assert 0.0 <= lo2 <= hi2 <= 1.0

    df = pd.DataFrame({"step": [1, 2], "aha_any_gpt": [1, 0]})
    with pytest.raises(KeyError):
        f1.bootstrap_problem_ratio(df, "missing")
    ratios = f1.bootstrap_problem_ratio(df.assign(dummy=[1, 0]), column="dummy", num_bootstrap=0)
    assert ratios["lo"].isna().all() and ratios["hi"].isna().all()

    small = pd.DataFrame({"step": [1], "ratio": [0.5], "n": [1]})
    slope, intercept, slope_1k, delta_range, r2, xfit, yfit = f1.fit_trend_wls(small)
    assert np.isnan(slope) and xfit.size == 1

    two = pd.DataFrame({"step": [1, 2], "ratio": [0.4, 0.6], "n": [2, 2]})
    slope, intercept, slope_1k, delta_range, r2, xfit, yfit = f1.fit_trend_wls(two)
    assert slope > 0 and intercept < 1 and np.all(np.isfinite([slope_1k, delta_range, r2]))


def test_positive_delta_and_flags():
    subset = pd.DataFrame(
        {
            "freq_correct": [0.3, 0.4],
            "aha_any_gpt": [1, 1],
            "p_correct_given_shift": [0.5, 0.7],
        }
    )
    assert f1._positive_delta_flag(subset) is True
    subset_bad = subset.copy()
    subset_bad["aha_any_gpt"] = [0, 0]
    assert f1._positive_delta_flag(subset_bad) is False

    problem_step = pd.DataFrame(
        {
            "step": [1, 1],
            "problem": ["p1", "p2"],
            "freq_correct": [0.1, 0.2],
            "aha_any_gpt": [1, 0],
            "p_correct_given_shift": [0.4, np.nan],
        }
    )
    flags = f1.build_positive_delta_flags(problem_step)
    assert set(flags.keys()) == {"All"}
    assert flags["All"][1] is True


def test_parse_float_list():
    assert f1.parse_float_list("1.0, 2 , , 3") == [1.0, 2.0, 3.0]
