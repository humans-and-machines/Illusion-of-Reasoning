from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.core as core_mod
from src.analysis.core import LoadRowsConfig, make_formal_thresholds


def test_make_formal_thresholds_coerces_numbers():
    thresholds = make_formal_thresholds(0.1, 0.2, 3, "0.3")
    assert thresholds.delta1 == 0.1
    assert thresholds.delta2 == 0.2
    assert thresholds.min_prior_steps == 3
    assert thresholds.delta3 == 0.3


def test_compute_correct_and_shift_carpark_and_regular(monkeypatch):
    monkeypatch.setattr(core_mod, "aha_gpt_for_rec", lambda *args, **kwargs: 1)
    carpark_result = core_mod.compute_correct_and_shift(
        "Carpark-foo",
        {"soft_reward": "0.5"},
        {},
        gpt_subset_native=True,
        gpt_keys=["a"],
        carpark_success_fn=lambda reward: 1 if reward is not None else None,
    )
    assert carpark_result == (1, 1)

    monkeypatch.setattr(core_mod, "aha_gpt_for_rec", lambda *args, **kwargs: 0)
    regular_result = core_mod.compute_correct_and_shift(
        "Math",
        {"is_correct_pred": True},
        {},
        gpt_subset_native=False,
        gpt_keys=[],
        carpark_success_fn=lambda _: None,
    )
    assert regular_result == (1, 0)

    missing = core_mod.compute_correct_and_shift(
        "Math",
        {"is_correct_pred": None},
        {},
        gpt_subset_native=False,
        gpt_keys=[],
        carpark_success_fn=lambda _: None,
    )
    assert missing is None


def test_aha_series_for_group_sorts_by_step():
    df = pd.DataFrame(
        {
            "step": [3, 1],
            "freq_correct": [0.5, 0.25],
            "aha_rate_gpt": [0.1, 0.2],
            "aha_any_gpt": [1, 0],
        }
    )
    freq, rate, shift = core_mod.aha_series_for_group(df)
    np.testing.assert_allclose(freq, np.array([0.25, 0.5]))
    np.testing.assert_allclose(rate, np.array([0.2, 0.1]))
    np.testing.assert_array_equal(shift, np.array([0, 1]))


def test_compute_formal_flags_for_group_respects_prior_ok(monkeypatch):
    df = pd.DataFrame(
        {
            "step": [2, 1, 3],
            "freq_correct": [0.4, 0.5, 0.6],
            "aha_rate_gpt": [0.1, 0.1, 0.1],
            "aha_any_gpt": [1, 1, 0],
        }
    )
    thresholds = make_formal_thresholds(0.1, 0.2, 1, None)

    def formal_prior_ok(freq, rate, idx, thresholds_arg):
        assert thresholds_arg == thresholds
        return idx == 1

    flags = core_mod.compute_formal_flags_for_group(df, thresholds, formal_prior_ok)
    np.testing.assert_array_equal(flags, np.array([0, 1, 0]))


def test_add_formal_flags_column_validates_columns():
    bad_frame = pd.DataFrame({"step": [1], "freq_correct": [0.1], "aha_rate_gpt": [0.1]})
    with pytest.raises(ValueError):
        core_mod.add_formal_flags_column(
            bad_frame,
            group_keys=["g"],
            out_column="flag",
            delta1=0.1,
            delta2=0.2,
            min_prior_steps=1,
            formal_prior_ok_fn=lambda *args, **kwargs: True,
        )


def test_add_formal_flags_column_grouped(monkeypatch):
    frame = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2"],
            "step": [2, 1, 1],
            "freq_correct": [0.1, 0.2, 0.3],
            "aha_rate_gpt": [0.1, 0.2, 0.3],
            "aha_any_gpt": [1, 0, 1],
        }
    )

    def fake_prior(freq, rate, idx, thresholds):
        # Flag rows where freq_correct is below 0.15.
        return freq[idx] < 0.15

    result = core_mod.add_formal_flags_column(
        frame,
        group_keys=["group"],
        out_column="flag",
        delta1=0.1,
        delta2=0.2,
        min_prior_steps=1,
        formal_prior_ok_fn=fake_prior,
    )
    assert result[["group", "step"]].to_dict(orient="records") == [
        {"group": "g1", "step": 1},
        {"group": "g1", "step": 2},
        {"group": "g2", "step": 1},
    ]
    np.testing.assert_array_equal(result["flag"].to_numpy(), np.array([0, 1, 0]))


def test_add_standard_formal_flags_passes_formal_prior_ok(monkeypatch):
    recorded_kwargs = {}

    def fake_add_formal_flags(frame, group_keys, **kwargs):
        recorded_kwargs.update(kwargs)
        return frame.assign(marker=1)

    monkeypatch.setattr(core_mod, "add_formal_flags_column", fake_add_formal_flags)
    frame = pd.DataFrame(
        {
            "step": [1],
            "freq_correct": [0.1],
            "aha_rate_gpt": [0.1],
            "aha_any_gpt": [1],
        }
    )
    result = core_mod.add_standard_formal_flags(
        frame,
        group_keys=["problem"],
        out_column="flag",
        delta1=0.1,
        delta2=0.2,
        min_prior_steps=2,
    )
    assert "marker" in result.columns
    assert recorded_kwargs["formal_prior_ok_fn"] == core_mod.formal_prior_ok


def test_iter_pass1_records_filters_and_parses(tmp_path):
    file_path = tmp_path / "step-123.jsonl"
    file_path.write_text('{"a": 1}\n\nnot json\n{"b": 2}\n', encoding="utf-8")
    records = list(core_mod.iter_pass1_records([str(file_path)]))
    assert len(records) == 2
    assert records[0] == (str(file_path), 123, {"a": 1})
    assert records[1][2] == {"b": 2}


def test_build_problem_step_from_samples_basic():
    samples = pd.DataFrame(
        {
            "domain": ["Math", "Math", "Math"],
            "step": [1, 1, 2],
            "problem": ["p1", "p1", "p1"],
            "correct": [1, 0, 1],
            "aha_gpt": [1, 0, 0],
        }
    )
    problem_step = core_mod.build_problem_step_from_samples(samples)
    assert problem_step.columns.tolist() == [
        "domain",
        "step",
        "problem",
        "n_samples",
        "freq_correct",
        "aha_any_gpt",
        "aha_rate_gpt",
        "p_correct_given_shift",
    ]
    assert problem_step.loc[0, "n_samples"] == 2
    assert problem_step.loc[0, "aha_any_gpt"] == 1
    assert problem_step.loc[0, "p_correct_given_shift"] == 1.0


def test_build_problem_step_for_formal_includes_native():
    samples = pd.DataFrame(
        {
            "step": [1, 1],
            "problem": ["p1", "p1"],
            "correct": [0, 1],
            "aha_gpt": [0, 1],
            "aha_words": [1, 0],
        }
    )
    result = core_mod.build_problem_step_for_formal(samples)
    assert set(result.columns) >= {
        "aha_any_native",
        "aha_rate_native",
        "aha_any_gpt",
        "aha_rate_gpt",
    }
    assert result["aha_any_native"].iloc[0] in (0, 1)


def test_mark_formal_pairs_with_gain(monkeypatch):
    data = pd.DataFrame(
        {
            "problem": ["p1", "p1", "p2"],
            "step": [2, 1, 1],
            "freq_correct": [0.5, 0.6, 0.7],
            "aha_rate_gpt": [0.1, 0.2, 0.3],
            "aha_any_gpt": [1, 1, 0],
            "p_correct_given_shift": [0.9, 0.8, 0.7],
        }
    )

    def fake_formal_flags(freq, rate, shift, p_plus, thresholds):
        return np.arange(len(freq)) % 2

    monkeypatch.setattr(core_mod, "formal_flags_with_gain", fake_formal_flags)
    thresholds = core_mod.FormalThresholds(0.1, 0.2, 1, None)
    result = core_mod.mark_formal_pairs_with_gain(data, thresholds)
    assert result[["problem", "step"]].to_dict(orient="records") == [
        {"problem": "p1", "step": 1},
        {"problem": "p1", "step": 2},
        {"problem": "p2", "step": 1},
    ]
    np.testing.assert_array_equal(result["aha_formal_pair"].to_numpy(), np.array([0, 1, 0]))


def test_iter_correct_and_shift_samples(monkeypatch):
    def fake_iter_pass1(files_by_domain, min_step=None, max_step=None):
        yield "Math", {"soft_reward": 0.1}, 1, {"id": 1}
        yield "Math", {"soft_reward": 0.2}, 2, {"id": 2}

    def fake_compute(domain, pass1_data, record, **kwargs):
        return (1, 0) if record["id"] == 1 else None

    monkeypatch.setattr(core_mod, "iter_pass1_samples_by_domain", fake_iter_pass1)
    monkeypatch.setattr(core_mod, "compute_correct_and_shift", fake_compute)
    results = list(
        core_mod.iter_correct_and_shift_samples(
            {"Math": ["file1"]},
            gpt_keys=["k"],
            gpt_subset_native=True,
            min_step=None,
            max_step=None,
            carpark_success_fn=lambda _: None,
        )
    )
    assert results == [("Math", 1, {"id": 1}, 1, 0)]


def test_iter_correct_and_shift_samples_for_config(monkeypatch):
    config = LoadRowsConfig(
        gpt_keys=["k1"],
        gpt_subset_native=True,
        min_step=1,
        max_step=2,
        carpark_success_fn=lambda _: None,
    )

    def fake_iter(files_by_domain, **kwargs):
        return iter([("D", 1, {"r": 1}, 0, 1)])

    monkeypatch.setattr(core_mod, "iter_correct_and_shift_samples", fake_iter)
    assert list(core_mod.iter_correct_and_shift_samples_for_config({}, config)) == [("D", 1, {"r": 1}, 0, 1)]


def test_classify_and_prefer_math_paths():
    assert core_mod._classify_domain_from_dir("xword_temp-0.7") == "Crossword"
    assert core_mod._classify_domain_from_dir("carpark_run") == "Carpark"
    assert core_mod._classify_domain_from_dir("math7b_low-temp") == "Math2"
    assert core_mod._classify_domain_from_dir("low_temp-1.5b") == "Math"
    assert core_mod._classify_domain_from_dir("unknown") is None

    better = "temp-0.7_math_1.5b"
    worse = "temp-0.7_math_llama"
    assert core_mod._prefer_math_path(better, worse) is True
    assert core_mod._prefer_math_path(worse, better) is False


def test_discover_roots_by_temp_prefers_best_math(tmp_path, capsys):
    (tmp_path / "math_llama_temp-0.7").mkdir()
    (tmp_path / "math_1.5b_temp-0.7").mkdir()
    (tmp_path / "carpark_temp-0.7").mkdir()
    (tmp_path / "xword_temp-0.7").mkdir()
    (tmp_path / "skipme_temp-0.7").mkdir()

    mapping = core_mod.discover_roots_by_temp(
        str(tmp_path),
        temps=[0.7],
        low_alias=0.1,
        skip_substrings={"skipme"},
    )
    out = capsys.readouterr().out
    assert "T=0.7" in out
    assert mapping[0.7]["Math"].endswith("math_1.5b_temp-0.7")
    assert mapping[0.7]["Carpark"].endswith("carpark_temp-0.7")
    assert mapping[0.7]["Crossword"].endswith("xword_temp-0.7")


def test_discover_roots_for_temp_args(monkeypatch):
    calls = {}

    def fake_discover(scan_root, temps, low_alias, skip_substrings):
        calls["args"] = (scan_root, tuple(temps), low_alias, set(skip_substrings))
        return {"from": "discover"}

    monkeypatch.setattr(core_mod, "discover_roots_by_temp", fake_discover)
    args = SimpleNamespace(
        scan_root="rootA",
        temps=[0.1],
        low_alias=0.05,
        math3_tpl=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        carpark_tpl=None,
    )
    result = core_mod.discover_roots_for_temp_args(args, skip_set={"skip"}, include_math3=False)
    assert result == {"from": "discover"}
    assert calls["args"] == ("rootA", (0.1,), 0.05, {"skip"})

    def fake_build_roots(**kwargs):
        return {"built": kwargs}

    monkeypatch.setattr(core_mod, "build_roots_by_temp_from_templates", fake_build_roots)
    args.scan_root = None
    args.temps = [0.2]
    args.crossword_tpl = "cw"
    args.math_tpl = "m"
    args.math2_tpl = "m2"
    args.math3_tpl = "m3"
    args.carpark_tpl = "c"
    built = core_mod.discover_roots_for_temp_args(args, skip_set=set(), include_math3=True)
    assert built == {
        "built": {
            "temps": [0.2],
            "crossword_tpl": "cw",
            "math_tpl": "m",
            "math2_tpl": "m2",
            "math3_tpl": "m3",
            "carpark_tpl": "c",
        }
    }


def test_log_discovered_roots_outputs(capsys):
    core_mod._log_discovered_roots({0.7: {"Math": "path/to/math"}})
    out = capsys.readouterr().out
    assert "Math" in out and "0.7" in out


def test_discover_roots_by_temp_skips_unlisted_temps(tmp_path):
    # Directory uses temp-0.8 but requested temps excludes it -> mapping should stay empty.
    (tmp_path / "temp-0.8_math").mkdir()
    mapping = core_mod.discover_roots_by_temp(
        str(tmp_path),
        temps=[0.7],
        low_alias=0.1,
        skip_substrings=set(),
    )
    assert mapping == {}


def test_discover_roots_by_temp_skips_unknown_domain(tmp_path):
    # Temperature matches but domain classification fails -> no entry.
    (tmp_path / "temp-0.7_unknown").mkdir()
    mapping = core_mod.discover_roots_by_temp(
        str(tmp_path),
        temps=[0.7],
        low_alias=0.1,
        skip_substrings=set(),
    )
    assert mapping == {}


def test_discover_roots_by_temp_mixed_domains(tmp_path, capsys):
    (tmp_path / "math_temp-0.7").mkdir()
    (tmp_path / "unknown_temp-0.7").mkdir()
    mapping = core_mod.discover_roots_by_temp(
        str(tmp_path),
        temps=[0.7],
        low_alias=0.1,
        skip_substrings=set(),
    )
    assert mapping[0.7]["Math"].endswith("math_temp-0.7")
    out = capsys.readouterr().out
    assert "T=0.7" in out
