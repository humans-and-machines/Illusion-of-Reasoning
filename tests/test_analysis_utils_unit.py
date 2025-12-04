import argparse

import numpy as np
import pandas as pd
import pytest

import src.analysis.utils as utils


def test_nat_step_and_parse_temp_from_dir():
    assert utils.nat_step_from_path(".../step_0123/file.jsonl") == 123
    assert utils.nat_step_from_path("checkpoint-9") == 9
    assert utils.parse_temp_from_dir("temp-low", low_alias=0.2) == pytest.approx(0.2)
    assert utils.parse_temp_from_dir("0.7-temp", low_alias=0.3) == pytest.approx(0.7)


def test_slugify_and_coercions():
    assert utils.slugify("Hello World!") == "Hello_World"
    assert utils.coerce_bool("Yes") == 1 and utils.coerce_bool("no") == 0
    assert utils.coerce_bool(None) is None
    assert utils.truthy_flag("True") is True
    assert utils.coerce_float("3.5") == 3.5 and utils.coerce_float("bad") is None


def test_entropy_helpers_and_uncertainty():
    pass1 = {"entropy": 1.0, "entropy_think": 0.5, "entropy_answer": 0.3}
    assert utils.combined_entropy_from_pass1(pass1) == pytest.approx(1.0)
    assert utils.entropy_from_pass1(pass1, mode="sum") == pytest.approx(0.8)
    assert utils.entropy_from_pass1({"entropy": 2.0}, mode="think") == 2.0
    assert utils.choose_uncertainty({"entropy_answer": 0.2, "entropy": 0.4}, pref="answer") == 0.2
    assert utils.choose_uncertainty({"entropy": 0.4}, pref="think") == 0.4


def test_extract_pass1_and_step_and_aha_flag():
    rec = {"pass1": {"entropy": 1}, "step": "5"}
    p1, step = utils.extract_pass1_and_step(rec, None)
    assert p1 == {"entropy": 1} and step == 5

    flag = utils.get_aha_gpt_flag({"shift_in_reasoning_v1": "true"}, {"rechecked": 0})
    assert flag == 1
    assert utils.get_aha_gpt_flag({}, {}) is None


def test_formal_flags_and_prior_ok():
    thresholds = utils.FormalThresholds(delta1=0.2, delta2=0.3, min_prior_steps=1, delta3=0.05)
    freq = np.array([0.0, 0.1, 0.4])
    rate = np.array([0.0, 0.2, 0.2])
    shift = np.array([0, 1, 1])
    p_plus = np.array([0.0, 0.5, np.nan])

    assert utils.formal_prior_ok(freq, rate, 1, thresholds) is True
    flags = utils.formal_flags_with_gain(freq, rate, shift, p_plus, thresholds)
    assert flags.tolist() == [0, 1, 0]


def test_build_roots_by_temp_from_templates(tmp_path, monkeypatch):
    (tmp_path / "cw_0.7").mkdir()
    (tmp_path / "m_0.7").mkdir()
    roots = utils.build_roots_by_temp_from_templates(
        temps=[0.7],
        crossword_tpl=str(tmp_path / "cw_{T}"),
        math_tpl=str(tmp_path / "m_{T}"),
        carpark_tpl=str(tmp_path / "none_{T}"),
    )
    assert 0.7 in roots and roots[0.7]["Crossword"].endswith("cw_0.7")
    assert "Carpark" not in roots[0.7]


def test_cli_arg_helpers_build_expected_flags():
    parser = argparse.ArgumentParser()
    utils.add_results_root_split_and_output_args(parser, dataset_default="ds", model_default="m")
    args = parser.parse_args(["/root", "--split", "test"])
    assert args.results_root == "/root" and args.dataset_name == "ds"

    parser2 = argparse.ArgumentParser()
    utils.add_temp_scan_args(parser2)
    args2 = parser2.parse_args(["--temps", "0.3", "0.7"])
    assert args2.temps == [0.3, 0.7]


def test_run_module_main_with_argv_and_parse_passes(monkeypatch):
    called = []

    def fake_main():
        called.append(list(sys.argv))

    import sys

    utils.run_module_main_with_argv(fake_main, ["a", "b"], prog="prog")
    assert called[0][0] == "prog" and called[0][1:] == ["a", "b"]

    with pytest.raises(SystemExit):
        utils.parse_passes_argument(" , ")
    assert utils.parse_passes_argument("p1, p2") == ["p1", "p2"]


def test_problem_and_split_helpers():
    rec = {"problem": "p", "split": "train"}
    assert utils.get_problem_id({"sample_idx": 3}) == "sample_3"
    assert utils.problem_key_from_record({}, missing_default="x") == "x"
    assert utils.step_within_bounds(5, min_step=1, max_step=10) is True
    assert utils.record_matches_split(rec, split_value="train") is True
    assert utils.record_matches_split(rec, split_value="test") is False


def test_standardize_uncertainty_with_stats_proxy(monkeypatch):
    # Ensure the lazy import path is exercised.
    monkeypatch.setattr(utils, "_UNCERTAINTY_MODULE", None)
    df = pd.DataFrame({"uncertainty": [1.0, 3.0]})
    standardized, mean_val, std_val = utils.standardize_uncertainty_with_stats(
        df,
        source_col="uncertainty",
        dest_col="unc_std",
    )
    assert "unc_std" in standardized
    assert mean_val == pytest.approx(2.0)
    assert std_val == pytest.approx(1.0)
    assert standardized["unc_std"].tolist() == pytest.approx([-1.0, 1.0])
