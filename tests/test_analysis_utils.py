#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import re
import sys

import numpy as np
import pytest

import src.analysis.utils as utils


def test_step_parsing_and_temp_parsing():
    assert utils.nat_step_from_path(".../step-123/file.jsonl") == 123
    assert utils.nat_step_from_path("checkpoint_456/") == 456
    assert utils.nat_step_from_path("no_step") is None

    rec = {"step": "7"}
    assert utils.step_from_rec_or_path(rec, "path") == 7
    rec2 = {}
    assert utils.step_from_rec_or_path(rec2, "step-3") == 3

    assert utils.parse_temp_from_dir("temp-0.7", low_alias=0.05) == 0.7
    assert utils.parse_temp_from_dir("low-temp", low_alias=0.1) == 0.1
    assert utils.parse_temp_from_dir("no-temp-here", low_alias=0.1) is None


def test_slugify_and_coercions():
    assert utils.slugify("Hello, World!") == "Hello_World"
    assert utils.coerce_bool("Yes") == 1
    assert utils.coerce_bool("0") == 0
    assert utils.coerce_bool(None) is None
    assert utils.truthy_flag("true")
    assert not utils.truthy_flag(0.0)
    assert utils.coerce_float("1.23") == pytest.approx(1.23)
    assert utils.coerce_float("x") is None


def test_nat_step_from_path_group1_fallback_and_value_error(monkeypatch):
    # Patterns without a named group force the group(1) fallback and still handle
    # ValueError before returning a later match.
    monkeypatch.setattr(
        utils,
        "STEP_PATS",
        [
            re.compile(r"step-([A-Za-z]+)"),  # matches but int() raises ValueError
            re.compile(r"step(\d+)"),  # fallback uses group(1) -> succeeds
        ],
    )
    assert utils.nat_step_from_path("dir/step-bad/then/step5.jsonl") == 5


def test_both_get_lighten_entropy_helpers():
    record = {"a": 2}
    pass1 = {"a": None}
    assert utils.both_get(pass1, record, "a", default=5) == 2

    light = utils.lighten_hex("#000000", factor=0.5)
    assert light == "#7f7f7f"

    p1 = {"entropy": "2.0", "entropy_think": "1.0", "entropy_answer": "3.0"}
    assert utils.combined_entropy_from_pass1(p1) == pytest.approx(2.0)
    assert utils.entropy_from_pass1(p1, mode="sum") == pytest.approx(4.0)
    assert utils.entropy_from_pass1({}, mode="combined") is None
    assert utils.choose_uncertainty({"entropy_answer": 0.0}, pref="answer") == 0.0
    assert utils.choose_uncertainty({"entropy": 1.2}, pref="overall") == 1.2
    assert utils.choose_uncertainty({"entropy_think": 1.3}, pref="think") == 1.3


def test_extract_pass1_and_step(monkeypatch):
    rec = {"pass1": {"x": 1}, "step": "5"}
    p1, step = utils.extract_pass1_and_step(rec, step_from_name=None)
    assert p1["x"] == 1 and step == 5

    rec_bad = {"p1": {}, "step": "bad"}
    p1b, stepb = utils.extract_pass1_and_step(rec_bad, step_from_name=None)
    assert p1b == {} and stepb is None

    rec_name = {"first_pass": {"y": 2}}
    p1c, stepc = utils.extract_pass1_and_step(rec_name, step_from_name=9)
    assert stepc == 9


def test_formal_flags_and_apply():
    freq = np.array([0.0, 0.1, 0.2, 0.05])
    rate = np.array([0.0, 0.05, 0.1, 0.01])
    shift = np.array([0, 1, 1, 1])
    thresholds = utils.FormalThresholds(delta1=0.15, delta2=0.2, min_prior_steps=1)

    assert not utils.formal_prior_ok(freq, rate, 0, thresholds)
    flags = utils.formal_flags_from_series(freq, rate, shift, thresholds)
    assert list(flags) == [0, 1, 1, 0]

    p_plus = np.array([0.0, 0.1, 0.5, 0.2])
    thresholds_gain = utils.FormalThresholds(delta1=0.15, delta2=0.2, min_prior_steps=1, delta3=0.05)
    flags_gain = utils.formal_flags_with_gain(freq, rate, shift, p_plus, thresholds_gain)
    assert list(flags_gain) == [0, 0, 1, 0]

    # apply_formal_marking forwards arguments to provided function
    called = {}

    def marker(df, delta1, delta2, min_prior_steps):
        called["args"] = (delta1, delta2, min_prior_steps)
        return "ok"

    assert utils.apply_formal_marking("df", delta1=0.1, delta2=0.2, min_prior_steps=2, mark_formal_fn=marker) == "ok"
    assert called["args"] == (0.1, 0.2, 2)


def test_gpt_keys_and_flags():
    assert "pivot_llm" in utils.gpt_keys_for_mode("broad")
    assert utils.gpt_keys_for_mode("canonical") == ["change_way_of_thinking", "shift_in_reasoning_v1"]

    pass1 = {"shift_in_reasoning_v1": "yes"}
    record = {"rechecked": False}
    assert utils.get_aha_gpt_flag(pass1, record) == 1
    assert utils.get_aha_gpt_flag({}, {"rechecked": "no"}) == 0
    assert utils.get_aha_gpt_flag({}, {}) is None


def test_roots_by_temp_and_step_filters(tmp_path):
    (tmp_path / "t0.0_math").mkdir()
    (tmp_path / "t0.0_carpark").mkdir()
    mapping = utils.build_roots_by_temp_from_templates(
        [0.0],
        math_tpl=str(tmp_path / "t{T}_math"),
        carpark_tpl=str(tmp_path / "t{T}_carpark"),
    )
    assert 0.0 in mapping
    assert mapping[0.0]["Math"].endswith("t0.0_math")
    assert mapping[0.0]["Carpark"].endswith("t0.0_carpark")

    assert utils.step_within_bounds(5, min_step=None, max_step=10)
    assert not utils.step_within_bounds(11, min_step=None, max_step=10)

    rec = {"split": "test", "sample_idx": 1}
    assert utils.record_matches_split(rec, "test")
    assert not utils.record_matches_split(rec, "train")

    # step_from_record_if_within_bounds combines split and bounds checks
    rec_step = {"split": "test", "step": 3}
    assert utils.step_from_record_if_within_bounds(rec_step, "p/step-3", "test", 1, 5) == 3
    assert utils.step_from_record_if_within_bounds(rec_step, "p/step-3", "train", 1, 5) is None


def test_strings_and_ids_helpers():
    assert utils.first_nonempty_str("", None, "  hi ") == "hi"
    assert utils.first_nonempty_str(None, "   ") is None

    assert utils.get_problem_id({"problem": "p"}) == "p"
    assert utils.get_problem_id({"sample_idx": 2}) == "sample_2"
    assert utils.problem_key_from_record({"dataset_index": 7}, missing_default="x") == "idx:7"
    assert utils.problem_key_from_record({}, missing_default="x") == "x"

    assert utils.canon_equal(" ans ", "ans") == 1
    assert utils.canon_equal("no", ["yes", "no"]) == 1
    assert utils.canon_equal(None, "x") is None
    assert utils.canon_equal("x", 123) is None


def test_cli_arg_helpers_and_run_module(monkeypatch):
    parser = utils.build_mixed_root_arg_parser()
    args = parser.parse_args(["--split", "test", "--dataset_name", "D", "--model_name", "M"])
    assert args.split == "test"
    assert args.dataset_name == "D"

    argv = utils.build_results_root_argv("root", split="dev")
    assert argv == ["root", "--split", "dev"]

    sys_argv_before = list(sys.argv)
    called = {}

    def fake_main():
        called["argv"] = list(sys.argv)

    utils.run_module_main_with_argv(fake_main, ["a", "b"], prog="prog")
    assert called["argv"][0] == "prog"
    assert sys.argv == sys_argv_before

    # parse passes
    assert utils.parse_passes_argument("p1,p2") == ["p1", "p2"]
    with pytest.raises(SystemExit):
        utils.parse_passes_argument(" ,, ")


def test_load_legacy_main_from_path(tmp_path):
    script = tmp_path / "legacy.py"
    script.write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    legacy_main = utils.load_legacy_main_from_path(script, "legacy_mod")
    assert legacy_main() == "ok"

    missing = tmp_path / "missing.py"
    with pytest.raises(SystemExit):
        utils.load_legacy_main_from_path(missing, "missing")

    bad_script = tmp_path / "bad.py"
    bad_script.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        utils.load_legacy_main_from_path(bad_script, "bad_mod")


def test_edge_cases_for_parsing_and_coercion(monkeypatch):
    # nat_step_from_path continues after ValueError (non-numeric match)
    assert utils.nat_step_from_path("path/step-notanumber/file") is None
    # step_from_rec_or_path handles non-castable steps
    assert utils.step_from_rec_or_path({"step": object()}, "fallback") == 0

    # Force parse_temp_from_dir through the ValueError path by monkeypatching patterns.
    monkeypatch.setattr(utils, "TEMP_PATS", [re.compile(r"(?P<t>bad)")])
    assert utils.parse_temp_from_dir("bad", low_alias=0.0) is None

    # truthy_flag returns False for non-standard types
    assert not utils.truthy_flag({"a": 1})

    # combined_entropy_from_pass1 averages think/answer when overall missing
    p1 = {"entropy_think": 1.0, "entropy_answer": 3.0}
    assert utils.combined_entropy_from_pass1(p1) == pytest.approx(2.0)

    # entropy_from_pass1 fallbacks
    assert utils.entropy_from_pass1({"entropy": "1.1"}, mode="sum") == pytest.approx(1.1)
    assert utils.entropy_from_pass1({"entropy": 2.0}, mode="answer") == pytest.approx(2.0)
    assert utils.entropy_from_pass1({"entropy": 0.5}, mode="other") == pytest.approx(0.5)

    # choose_uncertainty fallbacks and unknown pref
    assert utils.choose_uncertainty({"entropy_answer": "1.5"}, pref="overall") == pytest.approx(1.5)
    assert utils.choose_uncertainty({"entropy_answer": 0.7}, pref="think") == pytest.approx(0.7)
    assert utils.choose_uncertainty({"entropy": 1.0}, pref="unknown") is None


def test_extract_pass1_and_step_edge_cases():
    # Non-dict pass1 collapses to {} and yields None step
    p1, step = utils.extract_pass1_and_step({"pass1": ["bad"], "step": 1}, None)
    assert p1 == {} and step is None

    # Missing step even with pass1 present
    p1, step = utils.extract_pass1_and_step({"pass1": {"x": 1}}, None)
    assert p1 == {} and step is None

    # Invalid step value returns ({}, None)
    p1, step = utils.extract_pass1_and_step({"pass1": {"x": 1}, "step": "NaN"}, None)
    assert p1 == {} and step is None


def test_build_roots_by_temp_math_variants(tmp_path):
    (tmp_path / "t0.5_math2").mkdir()
    (tmp_path / "t0.5_math3").mkdir()
    mapping = utils.build_roots_by_temp_from_templates(
        [0.5],
        math2_tpl=str(tmp_path / "t{T}_math2"),
        math3_tpl=str(tmp_path / "t{T}_math3"),
    )
    assert 0.5 in mapping
    assert mapping[0.5]["Math2"].endswith("t0.5_math2")
    assert mapping[0.5]["Math3"].endswith("t0.5_math3")


def test_optional_and_output_arg_helpers():
    parser = argparse.ArgumentParser()
    utils.add_optional_results_root_and_split_args(parser)
    args = parser.parse_args([])
    assert args.results_root is None and args.split is None

    parser2 = argparse.ArgumentParser()
    utils.add_results_root_split_and_output_args(
        parser2,
        dataset_default="D",
        model_default="M",
        results_root_optional=True,
    )
    args2 = parser2.parse_args([])
    assert args2.results_root is None
    assert args2.dataset_name == "D" and args2.model_name == "M"


def test_split_and_gpt_related_arg_helpers():
    parser = argparse.ArgumentParser()
    utils.add_split_arg(parser)
    args = parser.parse_args([])
    assert args.split == "test"

    parser2 = argparse.ArgumentParser()
    utils.add_split_and_gpt_mode_args(parser2, split_default=None)
    args2 = parser2.parse_args([])
    assert args2.split is None and args2.gpt_mode == "canonical"

    parser3 = argparse.ArgumentParser()
    utils.add_gpt_step_and_carpark_args(parser3)
    args3 = parser3.parse_args([])
    assert args3.no_gpt_subset_native is False
    assert args3.min_step is None and args3.max_step is None
    assert args3.carpark_success_op == "gt" and args3.carpark_soft_threshold == 0.0


def test_load_legacy_main_missing_loader(monkeypatch, tmp_path):
    dummy = tmp_path / "dummy.py"
    dummy.write_text("x = 1\n", encoding="utf-8")
    # Force spec.loader to be None to hit the guarded branch.
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda name, path: type("Spec", (), {"loader": None}),
    )
    with pytest.raises(SystemExit):
        utils.load_legacy_main_from_path(dummy, "dummy")


def test_split_bounds_helpers_edge_cases():
    assert not utils.step_within_bounds(0, min_step=1, max_step=None)
    assert utils.record_matches_split({}, None) is True
    assert utils.record_matches_split({"split": ""}, "train") is True

    record = {"step": 2}
    assert utils.step_from_record_if_within_bounds(record, "any", None, 1, 3) == 2


def test_nat_step_from_path_recovers_after_bad_match():
    path = "weird/step-nan/global_step-5/file"
    assert utils.nat_step_from_path(path) == 5


def test_choose_uncertainty_overall_falls_back_to_think():
    assert utils.choose_uncertainty({"entropy_think": 0.9}, pref="overall") == pytest.approx(0.9)


def test_step_from_record_if_within_bounds_out_of_range():
    record = {"step": 10}
    assert (
        utils.step_from_record_if_within_bounds(record, "p/step-10", split_value=None, min_step=None, max_step=5)
        is None
    )
