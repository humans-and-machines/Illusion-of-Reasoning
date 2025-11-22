#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re

import pytest

text_utils = pytest.importorskip("src.inference.utils.text_utils")
math_utils = pytest.importorskip("src.inference.utils.math_pass_utils")


def test_find_markers_and_context_with_prefix_skip():
    think = "Intro text. Wait, we need to reconsider this plan."
    prompt = "Problem: x+1=2"
    pattern = re.compile(r"wait", re.I)

    markers, pos, context, excerpt = text_utils.find_markers_and_context(
        think_text=think,
        prompt_text=prompt,
        patterns=[("wait_marker", pattern)],
        skip_prefix_chars=0,
    )

    assert markers == ["wait_marker"]
    assert isinstance(pos, int) and pos >= 0
    assert prompt in context
    assert "wait" in excerpt.lower()


def test_extract_blocks_and_valid_tag_structure_roundtrip():
    full = "prefix <think> some reasoning </think> mid <answer> 42 </answer> tail"
    think, ans = math_utils.extract_blocks(full)
    assert think == "some reasoning"
    assert ans == "42"
    assert math_utils.valid_tag_structure(full) is True

    invalid = "<think>oops</think><think>dup</think><answer>1</answer>"
    assert math_utils.valid_tag_structure(invalid) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        (r"\frac{1}{2}", "1/2"),
        (r" (\pi + 1) ", "pi+1"),
        (r"{-1}", "-1"),
        (r"1//2", "1/2"),
        (r"+3", "3"),
    ],
)
def test_canon_math_normalizes_common_forms(raw, expected):
    assert math_utils.canon_math(raw) == expected


def test_contains_canon_uses_canonicalized_strings():
    hay = "The final answer is 1/2."
    needle = "1/2"
    hay_c = math_utils.canon_math(hay)
    needle_c = math_utils.canon_math(needle)
    assert math_utils.contains_canon(hay_c, needle_c) is True


def test_finite_mean_ignores_nans_and_infinities():
    values = [1.0, float("nan"), float("inf"), 3.0]
    mean_val = math_utils.finite_mean(values)
    assert mean_val == pytest.approx(2.0)

    assert math_utils.finite_mean([float("nan"), float("inf")]) is None


def test_build_second_pass_cue_strings_splits_and_trims():
    phrase = " first cue  ||| second cue "
    cues = math_utils.build_second_pass_cue_strings(phrase)
    assert cues == ["first cue ", "second cue "]
    assert math_utils.build_second_pass_cue_strings("") == []
    assert math_utils.build_second_pass_cue_strings(None) == []


def test_pack_math_pass_result_includes_reconsider_markers_and_correctness():
    full_text = (
        "<think>Wait, we need to reconsider. Let's think this through step by step.</think>"
        "<answer> 42 </answer>"
    )
    ent_think = [0.5, 0.6]
    ent_answer = [0.4]

    meta = math_utils.build_math_pass_meta(
        problem="x+2=44",
        canon_gold=math_utils.canon_math("42"),
        injected_cue=False,
        prev_output=None,
        cue_prefix_str="",
        stop_reason_think="eos",
        stop_reason_answer="eos",
    )
    result = math_utils.pack_math_pass_result(
        full_text=full_text,
        ent_think=ent_think,
        ent_answer=ent_answer,
        meta=meta,
    )

    assert result["pred_answer"] == "42"
    assert result["is_correct_pred"] is True
    assert result["tokens_total"] == len(ent_think) + len(ent_answer)
    assert isinstance(result["entropy"], (float, type(None)))
    assert result["has_reconsider_cue"] in (True, False)
    # When reconsideration markers are present, they should be a list.
    assert isinstance(result["reconsider_markers"], list)
