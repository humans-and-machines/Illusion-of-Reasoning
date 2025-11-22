#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


core_mod = pytest.importorskip("src.inference.domains.summarize.summarize_inference_core")


def test_maybe_recompute_correctness_modes():
    maybe = core_mod.maybe_recompute_correctness

    pass_data = {"pred_answer_canon": "abc_def"}

    # "none" and "original" should never override.
    assert maybe(pass_data, "abc", "none") is None
    assert maybe(pass_data, "abc", "original") is None

    # substring mode
    assert maybe(pass_data, "abc", "substring") is True
    assert maybe(pass_data, "xyz", "substring") is False

    # exact mode
    exact_data = {"pred_answer_canon": "gold"}
    assert maybe(exact_data, "gold", "exact") is True
    assert maybe(exact_data, "gold_", "exact") is False

    # gold_canon None ⇒ no override
    assert maybe(pass_data, None, "substring") is None

    # non-string pred_answer_canon ⇒ no override
    assert maybe({"pred_answer_canon": 123}, "123", "exact") is None

    # unrecognized mode ⇒ no override
    assert maybe(pass_data, "abc", "weird_mode") is None


def _build_sample_record(correct_p1: bool, correct_p2: bool, improved_p2: bool):
    pass1 = {
        "is_correct_pred": correct_p1,
        "pred_answer_canon": "p1_pred",
        "entropy": 1.0,
        "entropy_think": 0.5,
        "entropy_answer": 0.7,
        "tokens_think": 10,
        "tokens_answer": 5,
        "stop_reason_think": "stop_token",
        "stop_reason_answer": "stop_token",
        "valid_tag_structure": True,
        "has_reconsider_cue": False,
        "soft_reward": 0.1,
    }
    pass2 = {
        "is_correct_pred": correct_p2,
        "pred_answer_canon": "p2_pred",
        "entropy": 2.0,
        "entropy_think": 1.0,
        "entropy_answer": 1.0,
        "tokens_think": 12,
        "tokens_answer": 6,
        "stop_reason_think": "stop_token",
        "stop_reason_answer": "stop_token",
        "valid_tag_structure": True,
        "has_reconsider_cue": True,
        "improved_over_pass1": improved_p2,
        "soft_reward": 0.9,
    }
    return {
        "problem": "P1",
        "gold_answer_canon": "gold",
        "pass1": pass1,
        "pass2": pass2,
    }


def test_stepagg_add_record_and_finalize_updates_counts():
    StepAgg = core_mod.StepAgg

    agg = StepAgg(step=1)
    record = _build_sample_record(correct_p1=False, correct_p2=True, improved_p2=True)

    agg.add_record(record, recompute_mode="none")
    agg.finalize()

    # Pass counts
    assert agg.pass1.n_samples == 1
    assert agg.pass2.n_samples == 1

    # pass1 incorrect, pass2 correct
    assert agg.pass1.sample_correct == 0
    assert agg.pass2.sample_correct == 1
    assert agg.pass1.correct_by_problem["P1"] is False
    assert agg.pass2.correct_by_problem["P1"] is True

    # Tag and reconsider counts
    assert agg.pass1.tag_ok == 1
    assert agg.pass2.tag_ok == 1
    assert agg.pass1.reconsider_numer == 0
    assert agg.pass2.reconsider_numer == 1

    # Soft rewards and per-problem storage
    assert agg.pass1.soft_values == [0.1]
    assert agg.pass2.soft_values == [0.9]
    assert agg.pass1.soft_by_problem["P1"] == [0.1]
    assert agg.pass2.soft_by_problem["P1"] == [0.9]

    # Tokens and entropies captured
    assert agg.pass1.tokens_think == [10]
    assert agg.pass1.tokens_answer == [5]
    assert agg.pass2.tokens_think == [12]
    assert agg.pass2.tokens_answer == [6]
    assert agg.pass1.entropy_all == [1.0]
    assert agg.pass2.entropy_all == [2.0]

    # Improvement: pass2 correct when pass1 is not → ex_improved_p2=1, sample_improved=1.
    assert agg.pass2.sample_improved == 1
    assert agg.ex_improved_p2 == 1

    # Row/footer text should be non-empty and mention the step number.
    row = agg.row_text()
    footer = agg.footer_text()
    assert str(agg.step) in row
    assert "examples:" in footer
