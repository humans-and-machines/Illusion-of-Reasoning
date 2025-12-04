#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


router_mod = pytest.importorskip("src.training.grpo_rewards_router")
rush_rewards_mod = pytest.importorskip("src.training.rush_rewards")


def test_reward_router_routes_by_task_label(monkeypatch):
    # Stub reward functions to identify which path was taken.
    def fake_crossword(completion_texts, gold_answers, **_kwargs):
        return ["cross"] * len(completion_texts)

    def fake_math(completion_texts, gold_answers, **_kwargs):
        return ["math"] * len(completion_texts)

    def fake_rush(*, prompts, completions, **_kwargs):
        return ["rush"] * len(completions)

    monkeypatch.setattr(router_mod, "pure_accuracy_reward", fake_crossword)
    monkeypatch.setattr(router_mod, "pure_accuracy_reward_math", fake_math)
    monkeypatch.setattr(router_mod, "rush_solution_shaped", fake_rush)

    prompts = ["p"]
    completions = ["c"]

    out_rush = router_mod.reward_router(
        prompts=prompts,
        completions=completions,
        tasks=["RUSH"],
    )
    assert out_rush == ["rush"]

    out_math = router_mod.reward_router(
        prompts=prompts,
        completions=completions,
        tasks=["MATH"],
    )
    assert out_math == ["math"]

    out_cross = router_mod.reward_router(
        prompts=prompts,
        completions=completions,
        tasks=["CROSSWORD"],
    )
    assert out_cross == ["cross"]


def test_reward_router_infers_task_from_script_args(monkeypatch):
    def fake_crossword(completion_texts, gold_answers, **_kwargs):
        return ["cross"] * len(completion_texts)

    def fake_math(completion_texts, gold_answers, **_kwargs):
        return ["math"] * len(completion_texts)

    monkeypatch.setattr(router_mod, "pure_accuracy_reward", fake_crossword)
    monkeypatch.setattr(router_mod, "pure_accuracy_reward_math", fake_math)

    class Args:
        def __init__(self, reward_funcs):
            self.reward_funcs = reward_funcs

    prompts = ["p"]
    completions = ["c"]

    # Names mentioning math should route to math reward.
    out_math = router_mod.reward_router(
        prompts=prompts,
        completions=completions,
        script_args=Args("pure_accuracy_reward_math"),
    )
    assert out_math == ["math"]

    # Default pure_accuracy_reward routes to crossword.
    out_cross = router_mod.reward_router(
        prompts=prompts,
        completions=completions,
        script_args=Args("pure_accuracy_reward"),
    )
    assert out_cross == ["cross"]


def test_rush_rewards_exact_and_shaped_basic_behaviour():
    # Simple canonical move string; exact match should give 1.0.
    completions = ["Bv2,A>1"]
    gold = ["Bv2,A>1"]

    exact_scores = rush_rewards_mod.rush_solution_exact(
        prompts=None,
        completions=completions,
        gold=gold,
    )
    assert exact_scores == [pytest.approx(1.0)]

    # Shaped reward should give a score in [0,1] and
    # higher for the exact match than for a clearly different sequence.
    shaped_good = rush_rewards_mod.rush_solution_shaped(
        prompts=None,
        completions=completions,
        gold=gold,
        board_str=None,
        board_size=None,
        gold_moves=None,
    )
    shaped_bad = rush_rewards_mod.rush_solution_shaped(
        prompts=None,
        completions=["Av1"],
        gold=gold,
        board_str=None,
        board_size=None,
        gold_moves=None,
    )

    assert len(shaped_good) == 1 and len(shaped_bad) == 1
    assert 0.0 <= shaped_good[0] <= 1.0
    assert 0.0 <= shaped_bad[0] <= 1.0
    assert shaped_good[0] >= shaped_bad[0]
