#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
router = pytest.importorskip("src.training.grpo_rewards_router")


def test_flatten_nested_for_rewards_aligns_prompts_and_sizes():
    prompts = ["p1", "p2"]
    completions = [["a1", "a2"], ["b1"]]

    flat_prompts, flat_completions, sizes, batch_size = router._flatten_nested_for_rewards(
        prompts,
        completions,
    )

    assert batch_size == 2
    assert sizes == [2, 1]
    assert flat_completions == ["a1", "a2", "b1"]
    assert flat_prompts == ["p1", "p1", "p2"]


def test_wrap_reward_for_nested_flattens_and_repacks(monkeypatch):
    calls = []

    def base_reward_fn(*, prompts, completions, offsets):
        # Record shapes and return a simple numeric sequence.
        calls.append((list(prompts), list(completions), list(offsets)))
        return list(range(len(completions)))

    wrapped = router._wrap_reward_for_nested(base_reward_fn)
    prompts = ["p1", "p2"]
    completions = [["c11", "c12"], ["c21"]]
    offsets = [1, 2]

    nested_scores = wrapped(prompts=prompts, completions=completions, offsets=offsets)
    assert nested_scores == [[0, 1], [2]]

    assert calls, "Base reward function should have been invoked"
    flat_prompts, flat_completions, flat_offsets = calls[0]
    assert flat_prompts == ["p1", "p1", "p2"]
    assert flat_completions == ["c11", "c12", "c21"]
    assert flat_offsets == [1, 1, 2]


@pytest.mark.parametrize(
    "reward_funcs,expected",
    [
        (None, None),
        ("rush_solution_shaped", "RUSH"),
        (["pure_accuracy_reward_math"], "MATH"),
        (["pure_accuracy_reward"], "CROSSWORD"),
        ({"math": 1.0}, "MATH"),
        (["other"], None),
    ],
)
def test_task_from_script_args_handles_various_reward_func_types(reward_funcs, expected):
    args = SimpleNamespace(reward_funcs=reward_funcs) if reward_funcs is not None else None
    assert router._task_from_script_args(args) == expected


def test_default_task_uses_dataset_and_prompts_and_env(monkeypatch):
    # Reward funcs hint wins first.
    args = SimpleNamespace(reward_funcs=["rush_solution_shaped"], dataset_name=None)
    assert router._default_task(args) == "RUSH"

    # Dataset name hint.
    args = SimpleNamespace(reward_funcs=None, dataset_name="my-math-dataset")
    assert router._default_task(args) == "MATH"

    # dataset_name_hint override.
    args = SimpleNamespace(reward_funcs=None, dataset_name=None)
    assert router._default_task(args, dataset_name_hint="carpark") == "RUSH"

    # Environment fallback.
    monkeypatch.setenv("DEFAULT_TASK_HINT", "cryptic-crosswords")
    assert router._default_task(args) == "CROSSWORD"


def test_to_text_list_handles_strings_tensors_and_id_lists():
    class DummyProc:
        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return "D:" + ",".join(str(int(i)) for i in ids)

    tensor_ids = torch.tensor([1, 2, 3])
    id_list = [4, 5]
    items = ["s", tensor_ids, id_list, {"x": 1}]

    out = router._to_text_list(items, proc=DummyProc())
    assert out[0] == "s"
    assert out[1].startswith("D:1,2,3")
    assert out[2].startswith("D:4,5")
    assert "{" in out[3]


def test_strip_answer_like_keys_and_normalize_prompt_list():
    kwargs_in = {"answer": [1], "labels": [2], "other": 3}
    kwargs_out = router._strip_answer_like_keys(kwargs_in)
    assert "answer" not in kwargs_out
    assert "labels" not in kwargs_out
    assert kwargs_out["other"] == 3

    prompts = router._normalize_prompt_list("p", num_samples=3)
    assert prompts == ["p", "p", "p"]

    prompts2 = router._normalize_prompt_list(["p1", "p2"], num_samples=2)
    assert prompts2 == ["p1", "p2"]


def test_extract_prompt_hint_prefers_user_message():
    assert router._extract_prompt_hint(["plain"]) == "plain"

    prompts = [
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user-text"},
        ],
    ]
    assert router._extract_prompt_hint(prompts) == "user-text"


def test_reward_router_routes_to_pure_accuracy(monkeypatch):
    recorded = {}

    def fake_pure_accuracy(completion_texts, gold_answers, **kwargs):
        recorded["completion_texts"] = completion_texts
        recorded["gold_answers"] = gold_answers
        recorded["kwargs"] = kwargs
        return [0.5] * len(completion_texts)

    monkeypatch.setattr(router, "pure_accuracy_reward", fake_pure_accuracy)

    res = router.reward_router(
        prompts=["crossword clue"],
        completions=["<answer>FOO</answer>"],
        tasks=None,
        proc=None,
        answers=["BAR"],
    )
    assert res == [0.5]
    assert recorded["completion_texts"] == ["<answer>FOO</answer>"]
    assert recorded["gold_answers"] == ["BAR"]
    # Answer-like keys should be stripped before forwarding.
    assert "answers" not in recorded["kwargs"]
    assert "gold" not in recorded["kwargs"]
    assert "labels" not in recorded["kwargs"]


def test_reward_router_routes_to_math_accuracy(monkeypatch):
    recorded = {}

    def fake_math(completion_texts, gold_answers, **kwargs):
        recorded["completion_texts"] = completion_texts
        recorded["gold_answers"] = gold_answers
        recorded["kwargs"] = kwargs
        return [1.0] * len(completion_texts)

    monkeypatch.setattr(router, "pure_accuracy_reward_math", fake_math)
    script_args = SimpleNamespace(reward_funcs=["pure_accuracy_reward_math"])

    res = router.reward_router(
        prompts=["math problem"],
        completions=["c1"],
        tasks=None,
        proc=None,
        answers=["42"],
        script_args=script_args,
    )
    assert res == [1.0]
    assert recorded["gold_answers"] == ["42"]


def test_reward_router_routes_to_rush_scores(monkeypatch):
    calls = []

    def fake_rush_solution_shaped(
        prompts,
        completions,
        gold,
        gold_moves=None,
        board_str=None,
        N=None,
        **_kwargs,
    ):
        calls.append(
            {
                "prompts": prompts,
                "completions": completions,
                "gold": gold,
                "gold_moves": gold_moves,
                "board_str": board_str,
                "N": N,
            },
        )
        return [0.7]

    monkeypatch.setattr(router, "rush_solution_shaped", fake_rush_solution_shaped)
    script_args = SimpleNamespace(reward_funcs=["rush_solution_shaped"])

    res = router.reward_router(
        prompts="rush prompt",
        completions=["move1", "move2"],
        tasks=None,
        proc=None,
        answers=["G"],
        gold_moves=[3],
        board_str="BOARD",
        N=6,
        script_args=script_args,
    )
    assert res == pytest.approx([0.7, 0.7])
    # Called once per sample with broadcast metadata.
    assert len(calls) == 2
    assert all(call["board_str"] == "BOARD" for call in calls)
    assert all(call["N"] == 6 for call in calls)

