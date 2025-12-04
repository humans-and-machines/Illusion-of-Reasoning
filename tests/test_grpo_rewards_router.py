#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch):
    class FakeTensor:
        def __init__(self, data):
            self._data = data

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self._data)

    fake_torch = SimpleNamespace(Tensor=FakeTensor, SymFloat=FakeTensor, SymBool=FakeTensor)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    yield
    sys.modules.pop("torch", None)


def test_flatten_and_wrap_reward_for_nested(monkeypatch):
    import src.training.grpo_rewards_router as rr

    calls = {}

    def reward_fn(*, prompts, completions, weights):
        calls["prompts"] = prompts
        calls["completions"] = completions
        calls["weights"] = weights
        return [1.0 for _ in completions]

    wrapper = rr._wrap_reward_for_nested(reward_fn)
    prompts = ["p1", "p2"]
    completions = [["a1", "a2"], ["b1"]]
    weights = [0.1, 0.2]
    out = wrapper(prompts=prompts, completions=completions, weights=weights)
    assert calls["completions"] == ["a1", "a2", "b1"]
    # weights expanded to match flat completions
    assert calls["weights"] == [0.1, 0.1, 0.2]
    assert out == [[1.0, 1.0], [1.0]]


def test_default_task_hierarchy_and_task_from_args(monkeypatch):
    import src.training.grpo_rewards_router as rr

    args = SimpleNamespace(reward_funcs="rush_solution_shaped", dataset_name=None)
    assert rr._default_task(args) == "RUSH"
    args2 = SimpleNamespace(reward_funcs=["math"], dataset_name=None)
    assert rr._task_from_script_args(args2) == "MATH"
    args3 = SimpleNamespace(reward_funcs=None, dataset_name="my-cross-ds")
    assert rr._default_task(args3) == "CROSSWORD"
    # Prompt hint override
    assert rr._default_task(None, prompt_hint="carpark levels") == "RUSH"
    # Last resort uses env hint
    monkeypatch.setenv("DEFAULT_TASK_HINT", "math")
    assert rr._default_task(None, prompt_hint=None) == "MATH"


def test_to_text_list_handles_tensor_and_list(monkeypatch):
    import src.training.grpo_rewards_router as rr

    class FakeProc:
        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return f"decoded-{ids}"

    tensor = rr.torch.Tensor([1, 2, 3])
    texts = rr._to_text_list([tensor, [4, 5], "raw"], proc=FakeProc())
    assert texts[0].startswith("decoded-[1, 2, 3]")
    assert texts[1].startswith("decoded-[4, 5]")
    assert texts[2] == "raw"


def test_reward_router_routes_tasks(monkeypatch):
    import src.training.grpo_rewards_router as rr

    # Stub reward functions
    monkeypatch.setattr(rr, "pure_accuracy_reward_math", lambda comps, gold, **k: ["math"])
    monkeypatch.setattr(rr, "pure_accuracy_reward", lambda comps, gold, **k: ["cross"])
    monkeypatch.setattr(rr, "rush_solution_shaped", lambda **k: ["rush"])

    prompts = ["p"]
    comps = ["c"]
    gold = ["g"]

    # RUSH path via task label
    out_rush = rr.reward_router(prompts=prompts, completions=comps, tasks=["RUSH"], gold=gold)
    assert out_rush == ["rush"]

    # MATH via reward_funcs arg
    args = SimpleNamespace(reward_funcs="math")
    out_math = rr.reward_router(prompts=prompts, completions=comps, script_args=args, gold=gold)
    assert out_math == ["math"]

    # Default crossword path
    out_cross = rr.reward_router(prompts=prompts, completions=comps, tasks=None, gold=gold)
    assert out_cross == ["cross"]


def test_infer_task_label_uses_tasks(monkeypatch):
    import src.training.grpo_rewards_router as rr

    label = rr._infer_task_label(None, tasks=["math"], prompts=["user prompt"])
    assert label == "MATH"


def test_adapt_and_normalize_helpers(monkeypatch):
    import src.training.grpo_rewards_router as rr

    gold = rr._adapt_gold(["a", "b"], answer="G")
    assert gold == ["G", "G"]

    # list passthrough
    gold2 = rr._adapt_gold(["a", "b"], answers=["x", "y"])
    assert gold2 == ["x", "y"]

    # normalize prompt when scalar provided
    prompts = rr._normalize_prompt_list("prompt", 3)
    assert prompts == ["prompt", "prompt", "prompt"]

    cleaned = rr._strip_answer_like_keys({"answer": 1, "keep": 2, "labels": [3]})
    assert cleaned == {"keep": 2}

    flat, factor = rr._flatten_nested([["a", "b"], ["c", "d"]])
    assert flat == ["a", "b", "c", "d"] and factor == 2


def test_infer_task_label_uses_prompt_hint():
    import src.training.grpo_rewards_router as rr

    prompt = [{"role": "user", "content": "Carpark puzzle"}]
    label = rr._infer_task_label(None, tasks=None, prompts=[prompt])
    assert label == "RUSH"


def test_rush_scores_broadcasts_and_ith(monkeypatch):
    import src.training.grpo_rewards_router as rr

    calls = []

    def fake_rush_solution_shaped(prompts, completions, gold, gold_moves, board_str, N, **weights):
        calls.append((prompts, completions, gold, gold_moves, board_str, N, weights))
        return [float(len(completions[0]))]

    monkeypatch.setattr(rr, "rush_solution_shaped", fake_rush_solution_shaped)
    prompt_list = ["p1", "p2"]
    completions = ["c1", "c2"]
    gold_answers = ["g1"]
    scores = rr._rush_scores(
        prompt_list,
        completions,
        gold_answers,
        {"gold_moves": ["m1", "m2"], "board_str": "board", "N": 4},
    )
    assert scores == [2.0, 2.0]
    # gold_moves list should be indexed; board_str/N broadcast.
    assert calls[0][3] == "m1" and calls[1][3] == "m2"
    assert calls[0][4] == "board" and calls[1][4] == "board"


def test_reward_router_nested_matches_structure(monkeypatch):
    import src.training.grpo_rewards_router as rr

    monkeypatch.setattr(rr, "pure_accuracy_reward_math", lambda comps, gold, **k: [10.0] * len(comps))
    monkeypatch.setattr(rr, "pure_accuracy_reward", lambda comps, gold, **k: [1.0] * len(comps))
    monkeypatch.setattr(rr, "rush_solution_shaped", lambda **k: [5.0] * len(k["completions"]))

    nested_comps = [["a1", "a2"], ["b1"]]
    args = SimpleNamespace(reward_funcs="math")
    out_nested = rr.reward_router(prompts="p", completions=nested_comps, script_args=args, gold="g")
    assert out_nested == [10.0, 10.0, 10.0]


def test_wrap_reward_non_nested_and_expand_value(monkeypatch):
    import src.training.grpo_rewards_router as rr

    calls = {}

    def reward_fn(*, prompts, completions, weights=None):
        calls["prompts"] = prompts
        calls["completions"] = completions
        calls["weights"] = weights
        return [42]

    wrapped = rr._wrap_reward_for_nested(reward_fn)
    out = wrapped(prompts="p", completions=["c"], weights=[1])
    assert out == [42]
    assert calls["prompts"] == "p"
    assert calls["completions"] == ["c"]

    # Nested branch with kwargs expansion that are not length-matched.
    def reward_fn2(*, prompts, completions, weights=None):
        return list(range(len(completions)))

    wrapped2 = rr._wrap_reward_for_nested(reward_fn2)
    out2 = wrapped2(prompts=["p1", "p2"], completions=[["a"], ["b", "c"]], weights=[0.5, 0.6])
    assert out2 == [[0], [1, 2]]


def test_task_from_script_args_errors_and_prompt_hints(monkeypatch):
    import src.training.grpo_rewards_router as rr

    class Bad:
        def __str__(self):
            raise TypeError("nope")

    assert rr._task_from_script_args(SimpleNamespace(reward_funcs=Bad())) is None
    assert rr._default_task(None, system_prompt="algebra system") == "MATH"
    assert rr._extract_prompt_hint("not-a-list") is None
    assert rr._extract_prompt_hint([[]]) is None


def test_to_text_list_fallback_paths(monkeypatch):
    import src.training.grpo_rewards_router as rr

    class ProcFail:
        def decode(self, *a, **k):
            raise ValueError("boom")

    tensor = rr.torch.Tensor([9])
    as_text = rr._to_text_list(tensor, proc=ProcFail())
    assert as_text == ["[9]"]

    as_text_ints = rr._to_text_list([[1, 2, 3]], proc=ProcFail())
    assert as_text_ints == ["1 2 3"]

    as_text_other = rr._to_text_list(object())
    assert isinstance(as_text_other[0], str)


def test_prompt_hint_user_extraction_and_none():
    import src.training.grpo_rewards_router as rr

    prompt = [
        [
            {"role": "assistant", "content": "ignore"},
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hint"},
        ]
    ]
    assert rr._extract_prompt_hint(prompt) == "hint"
    # No user role -> None
    assert rr._extract_prompt_hint([[{"role": "assistant", "content": "x"}]]) is None


def test_wrap_reward_expand_value_passthrough():
    import src.training.grpo_rewards_router as rr

    calls = {}

    def reward_fn(*, prompts, completions, weights=None):
        calls["weights"] = weights
        return [0] * len(completions)

    wrapped = rr._wrap_reward_for_nested(reward_fn)
    weights_tuple = (0.5, 0.6)  # not a list → should pass through unchanged
    wrapped(prompts=["p1"], completions=[["a"], ["b"]], weights=weights_tuple)
    assert calls["weights"] is weights_tuple


def test_normalize_id_sequence_bool_nonint_and_error():
    import src.training.grpo_rewards_router as rr

    assert rr._normalize_id_sequence([True, False]) == [1, 0]
    assert rr._normalize_id_sequence([1.5]) == [1.5]

    class FlakySeq:
        def __init__(self):
            self.used = False

        def __iter__(self):
            if not self.used:
                self.used = True
                raise TypeError("fail first pass")
            return iter([1, 2])

    assert rr._normalize_id_sequence(FlakySeq()) == [1, 2]
