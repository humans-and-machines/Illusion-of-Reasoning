#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_strip_answer_blocks_removes_think_and_answer():
    from src.training import grpo_dataset as grpo_ds

    txt = "<think>internal</think> user text <answer>42</answer>"
    cleaned = grpo_ds._strip_answer_blocks(txt)
    assert "<think>" not in cleaned and "<answer>" not in cleaned
    assert "user text" in cleaned


def test_messages_from_dict_and_list_prompts_normalize_and_drop_assistant():
    from src.training import grpo_dataset as grpo_ds
    dict_prompt = {"role": ["system", "assistant", "user"], "content": ["S", "A", "U <answer>x</answer>"]}
    msgs, dropped = grpo_ds._messages_from_dict_prompt(dict_prompt)
    assert dropped == 1
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "<answer>" not in msgs[-1]["content"]

    list_prompt = [
        {"role": "assistant", "content": "ignored"},
        {"role": "something", "content": "user-ish"},
    ]
    msgs2, dropped2 = grpo_ds._messages_from_list_prompt(list_prompt)
    assert dropped2 == 1
    assert len(msgs2) == 1 and msgs2[0]["role"] == "user"


def test_build_base_messages_handles_string_and_board_fallback():
    from src.training import grpo_dataset as grpo_ds
    example = {"problem": "What is 1+1?"}
    msgs, dropped = grpo_ds._build_base_messages(example, "problem", system_prompt="SYS")
    assert dropped == 0
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    example2 = {"board": "BOARD-STATE"}
    msgs2, _ = grpo_ds._build_base_messages(example2, "missing_col", system_prompt=None)
    assert len(msgs2) == 1 and msgs2[0]["content"] == "BOARD-STATE"


def test_ensure_user_and_system_messages_injects_when_missing():
    from src.training import grpo_dataset as grpo_ds
    messages = []
    example = {"board": "B"}
    grpo_ds._ensure_user_and_system_messages(messages, example, system_prompt="SYS")
    roles = [m["role"] for m in messages]
    assert "system" in roles and "user" in roles


def test_augment_with_legacy_metadata_appends_size_and_moves():
    from src.training import grpo_dataset as grpo_ds
    msgs = [{"role": "user", "content": "base"}]
    example = {"size": 4, "moves": 3}
    grpo_ds._augment_with_legacy_metadata(msgs, example)
    content = msgs[0]["content"]
    assert "Board size: 4x4" in content
    assert "Minimal moves to solve: 3" in content


@pytest.mark.parametrize(
    "raw_sol,expected",
    [
        ("plain", "plain"),
        ("<answer> 7 </answer>", "7"),
        (["A", "B"], "A,B"),
    ],
)
def test_extract_solution_handles_list_and_answer_tags(raw_sol, expected):
    from src.training import grpo_dataset as grpo_ds
    example = {"solution": raw_sol}
    sol = grpo_ds._extract_solution(example, "solution")
    assert sol == expected

    with pytest.raises(ValueError):
        grpo_ds._extract_solution({}, "solution")


def test_estimate_prompt_tokens_uses_fallback_when_chat_template_missing():
    from src.training import grpo_dataset as grpo_ds
    class DummyTok:
        def __call__(self, text, return_tensors=None):
            # Encode length of text as simple token count proxy.
            n = max(1, min(5, len(text.split())))
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, n)))

    msgs = [{"role": "user", "content": "one two three"}]
    tokens = grpo_ds._estimate_prompt_tokens(msgs, DummyTok())
    assert 1 <= tokens <= 5


def test_make_conversation_builds_chat_like_sample_and_respects_max_tokens():
    from src.training import grpo_dataset as grpo_ds
    class DummyTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
            # Each message contributes one token plus prompt marker.
            n = len(messages) + 1
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, n)))

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 10)))

    example = {"problem": "P", "answer": "<answer>42</answer>"}
    out = grpo_ds._make_conversation(
        example,
        prompt_column="problem",
        solution_column="answer",
        tokenizer=DummyTok(),
        system_prompt="SYS",
        max_prompt_tokens=10,
    )
    assert out is not None
    assert out["answer"] == "42"
    assert out["task"] == "MATH"
    assert isinstance(out["prompt"], list) and out["prompt"][0]["role"] == "system"

    # Over-length path returns None.
    class LongTok(DummyTok):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 9999)))

    out2 = grpo_ds._make_conversation(
        example,
        prompt_column="problem",
        solution_column="answer",
        tokenizer=LongTok(),
        system_prompt="SYS",
        max_prompt_tokens=100,
    )
    assert out2 is None


def test_load_easy_pool_uses_datasets_and_marks_task(monkeypatch):
    from src.training import grpo_dataset as grpo_ds
    class FakeSplit:
        def __init__(self, rows):
            self._rows = list(rows)
            self.column_names = ["messages"]

        def map(self, fn):
            return FakeSplit([fn(r) for r in self._rows])

        def filter(self, fn):
            return FakeDict({"train": FakeSplit([r for r in self._rows if fn(r)])})

        def remove_columns(self, name):
            # keep same rows but change column_names
            self.column_names = []
            return self

        def __iter__(self):
            return iter(self._rows)

    class FakeDict(dict):
        def keys(self):
            return super().keys()

    def fake_load_dataset(name):
        assert name == "easy-repo"
        # messages field as chat history
        return FakeSplit(
            [
                {
                    "messages": [
                        {"role": "user", "content": "clue1"},
                    ],
                    "answer": "A1",
                },
            ]
        )

    monkeypatch.setattr(grpo_ds, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    class DummyTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
            n = len(messages) + 1
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, n)))

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 5)))

    script_args = SimpleNamespace(
        easy_dataset_name="easy-repo",
        dataset_prompt_column="messages",
        dataset_solution_column="answer",
        dataset_train_split="train",
    )
    training_args = SimpleNamespace(system_prompt="SYS")

    pool = grpo_ds._load_easy_pool(script_args, DummyTok(), training_args)
    assert isinstance(pool, list)
    assert pool and pool[0]["task"] == "EASY"
    assert "prompt" in pool[0] and "answer" in pool[0]


def _ensure_torch_and_pad(hr_mod, monkeypatch):
    # HierarchicalRollout may have imported without accelerate; patch torch/pad_sequence if needed.
    import importlib

    torch_mod = importlib.import_module("torch")
    if getattr(hr_mod, "torch", None) is None or not hasattr(hr_mod.torch, "tensor"):
        monkeypatch.setattr(hr_mod, "torch", torch_mod, raising=False)
        from torch.nn.utils.rnn import pad_sequence as real_pad
        monkeypatch.setattr(hr_mod, "pad_sequence", real_pad, raising=False)


def test_hierarchical_rollout_two_stage_hf(monkeypatch):
    torch = pytest.importorskip("torch")
    import src.training.utils.hierarchical_rollout as hr_mod

    _ensure_torch_and_pad(hr_mod, monkeypatch)

    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 99

        def encode(self, text, add_special_tokens=False):
            if "think" in text:
                return [10]
            if "answer" in text:
                return [11]
            return [1]

        def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return ["decoded"] * len(ids)

    class DummyModel:
        def __init__(self, reason_limit):
            self.reason_limit = reason_limit

        def generate(self, input_ids, max_new_tokens, eos_token_id=None, do_sample=True, **kwargs):
            bsz, _ = input_ids.shape
            if max_new_tokens == self.reason_limit:
                extra = torch.full((bsz, 1), 2, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1)
            extra = torch.full((bsz, 1), 3, dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)

    model = DummyModel(reason_limit=5)
    tok = DummyTokenizer()
    rollout = hr_mod.HierarchicalRollout(model=model, tokenizer=tok, vllm_client=None, max_reason_tokens=5)

    input_ids = torch.tensor([[7, 8]], dtype=torch.long)
    reason_ids, full_ids = rollout.generate(input_ids=input_ids)

    # Stage 1 should append </think> and <answer> tag ids.
    think_ids, answer_ids = rollout.get_tag_ids()
    assert reason_ids.shape[1] >= input_ids.shape[1] + len(think_ids) + len(answer_ids)
    assert reason_ids[0, -len(answer_ids):].tolist() == answer_ids

    # Stage 2 appends at least one extra token.
    assert full_ids.shape[1] == reason_ids.shape[1] + 1
