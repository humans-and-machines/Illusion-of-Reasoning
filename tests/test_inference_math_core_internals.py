#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
math_core = pytest.importorskip("src.inference.domains.math.math_core")


def _make_config(num_samples: int = 3) -> math_core.MathInferenceConfig:
    return math_core.MathInferenceConfig(
        split_name="test",
        output_dir=".",
        step=0,
        batch_size=1,
        num_samples=num_samples,
        temperature=0.0,
        top_p=0.95,
        entropy_mode="none",
        eos_ids=None,
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=1,
        think_cap=5,
        answer_cap=5,
    )


def test_build_work_items_for_slice_respects_existing_samples():
    examples = [
        {"problem": "p1", "answer": "a1"},
        {"problem": "p2", "answer": "a2"},
        {"question": "no-problem-field"},
    ]
    existing_samples = {"p1": {0, 2}}
    cfg = _make_config(num_samples=3)

    work_items = math_core._build_work_items_for_slice(  # type: ignore[attr-defined]
        examples_slice=examples,
        existing_samples=existing_samples,
        config=cfg,
    )

    assert len(work_items) == 2
    w0 = next(item for item in work_items if item["_normalized_problem"] == "p1")
    assert w0["_todo_samples"] == [1]
    w1 = next(item for item in work_items if item["_normalized_problem"] == "p2")
    assert w1["_todo_samples"] == [0, 1, 2]


def test_build_first_pass_choice_prefers_new_or_existing_samples():
    layout = math_core.BatchLayout(
        work_items=[
            {"_normalized_problem": "p1"},
            {"_normalized_problem": "p2"},
        ],
        row_to_ex_idx=[0, 0, 1],
        row_target_sample_idx=[0, 1, 0],
    )
    pass1_full_texts = ["p1-s0", "p1-s1", "p2-s0"]
    existing_state = math_core.ExistingPassState(
        existing_samples={"p1": {0}},
        existing_pass1={("p1", 0): "disk-prev"},
    )
    cfg = _make_config(num_samples=2)

    choice = math_core._build_first_pass_choice(  # type: ignore[attr-defined]
        layout=layout,
        pass1_full_texts=pass1_full_texts,
        existing_state=existing_state,
        config=cfg,
    )

    assert choice[0] == "p1-s1"
    assert choice[1] == "p2-s0"

    cfg_disabled = _make_config(num_samples=2)
    cfg_disabled.two_pass_cfg.enabled = False
    choice_disabled = math_core._build_first_pass_choice(  # type: ignore[attr-defined]
        layout=layout,
        pass1_full_texts=pass1_full_texts,
        existing_state=existing_state,
        config=cfg_disabled,
    )
    assert choice_disabled == ["", ""]


class _DummyTokenizerForPass2:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}:{m['content']}" for m in messages]
        text = "|".join(parts)
        if add_generation_prompt:
            text += "|GEN"
        return text


def test_build_second_pass_base_prompts_uses_chat_template():
    tokenizer = _DummyTokenizerForPass2()
    work_items = [
        {"_normalized_problem": "p1"},
        {"_normalized_problem": "p2"},
    ]
    firstpass = ["prev1", "prev2"]
    phrase = "cue"

    prompts = math_core._build_second_pass_base_prompts(  # type: ignore[attr-defined]
        tokenizer=tokenizer,
        work_items=work_items,
        firstpass_choice_text_per_ex=firstpass,
        phrase=phrase,
    )
    assert len(prompts) == 2
    assert "system" in prompts[0]
    assert "user:Problem: p1" in prompts[0]
    assert "assistant:prev1" in prompts[0]
    assert "user:cue" in prompts[0]
