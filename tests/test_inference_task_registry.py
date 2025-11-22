#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from src.inference.utils.task_registry import (
    TASK_REGISTRY,
    DatasetSpec,
    TaskSpec,
    _resolve_callable,
    MATH_SYSTEM_PROMPT,
)
from src.inference.utils.common import OPENR1_PROMPT_TEMPLATE


def dummy_callable():
    return "ok"


def dummy_loader():
    return ["loaded"]


def dummy_canon(value):
    return str(value).upper()


def test_resolve_callable_roundtrip_and_validation():
    path = f"{__name__}:dummy_callable"
    fn = _resolve_callable(path)
    assert fn is dummy_callable
    assert fn() == "ok"

    assert _resolve_callable(None) is None
    with pytest.raises(ValueError):
        _resolve_callable("not-a-valid-path")


def test_dataset_spec_loader_fn_uses_resolved_callable():
    spec = DatasetSpec(loader=f"{__name__}:dummy_loader", default_id="ID", split="train")
    loader_fn = spec.loader_fn()
    assert loader_fn is dummy_loader
    assert loader_fn() == ["loaded"]


def test_task_spec_property_helpers_and_canon_functions():
    config = {
        "system_prompt": "sys",
        "stop_think": ["</think>"],
        "stop_answer": ["</answer>"],
        "two_pass": True,
        "think_cap": 10,
        "answer_cap": 5,
        "max_output_tokens": 100,
        "canonicalize_pred": f"{__name__}:dummy_canon",
        "canonicalize_gold": f"{__name__}:dummy_canon",
    }
    task = TaskSpec(name="t", config=config)

    assert task.system_prompt == "sys"
    assert task.stop_think == ["</think>"]
    assert task.stop_answer == ["</answer>"]
    assert task.two_pass is True
    assert task.think_cap == 10
    assert task.answer_cap == 5
    assert task.max_output_tokens == 100

    pred_fn = task.canon_pred_fn()
    gold_fn = task.canon_gold_fn()
    assert pred_fn("x") == "X"
    assert gold_fn("y") == "Y"


def test_task_registry_contains_known_tasks():
    assert "math-qwen" in TASK_REGISTRY
    math_task = TASK_REGISTRY["math-qwen"]
    assert math_task.name == "math-qwen"
    assert math_task.system_prompt == MATH_SYSTEM_PROMPT
    assert math_task.dataset.default_id == "MATH-500"

    openr1_task = TASK_REGISTRY["openr1"]
    assert openr1_task.system_prompt == OPENR1_PROMPT_TEMPLATE
    assert openr1_task.dataset.default_id == "open-r1/OpenR1-Math-220k"
