#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest


# Stub heavy modules before importing the package to avoid pulling
# transformer/deepspeed dependencies during test collection.
for _mod_name in (
    "src.inference.domains.carpark.carpark_core",
    "src.inference.domains.carpark.carpark_cli",
    "src.inference.domains.carpark.carpark_solver",
):
    sys.modules.setdefault(_mod_name, ModuleType(_mod_name))

import src.inference.domains.carpark.carpark_data as cd  # noqa: E402
from src.inference.utils.task_registry import CARPARK_SYSTEM_PROMPT  # noqa: E402


def test_ensure_messages_handles_list_and_json_string():
    msgs = [{"role": "user", "content": "p"}]
    assert cd._ensure_messages(msgs) is msgs

    json_str = json.dumps(msgs)
    assert cd._ensure_messages(json_str) == msgs

    assert cd._ensure_messages("not-json") == []
    with pytest.raises(ValueError):
        cd._ensure_messages(123)


def test_norm_fields_fallback_builds_prompt():
    example = {"board": "B1"}
    messages, solution = cd.norm_fields(example, prompt_col="messages", solution_col="solution")
    assert messages[0]["content"] == CARPARK_SYSTEM_PROMPT
    assert "B1" in messages[1]["content"]
    assert solution is None


def test_load_existing_example_index(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        "\n".join(
            [
                json.dumps({"example_id": "ex1", "sample_idx": 0}),
                json.dumps({"example_id": "ex1", "sample_idx": 1}),
                json.dumps({"example_id": "bad", "sample_idx": "x"}),
            ]
        ),
        encoding="utf-8",
    )
    mapping = cd.load_existing_example_index(str(out))
    assert mapping == {"ex1": {0, 1}}
    assert cd.load_existing_example_index(str(out.with_suffix(".missing"))) == {}


def test_build_batch_items_for_range_respects_existing(monkeypatch):
    class FakeDS(list):
        def select(self, rng):
            return [self[i] for i in rng]

    examples = FakeDS(
        [
            {"id": "a", "messages": '[{"role":"user","content":"p"}]', "solution": "s"},
            {"id": "b", "messages": [{"role": "user", "content": "p2"}], "solution": "s2"},
        ]
    )
    existing = {"a": {0}}
    batch = cd.build_batch_items_for_range(
        examples,
        start_idx=0,
        batch_size=2,
        config=cd.BatchRangeConfig(
            prompt_col="messages",
            solution_col="solution",
            num_samples=2,
            existing_by_example=existing,
        ),
    )
    assert batch[0]["id"] == "a" and batch[0]["missing_indices"] == [1]
    assert batch[1]["id"] == "b" and batch[1]["missing_indices"] == [0, 1]

    empty_batch = cd.build_batch_items_for_range(
        examples,
        start_idx=0,
        batch_size=1,
        config=cd.BatchRangeConfig(
            prompt_col="messages",
            solution_col="solution",
            num_samples=1,
            existing_by_example={"a": {0}, "b": {0}},
        ),
    )
    assert empty_batch == []


def test_load_rush_dataset_validates_columns(monkeypatch):
    class FakeDS(list):
        def __init__(self, rows, cols):
            super().__init__(rows)
            self.column_names = cols

        def select(self, rng):
            return self

    def loader_ok(dataset_id, split=None, cache_dir=None):
        return FakeDS([], ["messages", "solution"])

    def loader_bad(dataset_id, split=None, cache_dir=None):
        return FakeDS([], ["foo"])

    monkeypatch.setattr(cd, "require_datasets", lambda: (None, loader_ok))
    ds = cd.load_rush_dataset("id", "split", cache_dir="c")
    assert isinstance(ds, FakeDS)

    monkeypatch.setattr(cd, "require_datasets", lambda: (None, loader_bad))
    with pytest.raises(ValueError):
        cd.load_rush_dataset("id", "split", cache_dir="c")
