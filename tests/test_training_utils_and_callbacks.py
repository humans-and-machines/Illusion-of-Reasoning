#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest

configs = pytest.importorskip("src.training.configs")
rewards_mod = pytest.importorskip("src.training.rewards")
data_utils = pytest.importorskip("src.training.utils.data")
callbacks_mod = pytest.importorskip("src.training.utils.callbacks")


def test_get_reward_funcs_maps_names_and_handles_strings():
    class Args:
        reward_funcs = ["crossword_accuracy", "pure_accuracy"]

    funcs = rewards_mod.get_reward_funcs(Args())
    assert len(funcs) == 2

    # String input should be promoted to single-element list.
    class ArgsStr:
        reward_funcs = "pure_accuracy"

    funcs_str = rewards_mod.get_reward_funcs(ArgsStr())
    assert len(funcs_str) == 1

    class BadArgs:
        reward_funcs = ["unknown_name"]

    with pytest.raises(KeyError):
        rewards_mod.get_reward_funcs(BadArgs())


def test_get_dataset_uses_dataset_name(monkeypatch):
    calls = []

    class FakeDatasets:
        @staticmethod
        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return {"loaded": args}

    monkeypatch.setattr(data_utils, "_require_datasets_module", lambda: FakeDatasets())

    args = SimpleNamespace(dataset_name="repo", dataset_config="cfg", dataset_mixture=None)
    ds = data_utils.get_dataset(args)
    assert ds == {"loaded": ("repo", "cfg")}
    assert calls[0][0] == ("repo", "cfg")


def test_get_dataset_raises_when_neither_dataset_nor_mixture():
    args = SimpleNamespace(dataset_name=None, dataset_mixture=None)
    with pytest.raises(ValueError):
        data_utils.get_dataset(args)


def test_get_callbacks_builds_and_validates(monkeypatch):
    # Unknown callback name should raise.
    train_cfg = SimpleNamespace(callbacks=["unknown"])
    with pytest.raises(ValueError):
        callbacks_mod.get_callbacks(train_cfg, model_cfg=SimpleNamespace())

    # SuccessCachingCallback requires replay_buffer.
    train_cfg = SimpleNamespace(callbacks=["caching_callback"])
    with pytest.raises(ValueError):
        callbacks_mod.get_callbacks(train_cfg, model_cfg=SimpleNamespace(), replay_buffer=None)

    # ReplayBufferCallback requires both replay_buffer and tokenizer.
    train_cfg = SimpleNamespace(callbacks=["replay_buffer_callback"])
    with pytest.raises(ValueError):
        callbacks_mod.get_callbacks(train_cfg, model_cfg=SimpleNamespace(), replay_buffer=object(), tokenizer=None)

    # Happy path: all three callbacks with minimal stubs.
    class DummyBuf:
        def __init__(self):
            self.added = []

        def add(self, *args, **_kwargs):
            self.added.append(args)

        def __len__(self):
            return len(self.added)

    dummy_buf = DummyBuf()

    class DummyTok:
        def decode(self, ids, skip_special_tokens=True):
            return f"decoded-{len(ids)}"

    train_cfg = SimpleNamespace(
        callbacks=[
            "push_to_hub_revision",
            "caching_callback",
            "replay_buffer_callback",
        ],
    )
    model_cfg = SimpleNamespace()
    cb_list = callbacks_mod.get_callbacks(
        train_cfg,
        model_cfg,
        replay_buffer=dummy_buf,
        tokenizer=DummyTok(),
    )
    assert len(cb_list) == 3
    assert any(isinstance(cb, callbacks_mod.PushToHubRevisionCallback) for cb in cb_list)
    assert any(isinstance(cb, callbacks_mod.SuccessCachingCallback) for cb in cb_list)
    assert any(isinstance(cb, callbacks_mod.ReplayBufferCallback) for cb in cb_list)
