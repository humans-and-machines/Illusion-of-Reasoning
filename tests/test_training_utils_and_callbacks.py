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


def test_callbacks_interact_with_buffer_and_tokenizer(capsys):
    class DummyBuf:
        def __init__(self):
            self.added = []

        def add(self, prompt):
            self.added.append(prompt)

        def __len__(self):
            return len(self.added)

    class DummyTok:
        def decode(self, ids, skip_special_tokens=True):
            return f"decoded-{len(ids)}"

    dummy_buf = DummyBuf()

    # SuccessCachingCallback adds prompts above threshold.
    cache_cb = callbacks_mod.SuccessCachingCallback(dummy_buf, acc_threshold=0.5)
    trainer = SimpleNamespace(
        textual_logs={
            "prompt": ["p1", "p2"],
            "rewards": {"foo_accuracy": [0.6, 0.1]},
        }
    )
    cache_cb.set_trainer(trainer)
    cache_cb.on_log(None, None, None)
    assert dummy_buf.added == ["p1"]

    # ReplayBufferCallback decodes and adds prompts with high rewards.
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    replay_cb = callbacks_mod.ReplayBufferCallback(
        replay_buffer=dummy_buf,
        tokenizer=DummyTok(),
        accuracy_key="acc_key",
        threshold=0.5,
    )
    args = SimpleNamespace(local_rank=-1)
    rewards = {"acc_key": FakeTensor([0.4, 0.8])}
    replay_cb.on_train_batch_end(
        args,
        outputs={"rewards": rewards},
        inputs={"input_ids": [[1, 2], [3, 4]]},
    )
    # One more prompt added from replay path.
    assert dummy_buf.added[-1] == "decoded-2"
    out = capsys.readouterr().out
    assert "buffer" in out
