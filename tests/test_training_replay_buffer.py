#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest

rb_mod = pytest.importorskip("src.training.utils.replay_buffer")


def test_prompt_key_normalizes_whitespace():
    prompt1 = [
        {"role": "user", "content": "Hello   world"},
        {"role": "assistant", "content": "Answer\n there"},
    ]
    prompt2 = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Answer there"},
    ]
    key1 = rb_mod._prompt_key(prompt1)
    key2 = rb_mod._prompt_key(prompt2)
    assert key1 == key2


def test_finite_float_handles_non_finite_and_bad_values():
    assert rb_mod._finite_float("1.5") == pytest.approx(1.5)
    assert rb_mod._finite_float(float("inf"), default=-1.0) == -1.0
    assert rb_mod._finite_float("not-a-number", default=2.0) == 2.0


def test_is_full_example_recognizes_prompt_answer_dict():
    assert rb_mod._is_full_example({"prompt": [], "answer": "x"}) is True
    assert rb_mod._is_full_example({"prompt": []}) is False
    assert rb_mod._is_full_example("not-dict") is False


def test_replay_buffer_add_and_deduplication():
    buf = rb_mod.ReplayBuffer(capacity=2, debug_steps=0)
    sample = {"prompt": [{"role": "user", "content": "hi"}], "answer": "a"}

    inserted, uid = buf.add(sample, reward=1.0)
    assert inserted is True
    assert uid >= 0
    assert len(buf) == 1

    inserted2, uid2 = buf.add(sample, reward=2.0)
    assert inserted2 is False
    assert uid2 == -1
    assert len(buf) == 1


def test_replay_buffer_capacity_and_replacement():
    buf = rb_mod.ReplayBuffer(capacity=1, debug_steps=0)
    s1 = {"prompt": [{"role": "user", "content": "p1"}], "answer": "a1"}
    s2 = {"prompt": [{"role": "user", "content": "p2"}], "answer": "a2"}
    s3 = {"prompt": [{"role": "user", "content": "p3"}], "answer": "a3"}

    inserted1, uid1 = buf.add(s1, reward=0.1)
    assert inserted1 is True
    assert len(buf) == 1

    # Worse reward than existing → skipped.
    inserted2, uid2 = buf.add(s2, reward=0.05)
    assert inserted2 is False
    assert uid2 == -1
    assert len(buf) == 1

    # Better reward → replaces the worst entry.
    inserted3, uid3 = buf.add(s3, reward=0.2)
    assert inserted3 is True
    assert len(buf) == 1
    assert uid3 != uid1


def test_replay_buffer_sample_and_sample_uid_and_get_group(monkeypatch):
    np.random.seed(0)
    buf = rb_mod.ReplayBuffer(capacity=10, debug_steps=0)
    group1 = [{"prompt": [{"role": "user", "content": "g1"}], "answer": "a"}]
    group2 = [{"prompt": [{"role": "user", "content": "g2"}], "answer": "b"}]

    uid1 = buf.add_group(group1)
    uid2 = buf.add_group(group2)
    assert len(buf) == 2

    samples, idxs, uids, isw = buf.sample(batch_size=2)
    assert len(samples) == len(idxs) == len(uids) == isw.shape[0]
    assert set(uids) == {uid1, uid2}

    uid_random = buf.sample_uid()
    assert uid_random in {uid1, uid2}

    retrieved = buf.get_group(uid1)
    assert retrieved == group1


def test_replay_buffer_update_priority_and_debug_state():
    buf = rb_mod.ReplayBuffer(capacity=2, debug_steps=0)
    sample = {"prompt": [{"role": "user", "content": "p"}], "answer": "a"}
    _, uid = buf.add(sample, reward=0.0)

    buf.update_priority_by_uid(uid, reward=1.0)
    state = buf.debug_state()
    assert state["len"] == 1
    assert state["capacity"] == buf.capacity
    assert state["next_uid"] >= 1
