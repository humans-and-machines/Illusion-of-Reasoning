import numpy as np
import pytest

from src.training.utils.replay_buffer import ReplayBuffer, _finite_float, _is_full_example, _prompt_key


def test_prompt_key_normalizes_whitespace():
    prompt = [
        {"role": "user", "content": "Hello   world"},
        {"role": "assistant", "content": "Hi\tthere"},
    ]
    key = _prompt_key(prompt)
    assert key == (("user", "Hello world"), ("assistant", "Hi there"))


def test_finite_float_handles_bad_inputs():
    assert _finite_float("3.5") == pytest.approx(3.5)
    assert _finite_float(float("inf"), default=1.1) == pytest.approx(1.1)
    assert _finite_float("not-a-number", default=-2.0) == pytest.approx(-2.0)


def test_is_full_example_checks_prompt_and_answer():
    assert _is_full_example({"prompt": [], "answer": "ok"}) is True
    assert _is_full_example({"prompt": []}) is False
    assert _is_full_example("not-a-dict") is False


def test_add_and_get_group_and_dedup():
    buf = ReplayBuffer(capacity=2)
    sample = {"prompt": [{"role": "user", "content": "Q"}], "answer": "A"}
    inserted, uid = buf.add(sample, reward=1.0)
    assert inserted is True and uid == 0
    assert len(buf) == 1
    assert buf.get_group(uid)[0]["answer"] == "A"

    # Deduped insert should fail and record last_error
    inserted, uid_dup = buf.add(sample, reward=0.5)
    assert inserted is False and uid_dup == -1
    assert buf.last_error()["why"] == "dedup"


def test_capacity_replacement_and_skip():
    buf = ReplayBuffer(capacity=1)
    buf.add({"prompt": [{"role": "user", "content": "p1"}], "answer": "low"}, reward=0.1)
    assert len(buf) == 1

    # Lower reward should be skipped
    inserted_low, _ = buf.add({"prompt": [{"role": "user", "content": "p2"}], "answer": "lower"}, reward=0.05)
    assert inserted_low is False
    assert len(buf) == 1

    # Higher reward replaces
    inserted_high, uid_high = buf.add({"prompt": [{"role": "user", "content": "p3"}], "answer": "high"}, reward=0.9)
    assert inserted_high is True
    assert buf.get_group(uid_high)[0]["answer"] == "high"


def test_update_priority_and_debug_state():
    buf = ReplayBuffer(capacity=2)
    buf.add({"prompt": [], "answer": "ans"}, reward=1.0)
    buf.update_priority_by_uid(0, reward=0.0)
    state = buf.debug_state()
    assert state["len"] == 1
    assert state["next_uid"] == 1
    assert state["tail_n"][-1] == 2  # count should have incremented
    # mean should now be halfway between 1 and 0
    assert state["tail_mu"][-1] == pytest.approx(0.5)

    # Unknown uid should be ignored gracefully
    buf.update_priority_by_uid(99, reward=1.0)


def test_sample_and_sample_uid_behaviour():
    buf = ReplayBuffer(capacity=3)
    buf.add({"prompt": [{"role": "user", "content": "p1"}], "answer": "a"}, reward=0.2)
    buf.add({"prompt": [{"role": "user", "content": "p2"}], "answer": "b"}, reward=0.4)
    samples, idxs, uids, isw = buf.sample(batch_size=2)
    assert len(samples) == len(idxs) == len(uids) == len(isw) == 2
    assert np.all(isw == 1.0)
    uid_pick = buf.sample_uid()
    assert uid_pick in uids or uid_pick in buf._storage.uids  # sanity check

    # Empty buffer sampling errors/returns None
    empty = ReplayBuffer(capacity=1)
    assert empty.sample_uid() is None
    with pytest.raises(ValueError):
        empty.sample()


def test_add_group_uses_mean_reward_and_dedup():
    buf = ReplayBuffer(capacity=3)
    group = [
        {"prompt": [], "answer": "a", "reward": 1.0},
        {"prompt": [], "answer": "b", "reward": 0.0},
    ]
    uid = buf.add_group(group, reward=None)
    state = buf.debug_state()
    assert uid == 0
    assert state["tail_mu"][-1] == pytest.approx(0.5)

    # Explicit reward overrides implicit mean
    uid2 = buf.add_group([{"prompt": [], "answer": "c"}], reward=0.8)
    assert uid2 == 1
    assert buf.debug_state()["tail_mu"][-1] == pytest.approx(0.8)

    # Deduped group should return -1
    uid_dup = buf.add_group(group, reward=None)
    assert uid_dup == -1
