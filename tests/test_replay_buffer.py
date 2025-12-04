import pytest

from src.training.utils.replay_buffer import ReplayBuffer, _finite_float, _prompt_key


def test_prompt_key_and_finite_float():
    prompt = [{"role": "user", "content": " hello   world "}]
    key = _prompt_key(prompt)
    assert key == (("user", "hello world"),)
    assert _finite_float(1.5) == 1.5
    assert _finite_float(float("inf"), default=7.0) == 7.0
    assert _finite_float("bad", default=2.0) == 2.0


def test_add_capacity_and_replacement_branches(capsys):
    buf = ReplayBuffer(capacity=1, ucb_coefficient=0.1, debug_steps=1)
    assert buf.ucb_coefficient == 0.1

    sample = {"prompt": [{"role": "user", "content": "hi"}], "answer": "a"}
    ok, uid0 = buf.add(sample, reward=0.5)
    assert ok and uid0 == 0 and len(buf) == 1

    # Lower reward should be skipped with error metadata.
    ok, uid_fail = buf.add({"prompt": [{"role": "user", "content": "hi2"}], "answer": "b"}, reward=0.1)
    assert not ok and uid_fail == -1
    assert buf.last_error()["why"] == "capacity_worse_mu"

    # Higher reward replaces worst.
    ok, uid1 = buf.add({"prompt": [{"role": "user", "content": "hi3"}], "answer": "c"}, reward=0.9)
    assert ok and uid1 == 2 and len(buf) == 1

    # Debug print paths exercised.
    out = capsys.readouterr().out
    assert "[RB][REPLACE]" in out or "[RB][SKIP]" in out


def test_add_capacity_zero_and_dedup():
    buf0 = ReplayBuffer(capacity=0)
    ok, uid = buf0.add({"prompt": [], "answer": ""}, reward=1.0)
    assert not ok and uid == -1

    buf_kw = ReplayBuffer(C=2.5)
    assert buf_kw.ucb_coefficient == pytest.approx(2.5)

    buf = ReplayBuffer(capacity=2)
    sample = {"prompt": [{"role": "user", "content": "x"}], "answer": "a"}
    ok, uid0 = buf.add(sample, reward=0.2)
    assert ok
    ok, uid_dup = buf.add(sample, reward=0.3)
    assert not ok and uid_dup == -1
    assert buf.last_error()["why"] == "dedup"


def test_key_for_sample_handles_bad_group():
    buf = ReplayBuffer(capacity=1)
    bad_group = {"group": [{"not_prompt": 1}]}
    # Should not throw on bad structure.
    ok, uid = buf.add(bad_group, reward=0.1)
    assert ok and uid >= 0

    # TypeError branch in _key_for_sample
    ok2, uid2 = buf.add({"group": None}, reward=0.2)
    assert ok2 and uid2 >= 0


def test_add_group_reward_calc_and_verbose(capsys):
    buf = ReplayBuffer(capacity=3, debug_steps=0)
    group = [
        {"prompt": [{"role": "user", "content": "p"}], "answer": "a", "reward": "bad"},
        {"prompt": [{"role": "assistant", "content": "r"}], "answer": "b"},
    ]
    uid = buf.add_group(group, reward=None, verbose=True)
    assert uid >= 0 and len(buf) == 1
    out = capsys.readouterr().out
    assert "[RB][add_group]" in out

    # Dedup path when adding the same group again.
    uid2 = buf.add_group(group, reward=0.5, verbose=True)
    assert uid2 == -1


def test_update_priority_and_debug_state():
    buf = ReplayBuffer(capacity=2)
    ok, uid = buf.add({"prompt": [{"role": "user", "content": "p"}], "answer": "a"}, reward=0.5)
    assert ok
    buf.update_priority_by_uid(uid, reward=1.5)
    state = buf.debug_state()
    assert state["tail_mu"][0] == pytest.approx(1.0)

    # Missing uid with debug enabled should be handled gracefully.
    buf_debug = ReplayBuffer(capacity=1, debug_steps=1)
    buf_debug.update_priority_by_uid(999, reward=1.0)
    buf.update_priority(idx=0, reward=2.0)
    assert buf.debug_state()["tail_mu"][0] == pytest.approx(1.333333, rel=1e-6)


def test_sampling_and_get_group_behavior():
    buf = ReplayBuffer(capacity=3, debug_steps=1)
    # sample should raise on empty
    with pytest.raises(ValueError):
        buf.sample()

    group = [{"prompt": [{"role": "user", "content": "p"}], "answer": "a"}]
    buf.add_group(group, reward=1.0)
    buf.add({"prompt": [{"role": "assistant", "content": "q"}], "answer": "b"}, reward=0.2)
    samples, idxs, uids, isw = buf.sample(batch_size=2)
    assert len(samples) == len(idxs) == len(uids) == len(isw)
    assert buf.sample_uid() in uids

    # get_group for missing and various stored types
    assert buf.get_group(999) == []
    uid_group = buf.add_group([{"prompt": [{"role": "assistant", "content": "r"}], "answer": "c"}])
    group_data = buf.get_group(uid_group)
    group_data.append({"extra": True})
    assert len(buf.get_group(uid_group)) == 1  # ensure deep copy

    uid_list = buf.add(["plain"], reward=0.1)[1]
    assert buf.get_group(uid_list) == ["plain"]

    uid_other = buf.add({"prompt": [{"role": "user", "content": "z"}], "answer": "d"}, reward=0.1)[1]
    assert isinstance(buf.get_group(uid_other), list)


def test_add_skip_branch_with_nan_reward(capsys):
    buf = ReplayBuffer(capacity=1, debug_steps=1)
    buf.add({"prompt": [{"role": "user", "content": "p"}], "answer": "a"}, reward=1.0)
    ok, uid = buf.add({"prompt": [{"role": "user", "content": "q"}], "answer": "b"}, reward=float("nan"))
    assert not ok and uid == -1
    out = capsys.readouterr().out
    assert "reward=" in out
