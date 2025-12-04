#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _install_stubs():
    if "src.training.grpo_trainer_replay_impl" in sys.modules:
        return sys.modules["src.training.grpo_trainer_replay_impl"]

    # Torch stub with a minimal Tensor.
    class _Tensor:
        def __init__(self, data=None, device=None, dtype=None):
            if data is None:
                self.data = np.array([])
            else:
                self.data = np.array(data, dtype=dtype)
            self.device = device
            self.dtype = dtype

        @property
        def shape(self):
            return self.data.shape

        def view(self, *shape):
            return _Tensor(self.data.reshape(*shape), device=self.device, dtype=self.dtype)

        def bool(self):
            """Return a boolean mask tensor."""
            return _Tensor(self.data.astype(bool), device=self.device, dtype=bool)

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return _Tensor(self.data.astype(float), device=self.device, dtype=float)

        def to(self, device=None, dtype=None):
            """Return a copy with optional dtype/device changes."""
            data = self.data
            if dtype is not None:
                data = data.astype(dtype)
            return _Tensor(data, device=device or self.device, dtype=dtype or self.dtype)

        def numpy(self):
            return self.data

        def tolist(self):
            return self.data.tolist()

        def __iter__(self):
            for item in self.data:
                yield item

        def __getitem__(self, item):
            if isinstance(item, _Tensor):
                item = item.data
            return _Tensor(self.data[item], device=self.device, dtype=self.dtype)

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _Tensor
    torch_stub.tensor = lambda data=None, device=None, dtype=None: _Tensor(data, device=device, dtype=dtype)

    def _no_grad(*_a, **_k):
        def decorator(fn):
            return fn

        return decorator

    torch_stub.no_grad = _no_grad

    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.is_initialized = lambda: False
    dist_stub.get_rank = lambda: 0
    dist_stub.get_world_size = lambda: 1
    dist_stub.all_gather_object = lambda gathered, winners_local: gathered.__setitem__(0, winners_local)
    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = dist_stub

    # runtime.env stub
    env_mod = types.ModuleType("src.training.runtime.env")
    env_mod.torch = torch_stub
    env_mod.dist = dist_stub
    env_mod.GRPOTrainer = type("DummyTrainer", (), {})
    env_mod.TrainerCallback = object
    env_mod.wandb = types.SimpleNamespace(log=lambda *a, **k: None)
    sys.modules["src.training.runtime.env"] = env_mod

    return importlib.import_module("src.training.grpo_trainer_replay_impl")


def make_trainer(
    grpo_replay,
    *,
    buffer=None,
    easy_pool=None,
    schedule=None,
    is_main=True,
    training=True,
):
    tr = object.__new__(grpo_replay.GRPOTrainerReplay)
    tr.accelerator = SimpleNamespace(is_main_process=is_main, process_index=0)
    tr.args = SimpleNamespace(steps_per_generation=4)
    tr.state = SimpleNamespace(global_step=0)
    tr.temp_end = 0.3
    tr.temperature_schedule = grpo_replay.TemperatureSchedule(
        start_temperature=1.0,
        end_temperature=0.5,
        anneal_steps=2,
        high_temperature_period=2,
    )
    tr.mix_settings = grpo_replay.MixSettings(
        easy_pool=easy_pool or [],
        schedule=schedule or grpo_replay.EASY_MIX_SCHEDULE,
    )
    tr.replay_settings = grpo_replay.ReplaySettings(
        buffer=buffer,
        warmup_steps=1,
        mix_exploit_ratio=0.5,
        constant_test_reward=None,
    )
    tr.runtime_state = grpo_replay.RuntimeState()
    tr.model = SimpleNamespace(
        training=training,
        generation_config=SimpleNamespace(temperature=None),
        config=SimpleNamespace(pad_token_id=None),
    )
    tr.tokenizer = SimpleNamespace(
        pad_token_id=None,
        eos_token="[EOS]",
        eos_token_id=7,
    )
    tr.processing_class = tr.tokenizer
    tr.replay_buffer = buffer
    return tr


def test_ensure_pad_token_on_adds_tokens_and_resize():
    grpo_replay = _install_stubs()

    class Tok:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = None

        def __len__(self):
            return 0

        def add_special_tokens(self, mapping):
            self.pad_token_id = 99
            self.mapping = mapping

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None)
            self.resized = False

        def resize_token_embeddings(self, _size):
            self.resized = True

    tok = Tok()
    model = Model()
    grpo_replay.GRPOTrainerReplay._ensure_pad_token_on(tok, model)
    assert tok.pad_token_id == 99 and model.config.pad_token_id == 99
    assert model.resized is True
    assert getattr(tok, "padding_side", "left") == "left"


def test_update_temperature_and_inject_flag():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    t1 = tr._update_generation_temperature(step=1)
    assert pytest.approx(t1) == 0.75
    t2 = tr._update_generation_temperature(step=5)
    tr.runtime_state.gen_round = -1
    tr.temp_end = t2
    gen_round, inject_now, is_rank0 = tr._compute_inject_now_flag(step=8, new_temperature=t2)
    assert gen_round == 2 and isinstance(inject_now, bool) and is_rank0 is True


def test_maybe_mix_easy_batch_inserts_group(monkeypatch):
    grpo_replay = _install_stubs()
    easy = [{"prompt": "p", "mix_group_id": 0}]
    tr = make_trainer(grpo_replay, easy_pool=easy, schedule=[(0, 1.0)])
    tr.runtime_state.mix_group_counter = 0
    monkeypatch.setattr(grpo_replay.random, "random", lambda: 0.0)
    batch = [{"task": "MATH"} for _ in range(4)]
    mixed = tr._maybe_mix_easy_batch(batch, step=0)
    assert mixed[0]["task"] == "EASY"
    assert tr.runtime_state.mix_group_counter == 1


def test_p_easy_handles_empty_schedule():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    tr.mix_settings.schedule = []
    assert tr._p_easy(step=5) == 0.0


def test_maybe_inject_replay_group_calls_inject(monkeypatch):
    grpo_replay = _install_stubs()

    class Buf:
        def __len__(self):
            return 2

        def sample_uid(self, mix_exploit_ratio):
            return 1

        def get_group(self, uid):
            return [{"prompt": "p"}]

    tr = make_trainer(grpo_replay, buffer=Buf())
    tr.replay_buffer = tr.replay_settings.buffer
    called = {}

    def fake_inject(batch, uid, group):
        called["uid"] = uid
        return batch

    monkeypatch.setattr(tr, "_inject_group", fake_inject)
    batch = tr._maybe_inject_replay_group([{"prompt": "x"} for _ in range(2)], step=0, inject_now=True, is_rank0=True)
    assert called["uid"] == 1
    assert isinstance(batch, list)


def test_maybe_inject_replay_group_handles_errors(capfd):
    grpo_replay = _install_stubs()

    class Buf:
        def __len__(self):
            return 3

        def sample_uid(self, mix_exploit_ratio):
            raise KeyError("missing")

    tr = make_trainer(grpo_replay, buffer=Buf())
    tr.replay_buffer = tr.replay_settings.buffer
    out = tr._maybe_inject_replay_group([{"prompt": "p"}], step=0, inject_now=True, is_rank0=True)
    assert isinstance(out, list)
    captured = capfd.readouterr().out
    assert "ReplayPrep][ERR" in captured


def test_compute_mean_reward_for_credit_handles_arrays():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay, buffer=None)
    tr.replay_settings.constant_test_reward = 0.25
    assert tr._compute_mean_reward_for_credit({"rewards": None}) == 0.25
    out = {"rewards": np.array([0.2, 0.4], dtype=np.float32)}
    assert tr._compute_mean_reward_for_credit(out) == pytest.approx(0.3)

    class BadArr:
        def __iter__(self):
            raise TypeError("bad")

    assert tr._compute_mean_reward_for_credit({"rewards": BadArr()}) == 0.25


def test_decode_replay_from_ids_and_text_keys():
    grpo_replay = _install_stubs()
    tok = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True, clean_up_tokenization_spaces=False: "".join(map(str, ids))
    )
    out = {
        "completion_ids": grpo_replay.torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]]),
    }
    dec = grpo_replay.GRPOTrainerReplay._decode_replay_from_ids(
        out["completion_ids"], out, batch_size=1, tokenizer=tok, is_rank0=True
    )
    assert dec == [["1200", "3400"]]
    assert (
        grpo_replay.GRPOTrainerReplay._decode_replay_from_ids(
            out["completion_ids"], out, batch_size=0, tokenizer=tok, is_rank0=True
        )
        is None
    )
    out_text = {"completions_text": ["a", "b", "c", "d"]}
    grouped = grpo_replay.GRPOTrainerReplay._decode_replay_from_text_keys(out_text, batch_size=2)
    assert grouped == [["a", "b"], ["c", "d"]]
    assert grpo_replay.GRPOTrainerReplay._decode_replay_from_text_keys({}, batch_size=1) is None


def test_decode_replay_from_ids_bad_shape(capfd):
    grpo_replay = _install_stubs()
    ids = grpo_replay.torch.tensor([[1, 2], [3, 4], [5, 6]])
    out = {"completion_mask": grpo_replay.torch.tensor([[1, 0], [1, 0], [1, 0]])}
    dec = grpo_replay.GRPOTrainerReplay._decode_replay_from_ids(
        ids,
        out,
        batch_size=2,
        tokenizer=SimpleNamespace(decode=lambda ids, **k: ids),
        is_rank0=True,
    )
    assert dec is None
    assert "unexpected shapes" in capfd.readouterr().out


def test_decode_replay_from_text_keys_zero_batch():
    grpo_replay = _install_stubs()
    assert (
        grpo_replay.GRPOTrainerReplay._decode_replay_from_text_keys({"completions_text": ["a"]}, batch_size=0) is None
    )


def test_align_and_patch_metadata():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    inputs = [{"_buf_uid": 1, "answer": "A1", "prompt": "p1"}, {"answer": "A2", "prompt": "p2"}]
    out = {"unsort_idx": grpo_replay.torch.tensor([1, 0]), "reward_kwargs": {}}
    batch_info = {"gold_answers": ["A1", "A2"], "prompts_list": ["p1", "p2"], "tasks_list": ["T", "T"]}
    completions = [["c1"], ["c2"]]
    comps2, info2 = tr._align_replay_metadata(inputs, out, completions, batch_info)
    assert info2["gold_answers"] == ["A2", "A2"]
    assert out["gold"] == ["A2", "A2"] and out["prompt"] == ["p2", "p2"]


def test_attach_rewards_and_flatten(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    tr.processing_class = SimpleNamespace()
    monkeypatch.setattr(grpo_replay, "reward_router", lambda **kwargs: [[1.0, 0.5]])
    out = {"advantages": grpo_replay.torch.tensor([0.0, 0.0])}
    comps = [["x", "y"]]
    batch_info = {
        "gold_answers": ["a"],
        "prompts_list": ["p"],
        "tasks_list": ["task"],
        "boards_list": None,
        "sizes_list": None,
        "moves_list": None,
    }
    tr._attach_rewards_from_completions(out, comps, batch_info)
    assert out["rewards"].tolist() == [1.0, 0.5]
    assert out["reward_kwargs"]["answer"] == ["a"]


def test_attach_rewards_tensor_passthrough(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    flat_tensor = grpo_replay.torch.tensor([0.2, 0.4])
    monkeypatch.setattr(grpo_replay, "reward_router", lambda **kwargs: flat_tensor)
    out = {"advantages": grpo_replay.torch.tensor([0.0, 0.0])}
    batch_info = {
        "gold_answers": ["a"],
        "prompts_list": ["p"],
        "tasks_list": ["task"],
        "boards_list": None,
        "sizes_list": None,
        "moves_list": None,
    }
    tr._attach_rewards_from_completions(out, [["x", "y"]], batch_info)
    assert isinstance(out["rewards"], grpo_replay.torch.Tensor)


def test_maybe_credit_injected_uids_guards_and_errors(capfd):
    grpo_replay = _install_stubs()

    class Buffer:
        def update_priority_by_uid(self, uid, reward):
            raise ValueError("fail")

    tr = make_trainer(grpo_replay, buffer=Buffer())
    tr.replay_buffer = tr.replay_settings.buffer
    tr.accelerator.is_main_process = True
    tr.runtime_state.latest_injected_uids = [1]
    tr._maybe_credit_injected_uids({"rewards": grpo_replay.torch.tensor([0.0])})
    out = capfd.readouterr().out
    assert "update_priority_by_uid failed" in out
    assert tr.runtime_state.latest_injected_uids == []


def test_build_and_gather_winners(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    completions = [["good", "bad"], ["nohit"]]
    golds = ["good", "other"]
    prompts = ["p1", "p2"]
    monkeypatch.setattr(
        grpo_replay,
        "pure_accuracy_reward",
        lambda preds, golds: [1.0 if p == g else 0.0 for p, g in zip(preds, golds)],
    )
    winners_local = tr._build_local_replay_winners(completions, golds, prompts, {"is_rank0": True, "rank": 0})
    assert winners_local and winners_local[0]["answer"] == "good"
    gathered = tr._gather_replay_winners(winners_local, {"world": 1, "is_rank0": True})
    assert gathered == winners_local


def test_push_replay_winners_to_buffer_and_credit():
    grpo_replay = _install_stubs()

    class Buffer:
        def __len__(self):
            return 1

        def add_group(self, examples, reward):
            self.added = examples
            return True, 9

        def update_priority_by_uid(self, uid, reward):
            self.updated = (uid, reward)

        def debug_state(self):
            return {"len": 1}

    buf = Buffer()
    tr = make_trainer(grpo_replay, buffer=buf)
    tr.replay_buffer = buf
    tr.model.training = True
    tr.runtime_state.latest_injected_uids = [9]
    out = {"rewards": [1.0], "reward_kwargs": {}}
    winners = [{"prompt": "u", "answer": "a"}, {"prompt": "u", "answer": "a"}]
    tr._push_replay_winners_to_buffer(out, winners, world=1, is_rank0=True)
    assert hasattr(buf, "added") and len(buf.added) == 1
    assert not tr.runtime_state.latest_injected_uids


def test_push_replay_winners_handles_truthy_empty_iterable(capfd):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay, buffer=object())
    tr.replay_buffer = tr.replay_settings.buffer
    tr.model.training = True

    empty_generator = (_ for _ in ())
    tr._push_replay_winners_to_buffer({}, empty_generator, world=1, is_rank0=True)
    assert "[Replay][DEBUG] gathered no unique winners to add." in capfd.readouterr().out


def test_inject_group_logs_head_user_for_small_batch(capfd):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    tr.accelerator.is_main_process = True

    generation_batch = [{"messages": [{"role": "user", "content": "hello user"}]}]
    group = [{"prompt": "p"}]
    new_batch = tr._inject_group(generation_batch, uid=7, group=group)

    captured = capfd.readouterr().out
    assert "hello user" in captured
    assert new_batch[0]["messages"][0]["content"] == "hello user"
    assert tr.runtime_state.latest_injected_uids == [7]


def test_extract_completions_and_rewards_padding():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    out = {"completions": ["a"], "rewards": [0.1]}
    comps, rewards = tr._extract_completions_and_rewards(out, batch_size=2)
    assert comps == ["a", None]
    assert rewards == [0.1, 0.0]


def test_ensure_pad_token_on_none_and_readonly():
    grpo_replay = _install_stubs()
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id="keep"))
    # tok is None -> early return
    grpo_replay.GRPOTrainerReplay._ensure_pad_token_on(None, model)

    class ReadOnlyTok:
        pad_token_id = 5
        eos_token = "<eos>"
        eos_token_id = 5

        def __len__(self):
            return 0

        @property
        def padding_side(self):  # no setter → AttributeError on assign
            return "right"

    tok = ReadOnlyTok()
    grpo_replay.GRPOTrainerReplay._ensure_pad_token_on(tok, model)
    assert model.config.pad_token_id == "keep"


def test_update_generation_temperature_public(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    monkeypatch.setattr(tr, "_update_generation_temperature", lambda step: step + 0.1)
    assert tr.update_generation_temperature(3) == pytest.approx(3.1)


def test_maybe_mix_easy_batch_guard_and_skip(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay, easy_pool=[])
    sentinel = object()
    assert tr._maybe_mix_easy_batch(sentinel, step=0) is sentinel

    tr.mix_settings.easy_pool = [{"prompt": "p"}]
    # batch smaller than num_replicas → do_mix False with rank0 logging path
    monkeypatch.setattr(grpo_replay, "random", SimpleNamespace(random=lambda: 1.0))
    batch = [{"task": "x"} for _ in range(2)]
    assert tr._maybe_mix_easy_batch(batch, step=0) == batch


def test_maybe_credit_injected_uids_guards(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    tr.runtime_state.latest_injected_uids = [1]
    monkeypatch.setattr(grpo_replay, "_is_rank0", lambda accel: False)
    tr._maybe_credit_injected_uids({"rewards": [0.0]})
    assert tr.runtime_state.latest_injected_uids == []

    tr.runtime_state.latest_injected_uids = [2]
    monkeypatch.setattr(grpo_replay, "_is_rank0", lambda accel: True)
    tr.replay_settings.buffer = None
    tr._maybe_credit_injected_uids({"rewards": [0.0]})
    assert tr.runtime_state.latest_injected_uids == []


def test_generate_and_score_completions_collated_batch(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    base_cls = grpo_replay.GRPOTrainerReplay.__mro__[1]
    monkeypatch.setattr(
        base_cls, "_generate_and_score_completions", lambda self, inputs: {"base": inputs}, raising=False
    )
    tr._ensure_pad_token_on = lambda *a, **k: None
    tr._maybe_throttle_vllm = lambda: None
    tr._build_replay_batch_info = lambda inputs: ({"orig_inputs": {"x": 1}, "batch_size": 0}, inputs)
    tr._normalize_batch_for_trl = lambda x: x
    tr._decode_replay_completions = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not decode"))
    out = tr._generate_and_score_completions({"prompt": "p"})
    assert out["base"] == {"prompt": "p"}


def test_generate_and_score_completions_none_completions(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    base_cls = grpo_replay.GRPOTrainerReplay.__mro__[1]
    monkeypatch.setattr(
        base_cls, "_generate_and_score_completions", lambda self, inputs: {"base": inputs}, raising=False
    )
    tr._ensure_pad_token_on = lambda *a, **k: None
    tr._maybe_throttle_vllm = lambda: None
    tr._build_replay_batch_info = lambda inputs: ({"orig_inputs": inputs, "batch_size": 1}, inputs)
    tr._normalize_batch_for_trl = lambda x: x
    tr._decode_replay_completions = lambda *a, **k: None
    out = tr._generate_and_score_completions(["p"])
    assert out["base"] == ["p"]


def test_maybe_throttle_vllm_returns_early(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    tr.accelerator.is_main_process = False
    tr.runtime_state.vllm_cooldown = 5
    tr.runtime_state.last_vllm_upload_ts = 0.0
    tr._maybe_throttle_vllm()  # guarded by is_main_process

    tr.accelerator.is_main_process = True
    tr.runtime_state.last_vllm_upload_ts = None
    tr._maybe_throttle_vllm()  # guarded by missing timestamp


def test_build_replay_batch_info_nonlist():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    batch_info, clean = tr._build_replay_batch_info({"prompt": "p"})
    assert batch_info["batch_size"] == 0 and clean["prompt"] == "p"


def test_align_metadata_anchor_and_reward_kwargs(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    inputs = [
        {"_buf_uid": 1, "answer": "A1", "prompt": "P1"},
        {"answer": "A2", "prompt": "P2"},
    ]
    out = {"reward_kwargs": {"gold": ["old"], "answer": ["old"], "tasks": "old"}}
    batch_info = {
        "gold_answers": ["A1", "A2"],
        "prompts_list": ["P1", "P2"],
        "tasks_list": ["T1", "T2"],
    }
    completions = [["c1"], ["c2"]]
    comps2, info2 = tr._align_replay_metadata(inputs, out, completions, batch_info)
    assert comps2[0][0] == "c1"
    assert info2["gold_answers"] == ["A1", "A1"]
    assert out["gold"] == ["A1", "A1"] and out["reward_kwargs"]["gold"] == ["A1", "A1"]


def test_build_local_replay_winners_skips_invalid():
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    winners = tr._build_local_replay_winners(
        completions_txt=[["x"], ["y"]],
        gold_answers=[None, ""],
        prompts_list=[None, "p"],
        dist_info={"is_rank0": False},
    )
    assert winners == []


def test_select_replay_winners_calls_helpers(monkeypatch):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    monkeypatch.setattr(tr, "_build_local_replay_winners", lambda *a, **k: ["local"])
    monkeypatch.setattr(tr, "_gather_replay_winners", lambda winners, dist_info: winners + ["gathered"])
    result = tr._select_replay_winners(
        [["c"]], {"gold_answers": [], "prompts_list": []}, {"rank": 0, "world": 1, "is_rank0": True}
    )
    assert result == ["local", "gathered"]


def test_push_replay_winners_guards_and_failures(monkeypatch, capfd):
    grpo_replay = _install_stubs()
    # Guard when not training / not rank0
    tr = make_trainer(grpo_replay, buffer=SimpleNamespace())
    tr.model.training = False
    tr._push_replay_winners_to_buffer({"rewards": []}, [{"prompt": "p", "answer": "a"}], world=1, is_rank0=True)

    class BufFail:
        def __len__(self):
            return 1

        def add_group(self, examples, reward):
            self.called = True
            return False, -1

        def debug_state(self):
            return {"len": 1}

    buf = BufFail()
    tr2 = make_trainer(grpo_replay, buffer=buf)
    tr2.replay_buffer = buf
    tr2.model.training = True
    monkeypatch.setattr(tr2, "_maybe_credit_injected_uids", lambda out: (_ for _ in ()).throw(ValueError("fail")))
    winners = [{"prompt": [{"role": "user", "content": " hi "}], "answer": "A"}]
    tr2._push_replay_winners_to_buffer({"rewards": [1.0], "reward_kwargs": {}}, winners, world=1, is_rank0=True)
    assert getattr(buf, "called", False) is True
    captured = capfd.readouterr().out
    assert "add_group failed" in captured and "credit failed" in captured


def test_push_replay_winners_add_group_scalar_success(monkeypatch):
    grpo_replay = _install_stubs()

    class Buf:
        def __len__(self):
            return 1

        def add_group(self, examples, reward):
            self.examples = examples
            return 7  # non-tuple path

        def update_priority_by_uid(self, uid, reward):
            self.updated = (uid, reward)

    buf = Buf()
    tr = make_trainer(grpo_replay, buffer=buf)
    tr.replay_buffer = buf
    tr.model.training = True
    tr._maybe_credit_injected_uids = lambda out: setattr(tr.runtime_state, "latest_injected_uids", [])
    tr._push_replay_winners_to_buffer(
        {"rewards": [1.0], "reward_kwargs": {}}, [{"prompt": "p", "answer": "a"}], world=1, is_rank0=True
    )
    assert hasattr(buf, "examples")


def test_inject_group_empty_and_head_user(monkeypatch, capfd):
    grpo_replay = _install_stubs()
    tr = make_trainer(grpo_replay)
    # Empty group → pass through
    orig = [{"prompt": "x"}]
    assert tr._inject_group(orig, uid=1, group=[]) == orig

    monkeypatch.setattr(grpo_replay, "_is_rank0", lambda accel: True)
    batch = [{"prompt": "orig"} for _ in range(4)]
    group = [
        {"messages": [{"role": "user", "content": "hello"}], "prompt": None, "answer": "a"},
        {"messages": [{"role": "assistant", "content": "irrelevant"}], "answer": "b"},
    ]
    new_batch = tr._inject_group(batch, uid=3, group=group)
    assert new_batch[0]["is_replay"] is True
    assert tr.runtime_state.latest_injected_uids == [3]
    out = capfd.readouterr().out
    assert "ReplayInject" in out
