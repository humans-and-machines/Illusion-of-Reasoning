#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.data


# Stub torch and runtime dependencies before importing the trainer.
fake_torch = SimpleNamespace(
    Tensor=_FakeTensor,
    no_grad=lambda *_a, **_k: (lambda fn: fn),
    tensor=lambda data=None, **_k: _FakeTensor(data),
    ones=lambda shape, **_k: _FakeTensor(np.ones(shape)),
    zeros=lambda shape, **_k: _FakeTensor(np.zeros(shape)),
)
sys.modules["torch"] = fake_torch
sys.modules["torch.distributed"] = SimpleNamespace(
    is_initialized=lambda: False, get_rank=lambda: 0, get_world_size=lambda: 1
)
sys.modules["torch.nn.functional"] = SimpleNamespace()
sys.modules.setdefault("torch.serialization", SimpleNamespace(add_safe_globals=lambda *_a, **_k: None))

# Stub runtime.env module so the trainer import succeeds without heavy deps.
env_mod = types.ModuleType("src.training.runtime.env")
env_mod.torch = fake_torch
env_mod.dist = sys.modules["torch.distributed"]
env_mod.GRPOTrainer = type(
    "FakeGRPOTrainer",
    (),
    {
        "__init__": lambda self, *a, **k: None,
        "_prepare_inputs": lambda self, gb: gb,
        "_generate_and_score_completions": lambda self, inputs: {},
    },
)
env_mod.wandb = SimpleNamespace()
env_mod.TrainerCallback = type("TrainerCallback", (), {})
sys.modules["src.training.runtime.env"] = env_mod

import src.training.grpo_trainer_replay_impl as impl  # noqa: E402


def _trainer_with_state():
    """Build a trainer instance with minimal state without running __init__."""
    inst = impl.GRPOTrainerReplay.__new__(impl.GRPOTrainerReplay)
    inst.mix_settings = SimpleNamespace(schedule=[(0, 0.1), (5, 0.5)])
    inst.temperature_schedule = SimpleNamespace(
        start_temperature=1.0,
        end_temperature=0.2,
        anneal_steps=4,
        high_temperature_period=3,
    )
    inst.model = SimpleNamespace(
        generation_config=SimpleNamespace(temperature=None),
        config=SimpleNamespace(pad_token_id=None),
    )
    inst.processing_class = None
    inst.tokenizer = None
    inst.temp_end = inst.temperature_schedule.end_temperature
    inst.args = SimpleNamespace(steps_per_generation=2, local_rank=0)
    inst.state = SimpleNamespace(global_step=0)
    inst.runtime_state = SimpleNamespace(
        gen_round=-1,
        mix_group_counter=0,
        latest_injected_uids=[],
        printed_out_keys_once=False,
        vllm_cooldown=0,
        last_vllm_upload_ts=None,
        paired_debug_once=False,
    )
    inst.replay_settings = SimpleNamespace(
        buffer=None,
        warmup_steps=1,
        mix_exploit_ratio=0.5,
        constant_test_reward=None,
    )
    inst.accelerator = SimpleNamespace(is_main_process=True)
    inst.replay_buffer = None
    return inst


def test_ensure_pad_token_on_sets_pad_and_config():
    tok = SimpleNamespace(
        pad_token_id=None,
        eos_token="[eos]",
        eos_token_id=5,
        add_special_tokens=lambda mapping: mapping,
    )
    resized = {}

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(pad_token_id=None)

        def resize_token_embeddings(self, n):
            resized["n"] = n

    model = _Model()
    impl.GRPOTrainerReplay._ensure_pad_token_on(tok, model)
    assert tok.pad_token == "[eos]"
    assert model.config.pad_token_id == tok.pad_token_id


def test_temperature_schedule_and_mix_probabilities(monkeypatch):
    inst = _trainer_with_state()
    # step inside anneal
    temp = inst._update_generation_temperature(step=2)
    assert inst.model.generation_config.temperature == temp
    # step beyond anneal triggers sprinkle
    temp2 = inst._update_generation_temperature(step=10)
    assert temp2 in (inst.temperature_schedule.start_temperature, inst.temperature_schedule.end_temperature)

    prob0 = inst._p_easy(step=0)
    prob6 = inst._p_easy(step=6)
    assert prob0 == 0.1 and prob6 == 0.5
    assert inst.p_easy(3) == inst._p_easy(3)


def test_compute_inject_flag_and_mean_reward(monkeypatch):
    inst = _trainer_with_state()
    gen_round, inject_now, _ = inst._compute_inject_now_flag(step=4, new_temperature=0.2)
    assert gen_round == 2
    assert inject_now is True

    # Rewards as list/ndarray/tensor
    out_list = {"rewards": [1.0, 2.0]}
    out_np = {"rewards": np.array([1.0, 2.0])}
    out_tensor = {"rewards": _FakeTensor([1.0, 2.0])}
    monkeypatch.setattr(impl, "torch", SimpleNamespace(Tensor=_FakeTensor, no_grad=lambda *a, **k: (lambda fn: fn)))
    assert inst._compute_mean_reward_for_credit(out_list) == pytest.approx(1.5)
    assert inst._compute_mean_reward_for_credit(out_np) == pytest.approx(1.5)
    assert inst._compute_mean_reward_for_credit(out_tensor) == pytest.approx(1.5)


def test_build_replay_batch_info_handles_injected():
    inst = _trainer_with_state()
    inputs = [
        {"_buf_uid": 1, "answer": "a1", "prompt": "p1"},
        {"answer": "a2", "prompt": "p2"},
    ]
    batch_info, clean = inst._build_replay_batch_info(inputs)
    assert batch_info["injected"] is True
    assert batch_info["gold_answers"] == ["a1", "a1"]
    assert all(ex["prompt"] == "p1" for ex in clean)


def test_normalize_batch_for_trl(monkeypatch):
    inst = _trainer_with_state()
    calls = []
    monkeypatch.setattr(impl, "_normalize_for_trl", lambda example, proc: calls.append(example) or {"norm": example})
    norm = inst._normalize_batch_for_trl([{"a": 1}, {"b": 2}])
    assert len(calls) == 2
    assert isinstance(norm, list) and norm[0]["norm"] == {"a": 1}


def test_throttle_vllm_and_dump_once(monkeypatch):
    inst = _trainer_with_state()
    inst.runtime_state.vllm_cooldown = 1
    inst.runtime_state.last_vllm_upload_ts = time.time()
    slept = {}
    monkeypatch.setattr(time, "sleep", lambda s: slept.setdefault("slept", s))
    inst._maybe_throttle_vllm()
    assert slept

    # _dump_out_once only prints once
    out = {"a": 1}
    inst._dump_out_once(out)
    inst._dump_out_once(out)
    assert inst.runtime_state.printed_out_keys_once is True


def test_maybe_mix_easy_batch_replaces_group(monkeypatch):
    inst = _trainer_with_state()
    inst.mix_settings.easy_pool = [{"prompt": "easy_prompt"}]
    inst.runtime_state.mix_group_counter = 3
    # Force mixing to trigger
    monkeypatch.setattr(impl, "_is_rank0", lambda accel: False)
    monkeypatch.setattr(impl.random, "random", lambda: 0.0)
    monkeypatch.setattr(impl.random, "choice", lambda pool: pool[0])

    def fake_label(ex, group_id, copy_idx, total=4):
        ex["mix_group_id"] = group_id
        ex["mix_copy_idx"] = copy_idx
        ex["total"] = total
        return ex

    monkeypatch.setattr(impl, "_label_easy_copy", fake_label)
    # Ensure probability gate is 1.0
    monkeypatch.setattr(inst, "_p_easy", lambda step: 1.0)
    batch = [{"orig": i} for i in range(4)]
    mixed = inst._maybe_mix_easy_batch(batch, step=10)
    assert all(ex["task"] == "EASY" for ex in mixed[:4])
    assert mixed[0]["mix_group_id"] == 3 and mixed[0]["mix_copy_idx"] == 1
    assert inst.runtime_state.mix_group_counter == 4


def test_maybe_inject_replay_group_injects_and_marks(monkeypatch):
    inst = _trainer_with_state()
    inst.accelerator.process_index = 0

    class _Buf:
        def __len__(self):
            return 5

        def sample_uid(self, mix_exploit_ratio):
            self.mix = mix_exploit_ratio
            return 42

        def get_group(self, uid):
            return [{"prompt": "p", "answer": "a"}]

    inst.replay_settings = SimpleNamespace(
        buffer=_Buf(),
        warmup_steps=1,
        mix_exploit_ratio=0.9,
        constant_test_reward=None,
    )
    batch = [{"orig": i} for i in range(4)]
    injected = inst._maybe_inject_replay_group(
        batch,
        step=0,
        inject_now=True,
        is_rank0=True,
    )
    assert injected[0]["_buf_uid"] == 42
    assert injected[0]["is_replay"] is True
    assert inst.runtime_state.latest_injected_uids == [42]


def test_maybe_inject_replay_group_skips_when_not_ready():
    inst = _trainer_with_state()

    class _Buf:
        def __len__(self):
            return 0

    inst.replay_settings = SimpleNamespace(
        buffer=_Buf(),
        warmup_steps=5,
        mix_exploit_ratio=0.5,
        constant_test_reward=None,
    )
    batch = [{"orig": 1}]
    same = inst._maybe_inject_replay_group(
        batch,
        step=0,
        inject_now=True,
        is_rank0=True,
    )
    assert same is batch


def test_maybe_credit_injected_uids_updates_buffer(monkeypatch):
    inst = _trainer_with_state()
    inst.runtime_state.latest_injected_uids = [7, 8]
    calls = []

    class _Buf:
        def update_priority_by_uid(self, uid, reward):
            calls.append((uid, reward))

    inst.replay_settings = SimpleNamespace(
        buffer=_Buf(),
        warmup_steps=1,
        mix_exploit_ratio=0.5,
        constant_test_reward=None,
    )
    monkeypatch.setattr(impl, "_is_rank0", lambda accel: True)
    inst._maybe_credit_injected_uids({"rewards": [1.0, 3.0]})
    assert calls == [(7, 2.0), (8, 2.0)]
    assert inst.runtime_state.latest_injected_uids == []


def test_compute_mean_reward_constant_fallback(monkeypatch):
    inst = _trainer_with_state()
    inst.replay_settings.constant_test_reward = 0.7
    mean = inst._compute_mean_reward_for_credit({})
    assert mean == pytest.approx(0.7)


def test_normalize_batch_for_trl_dict_and_passthrough(monkeypatch):
    inst = _trainer_with_state()
    called = {}

    def _fake_norm(example, proc):
        called["example"] = example
        return {"norm": example, "proc": proc}

    monkeypatch.setattr(impl, "_normalize_for_trl", _fake_norm)
    normalized = inst._normalize_batch_for_trl({"a": 1})
    assert normalized["norm"] == {"a": 1}
    assert called["example"] == {"a": 1}
    unchanged = inst._normalize_batch_for_trl("raw-string")
    assert unchanged == "raw-string"


def test_inject_group_replaces_half_and_sets_defaults(monkeypatch):
    inst = _trainer_with_state()
    inst.accelerator.process_index = 2
    monkeypatch.setattr(impl, "_is_rank0", lambda accel: False)
    group = [
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
    ]
    batch = [{"orig": i} for i in range(4)]
    injected = inst._inject_group(batch, uid=9, group=group)
    assert len(injected) == 4
    assert injected[0]["_buf_uid"] == 9 and injected[0]["_buf_rank"] == 2
    assert injected[0]["is_replay"] is True
    assert injected[0]["task"] == "MATH"
    assert inst.runtime_state.latest_injected_uids == [9]


def test_build_replay_batch_info_populates_metadata():
    inst = _trainer_with_state()
    inputs = [
        {"answer": "a1", "prompt": "p1", "task": "X", "board": "b1", "size": 3, "moves": [1]},
        {"answer": "a2", "prompt": "p2", "task": "Y", "board_str": "b2", "N": 5, "gold_moves": [2]},
    ]
    batch_info, clean = inst._build_replay_batch_info(inputs)
    assert batch_info["batch_size"] == 2
    assert batch_info["injected"] is False
    assert batch_info["gold_answers"] == ["a1", "a2"]
    assert batch_info["boards_list"] == ["b1", "b2"]
    assert clean[0]["answer"] == "a1" and clean[1]["prompt"] == "p2"


def test_decode_replay_from_ids_and_text_keys(monkeypatch):
    inst = _trainer_with_state()

    class _Tensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr)

        @property
        def shape(self):
            return self.arr.shape

        def view(self, *shape):
            return _Tensor(self.arr.reshape(*shape))

        def bool(self):
            return self

        def __getitem__(self, idx):
            if isinstance(idx, _Tensor):
                idx = idx.arr
            if isinstance(idx, np.ndarray) and idx.dtype != bool:
                idx = idx.astype(bool)
            return _Tensor(self.arr[idx])

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.arr.tolist()

    class _Tok:
        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return "|".join(str(i) for i in ids)

    monkeypatch.setattr(impl, "torch", SimpleNamespace(Tensor=_Tensor, no_grad=lambda *a, **k: (lambda fn: fn)))
    out = {"completion_mask": _Tensor(np.ones((2, 2, 3)))}
    ids = _Tensor(np.arange(12).reshape(4, 3))
    completions = inst._decode_replay_from_ids(ids, out, batch_size=2, tokenizer=_Tok(), is_rank0=True)
    assert completions and len(completions) == 2 and completions[0][0].startswith("0|")

    out_txt = {"texts": ["a1", "a2", "b1", "b2"]}
    completions_txt = inst._decode_replay_from_text_keys(out_txt, batch_size=2)
    assert completions_txt == [["a1", "a2"], ["b1", "b2"]]


def test_align_replay_metadata_orders_and_injected(monkeypatch):
    inst = _trainer_with_state()
    completions_txt = [["c1"], ["c2"]]
    batch_info = {
        "gold_answers": ["g1", "g2"],
        "prompts_list": ["p1", "p2"],
        "tasks_list": ["t1", "t2"],
    }

    class _TensorOrder:
        def tolist(self):
            return [1, 0]

    out = {"unsort_idx": _TensorOrder()}
    monkeypatch.setattr(impl, "torch", SimpleNamespace(Tensor=_TensorOrder))
    inputs = [{"_buf_uid": 1, "prompt": "p2", "answer": "g2"}, {"prompt": "p1", "answer": "g1"}]

    completions_aligned, batch_info_out = inst._align_replay_metadata(inputs, out, completions_txt, batch_info)
    assert completions_aligned[0] == ["c2"]
    assert batch_info_out["gold_answers"][0] == "g2"
    assert out["prompts_list"][0] == "p2"


def test_patch_out_for_injected_updates_reward_kwargs():
    out = {"reward_kwargs": {"answer": "old"}}
    batch_info = {
        "gold_answers": ["g"],
        "prompts_list": ["p"],
        "tasks_list": ["t"],
    }
    impl.GRPOTrainerReplay._patch_out_for_injected(out, batch_info)
    assert out["gold"] == ["g"]
    assert out["prompts"] == ["p"]
    assert out["tasks"] == ["t"]
    assert out["reward_kwargs"]["answer"] == ["g"]


def test_attach_rewards_from_completions_flattens_scores(monkeypatch):
    inst = _trainer_with_state()
    out = {}
    batch_info = {
        "gold_answers": ["g"],
        "prompts_list": ["p"],
        "tasks_list": ["t"],
        "boards_list": ["b"],
        "sizes_list": [3],
        "moves_list": [[1]],
    }
    completions = [["a", "b"], ["c", "d"]]

    monkeypatch.setattr(impl, "reward_router", lambda **kwargs: [[0.1, 0.2], [0.3, 0.4]])
    inst._attach_rewards_from_completions(out, completions, batch_info)
    assert out["rewards"] == [0.1, 0.2, 0.3, 0.4]
    assert out["reward_kwargs"]["gold_moves"] == [[1]]


def test_build_local_replay_winners_and_gather(monkeypatch):
    inst = _trainer_with_state()

    def _fake_accuracy(preds, golds):
        # Mark correct only when gold matches first pred
        return [1.0 if golds[0] == preds[0] else 0.0 for _ in preds]

    monkeypatch.setattr(impl, "pure_accuracy_reward", _fake_accuracy)
    completions = [["good", "bad"], ["miss", "hit"]]
    golds = ["good", "x"]
    prompts = ["p1", "p2"]

    winners_local = inst._build_local_replay_winners(completions, golds, prompts, {"is_rank0": True, "rank": 0})
    assert len(winners_local) == 1 and winners_local[0]["answer"] == "good"

    def _all_gather_object(dest, src):
        dest[0] = src

    fake_dist = SimpleNamespace(is_initialized=lambda: True, all_gather_object=_all_gather_object)
    monkeypatch.setattr(impl, "dist", fake_dist)
    winners = inst._gather_replay_winners(winners_local, {"world": 1, "is_rank0": False})
    assert winners == winners_local


def test_push_replay_winners_to_buffer_adds_and_filters(monkeypatch):
    inst = _trainer_with_state()
    inst.model.training = True
    inst.accelerator = SimpleNamespace(is_main_process=True)

    class _Buf:
        def __len__(self):
            return 3

        def add_group(self, examples, reward=1.0):
            return True, 5

        def debug_state(self):
            return {"x": 1}

    inst.replay_buffer = _Buf()
    out = {}
    winners_all = [
        {"prompt": "u1", "answer": "a1"},
        {"prompt": "u1", "answer": "a1"},  # duplicate
    ]
    inst._push_replay_winners_to_buffer(out, winners_all, world=1, is_rank0=True)
    assert inst.runtime_state.latest_injected_uids == []

    # No winners path
    inst.replay_buffer = SimpleNamespace(
        __len__=lambda self: 0, add_group=lambda *a, **k: (False, -1), debug_state=lambda: {}
    )
    inst._push_replay_winners_to_buffer(out, [], world=1, is_rank0=True)


def test_extract_completions_and_rewards(monkeypatch):
    inst = _trainer_with_state()
    monkeypatch.setattr(impl, "_extract_rewards_for_logging", lambda out: [1, 2])
    monkeypatch.setattr(impl, "_extract_text_completions", lambda out, tok: ["a"])
    comps, rewards = inst._extract_completions_and_rewards({}, batch_size=2)
    assert comps == ["a", None] and rewards == [1, 2]


def test_init_populates_defaults_and_warns(monkeypatch, caplog):
    class _Tok:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = None
            self.added = False
            self.padding_side = None

        def add_special_tokens(self, mapping):
            self.added = mapping
            self.pad_token = list(mapping.values())[0]
            self.pad_token_id = 11
            return mapping

        def __len__(self):
            return 7

    resized = {}

    class _Model(SimpleNamespace):
        def resize_token_embeddings(self, n):
            resized["n"] = n

    base_cls = impl.GRPOTrainerReplay.__mro__[1]

    def fake_base_init(self, *args, **kwargs):
        self.model = _Model(
            generation_config=SimpleNamespace(temperature=None),
            config=SimpleNamespace(pad_token_id=None),
            training=True,
        )
        self.state = SimpleNamespace(global_step=0)
        self.args = SimpleNamespace()
        self.accelerator = SimpleNamespace()

    monkeypatch.setattr(base_cls, "__init__", fake_base_init)

    tok = _Tok()
    with caplog.at_level("WARNING"):
        trainer = impl.GRPOTrainerReplay(
            tokenizer=tok,
            processing_class=None,
            temp_start=1.2,
            temp_end=0.8,
            anneal_steps=2,
            high_temp_period=5,
            easy_pool=[{"p": 1}],
            mix_schedule=[(2, 0.2)],
            inject_every_batch=True,
        )
    assert tok.pad_token == "[PAD]"
    assert trainer.model.config.pad_token_id == tok.pad_token_id
    assert resized["n"] == len(tok)
    assert trainer.mix_settings.schedule == [(2, 0.2)]
    assert trainer.temperature_schedule.start_temperature == 1.2
    assert any("inject_every_batch flag" in rec.message for rec in caplog.records)


def test_prepare_inputs_short_circuit_and_normalizes(monkeypatch):
    inst = _trainer_with_state()
    base_cls = impl.GRPOTrainerReplay.__mro__[1]
    monkeypatch.setattr(base_cls, "_prepare_inputs", lambda self, gb: gb)

    # Short-circuit when not training
    inst.model.training = False
    inst.replay_settings.buffer = None
    sentinel = inst._prepare_inputs([{"x": 1}])
    assert sentinel == [{"x": 1}]

    # Full path with hooks and normalization
    inst.model.training = True
    inst.replay_settings.buffer = SimpleNamespace(__len__=lambda self: 10)
    calls = {}
    monkeypatch.setattr(inst, "_update_generation_temperature", lambda step: calls.setdefault("temp", step) or 0.5)
    monkeypatch.setattr(inst, "_maybe_mix_easy_batch", lambda gb, step: calls.setdefault("mix", step) or gb)
    monkeypatch.setattr(inst, "_compute_inject_now_flag", lambda step, new_temp: (1, True, True))

    def _record_inject(gb, step, inject, rank0):
        calls["inject"] = (step, inject, rank0)
        return gb

    monkeypatch.setattr(inst, "_maybe_inject_replay_group", _record_inject)
    batch = [{} for _ in range(2)]
    out = inst._prepare_inputs(batch)
    assert all(ex["task"] == "MATH" and ex["mix_group_id"] == -1 for ex in out)
    assert calls["temp"] == inst.state.global_step
    assert calls["inject"][1] is True


def test_decode_replay_completions_updates_metadata(monkeypatch):
    inst = _trainer_with_state()
    inst.accelerator = SimpleNamespace(is_main_process=True)
    inst.runtime_state.last_vllm_upload_ts = None

    class _Tensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr)

        @property
        def shape(self):
            return self.arr.shape

        def view(self, *shape):
            return _Tensor(self.arr.reshape(*shape))

        def bool(self):
            return self

        def __getitem__(self, idx):
            if isinstance(idx, _Tensor):
                idx = idx.arr
            return _Tensor(self.arr[idx])

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.arr.tolist()

    class _Tok:
        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return ",".join(str(i) for i in ids)

    monkeypatch.setattr(impl, "torch", SimpleNamespace(Tensor=_Tensor, no_grad=lambda *a, **k: (lambda fn: fn)))
    out = {
        "completion_ids": _Tensor(np.arange(6).reshape(2, 3)),
        "completion_mask": _Tensor(np.ones((2, 3), dtype=bool)),
    }
    batch_info = {"batch_size": 1}
    completions = inst._decode_replay_completions(out, batch_info, tokenizer=_Tok(), is_rank0=True)
    assert completions and batch_info["num_candidates"] == 2
    assert inst.runtime_state.last_vllm_upload_ts is not None


def test_decode_replay_completions_none_sets_timestamp(monkeypatch):
    inst = _trainer_with_state()
    inst.accelerator = SimpleNamespace(is_main_process=True)
    inst.runtime_state.last_vllm_upload_ts = None
    batch_info = {"batch_size": 1}
    result = inst._decode_replay_completions({}, batch_info, tokenizer=None, is_rank0=True)
    assert result is None
    assert inst.runtime_state.last_vllm_upload_ts is not None


def test_generate_and_score_completions_calls_hooks(monkeypatch):
    inst = _trainer_with_state()
    base_cls = impl.GRPOTrainerReplay.__mro__[1]
    monkeypatch.setattr(base_cls, "_generate_and_score_completions", lambda self, inputs: {"base_out": inputs})
    flags = {}
    inst._ensure_pad_token_on = lambda *a, **k: flags.setdefault("pad", True)
    inst._maybe_throttle_vllm = lambda: flags.setdefault("throttle", True)
    inst._build_replay_batch_info = lambda inputs: ({"orig_inputs": inputs, "batch_size": 1}, inputs)

    def _normalize(clean):
        flags["norm"] = clean
        return "cleaned"

    inst._normalize_batch_for_trl = _normalize
    inst._decode_replay_completions = lambda out, info, tok, is_rank0: [["pred"]]
    inst._align_replay_metadata = lambda inputs, out, comps, info: (comps, info)
    inst._attach_rewards_from_completions = lambda out, comps, info: flags.setdefault("attach", True)
    inst._select_replay_winners = lambda comps, info, dist_info: [{"prompt": "p", "answer": "a"}]
    inst._push_replay_winners_to_buffer = lambda out, winners_all, world, is_rank0: flags.setdefault(
        "push", (world, is_rank0)
    )

    out = inst._generate_and_score_completions(["raw"])
    assert out["base_out"] == "cleaned"
    assert flags["pad"] and flags["attach"] and flags["push"]


def test_gather_replay_winners_multi_rank(monkeypatch):
    inst = _trainer_with_state()

    def _all_gather_object(dest, src):
        dest[0] = src
        dest[1] = [{"prompt": "p2"}]

    fake_dist = SimpleNamespace(is_initialized=lambda: True, all_gather_object=_all_gather_object)
    monkeypatch.setattr(impl, "dist", fake_dist)
    winners = inst._gather_replay_winners([{"prompt": "p1"}], {"world": 2, "is_rank0": True})
    assert len(winners) == 2
