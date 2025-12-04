import csv
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


class FakeTensor:
    def __init__(self, data, dtype="float32", shape=None):
        self.data = list(data)
        self.dtype = dtype
        self.shape = shape or (len(self.data),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def tolist(self):
        return list(self.data)


# Install lightweight stubs so import-time dependencies resolve in minimal test envs.
try:  # pragma: no cover - prefer real torch if present
    import torch as _torch  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in constrained envs
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = FakeTensor

    def _tensor(data=None, **_kwargs):
        return FakeTensor(data or [])

    torch_stub.tensor = _tensor
    sys.modules["torch"] = torch_stub
else:
    torch_stub = sys.modules["torch"]

# Stub runtime.env to avoid heavy imports and missing TrainerCallback in minimal envs.
env_stub = types.ModuleType("src.training.runtime.env")
env_stub.torch = torch_stub
env_stub.wandb = SimpleNamespace(log=lambda *a, **k: None)


class _TrainerCallback:
    def __init__(self, *a, **k): ...


env_stub.TrainerCallback = _TrainerCallback
sys.modules["src.training.runtime.env"] = env_stub

# Avoid importing the heavyweight runtime.main during test collection.
main_stub = types.ModuleType("src.training.runtime.main")
main_stub.main = lambda *_, **__: None
sys.modules["src.training.runtime.main"] = main_stub

import src.training.grpo_trainer_replay_support as h4  # noqa: E402


class FakeTokenizer:
    def __init__(self):
        self.decoded = []

    def batch_decode(self, seqs, skip_special_tokens=True):
        self.decoded.append((seqs, skip_special_tokens))
        # simple identity to strings
        return [f"decoded-{i}" for i, _ in enumerate(seqs)]


def test_replay_and_temperature_settings():
    replay = h4.ReplaySettings(buffer="buf", warmup_steps=3, mix_exploit_ratio=0.5, constant_test_reward=None)
    assert replay.as_dict()["buffer"] == "buf"
    assert replay.is_warmed_up(4) is True
    schedule = h4.TemperatureSchedule(1.0, 0.5, anneal_steps=10, high_temperature_period=2)
    assert schedule.as_dict()["start_temperature"] == 1.0
    assert schedule.fraction_complete(5) == 0.5
    assert schedule.fraction_complete(11) == 1.0
    zero = h4.TemperatureSchedule(1.0, 1.0, anneal_steps=0, high_temperature_period=1)
    assert zero.fraction_complete(5) == 1.0


def test_mix_and_runtime_state_helpers():
    mix = h4.MixSettings(easy_pool=[{"a": 1}], schedule=[(0, 1.0)])
    assert mix.is_enabled() and mix.as_dict()["easy_pool_size"] == 1
    state = h4.RuntimeState(latest_injected_uids=[1, 2], vllm_cooldown=2, last_vllm_upload_ts=123.0)
    state.reset_latest_uids()
    assert state.latest_injected_uids == []
    assert state.should_throttle_vllm() is True
    state.vllm_cooldown = 0
    assert state.should_throttle_vllm() is False
    assert h4._is_rank0(SimpleNamespace(is_main_process=False)) is False


def test_to_env_schema_and_logging_shortener():
    example = {"prompt": [{"role": "user", "content": "hello\nworld"}], "answer": "x", "metadata": {"k": 1}}
    env_schema = h4._to_env_schema(example)
    assert "prompt" in env_schema and env_schema["metadata"] == {"k": 1}
    long_prompt = [{"role": "user", "content": "a" * 200}]
    shortened = h4._shorten_for_log(long_prompt, max_chars=10)
    assert shortened.endswith("…")


def test_float_and_reward_extraction(monkeypatch):
    monkeypatch.setattr(h4, "torch", SimpleNamespace(Tensor=FakeTensor))
    assert h4._to_float_list(FakeTensor([1, 2])) == [1, 2]
    assert h4._to_float_list(np.array([3, 4])) == [3.0, 4.0]
    assert h4._to_float_list(["bad"]) is None
    out = {"scores": FakeTensor([0.2, 0.4])}
    assert h4._extract_rewards_for_logging(out) == [0.2, 0.4]


def test_pad_and_decode_batch_sequences(monkeypatch):
    padded = h4._pad_to_batch_size([1, 2], pad_value=0, batch_size=4)
    assert padded == [1, 2, 0, 0]
    trimmed = h4._pad_to_batch_size([1, 2, 3], pad_value=0, batch_size=2)
    assert trimmed == [1, 2]

    monkeypatch.setattr(h4, "torch", SimpleNamespace(Tensor=FakeTensor))
    tokenizer = FakeTokenizer()
    seqs = FakeTensor([[1, 2], [3, 4]])
    decoded = h4._decode_batch_sequences(seqs, tokenizer)
    assert decoded == ["decoded-0", "decoded-1"]
    assert tokenizer.decoded  # recorded call
    assert h4._decode_batch_sequences(None, tokenizer) is None


def test_extract_text_completions_paths(monkeypatch):
    tokenizer = FakeTokenizer()
    primary = h4._extract_text_completions({"completions_text": ["a", "b"]}, tokenizer)
    assert primary == ["a", "b"]
    decoded = h4._extract_text_completions({"completion_ids": [[1], [2]]}, tokenizer)
    assert decoded == ["decoded-0", "decoded-1"]

    class GenOut:
        def __init__(self):
            self.sequences = FakeTensor([[5], [6]])

    gen = GenOut()
    via_generation_outputs = h4._extract_text_completions({"generation_outputs": gen}, tokenizer)
    assert via_generation_outputs == ["decoded-0", "decoded-1"]

    none_case = h4._extract_text_completions({}, tokenizer)
    assert none_case is None


def test_label_easy_copy_and_summary(monkeypatch):
    monkeypatch.setattr(h4, "torch", SimpleNamespace(Tensor=FakeTensor))
    ex = {"prompt": [{"role": "user", "content": "[G:1] [COPY:1/4]\nold"}]}
    labeled = h4._label_easy_copy(ex, group_id=2, copy_idx=3, total=4)
    assert labeled["mix_group_id"] == 2 and labeled["task"] == "EASY"
    assert "[G:2]" in labeled["prompt"][0]["content"]
    summary = h4._summarize_val(FakeTensor([1, 2], shape=(2,), dtype="float32"))
    assert "Tensor" in summary and "float32" in summary
    assert h4._summarize_val([1, 2]).startswith("list")
    assert h4._summarize_val({"a": 1}).startswith("dict")
    assert h4._summarize_val({"a": 1, "b": 2}).startswith("dict")


def test_join_and_normalize_for_trl(monkeypatch):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    joined = h4._default_join_messages(msgs)
    assert "USER: hi" in joined and joined.endswith("ASSISTANT:")

    class Proc:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return f"TEMPLATE {len(messages)} {add_generation_prompt}"

    example = {"messages": msgs}
    normalized = h4._normalize_for_trl(example, proc=Proc(), add_gen_prompt=False)
    assert normalized["prompt"].startswith("TEMPLATE")

    normalized_prompt_list = h4._normalize_for_trl({"prompt": msgs}, proc=None)
    assert "USER: hi" in normalized_prompt_list["prompt"]
    assert h4._normalize_for_trl({"prompt": "raw text"})["prompt"] == "raw text"
    with pytest.raises(ValueError):
        h4._normalize_for_trl({"other": 1})

    # message list path with no proc uses default join
    normalized_msgs = h4._normalize_for_trl({"messages": msgs}, proc=None)
    assert "USER: hi" in normalized_msgs["prompt"]


def test_loss_logging_callback(monkeypatch, tmp_path):
    log_calls = []
    monkeypatch.setattr(h4, "wandb", SimpleNamespace(log=lambda payload, step: log_calls.append((payload, step))))

    callback = h4.LossLoggingCallback(output_dir=str(tmp_path))
    args = SimpleNamespace(local_rank=0)
    state = SimpleNamespace(global_step=5)
    logs = {
        "loss/policy_loss": 1.2,
        "loss/value_loss": 0.5,
        "beta": 0.9,
        "ignore": 99,
    }
    callback.on_log(args, state, None, logs=logs)

    assert log_calls == [({"policy_loss": 1.2, "value_loss": 0.5, "beta": 0.9}, 5)]
    csv_path = tmp_path / "loss_history.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["step"] == "5"
    assert rows[0]["policy_loss"] == "1.2"

    # Skips non-rank0 and missing payloads
    args_bad = SimpleNamespace(local_rank=2)
    callback.on_log(args_bad, state, None, logs=logs)
    args_payload = SimpleNamespace(local_rank=0)
    callback.on_log(args_payload, state, None, logs={"unrelated": 1})
    assert callback.on_train_begin() is None


def test_shorten_and_float_and_pad_paths(monkeypatch):
    assert h4._shorten_for_log([], max_chars=5) == ""
    assert h4._shorten_for_log([{"role": "assistant", "content": "hi"}], max_chars=5) == ""
    assert h4._to_float_list(None) is None
    assert h4._to_float_list([1, 2.5]) == [1.0, 2.5]
    assert h4._to_float_list([object()]) is None
    assert h4._pad_to_batch_size(None, pad_value=0, batch_size=2) is None


def test_decode_and_extract_errors(monkeypatch):
    class BadSeq:
        def detach(self):
            raise RuntimeError("fail")

    bad = h4._decode_batch_sequences(BadSeq(), FakeTokenizer())
    assert bad is None

    class BadTok:
        def batch_decode(self, seqs, skip_special_tokens=True):
            raise TypeError("fail")

    assert h4._decode_batch_sequences([[1, 2]], BadTok()) is None

    tok = FakeTokenizer()
    seq_obj = type("Seq", (), {"sequences": FakeTensor([[1], [2]])})
    decoded = h4._extract_text_completions({"generation_outputs": (seq_obj(),)}, tok)
    assert decoded == ["decoded-0", "decoded-1"]


def test_additional_branches_for_float_decode_and_summary(tmp_path):
    # _init_csv early-return guard (line 135)
    cb = h4.LossLoggingCallback(output_dir=str(tmp_path))
    payload = {"policy_loss": 1.0}
    cb._init_csv(payload)
    before = (tmp_path / "loss_history.csv").read_text()
    cb._init_csv(payload)
    assert (tmp_path / "loss_history.csv").read_text() == before

    # _to_float_list final None branch (line 247)
    assert h4._to_float_list("nope") is None

    # _decode_batch_sequences cpu-only path (line 285)
    class CpuOnly:
        def __init__(self, seqs):
            self._seqs = seqs

        def cpu(self):
            return self

        def tolist(self):
            return self._seqs

    tok = FakeTokenizer()
    decoded_cpu = h4._decode_batch_sequences(CpuOnly([[9], [8]]), tok)
    assert decoded_cpu == ["decoded-0", "decoded-1"]

    # _summarize_val default path (line 367)
    assert h4._summarize_val((1, 2)) == "tuple"

    # _normalize_for_trl prompt-as-list with proc (line 395)
    msgs = [{"role": "user", "content": "hi"}]

    class Proc:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return f"templated-{len(messages)}-{add_generation_prompt}"

    proc = Proc()
    normalized = h4._normalize_for_trl({"prompt": msgs}, proc=proc)
    assert normalized["prompt"].startswith("templated-1-True")
    assert proc.calls and proc.calls[0][1] is False and proc.calls[0][2] is True


def test_label_easy_copy_prompt_and_messages(monkeypatch):
    ex_prompt = {"prompt": "[G:1] [COPY:1/4]\nhello"}
    out = h4._label_easy_copy(ex_prompt, group_id=3, copy_idx=2, total=5)
    assert out["prompt"].startswith("[G:3] [COPY:2/5]")
    ex_messages = {"messages": [{"role": "user", "content": "[G:1] [COPY:1/4]\nold"}]}
    out2 = h4._label_easy_copy(ex_messages, group_id=4, copy_idx=1, total=2)
    assert "[G:4]" in out2["messages"][0]["content"]
