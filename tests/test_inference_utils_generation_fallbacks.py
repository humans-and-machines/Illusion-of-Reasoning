#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np
import pytest


gen = pytest.importorskip("src.inference.utils.generation")


class FakeTensor:
    """Lightweight tensor wrapper using numpy for arithmetic."""

    def __init__(self, arr):
        self.arr = np.array(arr)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, idx):
        return FakeTensor(self.arr[idx])

    def float(self):
        return self

    def any(self):
        return bool(np.any(self.arr))

    def exp(self):
        return FakeTensor(np.exp(self.arr))

    def sum(self, dim=None):
        axis = dim
        return FakeTensor(np.sum(self.arr, axis=axis))

    def item(self):
        return float(self.arr)

    def tolist(self):
        return self.arr.tolist()

    def max(self):
        return np.max(self.arr)

    def to(self, *_args, **_kwargs):
        return self

    def __mul__(self, other):
        other_arr = other.arr if isinstance(other, FakeTensor) else other
        return FakeTensor(self.arr * other_arr)

    __rmul__ = __mul__

    def __sub__(self, other):
        other_arr = other.arr if isinstance(other, FakeTensor) else other
        return FakeTensor(self.arr - other_arr)


class FakeTorch:
    """Minimal torch stand-in to exercise entropy fallbacks."""

    def __init__(self):
        self.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.nn = types.SimpleNamespace(functional=types.SimpleNamespace())

        def log_softmax(tensor, dim=-1):
            arr = tensor.arr
            arr = arr - np.max(arr, axis=dim, keepdims=True)
            exp = np.exp(arr)
            exp_sum = np.sum(exp, axis=dim, keepdims=True)
            return FakeTensor(np.log(exp / exp_sum))

        self.nn.functional.log_softmax = log_softmax

    def tensor(self, data=None, **_kwargs):
        return FakeTensor(data)

    def zeros(self, shape=None, **_kwargs):
        return FakeTensor(np.zeros(shape))

    def ones(self, shape=None, **_kwargs):
        return FakeTensor(np.ones(shape, dtype=int))

    def isnan(self, tensor):
        return FakeTensor(np.isnan(tensor.arr))

    def isinf(self, tensor):
        return FakeTensor(np.isinf(tensor.arr))

    def inference_mode(self, *_args, **_kwargs):
        class _CM:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        return _CM()


def test_row_entropy_fallbacks_to_entropy_from_start_index(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gen, "torch", fake_torch)

    captured = {}

    def fake_entropy_from_start_index(model, seqs, start_index):
        captured["args"] = (model, seqs.tolist(), start_index)
        return [0.5]

    monkeypatch.setattr(gen, "entropy_from_start_index", fake_entropy_from_start_index)

    ctx = gen._EntropyRowContext(
        scores=[FakeTensor([[np.nan, 0.0]])],
        sequences=np.array([[1, 2, 3]]),
        eos_ids=None,
        model="model",
    )
    entropies = gen._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=1)
    assert entropies == [0.5]
    assert captured["args"][2] == 0  # start_index = start_tok_idx - 1


def test_row_entropy_handles_nan_log_probs(monkeypatch):
    class TorchStub:
        class nn:
            class functional:
                @staticmethod
                def log_softmax(logits, dim=-1):
                    return np.array([[np.nan]])

        @staticmethod
        def isnan(val):
            return SimpleNamespace(any=lambda: True)

        @staticmethod
        def isinf(val):
            return SimpleNamespace(any=lambda: False)

    monkeypatch.setattr(gen, "torch", TorchStub)
    fallback_called = {}
    monkeypatch.setattr(gen, "entropy_from_start_index", lambda *a, **k: fallback_called.setdefault("res", [0.2]))

    class FakeScore:
        def __init__(self, arr):
            self.arr = np.array(arr, dtype=float)

        def __getitem__(self, idx):
            return FakeScore(self.arr[idx])

        def float(self):
            return self

    ctx = gen._EntropyRowContext(
        scores=[FakeScore([[1.0, 2.0]])],
        sequences=np.array([[1, 2]]),
        eos_ids=None,
        model="m",
    )
    ent = gen._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=1)
    assert ent == [0.2]
    assert fallback_called["res"] == [0.2]


def test_run_generate_batch_handles_inference_mode_typeerror(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gen, "torch", fake_torch)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, prefixes, return_tensors=None, padding=True, truncation=True, max_length=None):
            batch = len(prefixes)
            return {
                "input_ids": fake_torch.ones((batch, 2)),
                "attention_mask": fake_torch.ones((batch, 2)),
            }

        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    class Model:
        def __init__(self):
            self.called = False

        def generate(self, **_kwargs):
            self.called = True
            return types.SimpleNamespace(
                sequences=np.array([[1, 2, 3]]),
                scores=[fake_torch.zeros((1, 5))],
            )

    class TorchModule:
        def inference_mode(self):
            raise TypeError("not callable")

    config_like = types.SimpleNamespace(
        eos_ids=None,
        entropy_mode="none",
        temperature=0.0,
        top_p=1.0,
    )

    model = Model()
    decoded, entropies, input_lengths, sequences, stops = gen.run_generate_batch(
        prefixes=["hi"],
        cap=1,
        stop_strings=[],
        tokenizer=Tokenizer(),
        model=model,
        config_like=config_like,
        max_length=4,
        torch_module=TorchModule(),
        stopping_criteria_list_cls=lambda lst: lst,
    )

    assert model.called
    assert decoded == ["decoded"]
    assert entropies == [[]]
    assert input_lengths.tolist() == [2]
    assert sequences.tolist() == [[1, 2, 3]]
    assert stops == ["max_new_tokens"]


def test_entropy_from_start_index_renormalizes_non_finite(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gen, "torch", fake_torch)
    monkeypatch.setattr(gen.math, "isfinite", lambda _x: False)

    class FakeModel:
        def parameters(self):
            return iter([types.SimpleNamespace(device="cpu")])

        def __call__(self, **_kwargs):
            return types.SimpleNamespace(
                past_key_values=None,
                logits=FakeTensor([[[0.0, 0.0]]]),
            )

    ent = gen.entropy_from_start_index(FakeModel(), fake_torch.tensor([[1, 2, 3]]), start_idx=0)
    assert len(ent) == 2  # sequence_length - 1


def test_row_entropy_marks_bad_when_log_softmax_is_nan(monkeypatch):
    class TorchStub:
        class nn:
            class functional:
                @staticmethod
                def log_softmax(_logits, dim=-1):
                    return "nan-log-probs"

        @staticmethod
        def isnan(val):
            return SimpleNamespace(any=lambda: val == "nan-log-probs")

        @staticmethod
        def isinf(_val):
            return SimpleNamespace(any=lambda: False)

    class FakeScore:
        def __getitem__(self, _idx):
            return self

        def float(self):
            return self

    monkeypatch.setattr(gen, "torch", TorchStub)
    fallback = {}
    monkeypatch.setattr(
        gen,
        "entropy_from_start_index",
        lambda *a, **k: fallback.setdefault("called", [1.0]),
    )

    ctx = gen._EntropyRowContext(
        scores=[FakeScore()],
        sequences=np.array([[1, 2, 3]]),
        eos_ids=None,
        model="m",
    )
    entropies = gen._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=1)
    assert entropies == [1.0]
    assert fallback["called"] == [1.0]


def test_generate_and_score_batch_wraps_non_cm_inference_mode(monkeypatch):
    calls = {}

    def fake_tokenize(tokenizer, prefixes, max_length):
        calls["tokenize"] = (tuple(prefixes), max_length, tokenizer)
        return ({"input_ids": ["ids"], "attention_mask": ["mask"]}, [3])

    def fake_decode(_ctx):
        return (["decoded"], [[]], ["stop-token"])

    monkeypatch.setattr(gen, "tokenize_prefixes_for_generate", fake_tokenize)
    monkeypatch.setattr(gen, "decode_and_score_batch", fake_decode)

    class TorchModule:
        def inference_mode(self):
            return object()  # lacks __enter__

    class Model:
        def __init__(self):
            self.called = False

        def generate(self, **_kwargs):
            self.called = True
            return types.SimpleNamespace(sequences=["seqs"], scores=["scores"])

    runtime = gen.GenerateBatchRuntime(
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=1),
        model=Model(),
        torch_module=TorchModule(),
        stopping_criteria_list_cls=lambda items: items,
    )
    params = gen.GenerateBatchParams(
        prefixes=["p"],
        cap=1,
        stop_strings=[],
        config_like=SimpleNamespace(
            eos_ids=None,
            entropy_mode="none",
            temperature=None,
            top_p=None,
        ),
        max_length=4,
    )

    decoded, entropies, input_lengths, sequences, stop_reasons = gen.generate_and_score_batch(
        params,
        runtime,
    )

    assert calls["tokenize"][:2] == (("p",), 4)
    assert runtime.model.called is True
    assert decoded == ["decoded"]
    assert entropies == [[]]
    assert input_lengths == [3]
    assert sequences == ["seqs"]
    assert stop_reasons == ["stop-token"]
