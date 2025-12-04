import types

import numpy as np
import pytest

import src.inference.utils.generation as gen


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.array(arr)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, item):
        return FakeTensor(self.arr[item])

    def __eq__(self, other):
        other_arr = other.arr if isinstance(other, FakeTensor) else other
        return FakeTensor(self.arr == other_arr)

    def any(self):
        return bool(self.arr.any())

    def numel(self):
        return int(self.arr.size)

    def nonzero(self, as_tuple=False):
        idx = np.argwhere(self.arr)
        return FakeTensor(idx) if not as_tuple else tuple(idx.T)

    def float(self):
        return self

    def exp(self):
        return FakeTensor(np.exp(self.arr))

    def sum(self, dim=None, axis=None):
        axis = dim if dim is not None else axis
        return FakeTensor(self.arr.sum(axis=axis))

    def item(self):
        return float(self.arr.item())

    def __mul__(self, other):
        other_arr = other.arr if isinstance(other, FakeTensor) else other
        return FakeTensor(self.arr * other_arr)

    def __array__(self, dtype=None):
        return np.array(self.arr, dtype=dtype)

    def __iter__(self):
        for row in self.arr:
            yield FakeTensor(row)


class FakeTorch:
    def __init__(self):
        self.nn = type(
            "nn",
            (),
            {
                "functional": types.SimpleNamespace(
                    log_softmax=lambda logits, dim=-1: FakeTensor(
                        np.log(
                            np.exp(np.array(logits.arr)) / np.exp(np.array(logits.arr)).sum(axis=dim, keepdims=True)
                        )
                    )
                )
            },
        )

    @staticmethod
    def isnan(tensor):
        return FakeTensor(np.isnan(np.array(tensor.arr)))

    @staticmethod
    def isinf(tensor):
        return FakeTensor(np.isinf(np.array(tensor.arr)))


def test_row_entropy_from_scores_uses_eos_and_fallback(monkeypatch):
    monkeypatch.setattr(gen, "torch", FakeTorch())
    monkeypatch.setattr(gen, "entropy_from_start_index", lambda model, seq_ids, start_idx: [0.9])
    ctx = gen._EntropyRowContext(
        scores=[FakeTensor([[0.0, 1.0]])],
        sequences=FakeTensor([[0, 1, 2]]),
        eos_ids=[0],
        model="model",
    )
    ent = gen._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=0)
    assert ent == [0.9]


def test_row_entropy_from_scores_computes_entropies(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setattr(gen, "torch", fake_torch)

    scores = [
        FakeTensor([[0.0, 0.0]]),
        FakeTensor([[1.0, 0.0]]),
    ]
    ctx = gen._EntropyRowContext(
        scores=scores,
        sequences=FakeTensor([[1, 2]]),
        eos_ids=None,
        model="m",
    )
    ent = gen._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=0)
    assert len(ent) == 2
    assert pytest.approx(ent[0], rel=1e-3) == np.log(2)


def test_decode_and_score_batch_skips_entropy(monkeypatch):
    monkeypatch.setattr(gen, "_trim_and_classify", lambda *args, **kwargs: ("trimmed", "stop"))
    ctx = gen.DecodeBatchContext(
        tokenizer=types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "raw"),
        sequences=FakeTensor([[1, 2]]),
        scores=[],
        input_lengths=FakeTensor([1]),
        config=gen.DecodeBatchConfig(
            stop_strings=[],
            cap=5,
            eos_ids=None,
            entropy_mode="none",
            model=None,
        ),
    )
    texts, entropies, stops = gen.decode_and_score_batch(ctx)
    assert texts == ["trimmed"]
    assert entropies == [[]]
    assert stops == ["stop"]
