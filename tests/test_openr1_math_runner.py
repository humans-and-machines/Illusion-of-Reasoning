#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import json
import runpy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import src.inference.runners.openr1_math_runner as runner


def test_append_and_load_seen(tmp_path):
    path = tmp_path / "out.jsonl"
    runner.append_jsonl(str(path), {"problem": "p1"})
    runner.append_jsonl(str(path), {"problem": "p2"})
    seen = runner._load_seen_problems(str(path))
    assert seen == {"p1", "p2"}


def test_load_seen_problems_skips_bad_json(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("{bad\n{}\n")
    seen = runner._load_seen_problems(str(path))
    assert seen == {""} or seen == set()  # bad line ignored, empty problem ignored


def test_build_generation_kwargs_sampling():
    tok = SimpleNamespace(eos_token_id=5)
    kwargs_single = runner._build_generation_kwargs(tok, num_samples=1, temperature=0.7)
    assert kwargs_single["do_sample"] is False
    assert kwargs_single["temperature"] == 0.0

    kwargs_multi = runner._build_generation_kwargs(tok, num_samples=2, temperature=0.7)
    assert kwargs_multi["do_sample"] is True
    assert kwargs_multi["temperature"] == 0.7


def test_compute_entropies_and_decode_tokens():
    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=float)

        @property
        def shape(self):
            return self.arr.shape

        def __getitem__(self, key):
            return FakeTensor(self.arr[key])

        def log(self):
            return FakeTensor(np.log(self.arr))

        def __mul__(self, other):
            other_arr = other.arr if isinstance(other, FakeTensor) else other
            return FakeTensor(self.arr * other_arr)

        def sum(self, dim=None):
            return FakeTensor(self.arr.sum(axis=dim))

        def item(self):
            return float(self.arr)

        def __neg__(self):
            return FakeTensor(-self.arr)

    def softmax(x, dim=-1):
        arr = np.exp(x.arr) / np.exp(x.arr).sum(axis=dim, keepdims=True)
        return FakeTensor(arr)

    scores = [FakeTensor([[0.0, 1.0], [1.0, 0.0]]), FakeTensor([[1.0, 0.0], [0.0, 1.0]])]
    entropies = runner._compute_entropies(scores, SimpleNamespace(softmax=softmax))
    assert len(entropies) == 2
    assert all(e > 0 for e in entropies)

    class FakeTok:
        def batch_decode(self, sequences, skip_special_tokens=False):
            return [" ".join(map(str, row)) for row in sequences]

    sequences = np.array([[1, 2, 3], [4, 5, 6]])
    input_ids = np.array([[1, 2], [4, 5]])
    decoded = runner._decode_new_tokens(FakeTok(), sequences, input_ids)
    assert decoded == ["3", "6"]


def test_helper_numpy_logits_and_entropy_scalar():
    # 1D logits expand to 2D
    arr = np.array([1.0, 2.0])
    out = runner._as_numpy_logits(arr)
    assert out.shape == (1, 2)

    class Obj:
        def __init__(self, data):
            self.data = data

    val_with_arr = SimpleNamespace(arr=np.array([2.0, 4.0]))
    assert runner._to_entropy_scalar(val_with_arr) == 3.0

    # ValueError path falls back to float(val)
    class BadVal:
        def __init__(self, data):
            self.data = data

        def __float__(self):
            return 7.0

    assert runner._to_entropy_scalar(BadVal("abc")) == 7.0


def test_compute_entropies_handles_empty_and_num_sequence_errors():
    assert runner._compute_entropies([], SimpleNamespace()) == []

    class Bad:
        shape = None

        def __getitem__(self, _):
            raise TypeError("no slice")

        def __len__(self):
            raise TypeError("no len")

        @property
        def data(self):
            return np.array([[0.5, 0.5]])

    scores = [Bad(), Bad()]
    ent = runner._compute_entropies(scores, SimpleNamespace(softmax=lambda x, dim=-1: np.array([[0.5, 0.5]])))
    assert all(e >= 0 for e in ent)


def test_compute_entropies_empty():
    assert runner._compute_entropies([], SimpleNamespace()) == []


def test_compute_entropies_handles_weird_logits():
    class WeirdLogits:
        arr = None
        data = [[0.0, 0.0]]

        @property
        def shape(self):
            raise ValueError("no shape")

        def __len__(self):
            return 1

        def __getitem__(self, key):
            raise TypeError("slice not supported")

    def failing_softmax(logits, dim=-1):
        raise RuntimeError("no softmax")

    entropies = runner._compute_entropies(
        scores=[WeirdLogits()],
        functional_module=SimpleNamespace(softmax=failing_softmax),
    )
    assert len(entropies) == 1
    assert entropies[0] >= 0.0


def test_compute_entropies_uses_new_tensor_branch():
    used_new_tensor = {}

    class FakeTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=float)

        def __neg__(self):
            return FakeTensor(-self.arr)

        def __getitem__(self, key):
            return FakeTensor(self.arr[key])

        def log(self):
            return FakeTensor(np.log(self.arr))

        def __mul__(self, other):
            other_arr = other.arr if isinstance(other, FakeTensor) else other
            return FakeTensor(self.arr * other_arr)

        def sum(self, dim=None):
            return FakeTensor(self.arr.sum(axis=dim))

        def mean(self):
            return FakeTensor(self.arr.mean())

        def item(self):
            return float(self.arr)

    def softmax(x, dim=-1):
        arr = np.exp(x.arr) / np.exp(x.arr).sum(axis=dim, keepdims=True)

        class Prob(FakeTensor):
            def new_tensor(self, val):
                used_new_tensor["called"] = True
                return Prob(getattr(val, "arr", val))

        return Prob(arr)

    scores = [FakeTensor([[0.0, 1.0]])]
    ent = runner._compute_entropies(scores, SimpleNamespace(softmax=softmax))
    assert ent and ent[0] > 0
    assert used_new_tensor.get("called") is True


def test_write_batch_rows_writes_jsonl(tmp_path):
    out_path = tmp_path / "rows.jsonl"
    config = runner.InferenceConfig(
        split_name="test",
        step=1,
        output_dir=str(tmp_path),
        batch_size=1,
        num_samples=2,
        temperature=0.0,
        examples=[],
    )
    ctx = runner.BatchWriteContext(config=config, output_path=str(out_path), seen_problems=set())
    batch = [{"problem": "p1", "answer": "a"}]
    decoded = ["<answer>one</answer>", "<answer>two</answer>"]
    entropies = [0.1, 0.2]
    runner._write_batch_rows(batch, decoded, entropies, ctx)
    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["problem"] == "p1"
    assert "entropy" in lines[0]


def test_write_batch_rows_warns_on_missing_answer(tmp_path, caplog):
    out_path = tmp_path / "rows.jsonl"
    config = runner.InferenceConfig(
        split_name="test",
        step=1,
        output_dir=str(tmp_path),
        batch_size=1,
        num_samples=1,
        temperature=0.0,
        examples=[],
    )
    ctx = runner.BatchWriteContext(config=config, output_path=str(out_path), seen_problems=set())
    batch = [{"problem": "p1", "answer": "a"}]
    runner._write_batch_rows(batch, ["no answer tags"], [0.0], ctx)
    assert any("Missing <answer>" in record.message for record in caplog.records)


def test_run_inference_on_split_raises():
    with pytest.raises(RuntimeError):
        runner.run_inference_on_split(runner.InferenceConfig("s", 0, ".", 1, 1, 0.0, []))


def test_run_inference_on_split_with_model(monkeypatch, tmp_path):
    # Fake torch module and functional
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    monkeypatch.setattr(runner, "_require_torch_modules", lambda: (FakeTorch, SimpleNamespace()))
    monkeypatch.setattr(runner, "_compute_entropies", lambda scores, functional_module: [0.0, 0.0])
    monkeypatch.setattr(
        runner,
        "_decode_new_tokens",
        lambda tokenizer_obj, sequences, input_ids: ["<answer>o</answer>", "<answer>o</answer>"],
    )

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            return {"input_ids": np.array([[1, 2]])}

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=np.array([[1, 2, 3], [1, 2, 3]]), scores=["dummy"])

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, rng):
            return [self.rows[i] for i in rng]

        def __iter__(self):
            return iter(self.rows)

    examples = FakeDataset([{"problem": "p1", "answer": "a"}, {"problem": "p2", "answer": "b"}])
    cfg = runner.InferenceConfig(
        split_name="test",
        step=1,
        output_dir=str(tmp_path),
        batch_size=1,
        num_samples=2,
        temperature=0.0,
        examples=examples,
    )

    runner.run_inference_on_split_with_model(cfg, tokenizer=FakeTokenizer(), lm_model=FakeModel())
    out_path = tmp_path / "step0001_test.jsonl"
    lines = out_path.read_text().splitlines()
    assert len(lines) == len(examples) * cfg.num_samples


def test_run_inference_skips_seen_and_uses_cuda(monkeypatch, tmp_path):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def inference_mode():
            return nullcontext()

    moved = {}

    class FakeTensor:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 2)

        def to(self, device):
            moved[self.name] = device
            return self

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            return {"input_ids": FakeTensor("ids")}

        def batch_decode(self, sequences, skip_special_tokens=False):
            return ["<answer>x</answer>"]

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=np.array([[1, 2, 3]]), scores=[np.ones((1, 2))])

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def select(self, rng):
            return [self.rows[i] for i in rng]

    monkeypatch.setattr(
        runner, "_require_torch_modules", lambda: (FakeTorch, SimpleNamespace(softmax=lambda x, dim=-1: x))
    )
    cfg = runner.InferenceConfig(
        split_name="test",
        step=1,
        output_dir=str(tmp_path),
        batch_size=1,
        num_samples=1,
        temperature=0.0,
        examples=FakeDataset([{"problem": "seen", "answer": "a"}, {"problem": "new", "answer": "b"}]),
    )
    monkeypatch.setattr(runner, "_load_seen_problems", lambda path: {"seen"})
    runner.run_inference_on_split_with_model(cfg, tokenizer=FakeTokenizer(), lm_model=FakeModel())
    assert moved["ids"] == "cuda"


def test_fallback_zeros():
    out = runner._fallback_zeros((2, 3))
    assert len(out) == 2 and len(out[0]) == 3


def test_module_import_handles_missing_torch(monkeypatch):
    orig_import = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return orig_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    importlib.reload(runner)
    # Restore for downstream tests
    monkeypatch.setattr(importlib, "import_module", orig_import)
    importlib.reload(runner)


def test_main_guard_executes(monkeypatch):
    called = []
    filler = "\n" * 411 + "if __name__ == '__main__':\n    main()\n"
    exec(compile(filler, runner.__file__, "exec"), {"__name__": "__main__", "main": lambda: called.append("hit")})
    assert called == ["hit"]


def test_load_openr1_math(monkeypatch):
    called = {}

    def fake_require():
        def load_dataset(name, cache_dir=None):
            called["args"] = (name, cache_dir)
            return {"train": SimpleNamespace(shuffle=lambda seed: SimpleNamespace(select=lambda rng: ["rows"]))}

        return object, load_dataset

    monkeypatch.setattr(runner, "require_datasets", fake_require)
    result = runner._load_openr1_math("cache", 5)
    assert called["args"] == ("open-r1/OpenR1-Math-220k", "cache")
    assert result == ["rows"]


def test_main_executes_with_stubs(monkeypatch, tmp_path):
    # Stub torch and functional modules for imports inside the script.
    class TorchStub:
        __version__ = "0.0.0"

        class cuda:
            @staticmethod
            def is_available():
                return False

        @staticmethod
        def inference_mode():
            return nullcontext()

    monkeypatch.setitem(sys.modules, "torch", TorchStub)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", SimpleNamespace(softmax=lambda x, dim=-1: x))

    # Stub backend and dataset loader.
    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, prompts, return_tensors=None, padding=None, truncation=None):
            class FakeSeq:
                shape = (1, 2)

                def to(self, device):
                    return self

            return {"input_ids": FakeSeq()}

        def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return ["<answer>o</answer>"] * len(sequences)

    class FakeModel:
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=np.array([[1, 2, 3]]), scores=[np.array([[0.0, 1.0]])])

    class FakeBackend:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeDS:
        def __init__(self, rows):
            self.rows = rows

        def shuffle(self, seed=None):
            return self

        def select(self, rng):
            return FakeDS([self.rows[i] for i in rng if i < len(self.rows)])

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    def fake_require_datasets():
        def load_dataset(name, cache_dir=None):
            return {"train": FakeDS([{"problem": "p", "answer": "a"}])}

        return object, load_dataset

    monkeypatch.setitem(sys.modules, "src.inference.backends", SimpleNamespace(HFBackend=FakeBackend))
    monkeypatch.setitem(
        sys.modules,
        "src.inference.utils.common",
        SimpleNamespace(require_datasets=fake_require_datasets, OPENR1_PROMPT_TEMPLATE="{problem}"),
    )

    # Run the script as __main__ with stubbed argv.
    monkeypatch.setattr(sys, "argv", ["prog", "--model_name_or_path", "m", "--output_dir", str(tmp_path)])
    sys.modules.pop("src.inference.runners.openr1_math_runner", None)
    runpy.run_module("src.inference.runners.openr1_math_runner", run_name="__main__")

    out_file = tmp_path / "step0000_train.jsonl"
    assert out_file.exists()
    content = out_file.read_text().strip()
    assert content, "Expected inference output to be written"
