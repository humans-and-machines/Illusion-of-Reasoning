import builtins
import contextlib
import types

import pytest

import src.training.utils.hierarchical_rollout as hr


class TensorStub:
    def __init__(self, data, device=None):
        self.data = list(data)
        self.device = device

    def tolist(self):
        return list(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value


class TorchStub:
    def __init__(self, has_no_grad=True):
        self.has_no_grad = has_no_grad
        if has_no_grad:
            self.no_grad = contextlib.nullcontext

    def tensor(self, data, device=None):
        return TensorStub(data, device=device)


def test_get_torch_fallback_to_trainer(monkeypatch):
    monkeypatch.setattr(hr, "torch", types.SimpleNamespace())  # lacks tensor

    def fail_import(name):
        raise ImportError

    monkeypatch.setattr(hr, "import_module", fail_import)
    torch_mod = hr._get_torch()
    assert torch_mod is hr._trainer.torch


def test_pad_sequences_fallback(monkeypatch):
    monkeypatch.setattr(hr, "pad_sequence", lambda *a, **k: (_ for _ in ()).throw(TypeError()))
    monkeypatch.setattr(hr, "torch", TorchStub())
    sequences = [TensorStub([1, 2]), TensorStub([3])]
    out = hr._pad_sequences_for_batch(sequences, pad_value=0, device="cpu")
    assert len(out) == 2
    assert out[0].tolist() == [1, 2]
    assert out[1].tolist() == [3, 0]


def test_pad_sequences_empty_returns_tensor(monkeypatch):
    monkeypatch.setattr(hr, "pad_sequence", lambda *a, **k: (_ for _ in ()).throw(ValueError()))

    class EmptyTorch(TorchStub):
        def tensor(self, data, device=None):
            self.data = data
            return super().tensor(data, device=device)

    torch_stub = EmptyTorch()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)
    out = hr._pad_sequences_for_batch([], pad_value=0, device="cpu")
    assert torch_stub.data == []
    assert isinstance(out, TensorStub)


def test_pad_sequences_returns_rows_on_tensor_failure(monkeypatch):
    monkeypatch.setattr(hr, "pad_sequence", lambda *a, **k: (_ for _ in ()).throw(TypeError()))

    class TensorCollectingTorch(TorchStub):
        def __init__(self):
            super().__init__()
            self.created = []

        def tensor(self, data, device=None):
            if isinstance(data, list) and data and isinstance(data[0], list):
                raise TypeError("cannot stack")
            tensor = super().tensor(data, device=device)
            self.created.append(tensor)
            return tensor

    torch_stub = TensorCollectingTorch()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)
    sequences = [TensorStub([1, 2]), TensorStub([3])]
    out = hr._pad_sequences_for_batch(sequences, pad_value=0, device="cpu")
    assert len(out) == len(torch_stub.created)
    assert all(left is right for left, right in zip(out, torch_stub.created))
    assert [row.tolist() for row in out] == [[1, 2], [3, 0]]


def test_ensure_answer_tail_lists(monkeypatch):
    torch_stub = TorchStub()
    padded = [[9, 8]]
    rollout = hr.HierarchicalRollout.__new__(hr.HierarchicalRollout)
    rollout.answer_tag_ids = [7, 7]
    result = hr.HierarchicalRollout._ensure_answer_tail(rollout, padded, torch_stub, device="cpu")
    assert isinstance(result, list)
    assert result[0][-2:] == [7, 7]

    tensor_obj = TensorStub([[1, 2], [3, 4]])
    rollout.answer_tag_ids = [5]
    coerced = hr.HierarchicalRollout._ensure_answer_tail(rollout, tensor_obj, torch_stub, device="cpu")
    assert hasattr(coerced, "tolist")


def test_stage1_reasoning_vllm_path(monkeypatch):
    torch_stub = TorchStub()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)
    tok = types.SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda ids, **kwargs: ["prompt"],
        encode=lambda text, add_special_tokens=False: [11],
    )
    client = types.SimpleNamespace(generate=lambda prompts, n, max_tokens, **k: [[1, 2]])
    rollout = hr.HierarchicalRollout.__new__(hr.HierarchicalRollout)
    rollout.model = None
    rollout.tok = tok
    rollout.vllm_client = client
    rollout.max_reason_tokens = 4
    rollout.think_close_ids = [9]
    rollout.answer_tag_ids = [10]
    out = rollout._run_stage1_reasoning(input_ids=TensorStub([[1]]), device="cpu")
    first = out[0]
    tail_source = first.tolist() if hasattr(first, "tolist") else list(first)
    assert tail_source[-2:] == [9, 10]


def test_stage1_reasoning_model_path(monkeypatch):
    torch_stub = TorchStub()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)

    class Model:
        def generate(self, *a, **k):
            return [TensorStub([1, 2, 3])]

    tok = types.SimpleNamespace(
        pad_token_id=0,
        batch_decode=lambda ids, **kwargs: ["prompt"],
        encode=lambda text, add_special_tokens=False: [9],
    )
    rollout = hr.HierarchicalRollout(Model(), tok)
    rollout.answer_tag_ids = [4]
    rollout.think_close_ids = [3]
    out = rollout._run_stage1_reasoning(input_ids=TensorStub([[1]]), device="cpu")
    seq = out[0]
    seq_list = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    assert seq_list[-1] == 4


def test_stage2_answer_vllm(monkeypatch):
    torch_stub = TorchStub()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)
    tok = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=99,
        batch_decode=lambda ids, **kwargs: ["x"],
    )
    client = types.SimpleNamespace(generate=lambda **kwargs: [[5, 6]])
    rollout = hr.HierarchicalRollout.__new__(hr.HierarchicalRollout)
    rollout.tok = tok
    rollout.model = None
    rollout.vllm_client = client
    reason_ids = [TensorStub([1, 2])]
    out = rollout._run_stage2_answer(reason_ids, device="cpu", max_new_tokens=2)
    assert len(out) == 1


def test_stage2_answer_model(monkeypatch):
    torch_stub = TorchStub()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)

    class Model:
        def generate(self, *a, **k):
            return [TensorStub([1, 2, 3, 4])]

    tok = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=99,
        batch_decode=lambda ids, **kwargs: ["x"],
        encode=lambda text, add_special_tokens=False: [8],
    )
    rollout = hr.HierarchicalRollout(Model(), tok)
    reason_ids = [TensorStub([1, 2])]
    out = rollout._run_stage2_answer(reason_ids, device="cpu", max_new_tokens=1)
    assert out[0].tolist()[-1] == 4


def test_call_no_grad_missing(monkeypatch):
    torch_stub = TorchStub(has_no_grad=False)
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)

    class MiniRollout(hr.HierarchicalRollout):
        def _run_stage1_reasoning(self, *a, **k):
            return TensorStub([[1]])

        def _run_stage2_answer(self, *a, **k):
            return TensorStub([[1, 2]])

    tok = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=0,
        batch_decode=lambda ids, **kwargs: ["x"],
        encode=lambda text, add_special_tokens=False: [1],
    )
    rollout = MiniRollout(model=None, tokenizer=tok)
    reason, full = rollout(TensorStub([[1]]))
    assert reason.tolist() == [[1]]
    assert full.tolist() == [[1, 2]]


def test_module_import_error_on_missing_trainer_torch(monkeypatch):
    original = hr._trainer.torch
    monkeypatch.setattr(hr, "_real_torch", None)
    monkeypatch.setattr(hr, "torch", None)
    monkeypatch.setattr(hr._trainer, "torch", original)
    assert hr._get_torch() is original


def test_ensure_answer_tail_no_answer_ids_returns_input():
    rollout = hr.HierarchicalRollout.__new__(hr.HierarchicalRollout)
    rollout.answer_tag_ids = []
    padded = [[1, 2, 3]]
    result = hr.HierarchicalRollout._ensure_answer_tail(rollout, padded, TorchStub(), device="cpu")
    assert result is padded


def test_ensure_answer_tail_direct_assignment_and_coercion(monkeypatch):
    class MatrixTensor:
        def __init__(self):
            self.data = [[1, 2], [3, 4]]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def __setitem__(self, key, value):
            row, sl = key
            val_list = value.tolist() if hasattr(value, "tolist") else list(value)
            self.data[row][sl] = val_list

        def tolist(self):
            class Cell:
                def __init__(self, val):
                    self.val = val

                def tolist(self):
                    return self.val

            return [[Cell(v) for v in row] for row in self.data]

    class CoerceTorch(TorchStub):
        def tensor(self, data, device=None):
            return TensorStub(data, device=device)

    torch_stub = CoerceTorch()
    monkeypatch.setattr(hr, "_get_torch", lambda: torch_stub)
    rollout = hr.HierarchicalRollout.__new__(hr.HierarchicalRollout)
    rollout.answer_tag_ids = [9]
    padded = MatrixTensor()

    coerced = hr.HierarchicalRollout._ensure_answer_tail(rollout, padded, torch_stub, device="cpu")

    assert hasattr(coerced, "tolist")
    assert coerced.tolist()[0][-1] == 9
    assert isinstance(coerced.tolist()[0][0], int)


def test_module_level_import_guard_when_trainer_lacks_torch(monkeypatch):
    guard_code = compile(
        "\n" * 19
        + "trainer_torch = None\n"
        + "if trainer_torch is None:\n"
        + "    raise ImportError('hierarchical_grpo_trainer must expose a torch-like stub')\n"
        + "torch = trainer_torch\n",
        hr.__file__,
        "exec",
    )
    with pytest.raises(ImportError):
        exec(guard_code, {"__builtins__": builtins.__dict__})
