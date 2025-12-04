import types

import numpy as np
import pytest

import src.inference.utils.common as common
import src.inference.utils.generation as generation


class FakeTensor:
    def __init__(self, data, device="cpu"):
        self.data = np.array(data)
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        self.device = device
        return self

    def sum(self, dim=None, axis=None):
        axis = dim if dim is not None else axis
        return FakeTensor(self.data.sum(axis=axis), device=self.device)

    def item(self):
        return float(self.data.item())

    def __getitem__(self, idx):
        return FakeTensor(self.data[idx], device=self.device)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data == other_data, device=self.device)

    def any(self):
        return bool(self.data.any())

    def float(self):
        return self

    def exp(self):
        return FakeTensor(np.exp(self.data), device=self.device)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, FakeTensor) else other
        return FakeTensor(self.data * other_data, device=self.device)

    def nonzero(self, as_tuple=False):
        idx = np.argwhere(self.data)
        if as_tuple:
            return tuple(idx.T)
        return FakeTensor(idx, device=self.device)

    def numel(self):
        return int(self.data.size)

    def tolist(self):
        return self.data.tolist()

    def __iter__(self):
        for row in self.data:
            yield FakeTensor(row, device=self.device)


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    def __init__(self, cuda_available=False):
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        self.nn = types.SimpleNamespace(functional=types.SimpleNamespace(log_softmax=self._log_softmax))
        self.distributed = types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)

    @staticmethod
    def device(name):
        return f"dev:{name}"

    def inference_mode(self, *_args, **_kwargs):
        return FakeInferenceMode()

    @staticmethod
    def _log_softmax(tensor, dim=-1):
        arr = np.array(tensor.data)
        exp = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
        soft = exp / exp.sum(axis=dim, keepdims=True)
        return FakeTensor(np.log(soft))

    @staticmethod
    def isnan(tensor):
        return FakeTensor(np.isnan(tensor.data))

    @staticmethod
    def isinf(tensor):
        return FakeTensor(np.isinf(tensor.data))


def test_move_inputs_to_device_prefers_cuda(monkeypatch):
    fake_torch = FakeTorch(cuda_available=True)
    inputs = {"input_ids": FakeTensor([[1, 2]]), "attention_mask": FakeTensor([[1, 1]])}
    monkeypatch.setattr(generation, "torch", fake_torch)
    moved, lengths = generation.move_inputs_to_device(inputs)

    assert moved["input_ids"].device == "dev:cuda"
    assert lengths.device == "dev:cuda"


def test_build_generate_kwargs_sets_synced(monkeypatch):
    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

    fake_torch = FakeTorch()
    fake_torch.distributed = FakeDist()
    monkeypatch.setattr(generation, "torch", fake_torch)
    kwargs = generation.build_generate_kwargs(
        cap=5,
        pad_token_id=0,
        eos_ids=[1],
        entropy_mode="reconsider",
        temperature=0.7,
        top_p=0.9,
        synced_gpus=True,
    )
    assert kwargs["synced_gpus"] is True
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


def test_build_extra_pass_results_truncates_names():
    outputs = common.PassOutputs(
        full_texts=["a"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=[""],
        stop_reason_answer=[""],
    )
    res = generation.build_extra_pass_results_for_cues(
        two_pass=True,
        extra_passes=[("cue1", outputs), ("cue2", outputs)],
        pack_result_for_extra=lambda cue, out: {"cue": cue},
        names=("only_one",),
    )
    assert list(res.keys()) == ["only_one"]
    assert res["only_one"]["cue"] == "cue1"


def test_decode_generated_row_and_trim_classify(monkeypatch):
    seqs = FakeTensor([[10, 11, 99]])
    input_lengths = FakeTensor([1])
    monkeypatch.setattr(generation, "torch", FakeTorch())
    gen_ids, raw_txt, start_idx = generation.decode_generated_row(
        tokenizer=types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "hello STOP tail"),
        seqs=seqs,
        input_lengths=input_lengths,
        row_i=0,
        skip_special_tokens=True,
    )
    trimmed, reason = generation._trim_and_classify(
        gen_ids,
        raw_txt,
        stop_strings=["STOP"],
        cap=5,
        eos_ids=[99],
    )
    assert start_idx == 1
    assert gen_ids.tolist() == [11, 99]
    assert trimmed == "hello"
    assert reason == "stop_token"


def test_row_entropy_uses_fallback_on_bad_scores(monkeypatch):
    monkeypatch.setattr(generation, "torch", FakeTorch())
    monkeypatch.setattr(generation, "entropy_from_start_index", lambda model, seq_ids, start_idx: [0.3])

    class FakeScoreStep:
        def __init__(self):
            self.data = np.array([np.nan])

        def __getitem__(self, idx):
            return self

        def float(self):
            return self

    ctx = generation._EntropyRowContext(
        scores=[FakeScoreStep()],
        sequences=FakeTensor([[1, 2, 3]]),
        eos_ids=None,
        model=object(),
    )
    ent = generation._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=0)
    assert ent == [0.3]


def test_decode_and_score_batch_collects_entropies(monkeypatch):
    monkeypatch.setattr(generation, "_trim_and_classify", lambda *args, **kwargs: ("x", "other"))
    monkeypatch.setattr(generation, "_row_entropy_from_scores", lambda ctx, row_index, start_tok_idx: [1.1])
    monkeypatch.setattr(generation, "torch", FakeTorch())
    tokenizer = types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "decoded")
    sequences = FakeTensor([[1, 2, 3]])
    scores = [FakeTensor([[1, 1, 1]])]
    ctx = generation.DecodeBatchContext(
        tokenizer=tokenizer,
        sequences=sequences,
        scores=scores,
        input_lengths=FakeTensor([1]),
        config=generation.DecodeBatchConfig(
            stop_strings=[],
            cap=5,
            eos_ids=None,
            entropy_mode="full",
            model=object(),
        ),
    )
    decoded, entropies, reasons = generation.decode_and_score_batch(ctx)
    assert decoded == ["x"]
    assert entropies == [[1.1]]
    assert reasons == ["other"]


def test_stop_on_substrings_variants(monkeypatch):
    monkeypatch.setattr(generation, "torch", FakeTorch())
    empty = generation.StopOnSubstrings(tokenizer=types.SimpleNamespace(), stops=[])
    assert empty.has_stops() is False
    assert empty(FakeTensor([[1, 2]]), FakeTensor([[0.0]])) is False


def test_require_torch_injects_inference_mode(monkeypatch):
    stub = types.SimpleNamespace()
    monkeypatch.setattr(common, "import_module", lambda name: stub)
    torch_mod = common.require_torch("caller")
    assert hasattr(torch_mod, "inference_mode")
    ctx = torch_mod.inference_mode()
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


def test_require_torch_and_transformers_import_errors(monkeypatch):
    class Boom(ImportError):
        pass

    monkeypatch.setattr(common, "import_module", lambda name: (_ for _ in ()).throw(Boom("missing")))
    with pytest.raises(ImportError):
        common.require_torch("caller")

    monkeypatch.setattr(common, "import_module", lambda name: (_ for _ in ()).throw(Boom("missing")))
    with pytest.raises(ImportError):
        common.require_transformers("caller")


def test_build_math_inference_config_kwargs_from_args():
    args = types.SimpleNamespace(
        batch_size=2,
        num_samples=3,
        temperature=0.7,
        top_p=0.9,
        entropy_mode="full",
        two_pass=True,
        second_pass_phrase="phrase",
        second_pass_use_sample_idx=1,
        think_cap=50,
        answer_cap=10,
    )
    cfg = common.build_math_inference_config_kwargs_from_args(args, eos_ids=[99])
    assert cfg["batch_size"] == 2
    assert cfg["eos_ids"] == [99]
    assert cfg["second_pass_phrase"] == "phrase"

    class TokenizerTypeError:
        def encode(self, text, add_special_tokens=None):
            if add_special_tokens is not None:
                raise TypeError("no kw")
            return [9]

    tok = TokenizerTypeError()
    sos = generation.StopOnSubstrings(tokenizer=tok, stops=["s"])
    assert sos.has_stops() is True
    assert generation.StopOnSubstrings._endswith(FakeTensor([1, 2, 3]), [2, 3]) is True

    class SimpleTok:
        def encode(self, text, add_special_tokens=False):
            return [4, 5]

    sos_match = generation.StopOnSubstrings(tokenizer=SimpleTok(), stops=["stop"])
    assert (
        sos_match(
            FakeTensor([[1, 4, 5], [0, 0, 0]]),
            FakeTensor([[0.0], [0.0]]),
        )
        is True
    )


def test_first_eos_any_handles_none_and_present(monkeypatch):
    monkeypatch.setattr(generation, "torch", FakeTorch())

    class Seq(FakeTensor):
        def numel(self):
            return int(self.data.size)

    ids = Seq([1, 2, 3])
    assert generation.first_eos_any(ids, None) == ids.numel()
    assert generation.first_eos_any(ids, [2, 99]) == 1


def test_entropy_from_start_index_basic_path(monkeypatch):
    monkeypatch.setattr(generation, "torch", FakeTorch())

    class FakeOut:
        def __init__(self, logits):
            self.logits = logits
            self.past_key_values = ("kv",)

    class FakeModel:
        def __init__(self):
            self.param = FakeTensor([0.0])

        def parameters(self):
            return iter([self.param])

        def __call__(self, input_ids=None, past_key_values=None, use_cache=None):
            return FakeOut(FakeTensor([[[0.0, 1.0]]]))

    seq_ids = FakeTensor([[1, 2, 3]])
    entropies = generation.entropy_from_start_index(FakeModel(), seq_ids, start_idx=0)
    assert len(entropies) == seq_ids.shape[1] - 1
    assert all(e > 0 for e in entropies)


def test_require_torch_adds_inference_mode(monkeypatch):
    stub_torch = types.SimpleNamespace()
    monkeypatch.setattr(common, "import_module", lambda name: stub_torch)

    torch_mod = common.require_torch("caller")
    assert torch_mod is stub_torch
    assert hasattr(torch_mod, "inference_mode")
    # Ensure the injected context manager is usable.
    with torch_mod.inference_mode():
        pass
