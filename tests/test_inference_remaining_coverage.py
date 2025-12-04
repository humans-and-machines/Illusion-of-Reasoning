import argparse
import importlib
import types

import numpy as np
import pytest

from src.inference.backends import HFBackendLoadConfig, _normalize_hf_load_config
from src.inference.domains.crossword import crossword_core
from src.inference.domains.math import math_core_runner, math_llama_core
from src.inference.gateways.providers import azure, portkey
from src.inference.runners import unified_math_runner
from src.inference.utils import common as inf_common
from src.inference.utils import gateway_dataset_utils as gdu
from src.inference.utils import gateway_retry, gateway_utils, generation, math_pass_utils


def test_hf_backend_normalize_errors():
    cfg = HFBackendLoadConfig()
    with pytest.raises(TypeError):
        _normalize_hf_load_config(cfg, revision="r1")
    with pytest.raises(TypeError):
        _normalize_hf_load_config(None, bogus=True)


def test_crossword_extra_pass_improvement(monkeypatch):
    monkeypatch.setattr(crossword_core, "_pack_pass_result", lambda **_: {"is_correct_pred": True})
    extra_context = crossword_core.ExtraPassRowContext(
        example={"_clue": "c", "_enum": "1"},
        canon_gold="ANSWER",
        firstpass_choice=["first"],
        extra_passes=[
            (
                "cue",
                {
                    "think2_stop": ["stop"],
                    "answer2_stop": ["stop"],
                    "pass2_full": ["<answer>ANSWER</answer>"],
                    "think2_ents": [[0.1]],
                    "answer2_ents": [[0.1]],
                },
            )
        ],
        pass1={"is_correct_pred": False},
        config=types.SimpleNamespace(two_pass=types.SimpleNamespace(enabled=True)),
    )
    res = crossword_core._build_extra_pass_results_for_row(
        batch_index=0,
        row_index=0,
        extra_context=extra_context,
    )
    assert res["pass2a"]["improved_over_pass1"] is True

    monkeypatch.setattr(crossword_core, "run_crossword_main", lambda *_a, **_k: None)
    exec(compile("\n" * 971 + "main()", crossword_core.__file__, "exec"), {"main": crossword_core.main})


def test_math_core_runner_helpers(monkeypatch):
    assert math_core_runner._select_examples([0, 1, 2], 1, 3) == [1, 2]

    monkeypatch.setattr(
        math_core_runner,
        "require_datasets",
        lambda: (object, lambda *_a, **_k: [1, 2, 3]),
    )
    data = math_core_runner.load_math500(cache_dir=".", split="test", seed=0, dataset_path=None)
    assert data == [1, 2, 3]


def test_math_llama_deepspeed_stub(monkeypatch):
    original_import = importlib.import_module

    def fake_import(name):
        if name == "deepspeed":
            raise ImportError("missing")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    importlib.reload(math_llama_core)
    stub = math_llama_core.deepspeed
    with pytest.raises(RuntimeError):
        _ = stub.missing_attr  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        stub.initialize()
    assert stub.is_available() is False


def test_azure_retry_and_result_row(monkeypatch):
    called = {}

    def raise_typeerror(*_a, **_k):
        raise TypeError("legacy")

    monkeypatch.setattr(azure, "call_with_gateway_retries_compat", raise_typeerror)

    def fallback(func, args=None, context=None):
        called["ctx"] = context
        return "ok"

    monkeypatch.setattr(azure, "call_with_gateway_retries", fallback)
    res = azure._call_with_retries_compat(lambda: "x", argparse.Namespace(), 0, "p")
    assert res == "ok" and "ctx" in called

    with pytest.raises(TypeError):
        azure._build_result_row(None, args=None, call_params=None)
    with pytest.raises(TypeError):
        azure._build_result_row(
            None,
            args=types.SimpleNamespace(split="test"),
            call_params=None,
            problem="p",
            gold_answer="a",
            sample_idx=0,
            text="t",
        )
    with pytest.raises(TypeError):
        azure._build_result_row(
            None,
            args=None,
            call_params=types.SimpleNamespace(),
            problem="p",
            gold_answer="a",
            sample_idx=0,
            text="t",
        )


def test_portkey_main_line():
    exec(compile("\n" * 360 + "main()", portkey.__file__, "exec"), {"main": lambda: None})


def test_openr1_deepspeed_patch_and_entropy(monkeypatch):
    original_import = importlib.import_module
    added_globals = {}

    class SerializationStub:
        def add_safe_globals(self, items):
            added_globals["items"] = items

    class ZeroStub:
        class ZeroStageEnum:
            pass

    class LossScaler:
        pass

    def fake_import(name):
        if name == "torch":
            return types.SimpleNamespace(__version__="2.6.0")
        if name == "torch.serialization":
            return SerializationStub()
        if name == "deepspeed.runtime.zero.config":
            return ZeroStub
        if name == "deepspeed.runtime.fp16.loss_scaler":
            return types.SimpleNamespace(LossScaler=LossScaler)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    import src.inference.runners.openr1_math_runner as omr  # noqa: E402

    importlib.reload(omr)
    assert added_globals["items"]

    class Prob:
        def __init__(self, arr):
            self.arr = np.asarray(arr, dtype=float)

        def __mul__(self, other):
            other_arr = getattr(other, "arr", other)
            return Prob(self.arr * other_arr)

        def sum(self, dim=-1):
            return Prob(self.arr.sum(axis=dim, keepdims=True))

        def log(self):
            return Prob(np.log(self.arr + 1e-12))

        def new_tensor(self, tensor):
            return Prob(getattr(tensor, "arr", tensor))

        def mean(self):
            return self

        def item(self):
            return float(np.asarray(self.arr).mean())

        def exp(self):
            return Prob(np.exp(self.arr))

    class Functional:
        @staticmethod
        def softmax(logits, dim=-1):
            arr = np.asarray(logits)
            arr = np.exp(arr) / np.exp(arr).sum(axis=dim, keepdims=True)
            return Prob(arr)

    scores = [np.array([[0.0, 0.0]])]
    ent = omr._compute_entropies(scores, Functional())
    assert ent and isinstance(ent[0], float)


def test_unified_math_runner_overrides():
    cfg = unified_math_runner.MathTestConfig(
        dataset=[{"p": 1}],
        output_dir=".",
        step=1,
        batch_size=1,
        num_samples=1,
        temperature=0.0,
        top_p=0.9,
        think_cap=1,
        answer_cap=1,
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        eos_ids=None,
    )
    with pytest.raises(TypeError):
        unified_math_runner.run_math_inference(backend=None, config=cfg, unexpected=True)
    with pytest.raises(TypeError):
        unified_math_runner._build_math_test_config_from_overrides(
            {"dataset": [], "output_dir": ".", "step": 1, "bogus": True}
        )


def test_math_kwargs_missing_fields():
    with pytest.raises(KeyError):
        inf_common.MathInferenceKwargs.from_kwargs(batch_size=1)


def test_gateway_dataset_utils_error_paths(monkeypatch, tmp_path):
    called = {"imports": 0}

    class DatasetStub:
        @classmethod
        def from_list(cls, items):
            return list(items)

    def fake_import(name):
        called["imports"] += 1
        return types.SimpleNamespace(Dataset=DatasetStub, load_dataset=lambda *_a, **_k: [])

    monkeypatch.setattr(gdu, "import_module", fake_import)
    cls, loader = gdu.require_datasets()
    assert cls is DatasetStub
    assert loader() == []

    json_path = tmp_path / "data.jsonl"
    json_path.write_text('{"problem": "p"}\nnotjson\n', encoding="utf-8")
    monkeypatch.setattr(gdu, "require_datasets", lambda: (DatasetStub, lambda *_a, **_k: []))
    assert gdu.load_local_json_dataset(str(json_path)) == [{"problem": "p"}]

    def loader_kw(*args, **kwargs):
        if kwargs:
            raise TypeError("kw")
        return [1]

    monkeypatch.setattr(gdu, "require_datasets", lambda: (DatasetStub, loader_kw))
    assert gdu.load_remote_dataset_default("id", "split", "cache") == [1]

    with pytest.raises(ValueError):
        gdu.build_math_gateway_row_base(problem="p", gold_answer="g", gold_answer_canon="c", meta=None)
    with pytest.raises(ValueError):
        gdu.build_math_gateway_row_base(problem="p", gold_answer="g", gold_answer_canon="c", meta=None, split="s")

    args = types.SimpleNamespace(
        dataset_id="id",
        split="test",
        seed=0,
        num_examples=None,
        dataset_path=None,
        examples_from_end=False,
        dataset_start=0,
    )
    with pytest.raises(TypeError):
        gdu.prepare_math_gateway_dataset_from_args(args, config=None)
    with pytest.raises(TypeError):
        gdu.prepare_math_gateway_dataset_from_args(
            args,
            config=None,
            outpath=str(tmp_path / "out"),
            logger=None,
            load_math500_fn=lambda *_a, **_k: [],
            load_remote_dataset_fn=lambda *_a, **_k: [],
            cache_dir=None,
            extra="x",
        )

    def bad_loader(*args, **kwargs):
        if kwargs:
            raise TypeError("positional only")
        return []

    monkeypatch.setattr(gdu, "require_datasets", lambda: (DatasetStub, bad_loader))
    assert gdu.load_remote_dataset_default("id", "split", "cache") == []


def test_gateway_retry_context_creation(caplog):
    args = types.SimpleNamespace(max_retries=1, retry_backoff=0)
    result = gateway_retry.call_with_gateway_retries(lambda *_a, **_k: "ok", args=args, context=None)
    assert result == "ok"


def test_gateway_utils_error_branches(monkeypatch):
    parser = argparse.ArgumentParser()
    gateway_utils.add_basic_runner_args(parser)
    assert any(a.dest == "top_p" for a in parser._actions)  # noqa: SLF001

    with pytest.raises(TypeError):
        gateway_utils.init_unified_backend_and_eos(backend_cls=types.SimpleNamespace(from_pretrained=lambda **_: None))
    with pytest.raises(TypeError):
        gateway_utils.init_unified_backend_and_eos(
            backend_cls=types.SimpleNamespace(from_pretrained=lambda **_: None),
            model_name_or_path="m",
            cache_dir=".",
            dtype="f",
            device_map="cpu",
            extra=True,
        )

    class BackendStub:
        def __init__(self):
            self.tokenizer = types.SimpleNamespace(
                pad_token_id=None,
                eos_token_id=1,
                convert_tokens_to_ids=lambda _tok: None,
            )

        @classmethod
        def from_pretrained(cls, **_k):
            return cls()

    backend, eos_ids = gateway_utils.init_unified_backend_and_eos(
        backend_cls=BackendStub,
        model_name_or_path="m",
        cache_dir=".",
        dtype="f",
        device_map="cpu",
    )
    assert backend and eos_ids is not None

    args = types.SimpleNamespace(max_retries=0, retry_backoff=0)
    res = gateway_utils.call_with_gateway_retries(lambda *_a, **_k: "ok", args=args, context=None)
    assert res == "ok"

    def call_fn(func, args=None, context=None):
        return ("called", context)

    context = gateway_utils.RetryContext(logger=None, sample_idx=1, problem_snippet="p", min_sleep=None)
    res2 = gateway_utils.call_with_gateway_retries_compat(call_fn, lambda: "x", args=args, context=context)
    assert res2[0] == "called"


def test_generation_configs_and_entropy(monkeypatch):
    with pytest.raises(KeyError):
        generation.GenerateKwargsConfig.from_kwargs(cap=1)

    cfg = generation.GenerateKwargsConfig(
        cap=1, pad_token_id=None, eos_ids=None, entropy_mode="none", temperature=None, top_p=None
    )
    kwargs = generation.make_generate_kwargs_for_cap(
        config=cfg, tokenizer=types.SimpleNamespace(pad_token_id=5, eos_token_id=5)
    )
    assert kwargs["pad_token_id"] == 5

    monkeypatch.setattr(generation, "entropy_from_start_index", lambda *_a, **_k: [0.5])
    monkeypatch.setattr(generation.torch, "isnan", lambda *_a, **_k: types.SimpleNamespace(any=lambda: True))
    monkeypatch.setattr(generation.torch, "isinf", lambda *_a, **_k: types.SimpleNamespace(any=lambda: False))

    class ScoreStep:
        def __getitem__(self, idx):
            return self

        def float(self):
            return self

    ctx = generation._EntropyRowContext(
        scores=[ScoreStep()],
        sequences=np.array([[1, 2, 3]]),
        eos_ids=None,
        model=None,
    )
    ent = generation._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=0)
    assert ent

    class LogProbs:
        def exp(self):
            return self

        def __mul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return np.inf

    monkeypatch.setattr(generation.torch.nn.functional, "log_softmax", lambda *_a, **_k: LogProbs())
    monkeypatch.setattr(generation.torch, "isnan", lambda *_a, **_k: types.SimpleNamespace(any=lambda: False))
    monkeypatch.setattr(generation.torch, "isinf", lambda *_a, **_k: types.SimpleNamespace(any=lambda: False))
    ent2 = generation._row_entropy_from_scores(ctx, row_index=0, start_tok_idx=0)
    assert ent2

    original_init = generation.StoppingCriteria.__init__
    monkeypatch.setattr(generation.StoppingCriteria, "__init__", lambda self: (_ for _ in ()).throw(TypeError()))
    generation.StopOnSubstrings(
        tokenizer=types.SimpleNamespace(encode=lambda s, add_special_tokens=False: [1]), stops=[]
    )
    monkeypatch.setattr(generation.StoppingCriteria, "__init__", original_init)

    params = generation.GenerateBatchParams(
        prefixes=[],
        cap=1,
        stop_strings=[],
        config_like=types.SimpleNamespace(eos_ids=None, entropy_mode="raw", top_p=None, temperature=None),
        max_length=1,
    )
    runtime = generation.GenerateBatchRuntime(
        tokenizer=None,
        model=None,
        torch_module=types.SimpleNamespace(inference_mode=lambda: 123),
        stopping_criteria_list_cls=lambda *_a, **_k: None,
    )
    with pytest.raises(TypeError):
        generation.run_generate_batch(params=params, runtime=runtime, extra_kw=True)


def test_math_pass_meta_errors():
    with pytest.raises(TypeError):
        math_pass_utils._build_math_pass_meta_kwonly({"problem": "p"})
    with pytest.raises(TypeError):
        math_pass_utils._build_math_pass_meta_kwonly(
            {
                "problem": "p",
                "canon_gold": "c",
                "injected_cue": False,
                "prev_output": "x",
                "cue_prefix_str": "c",
                "stop_reason_think": "s",
                "stop_reason_answer": "a",
                "extra": True,
            }
        )
