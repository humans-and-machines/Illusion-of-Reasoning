from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace

import pytest


def _reload_math_llama_core(monkeypatch, *, torch_version="2.0.0", import_module_override=None):
    """Helper to reload math_llama_core with lightweight stubs."""
    sys.modules.pop("src.inference.domains.math.math_llama_core", None)

    import src.inference.utils.common as common

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch(types.SimpleNamespace):
        def __init__(self, version):
            super().__init__(
                __version__=version,
                bfloat16="bf16",
                float16="f16",
            )

        def inference_mode(self, *_args, **_kwargs):
            return _Ctx()

        def no_grad(self, *_args, **_kwargs):
            return _Ctx()

    fake_torch = FakeTorch(torch_version)

    class FakeConfig:
        def __init__(self):
            self.attn_implementation = None

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeConfig()

    class FakeTokenizer:
        def __init__(self):
            self.pad_token = None
            self.pad_token_id = 1
            self.eos_token = "<eos>"
            self.eos_token_id = 2
            self.padding_side = None
            self.truncation_side = None

        def convert_tokens_to_ids(self, tok):
            mapping = {"<|eot_id|>": 5, "<|end_of_text|>": 6}
            return mapping.get(tok)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return FakeTokenizer()

    class FakeParam:
        def __init__(self):
            self.requires_grad = True

    class FakeModel:
        def __init__(self):
            self.dtype = None
            self.eval_called = False
            self.params = [FakeParam()]

        def to(self, dtype):
            self.dtype = dtype
            return self

        def parameters(self):
            return iter(self.params)

        def eval(self):
            self.eval_called = True
            return self

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_config(cls, cfg, trust_remote_code=True):
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )

    def fake_load_dataset(dataset_id, split, cache_dir=None):
        return {"id": dataset_id, "split": split, "cache": cache_dir}

    monkeypatch.setattr(common, "require_torch", lambda caller: fake_torch)
    monkeypatch.setattr(common, "require_transformers", lambda caller: fake_transformers)
    monkeypatch.setattr(common, "require_datasets", lambda: (None, fake_load_dataset))

    class _Engine:
        def __init__(self, module):
            self.module = module
            self.load_calls = []

        def load_checkpoint(self, *args, **kwargs):
            self.load_calls.append((args, kwargs))

    ds_stub = types.ModuleType("deepspeed")
    ds_stub.initialize_calls = []

    def _initialize(model=None, config=None, model_parameters=None):
        ds_stub.initialize_calls.append((model, config, list(model_parameters)))
        return _Engine(model), None, None, None

    ds_stub.initialize = _initialize
    monkeypatch.setitem(sys.modules, "deepspeed", ds_stub)

    orig_import_module = importlib.import_module
    if import_module_override is not None:

        def patched_import_module(name, *args, **kwargs):
            result = import_module_override(name, orig_import_module)
            if result is not None:
                return result
            return orig_import_module(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", patched_import_module)

    module = importlib.reload(
        importlib.import_module("src.inference.domains.math.math_llama_core"),
    )

    return types.SimpleNamespace(
        module=module,
        ds_stub=ds_stub,
        fake_torch=fake_torch,
        fake_transformers=fake_transformers,
        fake_load_dataset=fake_load_dataset,
    )


@pytest.fixture
def math_llama(monkeypatch):
    """Import math_llama_core with lightweight stubs to avoid heavy deps."""
    return _reload_math_llama_core(monkeypatch)


def test_deepspeed_patch_runs_on_newer_torch(monkeypatch, caplog):
    caplog.set_level("INFO")

    def import_override(name, orig_import_module):
        if name == "torch.serialization":
            return types.SimpleNamespace(add_safe_globals=lambda *a, **k: caplog.records)
        if name == "deepspeed.runtime.zero.config":
            return types.SimpleNamespace(ZeroStageEnum="Z")
        if name == "deepspeed.runtime.fp16.loss_scaler":
            return types.SimpleNamespace(LossScaler="L")
        return None

    env = _reload_math_llama_core(
        monkeypatch,
        torch_version="2.6.0",
        import_module_override=import_override,
    )
    # Ensure patch path exercised without exceptions and with newer torch version
    assert env.fake_torch.__version__ == "2.6.0"
    assert any("DeepSpeed ZeRO patch enabled" in r.message for r in caplog.records)


def test_run_inference_on_split_errors(math_llama):
    m = math_llama.module
    with pytest.raises(TypeError):
        m.run_inference_on_split("positional", split_name="train")
    with pytest.raises(TypeError):
        m.run_inference_on_split(split_name="train", tokenizer="tok", model="m")


def test_run_inference_on_split_builds_config_and_calls_core(math_llama, monkeypatch):
    m = math_llama.module
    captured = {}

    class DummyConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    def dummy_run_inference_on_split(**kwargs):
        captured["run_kwargs"] = kwargs

    monkeypatch.setattr(m._math_core, "MathInferenceConfig", DummyConfig)
    monkeypatch.setattr(m._math_core, "run_inference_on_split", dummy_run_inference_on_split)

    m.run_inference_on_split(
        split_name="train",
        examples=[{"x": 1}],
        tokenizer="tok",
        model="model",
        step=7,
        outdir="/tmp/out",
        batch_size=2,
        temperature=0.3,
    )

    cfg = captured["config_kwargs"]
    assert cfg["split_name"] == "train"
    assert cfg["output_dir"] == "/tmp/out"
    assert cfg["step"] == 7
    # Overrides merged with defaults
    assert cfg["batch_size"] == 2
    assert cfg["temperature"] == 0.3
    assert cfg["num_samples"] == 1

    run_kwargs = captured["run_kwargs"]
    assert run_kwargs["examples"] == [{"x": 1}]
    assert run_kwargs["tokenizer"] == "tok"
    assert run_kwargs["model"] == "model"
    assert isinstance(run_kwargs["config"], DummyConfig)


def test_build_arg_parser_includes_deepspeed_flags(math_llama):
    m = math_llama.module
    parser = m._build_arg_parser()
    args = parser.parse_args(
        [
            "--model_name_or_path",
            "ckpt",
            "--output_dir",
            "/tmp/out",
            "--ds_config",
            "ds.json",
        ],
    )
    assert args.model_name_or_path == "ckpt"
    assert args.output_dir == "/tmp/out"
    assert args.dataset_id == "MATH-500"
    assert args.ds_tag is None


def test_init_tokenizer_and_eos_ids_sets_pad_and_eos_list(math_llama):
    m = math_llama.module
    args = Namespace(
        tokenizer_path=None,
        model_name_or_path="model",
        revision="r1",
    )
    tokenizer, eos_ids, cache_dir = m._init_tokenizer_and_eos_ids(args)
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token
    assert eos_ids == [2, 5, 6]
    assert cache_dir.endswith(".hf_cache")


def test_init_model_wraps_engine_and_loads_checkpoint(math_llama, monkeypatch):
    m = math_llama.module
    engine_records = {}

    class DummyWrapper:
        def __init__(self, engine):
            engine_records["engine"] = engine
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    monkeypatch.setattr(m, "DSModelWrapper", DummyWrapper)
    args = Namespace(
        model_name_or_path="ckpt",
        revision=None,
        attn_implementation="flash",
        dtype="bfloat16",
        ds_config="ds.json",
        ds_tag=None,
        step=5,
    )

    model, ds_tag = m._init_model(args, hf_cache_dir="/cache")

    # DeepSpeed initialization recorded the expected config/model parameters
    assert len(math_llama.ds_stub.initialize_calls) == 1
    init_model, init_cfg, init_params = math_llama.ds_stub.initialize_calls[0]
    assert init_cfg == "ds.json"
    assert list(init_params)[0].requires_grad is True
    # Check tag resolution and wrapper eval
    assert ds_tag == "global_step5"
    assert isinstance(model, DummyWrapper)
    assert model.eval_called is True
    # Engine should have loaded the checkpoint with resolved tag
    args_pos, kwargs = engine_records["engine"].load_calls[0]
    assert args_pos[0] == "ckpt"
    assert kwargs["tag"] == "global_step5"


def test_load_dataset_for_args_handles_math500_and_custom(math_llama, monkeypatch):
    m = math_llama.module
    monkeypatch.setattr(
        m, "load_math500", lambda cache_dir, split, seed: {"src": "math500", "split": split, "seed": seed}
    )
    monkeypatch.setattr(
        m,
        "limit_dataset_for_args",
        lambda dataset, args: {"limited": dataset, "num": getattr(args, "num_examples", None)},
    )

    args_math = Namespace(dataset_id="MATH-500", split="train", seed=1, num_examples=None)
    dataset, name = m._load_dataset_for_args(args_math, hf_cache_dir="/cache")
    assert name == "MATH-500"
    assert dataset["limited"]["src"] == "math500"
    assert dataset["num"] is None

    args_custom = Namespace(dataset_id="custom/path", split="dev", seed=0, num_examples=None)
    dataset2, name2 = m._load_dataset_for_args(args_custom, hf_cache_dir="/cache")
    assert name2 == "custom/path"
    assert dataset2["limited"]["id"] == "custom/path"
    assert dataset2["num"] is None


def test_load_math500_wraps_core(math_llama, monkeypatch):
    m = math_llama.module
    captured = {}

    def fake_load(cache_dir, split, seed, dataset_path=None):
        captured["args"] = (cache_dir, split, seed, dataset_path)
        return ["records"]

    monkeypatch.setattr(m._math_core, "load_math500", fake_load)

    result = m.load_math500("/c", "validation", 9, dataset_path="local.json")
    assert result == ["records"]
    assert captured["args"] == ("/c", "validation", 9, "local.json")


def test_main_wires_components(monkeypatch, tmp_path, math_llama):
    m = math_llama.module
    captured = {}

    monkeypatch.setattr(m, "_init_tokenizer_and_eos_ids", lambda args: ("tok", [7], "/cache"))
    monkeypatch.setattr(m, "_init_model", lambda args, cache_dir: ("model", "tagged"))
    monkeypatch.setattr(m, "_load_dataset_for_args", lambda args, cache_dir: (["d0"], "dataset-name"))
    monkeypatch.setattr(
        m, "build_math_inference_config_kwargs_from_args", lambda args, eos_ids: {"cfg_flag": True, "eos_ids": eos_ids}
    )

    def fake_run_inference_on_split(**kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(m, "run_inference_on_split", fake_run_inference_on_split)

    outdir = tmp_path / "math-out"
    argv = [
        "prog",
        "--model_name_or_path",
        "ckpt",
        "--output_dir",
        str(outdir),
        "--ds_config",
        "ds.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    m.main()

    assert outdir.exists()
    run_kwargs = captured["kwargs"]
    assert run_kwargs["split_name"] == "test"
    assert run_kwargs["examples"] == ["d0"]
    assert run_kwargs["tokenizer"] == "tok"
    assert run_kwargs["model"] == "model"
    assert run_kwargs["step"] == 0
    assert run_kwargs["outdir"] == str(outdir)
    assert run_kwargs["cfg_flag"] is True
    assert run_kwargs["eos_ids"] == [7]


def test_deepspeed_patch_logs_warning_on_failure(monkeypatch, caplog):
    def fail_imports(name, orig_import_module):
        if name in {
            "torch.serialization",
            "deepspeed.runtime.zero.config",
            "deepspeed.runtime.fp16.loss_scaler",
        }:
            raise ImportError("missing optional module")
        return None

    caplog.set_level("WARNING")
    env = _reload_math_llama_core(
        monkeypatch,
        torch_version="2.6.1",
        import_module_override=fail_imports,
    )

    warnings = [record.message for record in caplog.records if "DeepSpeed patch disabled" in record.message]
    assert warnings, "expected DeepSpeed patch warning when optional deps are missing"
    assert env.fake_torch.__version__ == "2.6.1"
