#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


def _install_stub_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install lightweight stubs into sys.modules so math_llama_core can run under __main__."""
    # Torch stub with minimal surface used by imports.
    torch_mod = types.SimpleNamespace(
        bfloat16=float,
        float16=float,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name=None: f"device:{name}",
    )

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    torch_mod.inference_mode = lambda *_a, **_k: _Ctx()
    torch_mod.no_grad = lambda *_a, **_k: _Ctx()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.cuda", torch_mod.cuda)

    # Transformers stubs
    class DummyConfig:
        def __init__(self, *args, **kwargs):
            self.attn_implementation = None

    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 1
            self.pad_token = None
            self.eos_token = "<eos>"
            self.truncation_side = None
            self.padding_side = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def convert_tokens_to_ids(self, tok):
            return 2 if tok else None

    class DummyParam:
        requires_grad = True

    class DummyModel:
        def __init__(self):
            self._params = [DummyParam()]
            self.config = types.SimpleNamespace()

        @classmethod
        def from_config(cls, *_args, **_kwargs):
            return cls()

        def to(self, *_args, **_kwargs):
            return self

        def parameters(self):
            return self._params

        def eval(self):
            return self

    transformers_mod = types.SimpleNamespace(
        AutoConfig=type("AC", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyConfig())}),
        AutoTokenizer=type("AT", (), {"from_pretrained": staticmethod(DummyTokenizer.from_pretrained)}),
        AutoModelForCausalLM=type("AM", (), {"from_config": staticmethod(DummyModel.from_config)}),
        StoppingCriteria=type("StoppingCriteria", (), {}),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # DeepSpeed stub
    class Engine:
        def __init__(self):
            self.module = DummyModel()
            self.device = "cpu"

        def load_checkpoint(self, *args, **kwargs):
            return None

        def eval(self):
            return self

    def initialize(*_args, **_kwargs):
        return Engine(), None, None, None

    deepspeed_mod = types.SimpleNamespace(initialize=initialize)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_mod)

    # datasets stub for require_datasets()
    datasets_mod = types.SimpleNamespace(
        Dataset=object,
        load_dataset=lambda *a, **k: [{"problem": "p"}],
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    # math_llama_utils stub so DSModelWrapper does not depend on real torch
    class StubWrapper:
        def __init__(self, engine):
            self.engine = engine
            self.module = engine.module

        def eval(self):
            return self

        def __getattr__(self, name):
            return getattr(self.module, name)

        def parameters(self):
            return []

    math_llama_utils_mod = types.SimpleNamespace(DSModelWrapper=StubWrapper)
    monkeypatch.setitem(
        sys.modules,
        "src.inference.utils.math_llama_utils",
        math_llama_utils_mod,
    )

    # math_core stub to intercept run_inference_on_split
    math_core_mod = types.SimpleNamespace(last_call=None)

    class MathInferenceConfig:
        def __init__(self, split_name=None, output_dir=None, step=None, **kwargs):
            self.split_name = split_name
            self.output_dir = output_dir
            self.step = step
            self.kwargs = kwargs

    def run_inference_on_split(**kwargs):
        math_core_mod.last_call = kwargs

    math_core_mod.MathInferenceConfig = MathInferenceConfig
    math_core_mod.run_inference_on_split = run_inference_on_split
    math_core_mod.load_math500 = lambda *_a, **_k: [{"problem": "p"}]
    monkeypatch.setitem(
        sys.modules,
        "src.inference.domains.math.math_core",
        math_core_mod,
    )

    return math_core_mod


def test_main_guard_executes_with_stubs(monkeypatch, tmp_path):
    math_core_mod = _install_stub_modules(tmp_path, monkeypatch)

    # Fresh import via run_module under __main__ to exercise the guard.
    sys.argv = [
        "math_llama_core",
        "--model_name_or_path",
        "model",
        "--output_dir",
        str(tmp_path / "out"),
        "--ds_config",
        "ds.json",
    ]
    sys.modules.pop("src.inference.domains.math.math_llama_core", None)
    runpy.run_module("src.inference.domains.math.math_llama_core", run_name="__main__")

    assert isinstance(math_core_mod.last_call["config"], math_core_mod.MathInferenceConfig)
    assert math_core_mod.last_call["config"].step == 0
    assert Path(math_core_mod.last_call["config"].output_dir).name == "out"
    assert math_core_mod.last_call["examples"] == [{"problem": "p"}]
