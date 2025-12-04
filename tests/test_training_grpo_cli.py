#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import is_dataclass
from types import SimpleNamespace

import pytest


configs_mod = pytest.importorskip("src.training.configs")


def test_grpo_main_wires_trl_parser_and_calls_impl(monkeypatch):
    import importlib
    import sys

    # Provide a minimal torch stub to satisfy optional imports.
    class FakeTensor:
        def __init__(self, data=None):
            self._data = data

        def to(self, *_args, **_kwargs):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

        @property
        def shape(self):
            return getattr(self._data, "shape", ())

    def _tensor(data=None, **_kwargs):
        return FakeTensor(data)

    fake_torch = SimpleNamespace(
        tensor=_tensor,
        zeros=lambda shape=None, **kwargs: FakeTensor(data=[[0] * shape[1]] if isinstance(shape, tuple) else [0]),
        ones=lambda shape=None, **kwargs: FakeTensor(data=[[1] * shape[1]] if isinstance(shape, tuple) else [1]),
        full=lambda shape, fill_value=0, **kwargs: FakeTensor(
            data=[[fill_value] * shape[1]] if isinstance(shape, tuple) else [fill_value]
        ),
        inference_mode=lambda *_a, **_k: (lambda fn: fn),
        no_grad=lambda *_a, **_k: (lambda fn: fn),
        SymFloat=FakeTensor,
        utils=SimpleNamespace(data=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", SimpleNamespace(data=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch.utils.data", SimpleNamespace(DataLoader=object, RandomSampler=object))

    # Build a fake trl module with ModelConfig and TrlParser.
    class FakeModelConfig:
        def __init__(self):
            self.model_name_or_path = "dummy-model"

    class FakeTrlParser:
        def __init__(self, config_classes):
            # Expect the tuple of config classes from grpo.main
            self.config_classes = config_classes

        def parse_args_and_config(self):
            # Instantiate each config class in order and return them as the parser would.
            instances = []
            for index, cls in enumerate(self.config_classes):
                if index == 0:
                    # GRPOScriptArguments needs a dataset field to pass validation.
                    instances.append(cls(dataset_name="dummy-dataset"))
                else:
                    instances.append(cls())
            assert len(instances) == 11
            # Map to the expected return tuple structure.
            return tuple(instances)

    fake_trl = SimpleNamespace(ModelConfig=FakeModelConfig, TrlParser=FakeTrlParser)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    # Stub grpo_impl so we don't pull in heavy runtime modules.
    recorded = {}

    def fake_impl_main(script_args, training_args, model_args):
        recorded["script_args"] = script_args
        recorded["training_args"] = training_args
        recorded["model_args"] = model_args

    fake_grpo_impl = SimpleNamespace(main=fake_impl_main)
    monkeypatch.setitem(sys.modules, "src.training.grpo_impl", fake_grpo_impl)

    # Import grpo after stubbing dependencies.
    grpo = importlib.import_module("src.training.grpo")

    # Wrap merge_dataclass_attributes so we can observe its usage.
    calls = []
    original_merge = grpo.merge_dataclass_attributes

    def wrapped_merge(target, *cfgs):
        calls.append((target, cfgs))
        return original_merge(target, *cfgs)

    monkeypatch.setattr(grpo, "merge_dataclass_attributes", wrapped_merge)

    # Run the CLI wiring.
    grpo.main()

    # Ensure underlying main was invoked with the expected objects.
    assert "script_args" in recorded
    assert "training_args" in recorded
    assert "model_args" in recorded

    # script_args should be an instance of GRPOScriptArguments.
    assert isinstance(recorded["script_args"], configs_mod.GRPOScriptArguments)
    # training_args should be a GRPOConfig instance; model_args a FakeModelConfig.
    assert isinstance(recorded["training_args"], configs_mod.GRPOConfig)
    assert isinstance(recorded["model_args"], FakeModelConfig)

    # merge_dataclass_attributes should have been used twice (script + training args).
    assert len(calls) == 2
    first_target, first_cfgs = calls[0]
    second_target, second_cfgs = calls[1]

    # First call flattens reward/dataset/span configs onto script_args.
    assert first_target is recorded["script_args"]
    assert any(is_dataclass(cfg) for cfg in first_cfgs)

    # Second call flattens chat/hub/wandb/grpo_only configs onto training_args.
    assert second_target is recorded["training_args"]
    assert any(is_dataclass(cfg) for cfg in second_cfgs)
