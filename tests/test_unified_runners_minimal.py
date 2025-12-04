import importlib
import sys

import pytest


def test_load_carpark_module(monkeypatch):
    # Ensure import_module is invoked with the expected path.
    calls = []

    def fake_import(name):
        calls.append(name)
        return "carpark_mod"

    mod = importlib.import_module("src.inference.runners.unified_carpark_runner")
    monkeypatch.setattr(mod, "import_module", fake_import)
    assert mod.load_carpark_module() == "carpark_mod"
    assert calls == ["src.inference.domains.carpark.carpark_core"]


def test_load_crossword_module(monkeypatch):
    fake_core = object()
    monkeypatch.setitem(sys.modules, "src.inference.domains.crossword.crossword_core", fake_core)
    mod = importlib.import_module("src.inference.runners.unified_crossword_runner")
    assert mod.load_crossword_module() is fake_core


def test_run_math_inference_builds_config(monkeypatch):
    mod = importlib.import_module("src.inference.runners.unified_math_runner")
    # Stub out the underlying run_math_inference to capture the config.
    captured = {}

    def fake_run_math_inference(backend, config):
        captured["backend"] = backend
        captured["config"] = config

    monkeypatch.setattr(mod, "_run_math_inference", fake_run_math_inference)
    backend = object()
    mod.run_math_inference(
        backend=backend,
        dataset="ds",
        output_dir="/tmp/out",
        step=123,
        batch_size=4,
        num_samples=2,
        temperature=0.5,
        top_p=0.9,
        think_cap=10,
        answer_cap=5,
        two_pass=True,
        second_pass_phrase="foo",
        second_pass_use_sample_idx=1,
        eos_ids=[1, 2],
    )
    cfg = captured["config"]
    assert captured["backend"] is backend
    assert cfg.dataset == "ds"
    assert cfg.output_dir == "/tmp/out"
    assert cfg.step == 123
    assert cfg.limits.batch_size == 4
    assert cfg.limits.num_samples == 2
    assert cfg.sampling.temperature == 0.5
    assert cfg.sampling.top_p == 0.9
    assert cfg.limits.think_cap == 10
    assert cfg.limits.answer_cap == 5
    assert cfg.sampling.two_pass is True
    assert cfg.sampling.second_pass_phrase == "foo"
    assert cfg.sampling.second_pass_use_sample_idx == 1
    assert cfg.eos_ids == [1, 2]

    with pytest.raises(TypeError):
        mod.run_math_inference(backend=backend)
