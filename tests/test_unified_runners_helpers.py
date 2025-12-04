#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


math_runner = pytest.importorskip("src.inference.runners.unified_math_runner")
math_cli = pytest.importorskip("src.inference.cli.unified_math")
carpark_cli = pytest.importorskip("src.inference.cli.unified_carpark")
crossword_cli = pytest.importorskip("src.inference.cli.unified_crossword")


def test_unified_math_runner_builds_config_and_delegates(monkeypatch, tmp_path):
    called = {}

    def fake_run_math_inference(*, backend, config):
        called["backend"] = backend
        called["config"] = config

    monkeypatch.setattr(math_runner, "_run_math_inference", fake_run_math_inference)

    class DummyBackend:
        pass

    dataset = ["problem1"]
    math_runner.run_math_inference(
        backend=DummyBackend(),
        dataset=dataset,
        output_dir=str(tmp_path),
        step=123,
        batch_size=2,
        num_samples=3,
        temperature=0.4,
        top_p=0.8,
        think_cap=10,
        answer_cap=5,
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=1,
        eos_ids=[42],
    )

    config = called["config"]
    assert config.dataset is dataset
    assert config.output_dir == str(tmp_path)
    assert config.step == 123
    assert config.limits.batch_size == 2
    assert config.limits.num_samples == 3
    assert config.limits.think_cap == 10
    assert config.limits.answer_cap == 5
    assert config.sampling.temperature == 0.4
    assert config.sampling.top_p == 0.8
    assert config.sampling.two_pass is True
    assert config.sampling.second_pass_phrase == "cue"
    assert config.sampling.second_pass_use_sample_idx == 1
    assert config.eos_ids == [42]


def test_unified_math_runner_main_uses_hfbackend(monkeypatch):
    called = {}

    def fake_run_math_main(*, backend_cls, argv=None):
        called["backend_cls"] = backend_cls
        called["argv"] = argv

    monkeypatch.setattr(math_cli, "run_math_main", fake_run_math_main)

    math_cli.main(argv=["--foo", "bar"])
    assert called["backend_cls"] is not None
    assert called["argv"] == ["--foo", "bar"]


def test_unified_carpark_runner_main_delegates(monkeypatch):
    called = {}

    def fake_run_carpark_main(*, load_module, backend_cls, argv=None):
        called["load_module"] = load_module
        called["backend_cls"] = backend_cls
        called["argv"] = argv

    monkeypatch.setattr(carpark_cli, "load_carpark_module", lambda: "module")
    monkeypatch.setattr(carpark_cli, "_load_carpark_module", carpark_cli._load_carpark_module)

    monkeypatch.setattr(carpark_cli, "run_carpark_main", fake_run_carpark_main)

    carpark_cli.main(argv=["--flag"])
    assert callable(called["load_module"])
    assert called["backend_cls"] is not None
    assert called["argv"] == ["--flag"]
    assert carpark_cli._load_carpark_module() == "module"


def test_unified_crossword_runner_main_delegates(monkeypatch):
    called = {}

    def fake_run_crossword_main(*, load_module, backend_cls, argv=None):
        called["load_module"] = load_module
        called["backend_cls"] = backend_cls
        called["argv"] = argv

    monkeypatch.setattr(crossword_cli, "load_crossword_module", lambda: "cross")
    monkeypatch.setattr(crossword_cli, "_load_crossword_module", crossword_cli._load_crossword_module)

    monkeypatch.setattr(crossword_cli, "run_crossword_main", fake_run_crossword_main)

    crossword_cli.main(argv=["--x"])
    assert callable(called["load_module"])
    assert called["backend_cls"] is not None
    assert called["argv"] == ["--x"]
    assert crossword_cli._load_crossword_module() == "cross"
