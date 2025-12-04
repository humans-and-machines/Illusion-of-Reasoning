#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import runpy
import sys
import types

import pytest

import src.training.generate as generate


def test_require_distilabel_deps_raises_when_missing(monkeypatch):
    monkeypatch.setattr(generate, "importlib", importlib)
    monkeypatch.setattr(
        generate.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(RuntimeError) as excinfo:
        generate._require_distilabel_deps()
    assert "distilabel" in str(excinfo.value)


def test_build_distilabel_pipeline_passes_config(monkeypatch):
    calls: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self):
            calls["pipeline_init"] = True

        def ray(self):
            calls["ray_called"] = True
            return self

        def __enter__(self):
            calls["enter"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["exit"] = (exc_type, exc, tb)

    class FakeStepResources:
        def __init__(self, replicas):
            calls["replicas"] = replicas

    class FakeTextGeneration:
        def __init__(self, **kwargs):
            calls["text_kwargs"] = kwargs

    monkeypatch.setattr(
        generate,
        "_require_distilabel_deps",
        lambda: (FakeOpenAI, FakePipeline, FakeStepResources, FakeTextGeneration),
    )

    cfg = generate.DistilabelPipelineConfig(
        prompt_column="prompt_field",
        prompt_template="tmpl: {{ instruction }}",
        generation=generate.GenerationSettings(
            temperature=0.3,
            top_p=0.7,
            max_new_tokens=99,
            num_generations=2,
        ),
        input_batch_size=8,
        client_replicas=3,
        timeout=10,
        retries=5,
    )

    pipeline = generate.build_distilabel_pipeline("m0", base_url="http://x", config=cfg)

    assert pipeline is not None
    gen_kwargs = calls["openai_kwargs"]["generation_kwargs"]  # type: ignore[index]
    assert gen_kwargs["max_new_tokens"] == 99
    assert gen_kwargs["temperature"] == 0.3
    assert gen_kwargs["top_p"] == 0.7
    assert calls["replicas"] == cfg.client_replicas
    text_kwargs = calls["text_kwargs"]  # type: ignore[assignment]
    assert text_kwargs["template"] == cfg.prompt_template
    assert text_kwargs["input_mappings"] == {"instruction": "prompt_field"}
    assert text_kwargs["num_generations"] == cfg.generation.num_generations


def test_build_distilabel_pipeline_defaults(monkeypatch):
    calls: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai_kwargs"] = kwargs

    class FakePipeline:
        def ray(self):
            return self

        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["exited"] = True

    class FakeStepResources:
        def __init__(self, replicas):
            calls["replicas"] = replicas

    class FakeTextGeneration:
        def __init__(self, **kwargs):
            calls["text_kwargs"] = kwargs

    monkeypatch.setattr(
        generate,
        "_require_distilabel_deps",
        lambda: (FakeOpenAI, FakePipeline, FakeStepResources, FakeTextGeneration),
    )

    pipeline = generate.build_distilabel_pipeline(model="m-default")
    assert pipeline is not None
    gen_kwargs = calls["openai_kwargs"]["generation_kwargs"]  # type: ignore[index]
    assert gen_kwargs == {"max_new_tokens": generate.GenerationSettings().max_new_tokens}
    text_kwargs = calls["text_kwargs"]  # type: ignore[assignment]
    assert text_kwargs["input_mappings"] == {}
    assert text_kwargs["num_generations"] == generate.GenerationSettings().num_generations
    assert calls["replicas"] == generate.DistilabelPipelineConfig().client_replicas
    assert calls.get("entered") and calls.get("exited")


def test_main_runs_pipeline_and_pushes(monkeypatch):
    calls: dict[str, object] = {}
    real_import_module = importlib.import_module

    dataset_mod = types.SimpleNamespace(load_dataset=lambda *a, **k: "DATASET")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai"] = kwargs

    class FakePipeline:
        def __init__(self):
            calls["pipeline"] = self

        def ray(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls["exit"] = True

        def run(self, dataset, dataset_batch_size=None, use_cache=None):
            calls["run"] = {
                "dataset": dataset,
                "dataset_batch_size": dataset_batch_size,
                "use_cache": use_cache,
            }
            return FakeDistiset()

    class FakeStepResources:
        def __init__(self, replicas):
            calls["replicas"] = replicas

    class FakeTextGeneration:
        def __init__(self, **kwargs):
            calls["text_kwargs"] = kwargs

    class FakeDistiset:
        def push_to_hub(self, repo_id, private=False):
            calls["pushed"] = {"repo_id": repo_id, "private": private}

    distilabel_modules = {
        "distilabel.llms": types.SimpleNamespace(OpenAILLM=FakeOpenAI),
        "distilabel.pipeline": types.SimpleNamespace(Pipeline=FakePipeline),
        "distilabel.steps": types.SimpleNamespace(StepResources=FakeStepResources),
        "distilabel.steps.tasks": types.SimpleNamespace(TextGeneration=FakeTextGeneration),
    }

    def fake_import_module(name, *args, **kwargs):
        if name == "datasets":
            return dataset_mod
        if name in distilabel_modules:
            return distilabel_modules[name]
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--hf-dataset",
            "ds",
            "--model",
            "m1",
            "--prompt-column",
            "instruction_col",
            "--prompt-template",
            "<{{ instruction }}>",
            "--temperature",
            "0.5",
            "--top-p",
            "0.4",
            "--max-new-tokens",
            "11",
            "--num-generations",
            "3",
            "--input-batch-size",
            "2",
            "--client-replicas",
            "4",
            "--timeout",
            "15",
            "--retries",
            "2",
            "--hf-output-dataset",
            "repo/name",
            "--private",
        ],
    )

    orig_generate_mod = sys.modules.pop("src.training.generate", None)
    try:
        runpy.run_module("src.training.generate", run_name="__main__", alter_sys=True)
    finally:
        if orig_generate_mod is not None:
            sys.modules["src.training.generate"] = orig_generate_mod

    gen_kwargs = calls["openai"]["generation_kwargs"]  # type: ignore[index]
    assert gen_kwargs["temperature"] == 0.5
    assert gen_kwargs["top_p"] == 0.4
    assert gen_kwargs["max_new_tokens"] == 11
    assert calls["replicas"] == 4
    assert calls["run"]["dataset"] == "DATASET"  # type: ignore[index]
    assert calls["run"]["dataset_batch_size"] == 2000  # 2 * 1000
    assert calls["pushed"] == {"repo_id": "repo/name", "private": True}
