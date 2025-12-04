import argparse
import types
from types import SimpleNamespace

import pytest

import src.inference.runners.unified_runner_base as urb


def test_parse_math_args_defaults():
    args = urb.parse_math_args(["--model_name_or_path", "m", "--output_dir", "out"])
    assert args.dataset_id == "MATH-500"
    assert args.dataset_path is None
    assert args.dtype == "float16"


def test_load_generic_dataset_uses_import(monkeypatch):
    called = {}

    class DummyDS:
        def load_dataset(self, dataset_id, split=None, cache_dir=None):
            called["args"] = (dataset_id, split, cache_dir)
            return ["ok"]

    monkeypatch.setattr(urb, "import_module", lambda name: DummyDS())
    out = urb._load_generic_dataset("ds", "test", "/cache")
    assert out == ["ok"]
    assert called["args"] == ("ds", "test", "/cache")


def test_run_math_main_uses_math500_and_generic(monkeypatch, tmp_path):
    captured = {}

    def fake_parse(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="float16",
            attn_implementation=None,
            dataset_id="MATH-500",
            dataset_path=None,
            split="test",
            seed=0,
            output_dir=str(tmp_path),
            step=1,
        )

    class FakeBackend:
        tokenizer = "tok"
        model = "model"

    monkeypatch.setattr(urb, "parse_math_args", fake_parse)
    monkeypatch.setattr(urb, "setup_hf_cache_dir_env", lambda path: path)
    monkeypatch.setattr(
        urb,
        "init_unified_backend_and_eos",
        lambda **kwargs: (FakeBackend(), [1]),
    )
    monkeypatch.setattr(
        urb.math_core,
        "load_math500",
        lambda cache, split, seed, dataset_path=None: ["math500"],
    )
    monkeypatch.setattr(urb, "_load_generic_dataset", lambda *a, **k: ["generic"])
    monkeypatch.setattr(urb, "limit_dataset_for_args", lambda ds, args: ds + ["limited"])
    monkeypatch.setattr(
        urb,
        "build_math_inference_config_kwargs_from_args",
        lambda args, eos_ids: {"extra": "cfg"},
    )
    monkeypatch.setattr(
        urb.math_core,
        "MathInferenceConfig",
        lambda **kwargs: SimpleNamespace(cfg=kwargs),
    )

    def fake_run(examples, tokenizer, model, config):
        captured["examples"] = examples
        captured["config"] = config

    monkeypatch.setattr(urb.math_core, "run_inference_on_split", fake_run)

    urb.run_math_main(FakeBackend, argv=[])
    assert captured["examples"] == ["math500", "limited"]
    assert captured["config"].cfg["extra"] == "cfg"

    # Generic dataset path
    def fake_parse_generic(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="float16",
            attn_implementation=None,
            dataset_id="custom",
            dataset_path=None,
            split="test",
            seed=0,
            output_dir=str(tmp_path),
            step=1,
        )

    monkeypatch.setattr(urb, "parse_math_args", fake_parse_generic)
    urb.run_math_main(FakeBackend, argv=[])
    assert captured["examples"][0] == "generic"

    # MATH-500 with dataset_path branch
    def fake_parse_path(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="float16",
            attn_implementation=None,
            dataset_id="MATH-500",
            dataset_path="local.json",
            split="test",
            seed=0,
            output_dir=str(tmp_path),
            step=1,
        )

    captured["examples"] = None
    monkeypatch.setattr(urb, "parse_math_args", fake_parse_path)
    urb.run_math_main(FakeBackend, argv=[])
    assert captured["examples"][0] == "math500"


def test_mathtestconfig_validates_and_defaults():
    cfg = urb.MathTestConfig(
        [{"problem": "p"}],
        "out",
        1,
        batch_size=2,
        num_samples=3,
        temperature=0.4,
    )
    assert cfg.limits.batch_size == 2
    assert cfg.limits.think_cap == 750
    assert cfg.sampling.temperature == 0.4
    with pytest.raises(TypeError):
        urb.MathTestConfig(["p"], "out", 1, unexpected=1)
    with pytest.raises(TypeError):
        urb.MathTestConfig([{}], "o", 0, "extra")


def test_run_math_inference_builds_config(monkeypatch):
    captured = {}

    class DummyBackend:
        tokenizer = "tok"
        model = "model"

    monkeypatch.setattr(
        urb.math_core,
        "MathInferenceConfig",
        lambda **kwargs: SimpleNamespace(cfg=kwargs),
    )

    def fake_run(examples, tokenizer, model, config):
        captured["examples"] = list(examples)
        captured["config"] = config

    monkeypatch.setattr(urb.math_core, "run_inference_on_split", fake_run)
    cfg = urb.MathTestConfig(
        [{"problem": "p"}],
        "out",
        1,
        batch_size=1,
        num_samples=1,
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=1,
        think_cap=10,
        answer_cap=5,
        eos_ids=[2],
    )
    urb.run_math_inference(backend=DummyBackend(), config=cfg)
    assert captured["examples"][0]["problem"] == "p"
    assert captured["config"].cfg["two_pass"] is True
    assert captured["config"].cfg["eos_ids"] == [2]


def test_build_crossword_run_kwargs_with_and_without_configs():
    class CwModWith:
        class CrosswordCapsConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CrosswordSamplingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CrosswordTwoPassConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CrosswordInferenceConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    class CwModWithout:
        pass

    args = argparse.Namespace(
        batch_size=1,
        num_samples=1,
        think_cap=10,
        answer_cap=5,
        temperature=0.1,
        top_p=0.9,
        entropy_mode="mode",
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        split="test",
        output_dir="out",
        step=1,
    )
    run_kwargs = urb._build_crossword_run_kwargs(
        CwModWith, args, backend=SimpleNamespace(tokenizer="tok", model="m"), eos_ids=[1], examples=["e"]
    )
    assert "config" in run_kwargs

    run_kwargs2 = urb._build_crossword_run_kwargs(
        CwModWithout, args, backend=SimpleNamespace(tokenizer="tok", model="m"), eos_ids=[1], examples=["e"]
    )
    assert run_kwargs2["examples"] == ["e"]
    assert run_kwargs2["two_pass"] is True


def test_run_crossword_main_local_path(monkeypatch, tmp_path):
    captured = {}

    def fake_parse(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="float16",
            attn_implementation=None,
            dataset_id="CROSSWORD-LOCAL",
            dataset_path=str(tmp_path / "data.jsonl"),
            split="test",
            batch_size=1,
            num_samples=1,
            think_cap=10,
            answer_cap=5,
            temperature=0.1,
            top_p=0.9,
            entropy_mode="mode",
            two_pass=False,
            second_pass_phrase="cue",
            second_pass_use_sample_idx=0,
            step=1,
            output_dir=str(tmp_path),
        )

    monkeypatch.setattr(urb, "parse_crossword_args", fake_parse)
    monkeypatch.setattr(
        urb, "init_unified_backend_and_eos", lambda **kwargs: (SimpleNamespace(tokenizer="tok", model="m"), [1])
    )
    cw_mod = SimpleNamespace(
        load_crossword_local=lambda path: ["rows"],
        run_inference_on_split=lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(urb, "require_datasets", lambda: (None, None))
    monkeypatch.setattr(urb, "limit_dataset_for_args", lambda ds, args: ds)
    monkeypatch.setattr(
        urb, "_build_crossword_run_kwargs", lambda *a, **k: {"examples": ["rows"], "tokenizer": "tok", "model": "m"}
    )
    urb.run_crossword_main(lambda: cw_mod, backend_cls=None, argv=[])
    assert captured["examples"] == ["rows"]


def test_run_crossword_main_errors_and_remote(monkeypatch):
    # Missing dataset_path for local
    def fake_parse(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="float16",
            attn_implementation=None,
            dataset_id="CROSSWORD-LOCAL",
            dataset_path=None,
            split="test",
            output_dir="out",
            step=0,
            batch_size=1,
            num_samples=1,
            think_cap=1,
            answer_cap=1,
            temperature=0.0,
            top_p=0.9,
            entropy_mode="sum",
            two_pass=False,
            second_pass_phrase="cue",
            second_pass_use_sample_idx=0,
        )

    monkeypatch.setattr(
        urb, "init_unified_backend_and_eos", lambda **k: (SimpleNamespace(tokenizer="tok", model="m"), [1])
    )
    monkeypatch.setattr(urb, "parse_crossword_args", fake_parse)
    with pytest.raises(ValueError):
        urb.run_crossword_main(lambda: object(), backend_cls=None, argv=[])

    # Remote dataset path via require_datasets
    class DummyLoad:
        def __call__(self, dataset_id, split=None, cache_dir=None):
            return ["rows"]

    def fake_parse_remote(argv=None):
        ns = fake_parse()
        ns.dataset_id = "REMOTE"
        return ns

    monkeypatch.setattr(urb, "parse_crossword_args", fake_parse_remote)
    monkeypatch.setattr(urb, "limit_dataset_for_args", lambda ds, args: ds)
    monkeypatch.setattr(urb, "require_datasets", lambda: (None, DummyLoad()))
    monkeypatch.setattr(
        urb, "_build_crossword_run_kwargs", lambda *a, **k: {"examples": ["rows"], "tokenizer": "tok", "model": "m"}
    )
    cw_mod = types.SimpleNamespace(run_inference_on_split=lambda **kwargs: kwargs)
    urb.run_crossword_main(lambda: cw_mod, backend_cls=None, argv=[])
