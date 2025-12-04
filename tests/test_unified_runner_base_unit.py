import argparse
import types

import pytest

import src.inference.runners.unified_runner_base as base


def test_parse_math_args_defaults(monkeypatch):
    called = {}
    monkeypatch.setattr(
        base,
        "add_model_and_output_args",
        lambda p: (
            p.add_argument("--model_name_or_path"),
            p.add_argument("--output_dir"),
            called.setdefault("add_model", True),
        ),
    )
    monkeypatch.setattr(
        base,
        "configure_unified_runner_common",
        lambda p, default_dtype: called.setdefault("config_common", default_dtype),
    )
    args = base.parse_math_args(["--model_name_or_path", "m", "--output_dir", "o"])
    assert args.model_name_or_path == "m" and args.output_dir == "o"
    assert called["add_model"] is True and called["config_common"] == "float16"


def test_simple_list_dataset_select():
    ds = base._SimpleListDataset([{"id": 0}, {"id": 1}, {"id": 2}])
    subset = ds.select([2, 0])
    assert len(subset) == 2
    assert list(subset)[0]["id"] == 2 and list(subset)[1]["id"] == 0


def test_parse_carpark_and_crossword_defaults(monkeypatch):
    called = {}

    def _add(parser):
        parser.add_argument("--model_name_or_path")
        parser.add_argument("--output_dir")
        called.setdefault("add", True)

    monkeypatch.setattr(base, "add_model_and_output_args", _add)
    monkeypatch.setattr(
        base,
        "configure_unified_runner_common",
        lambda parser, default_dtype: called.setdefault("dtype", default_dtype),
    )

    car_args = base.parse_carpark_args(["--model_name_or_path", "m", "--output_dir", "o"])
    assert car_args.dataset_id == "od2961/rush4-5-6-balanced"
    assert called["dtype"] == "bfloat16"

    called.pop("dtype", None)
    cw_args = base.parse_crossword_args(["--model_name_or_path", "m", "--output_dir", "o"])
    assert cw_args.dataset_id == "CROSSWORD-LOCAL"
    assert cw_args.dataset_path is None
    assert called["dtype"] == "float16"


def test_math_test_config_positional_and_kwargs():
    cfg = base.MathTestConfig(
        [{"p": "q"}],
        "out",
        1,
        batch_size=2,
        num_samples=3,
        temperature=0.4,
        top_p=0.8,
        think_cap=5,
        answer_cap=6,
        two_pass=True,
        second_pass_phrase="c",
        second_pass_use_sample_idx=1,
        eos_ids=[9],
    )
    assert cfg.output_dir == "out" and cfg.step == 1
    assert cfg.limits.batch_size == 2 and cfg.sampling.temperature == 0.4
    with pytest.raises(TypeError):
        base.MathTestConfig(dataset=[{}], output_dir="o", step=0, unexpected=1)


def test_run_math_inference_invokes_core(monkeypatch, tmp_path):
    records = [{"problem": "p", "answer": "a"}]
    cfg = base.MathTestConfig(records, str(tmp_path), 0)
    backend = types.SimpleNamespace(tokenizer="tok", model="model")
    called = {}

    def fake_run_inference_on_split(examples, tokenizer, model, config):
        called["examples"] = list(examples)
        called["tokenizer"] = tokenizer
        called["model"] = model
        called["config_step"] = config.step

    monkeypatch.setattr(base.math_core, "run_inference_on_split", fake_run_inference_on_split)

    base.run_math_inference(backend=backend, config=cfg)
    assert called["examples"][0]["problem"] == "p"
    assert called["config_step"] == 0


def test_build_crossword_run_kwargs_configured(monkeypatch):
    class CWMod:
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

    args = argparse.Namespace(
        batch_size=1,
        num_samples=1,
        think_cap=5,
        answer_cap=6,
        temperature=0.1,
        top_p=0.9,
        entropy_mode="sum",
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        split="test",
        output_dir="out",
        step=0,
    )
    backend = types.SimpleNamespace(tokenizer="tok", model="m")
    eos_ids = [1]
    kwargs = base._build_crossword_run_kwargs(CWMod, args, backend, eos_ids, examples=["ex"])
    assert "config" in kwargs and isinstance(kwargs["config"], CWMod.CrosswordInferenceConfig)


def test_build_crossword_run_kwargs_fallback():
    class CWMod:
        pass

    args = argparse.Namespace(
        batch_size=1,
        num_samples=1,
        think_cap=5,
        answer_cap=6,
        temperature=0.1,
        top_p=0.9,
        entropy_mode="sum",
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        split="test",
        output_dir="out",
        step=0,
    )
    backend = types.SimpleNamespace(tokenizer="tok", model="m")
    eos_ids = [1]
    kwargs = base._build_crossword_run_kwargs(CWMod, args, backend, eos_ids, examples=["ex"])
    assert kwargs["eos_ids"] == eos_ids and kwargs["examples"] == ["ex"]


def test_run_carpark_main_invokes_pipeline(monkeypatch, tmp_path):
    captured = {}

    def fake_parse(argv=None):
        return argparse.Namespace(
            model_name_or_path="m",
            revision="main",
            tokenizer_path=None,
            dtype="bf16",
            attn_implementation=None,
            dataset_id="rush",
            dataset_prompt_column="prompt",
            dataset_solution_column="sol",
            split="test",
            output_dir=str(tmp_path),
            step=2,
            batch_size=1,
            num_samples=1,
            think_cap=5,
            answer_cap=3,
            temperature=0.1,
            top_p=0.9,
            entropy_mode="mode",
            two_pass=True,
            second_pass_phrase="cue",
            second_pass_use_sample_idx=0,
        )

    monkeypatch.setattr(base, "parse_carpark_args", fake_parse)
    monkeypatch.setattr(
        base,
        "init_unified_backend_and_eos",
        lambda **kwargs: (types.SimpleNamespace(tokenizer="tok", model="m"), [9]),
    )

    def fake_load_module():
        def load_rush_dataset(**kwargs):
            captured["load_args"] = kwargs
            return [{"id": 1}]

        def run_inference_on_split(**kwargs):
            captured["run_kwargs"] = kwargs

        return types.SimpleNamespace(
            load_rush_dataset=load_rush_dataset,
            run_inference_on_split=run_inference_on_split,
        )

    monkeypatch.setattr(base, "limit_dataset_for_args", lambda ds, args: ds + [{"id": 99}])
    monkeypatch.setattr(
        base, "build_math_inference_config_kwargs_from_args", lambda args, eos_ids: {"extra_cfg": eos_ids}
    )

    base.run_carpark_main(fake_load_module, backend_cls=None, argv=[])
    assert captured["load_args"]["dataset_id"] == "rush"
    assert captured["run_kwargs"]["examples"][-1]["id"] == 99
    assert captured["run_kwargs"]["extra_cfg"] == [9]
    assert captured["run_kwargs"]["outdir"] == str(tmp_path)
