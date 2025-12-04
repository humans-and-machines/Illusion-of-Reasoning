import runpy
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace


def test_main_calls_merge_and_main(monkeypatch):
    # Stub heavy dependencies before importing module.
    class FakeTensor:
        def __init__(self, data=None):
            self._data = data

        def to(self, *_args, **_kwargs):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

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
        device=lambda *_a, **_k: "cpu",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch.utils", SimpleNamespace(data=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch.utils.data", SimpleNamespace(DataLoader=object, RandomSampler=object))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(
            ModelConfig=SimpleNamespace,
            TrlParser=None,
            ScriptArguments=SimpleNamespace,
            GRPOConfig=SimpleNamespace,
            SFTConfig=SimpleNamespace,
        ),
    )
    # Stub runtime.env module to satisfy imports during grpo import.
    env_stub = SimpleNamespace(
        datasets=None,
        transformers=None,
        get_last_checkpoint=lambda *_: None,
        get_peft_config=lambda *_: None,
        set_seed=lambda *_: None,
        torch=fake_torch,
        dist=SimpleNamespace(),
        GRPOTrainer=object,
        wandb=SimpleNamespace(),
        TrainerCallback=object,
    )
    monkeypatch.setitem(sys.modules, "src.training.runtime.env", env_stub)
    import src.training.cli.grpo as grpo_cli

    called = {}

    class FakeParser:
        def __init__(self, configs):
            self.configs = configs

        def parse_args_and_config(self):
            return (
                "script",
                "reward",
                "cosine",
                "dataset",
                "span_kl",
                "chat",
                "hub",
                "wandb",
                "grpo_only",
                "train",
                "model",
            )

    def fake_load():
        return SimpleNamespace, lambda configs: FakeParser(configs)

    def fake_merge(target, *others):
        called.setdefault("merge_calls", []).append((target, others))

    monkeypatch.setattr(grpo_cli, "_load_trl_parser", fake_load)
    monkeypatch.setattr(grpo_cli, "_main", lambda *args: called.setdefault("main_args", args))

    grpo_cli.main(merge_fn=fake_merge)

    assert len(called["merge_calls"]) == 2
    assert called["main_args"] == ("script", "train", "model")


def test_dunder_main_executes_with_stubbed_trl(monkeypatch):
    # Ensure we load a fresh module instance via runpy.
    sys.modules.pop("src.training.cli.grpo", None)

    @dataclass
    class ScriptCfg:
        script: str = "script"

    @dataclass
    class RewardCfg:
        reward: str = "reward"

    @dataclass
    class CosineCfg:
        cosine: str = "cosine"

    @dataclass
    class DatasetCfg:
        dataset: str = "dataset"

    @dataclass
    class SpanCfg:
        span: str = "span"

    @dataclass
    class ChatCfg:
        chat: str = "chat"

    @dataclass
    class HubCfg:
        hub: str = "hub"

    @dataclass
    class WandbCfg:
        wandb: str = "wandb"

    @dataclass
    class OnlyCfg:
        only: str = "only"

    @dataclass
    class TrainCfg:
        train: str = "train"

    @dataclass
    class ModelCfg:
        model: str = "model"

    class FakeParser:
        def __init__(self, configs):
            self.configs = configs

        def parse_args_and_config(self):
            return (
                ScriptCfg(),
                RewardCfg(),
                CosineCfg(),
                DatasetCfg(),
                SpanCfg(),
                ChatCfg(),
                HubCfg(),
                WandbCfg(),
                OnlyCfg(),
                TrainCfg(),
                ModelCfg(),
            )

    captured = {}
    trl_stub = SimpleNamespace(
        ModelConfig=ModelCfg,
        TrlParser=FakeParser,
        ScriptArguments=object,
        GRPOConfig=object,
        SFTConfig=object,
    )
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.setitem(
        sys.modules, "src.training.grpo_impl", SimpleNamespace(main=lambda *args: captured.setdefault("args", args))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("src.training.cli.grpo", run_name="__main__")

    script_args, training_args, model_args = captured["args"]
    assert script_args.reward == "reward" and script_args.span == "span"
    assert training_args.chat == "chat" and training_args.only == "only"
    assert isinstance(model_args, ModelCfg)
