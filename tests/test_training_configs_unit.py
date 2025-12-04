import types

import pytest

import src.training.configs as cfg_mod


def test_maybe_run_post_init_handles_import_error():
    called = {"count": 0}

    def bad_post_init():
        called["count"] += 1
        raise ImportError("nope")

    cfg_mod._maybe_run_post_init(bad_post_init)
    assert called["count"] == 1


def test_maybe_run_post_init_noop_when_not_callable():
    assert cfg_mod._maybe_run_post_init(None) is None


def test_maybe_run_post_init_reraises_outside_pytest(monkeypatch):
    class TorchWithBackends:
        backends = object()

    class Transformers:
        pass

    monkeypatch.setattr(cfg_mod, "_IS_PYTEST", False)
    monkeypatch.setitem(cfg_mod.sys.modules, "torch", TorchWithBackends())
    monkeypatch.setitem(cfg_mod.sys.modules, "transformers", Transformers())

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cfg_mod._maybe_run_post_init(boom)


def test_maybe_run_post_init_skips_errors_in_pytest(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_IS_PYTEST", True)
    called = {}

    def boom():
        called["hit"] = True
        raise RuntimeError("boom")

    cfg_mod._maybe_run_post_init(boom)
    assert called["hit"] is True


def test_script_arguments_dataset_mixture_parsed():
    args = cfg_mod.ScriptArguments(
        dataset_name=None,
        dataset_mixture={
            "datasets": [
                {"id": "ds1", "config": "c1", "split": "train", "columns": ["a"], "weight": 0.5},
                {"id": "ds2", "config": None, "split": "train", "columns": None, "weight": None},
            ],
            "seed": 42,
            "test_split_size": 0.1,
        },
    )
    assert isinstance(args.dataset_mixture, cfg_mod.DatasetMixtureConfig)
    assert args.dataset_mixture.seed == 42
    assert args.dataset_mixture.datasets[0].dataset_id == "ds1"


def test_trl_wrappers_skip_post_init(monkeypatch):
    transformers_stub = types.SimpleNamespace()
    torch_stub = types.SimpleNamespace(backends=None)
    monkeypatch.setitem(cfg_mod.sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(cfg_mod.sys.modules, "torch", torch_stub)

    class Boom:
        def __post_init__(self):
            raise RuntimeError("boom")

    class DummyGRPO(Boom):
        pass

    class DummySFT(Boom):
        pass

    monkeypatch.setattr(
        cfg_mod,
        "trl",
        types.SimpleNamespace(GRPOConfig=DummyGRPO, SFTConfig=DummySFT, ScriptArguments=cfg_mod.trl.ScriptArguments),
    )

    cfg_mod._IS_PYTEST = True
    cfg_mod.GRPOConfig().__post_init__()
    cfg_mod.SFTConfig().__post_init__()
