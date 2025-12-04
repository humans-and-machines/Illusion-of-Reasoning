import importlib
import sys
import types


def _make_pkg(name: str):
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    return mod


def test_grpo_runtime_env_reexports(monkeypatch):
    env_mod = types.ModuleType("src.training.runtime.env")
    env_mod.__all__ = ["foo"]
    env_mod.foo = 123
    runtime_pkg = _make_pkg("src.training.runtime")
    runtime_pkg.env = env_mod
    monkeypatch.setitem(sys.modules, "src.training.runtime", runtime_pkg)
    monkeypatch.setitem(sys.modules, "src.training.runtime.env", env_mod)
    sys.modules.pop("src.training.grpo_runtime_env", None)

    shim = importlib.reload(importlib.import_module("src.training.grpo_runtime_env"))
    assert shim.foo == 123
    assert "foo" in shim.__all__


def test_grpo_runtime_impl_and_main_forward_main(monkeypatch):
    main_mod = types.ModuleType("src.training.runtime.main")

    def fake_main():
        return "ok"

    main_mod.fake = fake_main
    main_mod.main = fake_main

    runtime_pkg = _make_pkg("src.training.runtime")
    runtime_pkg.main = main_mod

    monkeypatch.setitem(sys.modules, "src.training.runtime", runtime_pkg)
    monkeypatch.setitem(sys.modules, "src.training.runtime.main", main_mod)
    sys.modules.pop("src.training.grpo_runtime_impl", None)
    sys.modules.pop("src.training.grpo_runtime_main", None)

    impl = importlib.reload(importlib.import_module("src.training.grpo_runtime_impl"))
    main_shim = importlib.reload(importlib.import_module("src.training.grpo_runtime_main"))
    assert impl.main() == "ok"
    assert main_shim.main() == "ok"
    assert "main" in main_shim.__all__


def test_grpo_runtime_impl_full_reexports(monkeypatch):
    replay_impl_mod = types.ModuleType("src.training.grpo_trainer_replay_impl")

    class DummyReplay:
        pass

    replay_impl_mod.GRPOTrainerReplay = DummyReplay

    support_mod = types.ModuleType("src.training.grpo_trainer_replay_support")

    class RS: ...

    class TS: ...

    class MS: ...

    class RTS: ...

    class LLC: ...

    support_mod.ReplaySettings = RS
    support_mod.TemperatureSchedule = TS
    support_mod.MixSettings = MS
    support_mod.RuntimeState = RTS
    support_mod.LossLoggingCallback = LLC

    main_mod = types.ModuleType("src.training.runtime.main")
    main_mod.main = lambda: "main"

    runtime_pkg = _make_pkg("src.training.runtime")
    runtime_pkg.main = main_mod
    monkeypatch.setitem(sys.modules, "src.training.runtime", runtime_pkg)
    monkeypatch.setitem(sys.modules, "src.training.runtime.main", main_mod)
    monkeypatch.setitem(sys.modules, "src.training.grpo_trainer_replay_impl", replay_impl_mod)
    monkeypatch.setitem(sys.modules, "src.training.grpo_trainer_replay_support", support_mod)
    sys.modules.pop("src.training.grpo_runtime_impl_full", None)

    mod = importlib.reload(importlib.import_module("src.training.grpo_runtime_impl_full"))
    for name in [
        "ReplaySettings",
        "TemperatureSchedule",
        "MixSettings",
        "RuntimeState",
        "LossLoggingCallback",
        "GRPOTrainerReplay",
        "main",
    ]:
        assert hasattr(mod, name)
        assert name in mod.__all__


def test_grpo_trainer_replay_reexports_from_full(monkeypatch):
    full_mod = types.ModuleType("src.training.grpo_runtime_impl_full")

    class RS: ...

    class TS: ...

    class MS: ...

    class RTS: ...

    class LLC: ...

    class GR: ...

    full_mod.ReplaySettings = RS
    full_mod.TemperatureSchedule = TS
    full_mod.MixSettings = MS
    full_mod.RuntimeState = RTS
    full_mod.LossLoggingCallback = LLC
    full_mod.GRPOTrainerReplay = GR

    monkeypatch.setitem(sys.modules, "src.training.grpo_runtime_impl_full", full_mod)
    sys.modules.pop("src.training.grpo_trainer_replay", None)

    replay_mod = importlib.reload(importlib.import_module("src.training.grpo_trainer_replay"))
    expected_names = {cls.__name__ for cls in (RS, TS, MS, RTS, LLC, GR)}
    assert set(replay_mod.__all__) == expected_names
    assert replay_mod.GRPOTrainerReplay is GR
