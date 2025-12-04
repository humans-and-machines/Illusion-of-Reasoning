import importlib


def test_main_delegates_to_run_math(monkeypatch):
    calls = {}

    def fake_run_math_main(**kwargs):
        calls["kwargs"] = kwargs

    mod = importlib.import_module("src.inference.cli.unified_math")
    monkeypatch.setattr(mod, "run_math_main", fake_run_math_main)

    argv = ["--foo", "bar"]
    mod.main(argv=argv)

    assert "kwargs" in calls
    assert calls["kwargs"]["backend_cls"] is mod.HFBackend
    assert calls["kwargs"]["argv"] == argv
