import src.inference.cli.unified_carpark as cli


def test_main_delegates_with_argv(monkeypatch):
    captured = {}

    def fake_run_carpark_main(**kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run_carpark_main", fake_run_carpark_main)
    monkeypatch.setattr(cli, "load_carpark_module", lambda: "module")
    cli.main(argv=["--foo", "bar"])
    assert captured["kwargs"]["backend_cls"] is cli.HFBackend
    assert captured["kwargs"]["load_module"]() == "module"
    assert captured["kwargs"]["argv"] == ["--foo", "bar"]
