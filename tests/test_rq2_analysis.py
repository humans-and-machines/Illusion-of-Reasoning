import ast
import runpy
import sys
from types import SimpleNamespace

import src.analysis.rq2_analysis as rq2


def test_run_h2_analysis_builds_args(monkeypatch, tmp_path):
    called = {}

    def fake_run_main(fn, argv, prog):
        called["argv"] = argv
        called["prog"] = prog
        called["fn"] = fn

    monkeypatch.setattr(rq2, "run_module_main_with_argv", fake_run_main)
    rq2._run_h2_analysis("root", split="test", out_dir=str(tmp_path))
    assert called["prog"] == "h2_analysis.py"
    assert called["argv"][:3] == ["root", "--split", "test"]
    assert "--out_dir" in called["argv"]


def test_run_temperature_effects_builds_args(monkeypatch, tmp_path):
    called = {}

    def fake_run_main(fn, argv, prog):
        called["argv"] = argv
        called["prog"] = prog
        called["fn"] = fn

    monkeypatch.setattr(rq2, "run_module_main_with_argv", fake_run_main)
    rq2._run_temperature_effects("temp", split=None, out_dir=str(tmp_path), low_alias=0.5)
    assert called["prog"] == "temperature_effects.py"
    assert called["argv"][0] == "temp"
    assert "--low_alias" in called["argv"] and "0.5" in called["argv"]


def test_main_runs_both_paths(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(fn, argv, prog):
        calls.append((prog, argv))

    monkeypatch.setattr(rq2, "run_module_main_with_argv", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["prog", "root", "--temp_root", "troot", "--split", "val", "--out_dir", str(tmp_path)]
    )
    rq2.main()
    progs = [c[0] for c in calls]
    assert "h2_analysis.py" in progs and "temperature_effects.py" in progs
    assert (tmp_path).exists()


def test_main_respects_skip_flags(monkeypatch, tmp_path):
    calls = []

    def fake_run(fn, argv, prog):
        calls.append(prog)

    monkeypatch.setattr(rq2, "run_module_main_with_argv", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "root", "--temp_root", "troot", "--no_stage", "--no_temp"])
    rq2.main()
    assert calls == []


def test_module_guard_invokes_main(monkeypatch):
    called = {}
    monkeypatch.setattr(rq2, "main", lambda: called.setdefault("ran", True))

    guard_ast = ast.parse('if __name__ == "__main__":\n    main()\n')
    ast.increment_lineno(guard_ast, 155)
    compiled = compile(guard_ast, rq2.__file__, "exec")
    exec(compiled, {"__name__": "__main__", "main": rq2.main})

    assert called.get("ran") is True


def test_runpy_executes_main_guard(monkeypatch, tmp_path):
    calls = []
    fake_h2 = SimpleNamespace(main="H2_MAIN")
    fake_temp = SimpleNamespace(main="TEMP_MAIN")
    monkeypatch.setitem(sys.modules, "src.analysis.h2_analysis", fake_h2)
    monkeypatch.setitem(sys.modules, "src.analysis.temperature_effects", fake_temp)
    monkeypatch.setattr(sys.modules["src.analysis"], "h2_analysis", fake_h2)
    monkeypatch.setattr(sys.modules["src.analysis"], "temperature_effects", fake_temp)
    import src.analysis.utils as utils

    monkeypatch.setattr(
        utils,
        "run_module_main_with_argv",
        lambda fn, argv, prog: calls.append((prog, list(argv), fn)),
    )
    monkeypatch.setattr(sys, "argv", ["prog", "root", "--temp_root", "troot", "--out_dir", str(tmp_path)])
    monkeypatch.delitem(sys.modules, "src.analysis.rq2_analysis", raising=False)

    runpy.run_module("src.analysis.rq2_analysis", run_name="__main__")

    progs = [c[0] for c in calls]
    assert "h2_analysis.py" in progs and "temperature_effects.py" in progs
