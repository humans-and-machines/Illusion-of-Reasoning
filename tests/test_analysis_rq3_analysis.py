import ast
import runpy
import sys
from types import SimpleNamespace

import src.analysis.rq3_analysis as rq3


def test_run_h3_analysis_builds_argv(monkeypatch, tmp_path):
    called = {}

    monkeypatch.setattr(rq3, "build_results_root_argv", lambda rr, split: ["BASE", rr, split])

    def fake_run(main_fn, argv, prog):
        called["main_fn"] = main_fn
        called["argv"] = argv
        called["prog"] = prog

    fake_h3 = SimpleNamespace(main="MAINFN")
    monkeypatch.setattr(rq3, "h3_analysis", fake_h3)
    monkeypatch.setattr(rq3, "run_module_main_with_argv", fake_run)

    out_dir = tmp_path / "out"
    rq3._run_h3_analysis(str(tmp_path / "root"), "test", str(out_dir), "entropy_answer", 5)

    assert (out_dir / "h3_analysis").exists()
    assert called["main_fn"] == "MAINFN"
    assert called["prog"] == "h3_analysis.py"
    assert called["argv"][:3] == ["BASE", str(tmp_path / "root"), "test"]
    assert "--uncertainty_field" in called["argv"]
    assert "entropy_answer" in called["argv"]
    assert str(5) in called["argv"]


def test_export_cue_variants_invokes_helper(monkeypatch, tmp_path):
    recorded = {}

    def fake_export(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(rq3.export_cue_variants, "export_cue_variants", fake_export)
    rq3._export_cue_variants(str(tmp_path / "root"), "split", str(tmp_path / "out"), ["p1", "p2"])

    assert recorded["results_root"] == str(tmp_path / "root")
    assert recorded["split_substr"] == "split"
    assert recorded["passes"] == ["p1", "p2"]
    assert recorded["out_csv"].endswith("_split.csv")


def test_main_runs_h3_and_cue_export(monkeypatch, tmp_path):
    calls = {"h3": 0, "cues": 0}
    monkeypatch.setattr(rq3, "_run_h3_analysis", lambda **kwargs: calls.__setitem__("h3", calls["h3"] + 1))
    monkeypatch.setattr(rq3, "_export_cue_variants", lambda **kwargs: calls.__setitem__("cues", calls["cues"] + 1))
    monkeypatch.setattr(rq3, "parse_passes_argument", lambda s: ["passX"])

    out_dir = tmp_path / "out"
    argv = [
        "prog",
        str(tmp_path / "root"),
        "--out_dir",
        str(out_dir),
        "--export_cues",
        "--passes",
        "pass1,pass2",
        "--num_buckets",
        "6",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rq3.main()

    assert calls["h3"] == 1
    assert calls["cues"] == 1
    assert out_dir.exists()


def test_module_guard_invokes_main(monkeypatch):
    called = {}
    monkeypatch.setattr(rq3, "main", lambda: called.setdefault("ran", True))

    guard_ast = ast.parse('if __name__ == "__main__":\n    main()\n')
    ast.increment_lineno(guard_ast, 178)
    compiled = compile(guard_ast, rq3.__file__, "exec")
    exec(compiled, {"__name__": "__main__", "main": rq3.main})

    assert called.get("ran") is True


def test_runpy_executes_main_guard(monkeypatch, tmp_path):
    calls = []
    fake_h3 = SimpleNamespace(main="H3_MAIN")
    fake_export = SimpleNamespace(export_cue_variants=lambda **kwargs: calls.append(("export", kwargs)))
    monkeypatch.setitem(sys.modules, "src.analysis.h3_analysis", fake_h3)
    monkeypatch.setitem(sys.modules, "src.analysis.export_cue_variants", fake_export)
    monkeypatch.setattr(sys.modules["src.analysis"], "h3_analysis", fake_h3)
    monkeypatch.setattr(sys.modules["src.analysis"], "export_cue_variants", fake_export)

    import src.analysis.utils as utils

    monkeypatch.setattr(utils, "build_results_root_argv", lambda rr, split: ["ROOT", rr, split])
    monkeypatch.setattr(utils, "parse_passes_argument", lambda passes: ["pX"])
    monkeypatch.setattr(
        utils,
        "run_module_main_with_argv",
        lambda fn, argv, prog: calls.append((prog, list(argv), fn)),
    )
    monkeypatch.setattr(sys, "argv", ["prog", "root", "--out_dir", str(tmp_path), "--export_cues"])
    monkeypatch.delitem(sys.modules, "src.analysis.rq3_analysis", raising=False)

    runpy.run_module("src.analysis.rq3_analysis", run_name="__main__")

    progs = [c[0] for c in calls if isinstance(c, tuple)]
    assert "h3_analysis.py" in progs
    assert any(c[0] == "export" for c in calls)
