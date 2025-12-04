import os
import runpy
import sys
from types import ModuleType, SimpleNamespace

import src.analysis.rq1_analysis as rq1


def test_run_module_main_with_argv_restores(monkeypatch):
    called = {}

    def fake_main():
        called["argv"] = list(sys.argv)

    old = list(sys.argv)
    rq1._run_module_main_with_argv(fake_main, ["--foo", "bar"])
    assert called["argv"][0] == "h1_analysis.py"
    assert called["argv"][1:] == ["--foo", "bar"]
    assert sys.argv == old


def test_run_h1_glm_builds_args(monkeypatch, tmp_path):
    monkeypatch.setattr(rq1, "build_results_root_argv", lambda root, split: ["--root", root, "--split", split])
    captured = {}
    monkeypatch.setattr(rq1, "_run_module_main_with_argv", lambda main, argv: captured.update({"argv": argv}))
    rq1._run_h1_glm("root_dir", "split_val", str(tmp_path), "DS", "MODEL")
    assert (tmp_path / "h1_glm").exists()
    assert captured["argv"] == [
        "--root",
        "root_dir",
        "--split",
        "split_val",
        "--out_dir",
        os.path.join(str(tmp_path), "h1_glm"),
        "--dataset_name",
        "DS",
        "--model_name",
        "MODEL",
    ]


def test_run_shift_summary_calls(monkeypatch):
    called = {}

    def fake_summarize(root, split=None, max_examples=None):
        called["root"] = root
        called["split"] = split
        called["max"] = max_examples

    monkeypatch.setattr(rq1.shift_summary, "summarize_root", fake_summarize)
    rq1._run_shift_summary("res_root", split="sp", max_examples=7)
    assert called == {"root": os.path.abspath("res_root"), "split": "sp", "max": 7}


def test_main_runs_steps(monkeypatch, tmp_path):
    args = SimpleNamespace(
        results_root="root",
        split=None,
        out_dir=str(tmp_path),
        dataset_name="ds",
        model_name="model",
        no_h1_glm=False,
        no_shift_summary=False,
        shift_max_examples=0,
    )

    class FakeParser:
        def parse_args(self):
            return args

    called = {}
    monkeypatch.setattr(rq1, "build_argparser", lambda: FakeParser())
    monkeypatch.setattr(rq1, "_run_h1_glm", lambda **kwargs: called.setdefault("glm", kwargs))
    monkeypatch.setattr(rq1, "_run_shift_summary", lambda **kwargs: called.setdefault("shift", kwargs))
    rq1.main()
    assert "glm" in called and "shift" in called


def test_build_argparser_defaults():
    parser = rq1.build_argparser()
    args = parser.parse_args(["/results/root"])
    assert args.results_root == "/results/root"
    assert args.split is None
    assert args.out_dir is None
    assert args.dataset_name == "MATH-500"
    assert args.model_name == "Qwen2.5-1.5B"
    assert args.no_h1_glm is False
    assert args.no_shift_summary is False
    assert args.shift_max_examples == 0


def test_build_argparser_with_flags():
    parser = rq1.build_argparser()
    args = parser.parse_args(
        [
            "/root",
            "--split",
            "train",
            "--out_dir",
            "/tmp/out",
            "--dataset_name",
            "DS2",
            "--model_name",
            "MODEL2",
            "--no_h1_glm",
            "--no_shift_summary",
            "--shift_max_examples",
            "5",
        ],
    )
    assert args.results_root == "/root"
    assert args.split == "train"
    assert args.out_dir == "/tmp/out"
    assert args.dataset_name == "DS2"
    assert args.model_name == "MODEL2"
    assert args.no_h1_glm is True
    assert args.no_shift_summary is True
    assert args.shift_max_examples == 5


def test_main_guard_run_module(monkeypatch, tmp_path):
    # Provide lightweight stubs so the module can run through its guard safely.
    calls = {}
    fake_h1 = ModuleType("fake_h1_analysis")
    fake_h1.main = lambda: calls.setdefault("glm", True)
    fake_shift = ModuleType("fake_shift_summary")
    fake_shift.summarize_root = lambda root, split=None, max_examples=None: calls.setdefault(
        "shift",
        {"root": root, "split": split, "max": max_examples},
    )

    monkeypatch.setitem(sys.modules, "src.analysis.h1_analysis", fake_h1)
    monkeypatch.setitem(sys.modules, "src.analysis.shift_summary", fake_shift)
    # Ensure the package attributes also point at our stubs.
    monkeypatch.setattr(sys.modules["src.analysis"], "h1_analysis", fake_h1, raising=False)
    monkeypatch.setattr(sys.modules["src.analysis"], "shift_summary", fake_shift, raising=False)
    monkeypatch.delitem(sys.modules, "src.analysis.rq1_analysis", raising=False)

    results_root = tmp_path / "res_root"
    results_root.mkdir()
    out_dir = tmp_path / "out_dir"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(results_root),
            "--split",
            "dev",
            "--out_dir",
            str(out_dir),
            "--shift_max_examples",
            "3",
        ],
    )

    runpy.run_module("src.analysis.rq1_analysis", run_name="__main__")

    assert calls["glm"] is True
    assert calls["shift"] == {
        "root": os.path.abspath(str(results_root)),
        "split": "dev",
        "max": 3,
    }
    assert out_dir.is_dir()


def test_main_respects_skip_flags(monkeypatch, tmp_path):
    # _run_h1_glm and _run_shift_summary should be skipped when flags are set.
    args = SimpleNamespace(
        results_root=str(tmp_path),
        split=None,
        out_dir=None,
        dataset_name="ds",
        model_name="m",
        no_h1_glm=True,
        no_shift_summary=True,
        shift_max_examples=0,
    )

    class FakeParser:
        def parse_args(self):
            return args

    called = {}
    monkeypatch.setattr(rq1, "build_argparser", lambda: FakeParser())
    monkeypatch.setattr(rq1, "_run_h1_glm", lambda **_: called.setdefault("glm", True))
    monkeypatch.setattr(rq1, "_run_shift_summary", lambda **_: called.setdefault("shift", True))

    rq1.main()
    # out_dir default should be created under results_root/rq1
    assert (tmp_path / "rq1").is_dir()
    assert called == {}
