import importlib
import sys
import types


def _install_stubs(monkeypatch):
    """Install lightweight stub modules so the backcompat wrapper imports cleanly."""
    from pathlib import Path

    # Base annotate package with public API list
    annotate_pkg = types.ModuleType("src.annotate")
    annotate_pkg.__path__ = [str(Path("src/annotate").resolve())]  # mark as package
    annotate_pkg._ANNOTATION_PUBLIC_API = [
        "AnnotateOpts",
        "annotate_file",
        "scan_jsonl",
        "llm_judge_shift",
    ]

    # Core shift_core stub with expected symbols
    shift_core_mod = types.ModuleType("src.annotate.core.shift_core")
    shift_core_mod.AnnotateOpts = object
    shift_core_mod.annotate_file = lambda *a, **k: "annotated"
    shift_core_mod.llm_judge_shift = lambda *a, **k: "judged"
    shift_core_mod.scan_jsonl = lambda *a, **k: ["scan"]
    shift_core_mod.nat_step_from_path = lambda p: f"nat:{p}"
    shift_core_mod.record_id_for_logs = lambda rec: f"id:{rec}"
    shift_core_mod._annotate_record_for_pass = lambda *a, **k: {"pass": 1}
    shift_core_mod._json_from_text = lambda txt: {"txt": txt}
    shift_core_mod._sanitize_jsonish = lambda txt: txt.strip()

    # Core and CLI packages
    core_pkg = types.ModuleType("src.annotate.core")
    core_pkg.__path__ = []
    cli_pkg = types.ModuleType("src.annotate.cli")
    cli_pkg.__path__ = []
    backcompat_pkg = types.ModuleType("src.annotate.backcompat")
    backcompat_pkg.__path__ = [str(Path("src/annotate/backcompat").resolve())]

    # CLI stub
    shift_cli_mod = types.ModuleType("src.annotate.cli.shift_cli")
    shift_cli_mod.called = 0
    shift_cli_mod.build_argparser = lambda: "parser"

    def _main():
        shift_cli_mod.called += 1
        return "cli-main"

    shift_cli_mod.main = _main

    # Register modules
    monkeypatch.setitem(sys.modules, "src.annotate", annotate_pkg)
    monkeypatch.setitem(sys.modules, "src.annotate.core", core_pkg)
    monkeypatch.setitem(sys.modules, "src.annotate.cli", cli_pkg)
    monkeypatch.setitem(sys.modules, "src.annotate.backcompat", backcompat_pkg)
    # Stubs for other backcompat modules imported by __init__
    for name in [
        "clean_failed_shift_core",
        "clean_failed_shift_labels",
        "config",
        "llm_client",
        "prompts",
        "shift_cli",
    ]:
        stub = types.ModuleType(f"src.annotate.backcompat.{name}")
        monkeypatch.setitem(sys.modules, f"src.annotate.backcompat.{name}", stub)
    monkeypatch.setitem(sys.modules, "src.annotate.core.shift_core", shift_core_mod)
    monkeypatch.setitem(sys.modules, "src.annotate.cli.shift_cli", shift_cli_mod)

    # Remove previous import to force reload with stubs
    sys.modules.pop("src.annotate.backcompat.tasks.shifts", None)

    return shift_core_mod, shift_cli_mod


def test_reexports_point_to_core(monkeypatch):
    shift_core_mod, shift_cli_mod = _install_stubs(monkeypatch)
    shifts = importlib.reload(importlib.import_module("src.annotate.backcompat.tasks.shifts"))

    assert shifts.annotate_file is shift_core_mod.annotate_file
    assert shifts.llm_judge_shift() == "judged"
    assert shifts.nat_step_from_path("p") == "nat:p"
    assert shifts._sanitize_jsonish(" x ") == "x"
    assert shifts.build_argparser() == "parser"
    shifts.main()
    assert shift_cli_mod.called == 1


def test___all___includes_public_api(monkeypatch):
    _install_stubs(monkeypatch)
    shifts = importlib.reload(importlib.import_module("src.annotate.backcompat.tasks.shifts"))

    for name in ["AnnotateOpts", "annotate_file", "scan_jsonl", "llm_judge_shift", "build_argparser", "main"]:
        assert name in shifts.__all__
