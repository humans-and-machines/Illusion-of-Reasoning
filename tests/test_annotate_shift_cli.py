#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
import types
from types import SimpleNamespace


def test_shift_cli_main_calls_components(monkeypatch):
    import src.annotate.cli.shift_cli as shift_cli

    calls = {"clean": 0, "scan": 0, "annotate": 0}

    def fake_clean_root(root):
        calls["clean"] += 1
        calls["clean_root"] = root

    def fake_scan(root, split):
        calls["scan"] += 1
        return ["f1"]

    def fake_count_progress(path):
        return 10, 5

    def fake_annotate_file(path, opts):
        calls["annotate"] += 1
        calls["annotate_opts"] = opts

    monkeypatch.setattr(shift_cli, "_clean_failed_root", fake_clean_root)
    monkeypatch.setattr(shift_cli, "scan_jsonl", fake_scan)
    monkeypatch.setattr(shift_cli, "count_progress", fake_count_progress)
    monkeypatch.setattr(shift_cli, "annotate_file", fake_annotate_file)

    # Provide parsed args directly
    args = SimpleNamespace(
        results_root="root",
        split=None,
        seed=1,
        max_calls=None,
        dry_run=False,
        jitter=0.0,
        loglevel="INFO",
        force_relabel=False,
        clean_failed_first=True,
        passes="pass1,pass2",
        backend="azure",
        endpoint="http://e",
        api_version="v1",
        use_v1=0,
        deployment="dep",
    )
    monkeypatch.setattr(shift_cli, "build_argparser", lambda: shift_cli.argparse.ArgumentParser())
    monkeypatch.setattr(shift_cli.argparse.ArgumentParser, "parse_args", lambda self: args)

    shift_cli.main()

    assert calls["clean"] == 1
    assert calls["scan"] == 1
    assert calls["annotate"] == 1
    opts = calls["annotate_opts"]
    assert opts.passes == ["pass1", "pass2"]
    assert opts.client_cfg["deployment"] == "dep"


def test_shift_cli_main_no_files_prints_message(monkeypatch, capsys):
    import src.annotate.cli.shift_cli as shift_cli

    args = SimpleNamespace(
        results_root="root",
        split="sp",
        seed=1,
        max_calls=None,
        dry_run=False,
        jitter=0.1,
        loglevel="INFO",
        force_relabel=False,
        clean_failed_first=False,
        passes="pass1",
        backend="azure",
        endpoint="http://e",
        api_version="v1",
        use_v1=0,
        deployment="dep",
    )

    monkeypatch.setattr(shift_cli, "build_argparser", lambda: shift_cli.argparse.ArgumentParser())
    monkeypatch.setattr(shift_cli.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(shift_cli, "scan_jsonl", lambda *_: [])

    called = {}
    monkeypatch.setattr(shift_cli, "annotate_file", lambda *a, **k: called.setdefault("annotated", True))
    monkeypatch.setattr(shift_cli, "_clean_failed_root", lambda *_: called.setdefault("cleaned", True))

    shift_cli.main()

    out = capsys.readouterr().out
    assert "No JSONL files found" in out
    assert "annotated" not in called
    assert "cleaned" not in called


def test_shift_cli_main_guard_runpy(monkeypatch, capsys):
    stub_clean_core = types.ModuleType("src.annotate.core.clean_core")
    stub_clean_core.clean_root = lambda *_: None

    stub_shift_core = types.ModuleType("src.annotate.core.shift_core")
    stub_shift_core.AnnotateOpts = lambda **kwargs: SimpleNamespace(**kwargs)
    stub_shift_core.DEFAULT_API_VERSION = "api"
    stub_shift_core.DEFAULT_DEPLOYMENT = "dep"
    stub_shift_core.DEFAULT_ENDPOINT = "ep"
    stub_shift_core.DEFAULT_USE_V1 = 1
    stub_shift_core.annotate_file = lambda *_, **__: None
    stub_shift_core.scan_jsonl = lambda *_a, **_k: []

    stub_progress = types.ModuleType("src.annotate.core.progress")
    stub_progress.count_progress = lambda path: (0, 0)

    monkeypatch.setitem(sys.modules, "src.annotate.core.clean_core", stub_clean_core)
    monkeypatch.setitem(sys.modules, "src.annotate.core.shift_core", stub_shift_core)
    monkeypatch.setitem(sys.modules, "src.annotate.core.progress", stub_progress)

    monkeypatch.delitem(sys.modules, "src.annotate.cli.shift_cli", raising=False)
    monkeypatch.setattr(sys, "argv", ["shift_cli.py", "rootdir"])

    runpy.run_module("src.annotate.cli.shift_cli", run_name="__main__", alter_sys=True)

    out = capsys.readouterr().out
    assert "No JSONL files found" in out
