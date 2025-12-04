import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from src.analysis import annotate_progress
from src.annotate.core import progress as core_progress


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_reexports_core_progress_functions():
    assert annotate_progress.count_progress is core_progress.count_progress
    assert annotate_progress.main is core_progress.main
    assert annotate_progress.parse_args is core_progress.parse_args
    assert annotate_progress.__all__ == ["count_progress", "main", "parse_args"]


def test_count_progress_counts_judge_tags(tmp_path):
    jsonl = tmp_path / "sample.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"pass2a": {"shift_rationale_gpt_model": "gpt-4o"}},  # counted
            {"pass2b": {"shift_rationale_gpt_model": None}},  # not counted
        ],
    )
    total, seen = annotate_progress.count_progress(jsonl)
    assert total == 2
    assert seen == 1


def test_count_progress_require_shift_flag(tmp_path):
    jsonl = tmp_path / "sample_flags.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"pass2a": {"shift_in_reasoning_v1": True}},  # counted
            {"pass2c": {"shift_in_reasoning_v1": 0}},  # falsey
        ],
    )
    total, seen = annotate_progress.count_progress(jsonl, require_shift_flag=True)
    assert total == 2
    assert seen == 1


def test_main_prints_counts(monkeypatch, capsys, tmp_path):
    jsonl = tmp_path / "sample_main.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"pass2a": {"shift_rationale_gpt_model": "gpt"}},
            {"pass2a": {}},
        ],
    )

    def fake_parse_args():
        return SimpleNamespace(jsonl=jsonl, require_shift_flag=False)

    monkeypatch.setattr(core_progress, "parse_args", fake_parse_args)
    annotate_progress.main()
    output = capsys.readouterr().out
    assert "total rows: 2" in output
    assert "processed by GPT-4o judge: 1" in output
    assert "pending: 1" in output


def test_module_executes_main_when_run_as_script(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr(core_progress, "main", fake_main)
    # Remove cached module to force __main__ execution path
    monkeypatch.delitem(sys.modules, "src.analysis.annotate_progress", raising=False)

    runpy.run_module("src.analysis.annotate_progress", run_name="__main__")
    assert called["ran"] is True
