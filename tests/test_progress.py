import json
import runpy
import sys

import src.annotate.core.progress as progress


def test_iter_records_skips_blank(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"a":1}\n \n', encoding="utf-8")
    recs = list(progress._iter_records(path))
    assert recs == [{"a": 1}]


def test_has_judge_tag_and_require_shift(tmp_path):
    rec = {
        "pass2a": {"shift_rationale_gpt_model": "gpt"},
        "pass2b": {},
        "pass2c": None,
    }
    assert progress._has_judge_tag(rec) is True
    rec2 = {"pass2a": {}, "pass2b": {}, "pass2c": {}}
    assert progress._has_judge_tag(rec2) is False

    path = tmp_path / "rows.jsonl"
    rows = [
        {"pass2a": {"shift_in_reasoning_v1": True, "shift_rationale_gpt_model": "gpt"}},
        {"pass2b": {"shift_in_reasoning_v1": False}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    total, seen = progress.count_progress(path, require_shift_flag=False)
    assert total == 2 and seen == 1
    total2, seen2 = progress.count_progress(path, require_shift_flag=True)
    assert seen2 == 1  # only first row has shift flag


def test_main_and_parse_args(monkeypatch, tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"pass2a": {"shift_rationale_gpt_model": "gpt"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", str(path)])
    progress.main()
    out = capsys.readouterr().out
    assert "total rows" in out and "processed" in out


def test_module_entrypoint_executes(monkeypatch, tmp_path, capsys):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(json.dumps({"pass2a": {"shift_rationale_gpt_model": "gpt"}}) + "\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "src.annotate.core.progress", raising=False)
    monkeypatch.setattr(sys, "argv", ["progress.py", str(jsonl)])
    runpy.run_module("src.annotate.core.progress", run_name="__main__")
    out = capsys.readouterr().out
    assert "pending" in out
