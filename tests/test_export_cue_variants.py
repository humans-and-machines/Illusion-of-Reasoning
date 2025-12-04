import json
import runpy
import sys

import pytest

import src.analysis.export_cue_variants as export_mod


def test_iter_flat_rows_filters_empty_passes(tmp_path, monkeypatch):
    data = [
        {"problem": "p1", "pass1": {"is_correct_pred": True}, "step": 1, "split": "train", "sample_idx": 0},
        {"problem": "p2", "pass1": {}, "pass2": {"is_correct_pred": False}, "step": 2, "sample_idx": 1},
        "not a dict",
    ]
    path = tmp_path / "file.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    rows = list(export_mod.iter_flat_rows([str(path)], ["pass1", "pass2"]))
    assert len(rows) == 2  # skips empty pass1 in second record
    assert rows[0]["variant"] == "pass1" and rows[1]["variant"] == "pass2"
    assert rows[0]["is_baseline"] == 1 and rows[1]["is_reconsider_variant"] == 1


def test_export_cue_variants_writes_and_errors(monkeypatch, tmp_path, capsys):
    # No files found should exit.
    with pytest.raises(SystemExit):
        export_mod.export_cue_variants("missing", None, str(tmp_path / "out.csv"), ["pass1"])

    data = [{"problem": "p1", "pass1": {"is_correct_pred": True}, "sample_idx": 0}]
    path = tmp_path / "f.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    monkeypatch.setattr(export_mod, "scan_jsonl_files", lambda root, split_substr=None: [str(path)])
    export_mod.export_cue_variants(str(tmp_path), None, str(tmp_path / "out.csv"), ["pass1"])
    out_text = (tmp_path / "out.csv").read_text()
    assert "problem_id" in out_text and "p1" in out_text


def test_main_default_out_path(monkeypatch, tmp_path):
    data = [{"problem": "p1", "pass1": {"is_correct_pred": True}, "sample_idx": 0}]
    path = tmp_path / "f.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")

    def fake_scan(root, split_substr=None):
        return [str(path)]

    monkeypatch.setattr(export_mod, "scan_jsonl_files", fake_scan)
    monkeypatch.setattr(export_mod, "parse_passes_argument", lambda p: ["pass1"])
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path)])
    export_mod.main()
    out_path = tmp_path / "cue_variants.csv"
    assert out_path.exists()


def test_as_json_and_main_with_explicit_out(monkeypatch, tmp_path):
    assert export_mod._as_json(None) == ""
    assert export_mod._as_json(True) == "True"
    assert export_mod._as_json({"a": [1, 2]}) == json.dumps({"a": [1, 2]}, ensure_ascii=False)

    data = [{"problem": "p1", "pass1": {"is_correct_pred": True}, "sample_idx": 0}]
    path = tmp_path / "g.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")

    monkeypatch.setattr(export_mod, "scan_jsonl_files", lambda root, split_substr=None: [str(path)])
    monkeypatch.setattr(export_mod, "parse_passes_argument", lambda p: ["pass1"])

    out_csv = tmp_path / "custom.csv"
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path), "--out_csv", str(out_csv)])
    sys.modules.pop("src.analysis.export_cue_variants", None)
    runpy.run_module("src.analysis.export_cue_variants", run_name="__main__")
    assert out_csv.exists()
