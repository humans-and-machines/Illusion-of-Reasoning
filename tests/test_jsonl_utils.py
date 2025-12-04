import json

import pytest

import src.common.jsonl_utils as ju


def test_iter_jsonl_lines_strict_and_lenient(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"a":1}\nnotjson\n{"b":2}\n', encoding="utf-8")
    with path.open("r", encoding="utf-8") as handle:
        with pytest.raises(json.JSONDecodeError):
            list(ju.iter_jsonl_lines(handle, strict=True))

    with path.open("r", encoding="utf-8") as handle:
        lenient = list(ju.iter_jsonl_lines(handle, strict=False))
    assert lenient == [{"a": 1}, {"b": 2}]


def test_scan_jsonl_files_filters_split(tmp_path):
    (tmp_path / "a_train.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "b_test.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("{}", encoding="utf-8")
    paths = ju.scan_jsonl_files(str(tmp_path), split_substr="test")
    assert len(paths) == 1 and paths[0].endswith("b_test.jsonl")


def test_iter_jsonl_lines_skips_blank_lines():
    lines = ["   ", '{"ok":1}\n', "\n", '  {"bad": }']
    parsed = list(ju.iter_jsonl_lines(lines))
    assert parsed == [{"ok": 1}]
