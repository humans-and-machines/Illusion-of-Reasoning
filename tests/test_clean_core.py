import json

import pytest

import src.annotate.core.clean_core as clean


def test_should_clear_prefix_and_nonstring():
    assert clean._should_clear({"shift_rationale_gpt": clean.BAD_RATIONALE_PREFIXES[0] + " extra"})
    assert not clean._should_clear({"shift_rationale_gpt": "all good"})
    assert not clean._should_clear({"shift_rationale_gpt": 123})


def test_clean_file_removes_shift_fields(tmp_path):
    path = tmp_path / "data.jsonl"
    recs = [
        {"pass1": {"shift_rationale_gpt": clean.BAD_RATIONALE_PREFIXES[0], "shift_in_reasoning_v1": True, "other": 1}},
        {"pass1": {"shift_rationale_gpt": "ok", "shift_in_reasoning_v1": True}},
        "not a dict",
        {},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    total, cleared = clean.clean_file(str(path))
    assert total == 3  # only dict records counted (skips invalid JSON/non-dicts)
    assert cleared == 1
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first_pass1 = lines[0]["pass1"]
    for field in clean.SHIFT_FIELDS:
        assert field not in first_pass1
    assert first_pass1["other"] == 1
    # Second record should remain unchanged
    assert "shift_in_reasoning_v1" in lines[1]["pass1"]


def test_clean_root_walks_dir(monkeypatch, tmp_path, capsys):
    bad_file = tmp_path / "a.jsonl"
    bad_file.write_text(
        json.dumps({"pass1": {"shift_rationale_gpt": clean.BAD_RATIONALE_PREFIXES[0]}}) + "\n", encoding="utf-8"
    )
    other_dir = tmp_path / "sub"
    other_dir.mkdir()
    (other_dir / "skip.txt").write_text("x", encoding="utf-8")

    files, records, cleared = clean.clean_root(str(tmp_path))
    assert files == 1 and records == 1 and cleared == 1
    out = capsys.readouterr().out
    assert "clean_shift_fallbacks" in out

    with pytest.raises(SystemExit):
        clean.clean_root(str(tmp_path / "missing_dir"))
