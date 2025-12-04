import gzip
import io
import types
from pathlib import Path

import pytest

import src.analysis.io as aio


def test_scan_files_step_only_filters_extensions(tmp_path):
    skip_dir = tmp_path / "skipdir"
    skip_dir.mkdir()
    (skip_dir / "note.txt").write_text("x")
    root = tmp_path / "keep"
    root.mkdir()
    (root / "step1.jsonl").write_text("{}\n")
    (root / "ignore.txt").write_text("nope")
    out = aio.scan_files_step_only(str(tmp_path), split_substr=None, skip_substrings=["skipdir"])
    assert any("step1.jsonl" in p for p in out)
    assert all("ignore.txt" not in p for p in out)


def test_scan_files_step_only_requires_step_and_split(tmp_path):
    nostep = tmp_path / "nostep"
    nostep.mkdir()
    (nostep / "train.jsonl").write_text("{}")
    step_dir = tmp_path / "step-0003"
    step_dir.mkdir()
    (step_dir / "keep_train.jsonl").write_text("{}")
    (step_dir / "drop.jsonl").write_text("{}")
    step_file = tmp_path / "global_step-2_train.jsonl"
    step_file.write_text("{}")

    out = aio.scan_files_step_only(str(tmp_path), split_substr="train", skip_substrings=None)
    names = [Path(p).name for p in out]
    assert "keep_train.jsonl" in names
    assert "global_step-2_train.jsonl" in names
    assert "drop.jsonl" not in names
    assert all("nostep" not in p for p in out)


def test_build_jsonl_files_by_domain_sets_first_root(monkeypatch, tmp_path):
    monkeypatch.setattr(aio, "scan_jsonl_files", lambda root, split_substr=None: [str(tmp_path / "file.jsonl")])
    files_by_domain, first = aio.build_jsonl_files_by_domain({"Math": str(tmp_path)}, split_substr=None)
    assert first == str(tmp_path)
    assert "Math" in files_by_domain


def test_build_files_by_domain_skips_missing(monkeypatch):
    monkeypatch.setattr(
        aio,
        "scan_files_step_only",
        lambda root, split_substr=None, skip_substrings=None: ["a.jsonl"] if "good" in root else [],
    )
    domain_map = {"Crossword": "good_root", "Math": "empty_root", "Carpark": None}
    files = aio.build_files_by_domain(
        domain_map, ["Crossword", "Math", "Carpark"], split_substr=None, skip_substrings=[]
    )
    assert list(files.keys()) == ["Crossword"]


def test_build_standard_domain_roots():
    roots = aio.build_standard_domain_roots("c", "m", None, "p")
    assert roots["Crossword"] == "c"
    assert roots["Math2"] is None


def test_build_jsonl_files_by_domain_skips_empty(monkeypatch, tmp_path):
    file_path = tmp_path / "file.jsonl"
    file_path.write_text("{}")

    def fake_scan(root, split_substr=None):
        return [str(file_path)] if root == "nonempty" else []

    monkeypatch.setattr(aio, "scan_jsonl_files", fake_scan)
    files_by_domain, first = aio.build_jsonl_files_by_domain(
        {"Math": "empty", "Crossword": "nonempty", "Carpark": None}, split_substr=None
    )
    assert "Crossword" in files_by_domain and "Math" not in files_by_domain
    assert first == "nonempty"


def test_collect_jsonl_files_for_domains_first_root_missing(monkeypatch):
    # Force files but missing first_root to trigger the guard.
    monkeypatch.setattr(aio, "build_jsonl_files_by_domain", lambda roots, split: ({"X": ["f"]}, None))
    with pytest.raises(SystemExit):
        aio.collect_jsonl_files_for_domains({"X": "r"}, split_substr=None, results_root=None)


def test_collect_jsonl_files_for_domains_fallback_results_root(monkeypatch):
    monkeypatch.setattr(aio, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    monkeypatch.setattr(aio, "scan_jsonl_files", lambda root, split_substr=None: ["f1"])
    files_by_domain, first_root = aio.collect_jsonl_files_for_domains(
        {}, split_substr="train", results_root="fallback"
    )
    assert files_by_domain == {"All": ["f1"]}
    assert first_root == "fallback"


def test_collect_jsonl_files_for_domains_raises_when_empty(monkeypatch):
    monkeypatch.setattr(aio, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    monkeypatch.setattr(aio, "scan_jsonl_files", lambda root, split_substr=None: [])
    with pytest.raises(SystemExit):
        aio.collect_jsonl_files_for_domains({}, split_substr=None, results_root="fallback")


def test_collect_jsonl_files_for_domains_requires_any_root(monkeypatch):
    monkeypatch.setattr(aio, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    with pytest.raises(SystemExit, match="Provide --root_\\* folders"):
        aio.collect_jsonl_files_for_domains({}, split_substr=None, results_root=None)


def test_build_files_by_domain_for_args(monkeypatch):
    captured = {}

    def fake_collect(domain_roots, split_substr=None, results_root=None):
        captured["domain_roots"] = domain_roots
        captured["split_substr"] = split_substr
        captured["results_root"] = results_root
        return {"Math": ["f"]}, "root"

    monkeypatch.setattr(aio, "collect_jsonl_files_for_domains", fake_collect)
    args = types.SimpleNamespace(
        root_crossword="c",
        root_math="m",
        root_math2=None,
        root_carpark="p",
        split="train",
        results_root="fallback",
    )
    files, first = aio.build_files_by_domain_for_args(args)
    assert files == {"Math": ["f"]} and first == "root"
    assert captured["domain_roots"]["Crossword"] == "c"


def test_scan_files_with_steps_or_meta_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(aio, "scan_files_step_only", lambda *a, **k: [])
    monkeypatch.setattr(aio, "nat_step_from_path", lambda path: 1)
    keep_dir = tmp_path / "data"
    keep_dir.mkdir()
    (keep_dir / "keep.json").write_text("{}")
    (keep_dir / "skip.json").write_text("{}")
    out = aio.scan_files_with_steps_or_meta(str(tmp_path), split_substr="keep", skip_substrings=None)
    assert len(out) == 1 and "keep.json" in out[0]


def test_scan_files_with_steps_or_meta_returns_precomputed(monkeypatch):
    monkeypatch.setattr(aio, "scan_files_step_only", lambda *a, **k: ["pre"])
    out = aio.scan_files_with_steps_or_meta("root")
    assert out == ["pre"]


def test_scan_files_with_steps_or_meta_skips_dirs_and_extensions(monkeypatch, tmp_path):
    monkeypatch.setattr(aio, "scan_files_step_only", lambda *a, **k: [])
    monkeypatch.setattr(aio, "nat_step_from_path", lambda path: 1 if "good" in path else None)
    skip_dir = tmp_path / "hf_cache"
    skip_dir.mkdir()
    (skip_dir / "good_skip.json").write_text("{}")
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    (ok_dir / "ignore.txt").write_text("x")
    keep_file = ok_dir / "good_keep.json"
    keep_file.write_text("{}")

    out = aio.scan_files_with_steps_or_meta(str(tmp_path), split_substr="keep", skip_substrings=None)
    assert str(keep_file) in out
    assert all("hf_cache" not in p for p in out)
    assert all("ignore.txt" not in p for p in out)


def test_iter_json_from_text_mixed_lines():
    text = 'not-json\n\n{"a":1}\n{"b":2}\n'
    records = list(aio._iter_json_from_text(text))
    assert len(records) == 2 and records[0]["a"] == 1


def test_iter_json_from_text_empty_string():
    assert list(aio._iter_json_from_text("  \n")) == []


def test_iter_json_from_text_list_and_dict():
    list_records = list(aio._iter_json_from_text('[{"a": 1}, {"b": 2}]'))
    assert [r["a"] if "a" in r else r["b"] for r in list_records] == [1, 2]
    dict_record = list(aio._iter_json_from_text('{"c": 3}'))
    assert dict_record == [{"c": 3}]


def test_iter_json_lines_skips_invalid():
    buffer = io.StringIO('bad\n{"x":1}\n\n')
    out = list(aio._iter_json_lines(buffer))
    assert out == [{"x": 1}]


def test_iter_records_from_file_jsonl_gz(tmp_path):
    gz_path = tmp_path / "data.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        handle.write('{"a":1}\n{"b":2}\n')
    out = list(aio.iter_records_from_file(gz_path))
    assert len(out) == 2 and out[0]["a"] == 1


def test_iter_records_from_file_json(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text('{"a": 1}', encoding="utf-8")
    out = list(aio.iter_records_from_file(json_path))
    assert out == [{"a": 1}]


def test_iter_records_from_file_handles_oserror(tmp_path):
    missing = tmp_path / "missing.jsonl"
    assert list(aio.iter_records_from_file(missing)) == []


def test_iter_pass1_samples_by_domain(monkeypatch):
    records = [{"id": 1}, {"id": 2}, {"id": 3}]

    def fake_iter_records(path):
        return iter(records)

    monkeypatch.setattr(aio, "iter_records_from_file", fake_iter_records)
    monkeypatch.setattr(aio, "nat_step_from_path", lambda path: 5)

    def fake_extract(rec, step_from_name):
        if rec["id"] == 1:
            return None, None
        return {"correct": True}, step_from_name + rec["id"] - 2

    monkeypatch.setattr(aio, "extract_pass1_and_step", fake_extract)
    monkeypatch.setattr(
        aio,
        "step_within_bounds",
        lambda step, min_step, max_step: (min_step is None or step >= min_step)
        and (max_step is None or step <= max_step),
    )

    files_by_domain = {"Math": ["p1"]}
    out = list(aio.iter_pass1_samples_by_domain(files_by_domain, min_step=5, max_step=6))
    assert len(out) == 2
    assert all(item[0] == "Math" for item in out)
    assert all(item[1] == {"correct": True} for item in out)


def test_iter_pass1_samples_by_domain_respects_bounds(monkeypatch):
    records = [{"id": 1}]
    monkeypatch.setattr(aio, "iter_records_from_file", lambda path: iter(records))
    monkeypatch.setattr(aio, "nat_step_from_path", lambda path: 10)
    monkeypatch.setattr(
        aio,
        "extract_pass1_and_step",
        lambda rec, step_from_name: ({"p": 1}, step_from_name),
    )
    # step_within_bounds returns False to trigger skip at line 393
    monkeypatch.setattr(aio, "step_within_bounds", lambda step, min_step, max_step: False)
    out = list(aio.iter_pass1_samples_by_domain({"X": ["p"]}, min_step=0, max_step=5))
    assert out == []
