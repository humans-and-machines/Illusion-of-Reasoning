import json
import runpy
import sys
from pathlib import Path

import src.analysis.shift_summary as ss


def test_compute_shift_counts_and_joint_table(capsys, monkeypatch, tmp_path):
    # Build two files with simple records.
    paths = []
    for idx in range(2):
        file_path = tmp_path / f"file{idx}.jsonl"
        rows = [
            {"pass1": {"shift_in_reasoning_v1": 1, "is_correct_pred": True}},
            {"pass1": {"change_way_of_thinking": True, "is_correct_pred": False}},
            {"pass1": {"is_correct_pred": True}},
        ]
        file_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        paths.append(str(file_path))

    monkeypatch.setattr(
        ss,
        "iter_records_from_file",
        lambda path: json.loads(Path(path).read_text()).splitlines()
        if False
        else [json.loads(line) for line in Path(path).read_text().splitlines()],
    )
    total_examples, total_shifts, total_correct, joint_counts, shift_examples = ss._compute_shift_counts(paths)
    assert total_examples == 6  # 3 per file * 2
    assert total_shifts == 4  # two shift flags per file
    assert total_correct == 4  # two correct per file
    assert joint_counts[(1, 1)] >= 2
    assert len(shift_examples) == 4

    ss._print_joint_table(joint_counts)
    out = capsys.readouterr().out
    assert "Correctness × shift table" in out


def test_compute_shift_counts_skips_invalid_and_none_correct(monkeypatch):
    monkeypatch.setattr(
        ss,
        "iter_records_from_file",
        lambda _path: iter(
            [
                "bad_record",
                {"pass1": {"flag": 1, "is_correct_pred": None}},
                {"pass1": {"flag": 0, "is_correct_pred": True}},
            ]
        ),
    )
    monkeypatch.setattr(ss, "aha_gpt_canonical", lambda pass1, _rec: 1 if pass1.get("flag") else 0)
    monkeypatch.setattr(ss, "extract_correct", lambda pass1, _rec: pass1.get("is_correct_pred"))

    total_examples, total_shifts, total_correct, joint_counts, shift_examples = ss._compute_shift_counts(["f.jsonl"])

    assert total_examples == 2  # skips non-dict
    assert total_shifts == 1  # only one shift flag
    assert total_correct == 1  # ignores None correct
    assert joint_counts == {(1, 0): 1}
    assert len(shift_examples) == 1


def test_format_and_build_example_metadata():
    long_text = "A" * 500
    assert ss._format_text_snippet(long_text, 10).endswith("...[truncated]...")
    record = {
        "problem": "P",
        "step": 1,
        "sample_idx": 0,
        "endpoint": "ep",
        "deployment": "dep",
        "pass1": {"output": "O"},
        "shift_in_reasoning_v1": True,
    }
    meta = ss._build_example_metadata(record, record["pass1"])
    assert meta["problem_snippet"] == "P"
    assert meta["reasoning_snippet"] == "O"
    assert meta["shift_flag"] is None


def test_print_shift_examples_includes_reasoning_snippet(capsys):
    examples = [
        (
            "p",
            {
                "pass1": {"shift_in_reasoning_v1": True, "output": "reasoning text"},
                "problem": "prob",
                "step": 2,
                "sample_idx": 1,
            },
        )
    ]
    ss._print_shift_examples(examples, max_examples=2)
    out = capsys.readouterr().out
    assert "Pass-1 output (reasoning excerpt):" in out
    assert "reasoning text" in out


def test_print_shift_examples_truncation(capsys):
    examples = [("p", {"pass1": {"shift_in_reasoning_v1": True}, "problem": f"prob{i}"}) for i in range(3)]
    ss._print_shift_examples(examples, max_examples=1)
    out = capsys.readouterr().out
    assert "Shift examples" in out
    assert "truncated" in out


def test_summarize_root_handles_missing_and_examples(monkeypatch, tmp_path, capsys):
    # No files found path
    monkeypatch.setattr(ss, "scan_jsonl_files", lambda root, split_substr=None: [])
    ss.summarize_root("root", split=None, max_examples=None)
    out = capsys.readouterr().out
    assert "No JSONL files" in out

    # With files and examples printed
    file_path = tmp_path / "a.jsonl"
    file_path.write_text(
        json.dumps(
            {
                "pass1": {"shift_in_reasoning_v1": True, "is_correct_pred": True},
                "problem": "p",
                "step": 1,
                "sample_idx": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ss, "scan_jsonl_files", lambda root, split_substr=None: [str(file_path)])
    monkeypatch.setattr(ss, "iter_records_from_file", lambda path: [json.loads(Path(path).read_text())])
    ss.summarize_root("root", split="test", max_examples=1)
    out = capsys.readouterr().out
    assert "shift_summary" in out
    assert "Shift examples" in out


def test_summarize_root_reports_no_shift_examples(monkeypatch, capsys):
    monkeypatch.setattr(ss, "scan_jsonl_files", lambda root, split_substr=None: ["p"])
    monkeypatch.setattr(ss, "_compute_shift_counts", lambda *_a, **_k: (1, 0, 1, {(1, 0): 1}, []))

    ss.summarize_root("root", split="s", max_examples=5)

    out = capsys.readouterr().out
    assert "No shift examples" in out


def test_main_uses_argparser(monkeypatch, tmp_path):
    file_path = tmp_path / "a.jsonl"
    file_path.write_text(
        json.dumps({"pass1": {"shift_in_reasoning_v1": True, "is_correct_pred": True}}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path)])
    monkeypatch.setattr(ss, "scan_jsonl_files", lambda root, split_substr=None: [str(file_path)])
    monkeypatch.setattr(ss, "iter_records_from_file", lambda path: [json.loads(Path(path).read_text())])
    ss.main()  # should not raise


def test_main_guard_runs_via_runpy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["shift_summary.py", str(tmp_path)])

    monkeypatch.delitem(sys.modules, "src.analysis.shift_summary", raising=False)

    runpy.run_module("src.analysis.shift_summary", run_name="__main__", alter_sys=True)

    out = capsys.readouterr().out
    assert "shift_summary" in out
