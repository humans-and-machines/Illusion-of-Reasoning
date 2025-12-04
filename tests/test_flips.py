import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.analysis.flips as flips


def test_load_all_scores_raises_without_files(tmp_path):
    with pytest.raises(SystemExit):
        flips._load_all_scores(tmp_path)


def test_compute_flips_filters_wrong_to_correct():
    df = pd.DataFrame(
        [
            {"problem": "p1", "sample_idx": 0, "step": 50, "correct": False},
            {"problem": "p1", "sample_idx": 0, "step": 850, "correct": True},
            {"problem": "p2", "sample_idx": 1, "step": 50, "correct": True},
            {"problem": "p2", "sample_idx": 1, "step": 850, "correct": True},
        ]
    )
    flips_df, total = flips._compute_flips(df, init_step=50, final_step=850)
    assert total == 2
    assert list(flips_df.index.get_level_values("problem")) == ["p1"]


def _write_scored(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_main_runs_and_writes(tmp_path, capsys, monkeypatch):
    root = tmp_path / "run"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True)
    file_path = analysis_dir / "step_scored.jsonl"
    rows = [
        {
            "problem": "p1",
            "sample_idx": 0,
            "step": 50,
            "correct": False,
            "output": "a",
            "entropy": 0.1,
            "has_recheck": False,
            "reason": "init",
        },
        {
            "problem": "p1",
            "sample_idx": 0,
            "step": 850,
            "correct": True,
            "output": "b",
            "entropy": 0.2,
            "has_recheck": True,
            "reason": "final",
        },
    ]
    _write_scored(file_path, rows)

    out_csv = tmp_path / "flips.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--results_root", str(root), "--init_step", "50", "--final_step", "850", "--output", str(out_csv)],
    )
    flips.main()
    captured = capsys.readouterr().out
    assert "Found 1 flips" in captured
    assert out_csv.exists()
    df_out = pd.read_csv(out_csv)
    assert list(df_out["problem"]) == ["p1"]
    assert "output_50" in df_out.columns and "output_850" in df_out.columns


def test_main_no_flips_returns_early(tmp_path, capsys, monkeypatch):
    root = tmp_path / "run"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True)
    file_path = analysis_dir / "step_scored.jsonl"
    rows = [
        {"problem": "p1", "sample_idx": 0, "step": 50, "correct": True},
        {"problem": "p1", "sample_idx": 0, "step": 850, "correct": True},
    ]
    _write_scored(file_path, rows)

    out_csv = tmp_path / "flips.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--results_root", str(root), "--init_step", "50", "--final_step", "850", "--output", str(out_csv)],
    )
    flips.main()
    captured = capsys.readouterr().out
    assert "Found 0 flips" in captured
    assert not out_csv.exists()


def test_main_missing_analysis_dir_raises(tmp_path, monkeypatch):
    root = tmp_path / "nope"
    monkeypatch.setattr(sys, "argv", ["prog", "--results_root", str(root)])
    with pytest.raises(SystemExit):
        flips.main()


def test_module_entrypoint_invokes_main(monkeypatch, tmp_path):
    # Prepare minimal scored file to allow script to complete.
    root = tmp_path / "mod"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True)
    file_path = analysis_dir / "step_scored.jsonl"
    rows = [
        {
            "problem": "p1",
            "sample_idx": 0,
            "step": 50,
            "correct": False,
            "output": "a",
            "entropy": 0.1,
            "has_recheck": False,
            "reason": "init",
        },
        {
            "problem": "p1",
            "sample_idx": 0,
            "step": 850,
            "correct": True,
            "output": "b",
            "entropy": 0.2,
            "has_recheck": True,
            "reason": "final",
        },
    ]
    _write_scored(file_path, rows)

    out_csv = root / "flips.csv"
    monkeypatch.setattr(sys, "argv", ["prog", "--results_root", str(root), "--output", str(out_csv)])
    sys.modules.pop("src.analysis.flips", None)
    runpy.run_module("src.analysis.flips", run_name="__main__")
    assert out_csv.exists()
