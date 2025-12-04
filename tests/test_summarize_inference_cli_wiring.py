#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv

import pytest


summ_mod = pytest.importorskip("src.inference.runners.summarize_inference_runner")
core_mod = pytest.importorskip("src.inference.domains.summarize.summarize_inference_core")


def test_compute_prompt_drop_groups_respects_max_versions(monkeypatch, tmp_path):
    # Patch iter_jsonl_objects to yield a fixed number of records.
    def fake_iter_jsonl_objects(_path):
        yield {"problem": "p1"}
        yield {"problem": "p1"}

    monkeypatch.setattr(summ_mod, "iter_jsonl_objects", fake_iter_jsonl_objects)

    # Patch accumulate_prompt_variants to populate variants_by_group.
    def fake_accumulate_prompt_variants(record, args, variants_by_group, counts):
        group = "group-1"
        counts["seen"] += 1
        if summ_mod.extract_group(record, args.group_key):
            counts["with_prompt"] += 1
        variants_by_group.setdefault(group, set()).add(f"v{counts['seen']}")

    monkeypatch.setattr(summ_mod, "accumulate_prompt_variants", fake_accumulate_prompt_variants)

    args = argparse.Namespace(
        max_prompt_versions=1,
        filter_scope="per_problem",
        group_key="problem",
    )
    files = [str(tmp_path / "dummy.jsonl")]

    drop_groups = summ_mod._compute_prompt_drop_groups(files, args)
    assert "group-1" in drop_groups


def _build_simple_stepagg():
    StepAgg = core_mod.StepAgg
    agg = StepAgg(step=1)
    record = {
        "problem": "P1",
        "gold_answer_canon": "gold",
        "pass1": {
            "is_correct_pred": True,
            "pred_answer_canon": "gold",
            "entropy": 1.0,
            "entropy_think": 0.5,
            "entropy_answer": 0.5,
            "tokens_think": 10,
            "tokens_answer": 5,
            "stop_reason_think": "stop_token",
            "stop_reason_answer": "stop_token",
            "valid_tag_structure": True,
            "has_reconsider_cue": False,
            "soft_reward": 0.3,
        },
        "pass2": {
            "is_correct_pred": True,
            "pred_answer_canon": "gold",
            "entropy": 1.2,
            "entropy_think": 0.6,
            "entropy_answer": 0.6,
            "tokens_think": 11,
            "tokens_answer": 6,
            "stop_reason_think": "stop_token",
            "stop_reason_answer": "stop_token",
            "valid_tag_structure": True,
            "has_reconsider_cue": False,
            "improved_over_pass1": False,
            "soft_reward": 0.4,
        },
    }
    agg.add_record(record, recompute_mode="none")
    agg.finalize()
    return agg


def test_print_step_summaries_prints_header_and_rows(capsys):
    agg = _build_simple_stepagg()
    summ_mod._print_step_summaries([agg])
    out = capsys.readouterr().out
    assert "step" in out
    assert "acc1S" in out  # header token
    assert str(agg.step) in out


def test_write_csv_outputs_writes_step_and_example_csv(tmp_path):
    agg = _build_simple_stepagg()

    step_csv = tmp_path / "steps.csv"
    ex_csv = tmp_path / "examples.csv"
    args = argparse.Namespace(
        save_csv=str(step_csv),
        per_example_csv=str(ex_csv),
    )

    summ_mod._write_csv_outputs(args, [agg])

    # Step-level CSV should have header + one row.
    with step_csv.open("r", encoding="utf-8") as file_handle:
        rows = list(csv.reader(file_handle))
    assert rows[0][0] == "step"
    assert int(rows[1][0]) == agg.step

    # Per-example CSV should have header + one row for P1.
    with ex_csv.open("r", encoding="utf-8") as file_handle:
        rows = list(csv.reader(file_handle))
    assert rows[0][:3] == ["step", "problem", "p1_correct"]
    assert rows[1][1] == "P1"
