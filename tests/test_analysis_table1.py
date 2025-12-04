#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import src.analysis.table_1 as table1


def test_aha_gpt_gating_and_markers(monkeypatch):
    cfg = table1.GptAhaConfig(
        gpt_keys=["shift_in_reasoning_v1"],
        gpt_subset_native=True,
        allow_judge_cues_non_xword=False,
        broad_counts_marker_lists=True,
    )
    # Gate returns False -> shift should be 0 even if raw 1
    monkeypatch.setattr(table1, "any_keys_true", lambda pass1, rec, keys: 1)
    monkeypatch.setattr(table1, "cue_gate_for_llm", lambda *args, **kwargs: 0)
    assert table1._aha_gpt_for_rec({}, {}, "Math", cfg) == 0

    # Raw false, but marker list present in broad mode
    monkeypatch.setattr(table1, "any_keys_true", lambda *a, **k: 0)
    monkeypatch.setattr(table1, "cue_gate_for_llm", lambda *a, **k: 1)
    assert table1._aha_gpt_for_rec({"shift_markers_v1": ["x"]}, {}, "Math", cfg) == 1

    # Ungated broad config keeps raw value
    cfg2 = table1.GptAhaConfig(
        gpt_keys=["shift_in_reasoning_v1"],
        gpt_subset_native=False,
        allow_judge_cues_non_xword=False,
        broad_counts_marker_lists=False,
    )
    monkeypatch.setattr(table1, "any_keys_true", lambda *a, **k: 1)
    assert table1._aha_gpt_for_rec({}, {}, "Math", cfg2) == 1


def test_build_row_for_record_math_and_carpark(monkeypatch):
    cfg = table1.LoadRowsConfig(
        gpt_config=table1.GptAhaConfig(["k"], True, False, False),
        min_step=None,
        max_step=10,
        carpark_success_fn=lambda soft: 1 if soft and soft > 0 else 0,
    )
    monkeypatch.setattr(table1, "extract_pass1_and_step", lambda rec, step_from_name: (rec.get("pass1", {}), 5))
    monkeypatch.setattr(table1, "step_within_bounds", lambda step, min_step, max_step: step <= max_step)
    monkeypatch.setattr(table1, "_aha_gpt_for_rec", lambda *a, **k: 1)
    # Math branch uses is_correct_pred
    rec_math = {"pass1": {"is_correct_pred": True}}
    row_math = table1._build_row_for_record("Math", rec_math, None, cfg)
    assert row_math["correct"] == 1 and row_math["shift"] == 1

    # Carpark branch uses soft_reward comparator
    rec_car = {"pass1": {"soft_reward": 0.2}}
    row_car = table1._build_row_for_record("Carpark", rec_car, None, cfg)
    assert row_car["correct"] == 1

    # Missing pass1 or step returns None
    monkeypatch.setattr(table1, "extract_pass1_and_step", lambda rec, step_from_name: ({}, None))
    assert table1._build_row_for_record("Math", {}, None, cfg) is None


def test_build_row_respects_step_bounds_and_none_correct(monkeypatch):
    cfg = table1.LoadRowsConfig(
        gpt_config=table1.GptAhaConfig(["k"], True, False, False),
        min_step=0,
        max_step=1,
        carpark_success_fn=lambda soft: soft,
    )
    # Step out of bounds short-circuits.
    monkeypatch.setattr(table1, "extract_pass1_and_step", lambda rec, name: (rec.get("pass1", {}), 5))
    monkeypatch.setattr(table1, "step_within_bounds", lambda step, min_step, max_step: False)
    assert table1._build_row_for_record("Math", {"pass1": {}}, None, cfg) is None

    # None correctness short-circuits for non-carpark domains.
    monkeypatch.setattr(table1, "step_within_bounds", lambda step, min_step, max_step: True)
    monkeypatch.setattr(table1, "coerce_bool", lambda value: None)
    assert table1._build_row_for_record("Math", {"pass1": {"is_correct_pred": None}}, None, cfg) is None


def test_load_rows_and_empty(monkeypatch):
    calls = {"iter": 0}

    def fake_iter_records(path):
        calls["iter"] += 1
        return [
            {"pass1": {"is_correct_pred": 1}, "split": "test"},
            {"pass1": {"is_correct_pred": 0}, "split": "test"},
        ]

    monkeypatch.setattr(table1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(table1, "iter_records_from_file", fake_iter_records)
    monkeypatch.setattr(
        table1,
        "_build_row_for_record",
        lambda domain, rec, step, cfg: {"domain": domain, "step": 1, "correct": 1, "shift": 0},
    )

    cfg = table1.LoadRowsConfig(
        gpt_config=table1.GptAhaConfig(["k"], True, False, False),
        min_step=None,
        max_step=10,
        carpark_success_fn=lambda soft: 1,
        debug=True,
    )
    df = table1.load_rows({"Math": ["a.jsonl"]}, cfg)
    assert len(df) == 2
    assert calls["iter"] == 1

    # No rows triggers SystemExit
    monkeypatch.setattr(table1, "_build_row_for_record", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        table1.load_rows({"Math": ["a.jsonl"]}, cfg)


def test_agg_domain_conditionals_order_and_counts():
    df = pd.DataFrame(
        [
            {"domain": "A", "correct": 1, "shift": 1},
            {"domain": "A", "correct": 0, "shift": 0},
            {"domain": "B", "correct": 1, "shift": 0},
        ],
    )
    out = table1.agg_domain_conditionals(df)
    assert list(out["domain"]) == ["A", "B"]
    a_row = out[out["domain"] == "A"].iloc[0]
    assert a_row["n_total"] == 2
    assert a_row["k_correct_shift"] == 1
    assert a_row["k_correct_noshift"] == 0


def test_build_arg_parser_and_gpt_config_defaults():
    parser = table1._build_arg_parser()
    args = parser.parse_args([])
    assert args.gpt_mode == "canonical"
    assert args.carpark_success_op == "gt"

    cfg = table1._build_gpt_config(args)
    assert cfg.gpt_subset_native is True
    assert cfg.broad_counts_marker_lists is False


def test_build_files_by_domain(monkeypatch):
    args = SimpleNamespace(
        root_crossword="c1",
        root_math=None,
        root_math2=None,
        root_math3=None,
        root_carpark=None,
        label_math2="Math2",
        label_math3="Math3",
        split=None,
        results_root=None,
    )
    monkeypatch.setattr(table1, "build_jsonl_files_by_domain", lambda roots, split: ({"Crossword": ["f1"]}, "c1"))
    files_by_domain, first = table1._build_files_by_domain(args)
    assert files_by_domain == {"Crossword": ["f1"]}
    assert first == "c1"

    # No roots -> fallback to results_root
    args.root_crossword = None
    args.results_root = "r"
    monkeypatch.setattr(table1, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    monkeypatch.setattr(table1, "scan_jsonl_files", lambda root, split: ["r1"])
    files_by_domain2, first2 = table1._build_files_by_domain(args)
    assert files_by_domain2 == {"All": ["r1"]}
    assert first2 == "r"

    # Still empty should raise
    monkeypatch.setattr(table1, "scan_jsonl_files", lambda root, split: [])
    args.results_root = "r_empty"
    with pytest.raises(SystemExit):
        table1._build_files_by_domain(args)


def test_compute_effective_max_step(capsys):
    args = SimpleNamespace(max_step=None)
    assert table1._compute_effective_max_step(args) == table1.HARD_STEP_CAP
    captured = capsys.readouterr()
    assert "Capping max_step" in captured.out

    args2 = SimpleNamespace(max_step=500)
    assert table1._compute_effective_max_step(args2) == 500


def test_main_happy_path(monkeypatch, tmp_path):
    args = SimpleNamespace(
        root_crossword=None,
        root_math=None,
        root_math2=None,
        root_math3=None,
        root_carpark=None,
        results_root="r",
        split=None,
        out_dir=str(tmp_path / "out"),
        dataset_name="D",
        model_name="M",
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        allow_judge_cues_non_xword=False,
        min_step=None,
        max_step=None,
        carpark_soft_threshold=0.1,
        carpark_success_op="gt",
        debug=False,
        label_math2="Math2",
        label_math3="Math3",
    )
    # Ensure parse_args consumes only our synthetic args
    monkeypatch.setattr(table1, "_build_arg_parser", lambda: argparse.ArgumentParser())
    monkeypatch.setattr(table1.argparse.ArgumentParser, "parse_args", lambda self: args)

    monkeypatch.setattr(table1, "_build_files_by_domain", lambda a: ({"Math": ["f"]}, "r"))
    monkeypatch.setattr(
        table1, "load_rows", lambda files, cfg: pd.DataFrame([{"domain": "Math", "step": 1, "correct": 1, "shift": 0}])
    )
    monkeypatch.setattr(
        table1,
        "agg_domain_conditionals",
        lambda df: df.assign(
            n_total=1,
            k_correct_shift=0,
            n_shift=0,
            p_correct_given_shift=pd.NA,
            k_correct_noshift=1,
            n_noshift=1,
            p_correct_given_noshift=1.0,
            shift_rate=0.0,
        ),
    )

    table1.main()
    out_csv = tmp_path / "out" / "aha_conditionals__D__M.csv"
    assert out_csv.exists()


def test_aha_gpt_prefilter_markers(monkeypatch):
    cfg = table1.GptAhaConfig(
        gpt_keys=["shift_in_reasoning_v1"],
        gpt_subset_native=False,
        allow_judge_cues_non_xword=False,
        broad_counts_marker_lists=True,
    )
    monkeypatch.setattr(table1, "any_keys_true", lambda *a, **k: 0)
    monkeypatch.setattr(table1, "coerce_bool", lambda val: 1 if val else 0)

    result = table1._aha_gpt_for_rec({}, {"_shift_prefilter_markers": ["m1"]}, "Math", cfg)
    assert result == 1


def test_build_row_for_record_filters(monkeypatch):
    cfg = table1.LoadRowsConfig(
        gpt_config=table1.GptAhaConfig(["k"], True, False, False),
        min_step=None,
        max_step=5,
        carpark_success_fn=lambda soft: 1,
    )
    monkeypatch.setattr(table1, "extract_pass1_and_step", lambda rec, step_from_name: (rec.get("pass1", {}), 7))
    monkeypatch.setattr(table1, "step_within_bounds", lambda step, min_step, max_step: False)
    assert table1._build_row_for_record("Math", {"pass1": {}}, None, cfg) is None

    monkeypatch.setattr(table1, "step_within_bounds", lambda step, min_step, max_step: True)
    monkeypatch.setattr(table1, "coerce_bool", lambda value: None)
    assert table1._build_row_for_record("Math", {"pass1": {}}, None, cfg) is None


class _SeededRoot:
    def __init__(self, label: str, path: str):
        self.label = label
        self.path = path

    def __bool__(self) -> bool:
        frame = inspect.currentframe().f_back
        frame.f_locals["files_by_domain"][self.label] = ["existing"]
        return True

    def __str__(self) -> str:
        return self.path


def test_build_files_by_domain_label_dedup(monkeypatch):
    captured = {}

    def fake_build_jsonl_files_by_domain(domain_roots, split):
        captured["domain_roots"] = dict(domain_roots)
        return ({k: [f"{v}/file.jsonl"] for k, v in domain_roots.items()}, "first_root")

    args = SimpleNamespace(
        root_crossword=None,
        root_math="math_root",
        root_math2=_SeededRoot("Math2", "math2_root"),
        root_math3=_SeededRoot("Math3", "math3_root"),
        root_carpark="car_root",
        label_math2="Math2",
        label_math3="Math3",
        split=None,
        results_root=None,
    )
    monkeypatch.setattr(table1, "build_jsonl_files_by_domain", fake_build_jsonl_files_by_domain)

    files_by_domain, first_root = table1._build_files_by_domain(args)
    assert first_root == "first_root"
    assert set(files_by_domain.keys()) == {"Math", "Math2-2", "Math3-2", "Carpark"}
    assert captured["domain_roots"]["Math2-2"] == args.root_math2
    assert captured["domain_roots"]["Math3-2"] == args.root_math3


def test_build_files_by_domain_needs_roots(monkeypatch):
    args = SimpleNamespace(
        root_crossword=None,
        root_math=None,
        root_math2=None,
        root_math3=None,
        root_carpark=None,
        label_math2="Math2",
        label_math3="Math3",
        split=None,
        results_root=None,
    )
    monkeypatch.setattr(table1, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    with pytest.raises(SystemExit):
        table1._build_files_by_domain(args)


def test_module_entrypoint_runs(monkeypatch, tmp_path):
    stub_io = ModuleType("src.analysis.io")
    stub_metrics = ModuleType("src.analysis.metrics")
    stub_utils = ModuleType("src.analysis.utils")
    stub_aha = ModuleType("src.analysis.aha_utils")

    stub_io.build_jsonl_files_by_domain = lambda roots, split: ({"Math": ["stub.jsonl"]}, str(tmp_path))
    stub_io.iter_records_from_file = lambda path: [{"pass1": {"is_correct_pred": 1}}]
    stub_io.scan_jsonl_files = lambda root, split: ["stub.jsonl"]

    stub_metrics.make_carpark_success_fn = lambda op, threshold: lambda reward: 1

    stub_utils.add_split_and_out_dir_args = lambda parser, out_dir_help=None: (
        parser.add_argument("--split", default=None),
        parser.add_argument("--out_dir", default=None),
    )

    def add_standard_domain_root_args(parser):
        parser.add_argument("--root_crossword", default=None)
        parser.add_argument("--root_math", default=None)
        parser.add_argument("--root_math2", default=None)
        parser.add_argument("--root_carpark", default=None)
        parser.add_argument("--results_root", default=None)

    stub_utils.add_standard_domain_root_args = add_standard_domain_root_args
    stub_utils.both_get = lambda pass1_fields, rec, key: pass1_fields.get(key, rec.get(key))
    stub_utils.coerce_bool = lambda value: 1 if value is not None else None
    stub_utils.coerce_float = lambda value: float(value) if value is not None else None
    stub_utils.extract_pass1_and_step = lambda rec, step_from_name: (rec.get("pass1", {}), step_from_name or 1)
    stub_utils.gpt_keys_for_mode = lambda mode: ["shift_in_reasoning_v1"]
    stub_utils.nat_step_from_path = lambda path: 1
    stub_utils.step_within_bounds = lambda step, min_step, max_step: True

    stub_aha.any_keys_true = lambda pass1_fields, rec, keys: 0
    stub_aha.cue_gate_for_llm = lambda *args, **kwargs: 1

    monkeypatch.setitem(sys.modules, "src.analysis.io", stub_io)
    monkeypatch.setitem(sys.modules, "src.analysis.metrics", stub_metrics)
    monkeypatch.setitem(sys.modules, "src.analysis.utils", stub_utils)
    monkeypatch.setitem(sys.modules, "src.analysis.aha_utils", stub_aha)

    monkeypatch.setattr(sys, "argv", ["table_1"])
    orig_table1 = sys.modules.pop("src.analysis.table_1", None)
    runpy.run_module("src.analysis.table_1", run_name="__main__")
    if orig_table1 is not None:
        sys.modules["src.analysis.table_1"] = orig_table1

    out_dir = tmp_path / "aha_conditionals"
    assert (out_dir / "aha_conditionals__MIXED__MIXED_MODELS.csv").exists()


def test_build_row_for_record_filters_extract_none(monkeypatch):
    import src.analysis.table_1 as table1_mod

    monkeypatch.setattr(
        table1_mod,
        "extract_pass1_and_step",
        lambda rec, step: (None, None),
    )
    cfg = table1_mod.LoadRowsConfig(
        min_step=0,
        max_step=10,
        gpt_config=table1_mod.GptAhaConfig(
            gpt_keys=[],
            gpt_subset_native=False,
            allow_judge_cues_non_xword=False,
            broad_counts_marker_lists=True,
        ),
        carpark_success_fn=lambda val: 1,
    )
    out = table1_mod._build_row_for_record(
        "Math",
        {"dummy": 1},
        None,
        cfg,
    )
    assert out is None


def test_build_row_respects_step_bounds(monkeypatch):
    import src.analysis.table_1 as table1_mod

    monkeypatch.setattr(
        table1_mod,
        "extract_pass1_and_step",
        lambda rec, step: ({"is_correct_pred": True}, 5),
    )
    monkeypatch.setattr(table1_mod, "step_within_bounds", lambda step, min_step, max_step: False)
    cfg = table1_mod.LoadRowsConfig(
        min_step=10,
        max_step=20,
        gpt_config=table1_mod.GptAhaConfig(
            gpt_keys=[],
            gpt_subset_native=False,
            allow_judge_cues_non_xword=False,
            broad_counts_marker_lists=True,
        ),
        carpark_success_fn=lambda *_a, **_k: 1,
    )
    out = table1_mod._build_row_for_record("Math", {"pass1": {"is_correct_pred": True}}, None, cfg)
    assert out is None
