#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pytest


core_mod = pytest.importorskip("src.inference.domains.summarize.summarize_inference_core")


def test_nat_step_from_path_parses_step_number_and_none():
    assert core_mod.nat_step_from_path("step00050_test.jsonl") == 50
    assert core_mod.nat_step_from_path("no_step_here.jsonl") is None


def test_scan_files_sorts_and_filters(tmp_path):
    root = tmp_path / "results"
    (root / "sub").mkdir(parents=True)
    paths = [
        root / "step00010_test.jsonl",
        root / "step00005_train.jsonl",
        root / "nostep_test.jsonl",
        root / "sub" / "step00020_test.jsonl",
    ]
    for path in paths:
        path.write_text("{}", encoding="utf-8")

    all_files = core_mod.scan_files(str(root), split=None)
    # Sorted ascending by step then name.
    assert all_files[0].endswith("step00005_train.jsonl")
    assert all_files[-1].endswith("step00020_test.jsonl")

    test_files = core_mod.scan_files(str(root), split="test")
    assert all(p.endswith("test.jsonl") for p in test_files)
