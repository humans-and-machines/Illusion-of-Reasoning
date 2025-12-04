#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

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


def test_mean_safe_pct_fmt_float_and_extract_prompt_behaviors():
    assert core_mod.mean_safe([None, "bad"]) is None
    assert core_mod.pct(1, 0) == "-"
    assert core_mod.fmt_float(None) == "-"

    record = {"prompt": "base", "nested": {"p": "nested_prompt"}}
    # strict preferred key missing → None
    assert core_mod.extract_prompt(record, "missing", strict=True) is None
    # non-strict falls back to DEFAULT_PROMPT_KEYS
    assert core_mod.extract_prompt(record, "missing", strict=False) == "base"
    # preferred dotted key wins when present
    assert core_mod.extract_prompt(record, "nested.p", strict=False) == "nested_prompt"


def test_extract_group_and_drop_group_logic():
    rec = {"problem": "P", "prompt": "text"}
    assert core_mod.extract_group(rec, "problem") == "P"
    assert core_mod.extract_group(rec, "prompt") == "text"
    assert core_mod.extract_group({"problem": "P"}, "prompt") == "P"
    assert core_mod.extract_group(rec, "other") == ""

    assert core_mod.should_drop_group(rec, set(), "per_problem") is False
    assert core_mod.should_drop_group(rec, {"P"}, "per_problem") is True
    assert core_mod.should_drop_group(rec, {"__GLOBAL__"}, "global") is True


def test_accumulate_prompt_variants_handles_missing_and_regex():
    variants_by_group = {"__GLOBAL__": set()}
    counts = {"seen": 0, "with_prompt": 0}
    args = SimpleNamespace(prompt_key="", strict_prompt_key=False, prompt_family_regex=None, filter_scope="global")

    # Missing prompt → only seen increments
    core_mod.accumulate_prompt_variants({}, args, variants_by_group, counts)
    assert counts == {"seen": 1, "with_prompt": 0}

    # With prompt and regex stripping
    args.prompt_family_regex = "X"
    rec = {"prompt": "fooXbar"}
    core_mod.accumulate_prompt_variants(rec, args, variants_by_group, counts)
    assert counts["with_prompt"] == 1
    assert "foobar" in variants_by_group["__GLOBAL__"]


def test_group_name_for_record_global_and_per_problem():
    rec = {"problem": "P1"}
    assert core_mod.group_name_for_record(rec, "per_problem") == "P1"
    assert core_mod.group_name_for_record(rec, "global") == "__GLOBAL__"


def test_helper_edge_cases_and_fallbacks():
    # mean_safe should ignore items that cannot be coerced to float.
    assert core_mod.mean_safe([1, "bad", object(), "3"]) == pytest.approx(2.0)
    # pct denominator zero branch.
    assert core_mod.pct(0, 0) == "-"
    # _get_nested should return None when traversal hits a non-dict.
    assert core_mod._get_nested({"outer": {"inner": 1}}, "outer.inner.missing") is None

    prompt_record = {"fmt_prompt": "from_defaults"}
    # Preferred key missing, non-strict → falls back to DEFAULT_PROMPT_KEYS.
    assert core_mod.extract_prompt(prompt_record, preferred_key="", strict=False) == "from_defaults"

    rec = {"problem": "P_missing"}
    assert core_mod.extract_group(rec, "prompt") == "P_missing"
    assert core_mod.extract_group(rec, "nonexistent") == ""
    assert core_mod.should_drop_group(rec, {"__GLOBAL__"}, "global") is True
    assert core_mod.group_name_for_record(rec, "global") == "__GLOBAL__"

    # Substring/exact helpers handle missing inputs.
    assert core_mod._substr(None, "needle") is False
    assert core_mod._exact(None, "needle") is False


def test_accumulate_prompt_variants_early_return_and_regex_strip():
    variants_by_group = {"__GLOBAL__": set()}
    counts = {"seen": 0, "with_prompt": 0}
    args = SimpleNamespace(
        prompt_key="missing",
        strict_prompt_key=False,
        prompt_family_regex="strip",
        filter_scope="global",
    )

    # Missing prompt triggers early return after incrementing 'seen'.
    core_mod.accumulate_prompt_variants({}, args, variants_by_group, counts)
    assert counts == {"seen": 1, "with_prompt": 0}

    # Regex stripping removes the family marker before storing.
    core_mod.accumulate_prompt_variants(
        {"prompt": "prefix_strip_suffix"},
        args,
        variants_by_group,
        counts,
    )
    assert counts["seen"] == 2 and counts["with_prompt"] == 1
    assert "prefix__suffix" in variants_by_group["__GLOBAL__"]
