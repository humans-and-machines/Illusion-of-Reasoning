from src.inference.cli.backfill_math_pass2 import (
    _build_row_key,
    _is_missing_pass,
    _merge_generated_passes,
    _parse_group_by,
    _record_matches_filters,
    _select_prev_output,
)


def test_parse_group_by_requires_fields():
    try:
        _parse_group_by("")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty group_by")


def test_is_missing_pass_handles_absent_none_empty():
    assert _is_missing_pass({}, "pass2") is True
    assert _is_missing_pass({"pass2": None}, "pass2") is True
    assert _is_missing_pass({"pass2": {}}, "pass2") is True
    assert _is_missing_pass({"pass2": {"x": 1}}, "pass2") is False


def test_build_row_key_casts_sample_idx():
    record = {"problem": "p", "sample_idx": "3", "split": "test"}
    assert _build_row_key(record, ["split", "problem"]) == ("test", "p", 3)


def test_select_prev_output_prefers_requested_sample():
    group = ("m", "s")
    pass1_by_group = {group: {0: "p0", 3: "p3"}}
    assert _select_prev_output(pass1_by_group, group, preferred_sample_idx=3) == "p3"
    assert _select_prev_output(pass1_by_group, group, preferred_sample_idx=2) == "p0"
    assert _select_prev_output(pass1_by_group, ("other",), preferred_sample_idx=0) is None


def test_record_matches_filters_stringifies_values():
    record = {"model": "Llama", "temperature": 0.3}
    assert _record_matches_filters(record, {"model": "Llama"}) is True
    assert _record_matches_filters(record, {"temperature": "0.3"}) is True
    assert _record_matches_filters(record, {"temperature": "0.7"}) is False
    assert _record_matches_filters(record, {"missing": "x"}) is False


def test_merge_generated_passes_respects_force_and_missing():
    record = {"pass2": {"already": True}, "pass2a": None}
    updates = {"pass2": {"new": True}, "pass2a": {"a": 1}}

    merged = _merge_generated_passes(record, updates, force=False)
    assert merged["pass2"] == {"already": True}
    assert merged["pass2a"] == {"a": 1}

    forced = _merge_generated_passes(record, updates, force=True)
    assert forced["pass2"] == {"new": True}
    assert forced["pass2a"] == {"a": 1}

