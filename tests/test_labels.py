import pytest

import src.analysis.labels as labels


def test_aha_words_respects_injected_marker():
    base = {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"]}
    assert labels.aha_words(base) == 0
    assert labels.aha_words({"has_reconsider_cue": False}) == 0
    assert labels.aha_words({"has_reconsider_cue": True, "reconsider_markers": []}) == 1


def test_any_keys_true_and_cue_gate():
    pass1 = {"shift_in_reasoning_v1": True}
    rec = {"shift_llm": False}
    assert labels._any_keys_true(pass1, rec, labels.AHA_KEYS_CANONICAL) == 1
    assert labels._any_keys_true({}, {}, ["missing"]) == 0

    # Crossword allows judge/prefilter markers to open gate
    crossword_pass = {
        "has_reconsider_cue": False,
        "reconsider_markers": [],
        "_shift_prefilter_markers": ["p"],
        "shift_markers_v1": [],
    }
    assert labels._cue_gate_for_llm(crossword_pass, domain="Crossword") == 1
    # Non-crossword ignores judge markers, relies on reconsider or prefilter
    assert labels._cue_gate_for_llm(crossword_pass, domain="Math") == 1
    assert labels._cue_gate_for_llm({"has_reconsider_cue": False}, domain="Math") == 0


def test_aha_gpt_for_rec_gating():
    pass1 = {"change_way_of_thinking": True, "has_reconsider_cue": True}
    rec = {}
    keys = ["change_way_of_thinking"]
    assert labels.aha_gpt_for_rec(pass1, rec, gpt_subset_native=False, gpt_keys=keys, domain=None) == 1
    # When gating, reconsider flag still lets it through
    assert labels.aha_gpt_for_rec(pass1, rec, gpt_subset_native=True, gpt_keys=keys, domain="Math") == 1
    # Without reconsider marker, gate drops to 0
    pass1["has_reconsider_cue"] = False
    assert labels.aha_gpt_for_rec(pass1, rec, gpt_subset_native=True, gpt_keys=keys, domain="Math") == 0


def test_aha_gpt_modes_and_gate_by_words():
    pass1 = {"change_way_of_thinking": True, "shift_llm": True, "has_reconsider_cue": True}
    rec = {}
    assert labels.aha_gpt(pass1, rec, mode="canonical", gate_by_words=False) == 1
    assert labels.aha_gpt(pass1, rec, mode="broad", gate_by_words=False) == 1
    # Gating by words uses aha_words and cue gate
    assert labels.aha_gpt(pass1, rec, mode="canonical", gate_by_words=True, domain="Math") == 1
    # Missing reconsider cue removes gate
    pass1["has_reconsider_cue"] = False
    assert labels.aha_gpt(pass1, rec, mode="canonical", gate_by_words=True, domain="Math") == 0
    with pytest.raises(ValueError):
        labels.aha_gpt(pass1, rec, mode="bad")


def test_aha_gpt_broad_returns_zero_when_no_keys():
    # No keys set in pass1 or record should yield 0
    pass1 = {"has_reconsider_cue": True}
    rec = {}
    assert labels.aha_gpt_broad(pass1, rec) == 0
