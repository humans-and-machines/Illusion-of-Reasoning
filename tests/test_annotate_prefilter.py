#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


prefilter_mod = pytest.importorskip("src.annotate.core.prefilter")


def test_extract_think_returns_inner_block():
    text = "foo <think>  inner reasoning  </think> bar"
    out = prefilter_mod.extract_think(text)
    assert out == "inner reasoning"


def test_extract_think_none_when_missing():
    assert prefilter_mod.extract_think("no tags here") is None


def test_find_shift_cues_detects_cue_and_position():
    think = "I started, but wait, this seems wrong."
    cues, pos = prefilter_mod.find_shift_cues(think)
    assert isinstance(cues, list)
    assert "wait" in cues
    assert isinstance(pos, int)
    assert 0 <= pos < len(think)


def test_find_shift_cues_empty_when_no_hits():
    cues, pos = prefilter_mod.find_shift_cues("completely neutral text")
    assert cues == []
    assert pos is None


def test_find_cue_hits_returns_names_only():
    think = "On second thought, actually this is better."
    hits = prefilter_mod.find_cue_hits(think)
    assert isinstance(hits, list)
    assert hits  # at least one cue
    # verify it aligns with find_shift_cues' first element list
    cues, _ = prefilter_mod.find_shift_cues(think)
    assert hits == cues

