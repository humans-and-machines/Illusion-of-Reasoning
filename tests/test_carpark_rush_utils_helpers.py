#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


rush_utils = pytest.importorskip("src.inference.utils.carpark_rush_utils")


def test_canon_move_normalizes_case_and_direction():
    assert rush_utils._canon_move("bV2") == "Bv2"
    assert rush_utils._canon_move("A>1") == "A>1"
    assert rush_utils._canon_move("invalid") is None


def test_canon_rush_string_handles_whitespace_and_commas():
    text = " bV2 , A>1  "
    canon = rush_utils._canon_rush_string(text)
    assert canon == "Bv2,A>1"


def test_canon_rush_generic_accepts_string_and_list():
    assert rush_utils._canon_rush_generic("bV2, A>1") == "Bv2,A>1"
    assert rush_utils._canon_rush_generic(["bV2", "A>1"]) == "Bv2,A>1"
    assert rush_utils._canon_rush_generic(None) is None


def test_canon_rush_gold_collects_unique_canonical_answers():
    gold = ["bV2, A>1", ["A>1", "Bv2"], "invalid"]
    out = rush_utils._canon_rush_gold(gold)
    assert "Bv2,A>1" in out
    # invalid alternative should be ignored
    assert all(rush_utils._canon_rush_string(x) is not None for x in out)


def test_piece_dir_match_and_step_score_behaves_as_expected():
    match, score = rush_utils._piece_dir_match_and_step_score("Bv2", "Bv2")
    assert match is True
    assert score == pytest.approx(1.0)

    match, score = rush_utils._piece_dir_match_and_step_score("Bv1", "Bv3")
    assert match is True
    assert 0.0 <= score < 1.0

    match, score = rush_utils._piece_dir_match_and_step_score("Av1", "Bv1")
    assert match is False
    assert score == 0.0
