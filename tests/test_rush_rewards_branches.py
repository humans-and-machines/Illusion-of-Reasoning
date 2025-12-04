#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import types

import src.training.rush_rewards as rr


def test_parse_and_canon_invalid_tokens(monkeypatch):
    assert rr._parse_and_normalize_token("Avbad") is None  # non-numeric steps
    assert rr._canon_seq(["A>1", "bad"]) is None
    # Int conversion failure path
    monkeypatch.setattr(rr, "int", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")), raising=False)
    assert rr._parse_and_normalize_token("Av1") is None


def test_canon_seq_from_text_list_and_prompt_fallback():
    seq = rr._canon_seq_from_text(["Av1", "B>2"])
    assert seq == ["Av1", "B>2"]

    prompt = "WXYZ\nBoard size: 2x2\nBoard: A\n123\nNumbers\nOptimal moves: 10"
    board, size, gold = rr._extract_puzzle_from_prompt(prompt)
    assert board == "WXYZ" and size == 2 and gold == 10


def test_stringify_prompt_branches():
    assert rr._stringify_prompt({"content": "c"}) == "c"
    msgs = [{"content": ["x", "y"]}, 123]
    out = rr._stringify_prompt(msgs)
    assert "x" in out and "123" in out


def test_board_edge_cases():
    b = rr.Board("AxCoooooo", size=3)
    assert b.is_solved() is False  # missing A at edge
    blockers, _ = b.blockers_and_distance()
    assert blockers == 2  # wall + car
    assert b._step_move("Z", ">") is False  # missing car
    # Orientation mismatch
    b_vert = rr.Board("AAABBBooC", size=3)
    assert b_vert._step_move("B", ">") is False
    # Blocked downward move
    b_block = rr.Board("AooDooDoo", size=3)
    assert b_block._step_move("D", "v") is False
    # No target car present
    b_empty = rr.Board("oooo", size=2)
    assert b_empty.is_solved() is False
    # Orientation mismatch for vertical car
    b_vert = rr.Board("AoAo", size=2)
    assert b_vert._step_move("A", ">") is False
    # Upward move branch for vertical car at top edge
    assert b_vert._step_move("A", "^") is False


def test_formatting_bonus_and_think_count(monkeypatch):
    calls = {"scan": 0}

    def fake_finditer(text):
        calls["scan"] += 1
        raise re.error("bad")

    monkeypatch.setattr(rr, "_TOKEN_SCAN", types.SimpleNamespace(finditer=fake_finditer))
    bonus = rr._rush_formatting_bonus("<think>one two three</think>", 0, 1, 0.02)
    assert bonus <= 0.02 and calls["scan"] == 1
    assert rr._count_think_tokens("no think tags") == 0


def test_rush_build_board_and_moves_error_paths(monkeypatch):
    board, moves = rr._rush_build_board_and_moves(prompts=None, board_str="abcd", size="bad")
    assert board is None and moves is None

    monkeypatch.setattr(rr, "_extract_puzzle_from_prompt", lambda prompt: ("ABC", 2, 3))
    board2, moves2 = rr._rush_build_board_and_moves(prompts="p", board_str=None, size=None)
    assert board2 is None and moves2 == 3


def test_rush_prefix_credit_skips_empty():
    score = rr._rush_prefix_credit(["A>1"], [[], ["A>2"]])
    assert 0.0 <= score <= 1.0


def test_rush_solution_shaped_string_and_target_moves(monkeypatch):
    monkeypatch.setattr(rr, "_canon_seq_from_text", lambda text: None)
    monkeypatch.setattr(rr, "_rush_formatting_bonus", lambda *a, **k: 0.5)
    monkeypatch.setattr(rr, "_canon_gold_candidates", lambda gold: [["A>1"]])
    monkeypatch.setattr(rr, "_rush_build_board_and_moves", lambda prompts, board_str, board_size: (None, 4))
    scores = rr.rush_solution_shaped(
        prompts="p",
        completions="pred",
        gold=None,
        gold_moves=None,
        board_str=None,
        board_size=None,
    )
    assert scores == [0.5]


def test_rush_solution_exact_string_and_bonus(monkeypatch):
    monkeypatch.setattr(rr, "_canon_seq_from_text", lambda txt: None)
    monkeypatch.setattr(rr, "_rush_formatting_bonus", lambda *a, **k: 0.25)
    scores = rr.rush_solution_exact(prompts="p", completions="pred", gold=None)
    assert scores == [0.25]
