# src/tests/test_rewards.py

import pytest
from open_r1.rewards_core import (
    crossword_accuracy_reward,
    crossword_format_reward,
    crossword_length_reward,
    formating,
)

def make_comp(ans: str, think: str = None):
    content = ""
    if think is not None:
        content += f"<think>\n{think}\n</think>\n"
    content += f"<answer>{ans}</answer>"
    return [[{"content": content}]]

def make_prompt(clue: str):
    return [
        {"role": "system", "content": ""},
        {"role": "user",   "content": clue},
    ]

def test_accuracy_exact_match_and_mismatch():
    c = make_comp("FOO")
    assert crossword_accuracy_reward([c], ["foo"]) == [1.0]
    assert crossword_accuracy_reward([c], ["bar"]) == [0.0]

def test_formatting_and_uppercase_bonus():
    gold = ["RIGHT"]
    # wrong but well-formatted & uppercase → +1
    c1 = make_comp("WRONG", think="reason")
    assert crossword_format_reward([c1], gold, [make_prompt("clue (5)")]) == [1.0]
    # lowercase inside <answer> → 0
    c2 = make_comp("wrong", think="x")
    assert crossword_format_reward([c2], gold, [make_prompt("clue (5)")]) == [0.0]
    # correct answer → 0
    c3 = make_comp("RIGHT", think="x")
    assert crossword_format_reward([c3], gold, [make_prompt("clue (5)")]) == [0.0]

def test_length_reward():
    c = make_comp("FOUR", think="x")
    # matches (4)
    assert crossword_length_reward([c], [make_prompt("def (4)")]) == [1.0]
    # mismatch (5)
    assert crossword_length_reward([c], [make_prompt("def (5)")]) == [0.0]
    # no parentheses
    assert crossword_length_reward([c], [make_prompt("def no num")]) == [0.0]
