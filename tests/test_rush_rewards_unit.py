import pytest

import src.training.rush_rewards as rr


BOARD_4X4 = "AAooBBCoooCoDDEo"  # 4x4 board with horizontal A.


def test_token_parsing_and_canon_seq():
    assert rr._parse_and_normalize_token("Bv3") == ("B", "v", 3)
    assert rr._parse_and_normalize_token("C^0") is None
    assert rr._parse_and_normalize_token("Dv-1") is None
    assert rr._parse_and_normalize_token("EV2") == ("E", "v", 2)
    assert rr._parse_and_normalize_token(object()) is None
    assert rr._looks_like_token_list("Av1,B>2") is True
    assert rr._looks_like_token_list("nope") is False
    assert rr._looks_like_token_list("  ") is False

    merged = rr._canon_seq(["A>1", "A>2", "Bv1"])
    assert merged == ["A>3", "Bv1"]
    assert rr._canon_seq(123) is None
    assert rr._canon_seq("") is None


def test_extract_and_canon_from_text_prefers_answer_block():
    text = "junk <answer>Av1, Av2</answer> trailing B>1"
    assert rr._extract_answer_block(text) == "Av1, Av2"
    seq = rr._canon_seq_from_text(text)
    assert seq == ["Av3"]  # consecutive same move merged

    text2 = "Moves: BU1, B U1; cR2"
    seq2 = rr._canon_seq_from_text(text2)
    assert seq2 == ["B^2", "C>2"]

    assert rr._extract_answer_block(None) is None
    fallback = rr._extract_answer_block("Tokens: Av1 ; B>2")
    assert fallback == "Av1"
    assert rr._canon_seq_from_text("nonsense") is None
    seq3 = rr._canon_seq_from_text("AU1, AD1, AL2, AR2, AV0")
    # u/d/l/r map to ^/v/</>, and zero step skipped
    assert seq3 == ["A^1", "Av1", "A<2", "A>2"]


def test_stringify_and_extract_puzzle_from_prompt():
    prompt_parts = [
        {"content": "Board size: 3x3"},
        {"content": "Board: A A o  B x o  A x o"},
        {"content": "Optimal moves: 5"},
    ]
    prompt_text = rr._stringify_prompt(prompt_parts)
    board, size, gold_moves = rr._extract_puzzle_from_prompt(prompt_text)
    assert size == 3 and gold_moves == 5
    assert board == "AAoBxoAxo"

    assert rr._stringify_prompt({"prompt": "hi"}) == "hi"
    assert "a" in rr._stringify_prompt(["a", {"content": ["b", "c"]}])
    assert rr._stringify_prompt(123).startswith("123")

    # Crops board longer than N*N and infers square size when missing
    prompt_text2 = "Board size: 2x2\nBoard: A A B B C"
    board2, size2, _ = rr._extract_puzzle_from_prompt(prompt_text2)
    assert board2 == "AABB" and size2 == 2

    prompt_text3 = "Board: A A A B B B C C C"
    board3, size3, _ = rr._extract_puzzle_from_prompt(prompt_text3)
    assert board3.replace(" ", "") == "AAABBBCCC"
    assert size3 == 3


def test_board_moves_and_simulation():
    board = rr.Board(BOARD_4X4, 4)
    assert board.orient["A"] == "H"
    assert board.is_solved() is False
    assert board.blockers_and_distance() == (0, 2)

    with pytest.raises(ValueError):
        rr.Board("short", 3)
    # No A present
    empty_board = rr.Board("oooooooooooooooo", 4)
    assert empty_board.blockers_and_distance() == (0, 0)

    clone = board.clone()
    assert clone.apply_token("A>2") is True
    assert clone.is_solved() is True
    assert clone.apply_token("B^1") is False  # wrong orientation

    valid, solved, _final = rr._simulate_prefix(rr.Board(BOARD_4X4, 4), ["A>1", "A>1", "B>1"])
    assert valid == 2 and solved is True
    # Invalid move early stops
    invalid_board = rr.Board(BOARD_4X4, 4)
    bad_valid, bad_solved, _ = rr._simulate_prefix(invalid_board, ["B>9"])
    assert bad_valid == 0 and bad_solved is False

    # Orientation fallback path
    diag_board = rr.Board("AoooAoooo", 3)
    assert diag_board.orient["A"] == "H"
    vertical_board = rr.Board("BBooAoooC", 3)
    assert vertical_board._step_move("B", "<") is False


def test_gold_candidates_and_prefix_credit():
    gold = rr._canon_gold_candidates([["A>1", "B>1", "C>1"], "Bv2, C<1", 123])
    assert gold[0] == ["A>1", "B>1", "C>1"]
    assert gold[1] == ["Bv2", "C<1"]
    assert rr._canon_gold_candidates(None) == []

    pred = ["A>1", "B>2", "X>9"]
    score = rr._rush_prefix_credit(pred, gold)
    assert 0 < score < 1  # longest common prefix shorter than pred
    assert rr._rush_prefix_credit([], gold) == 0
    assert rr._solve_term_for_target(5, 2) < 1


def test_progress_terms_and_solve_terms():
    base_board = rr.Board(BOARD_4X4, 4)
    solved_board = base_board.clone()
    solved_board.apply_token("A>2")
    phi = rr._phi_progress(base_board, solved_board, solved=True)
    assert 0 < phi <= 1
    assert rr._phi_progress(rr.Board("oooooooooooooooo", 4), rr.Board("oooooooooooooooo", 4), solved=False) == 0

    prefix, solve_term, phi_term = rr._rush_solve_terms(["A>2"], rr.Board(BOARD_4X4, 4), target_moves=2)
    assert prefix == pytest.approx(1.0)
    assert solve_term == pytest.approx(1.0)
    assert phi_term > 0

    prefix2, solve_term2, phi_term2 = rr._rush_solve_terms(["A>1"], None, target_moves=1)
    assert prefix2 == 0 and solve_term2 == pytest.approx(1.0) and phi_term2 == 0
    prefix3, solve_term3, phi_term3 = rr._rush_solve_terms(["A>9"], rr.Board(BOARD_4X4, 4), target_moves=1)
    assert prefix3 < 1 and solve_term3 in (0, 1) and phi_term3 >= 0


def test_formatting_bonus_and_rewards_paths():
    pred_text = "A>0 <think>one two three four five</think>"
    bonus = rr._rush_formatting_bonus(pred_text, think_min_tokens=1, think_full_tokens=3, think_bonus_cap=0.5)
    assert bonus == pytest.approx(0.01)  # capped

    scores_exact = rr.rush_solution_exact(prompts=None, completions=[pred_text], gold=["A>2"])
    assert scores_exact == [pytest.approx(bonus)]

    shaped = rr.rush_solution_shaped(
        prompts=None,
        completions=[pred_text],
        gold=["A>2"],
        board_str=None,
        board_size=None,
        gold_moves=None,
    )
    assert shaped[0] == pytest.approx(bonus)

    # Formatting bonus when gold is missing and tokens parsed fails
    exact_none_gold = rr.rush_solution_exact(
        prompts=None,
        completions=["junk <think>many words here</think>"],
        gold=None,
        think_min_tokens=0,
        think_full_tokens=1,
    )
    assert exact_none_gold[0] > 0

    # Shaped reward with no gold/board returns 0 for valid token seq
    shaped_no_gold = rr.rush_solution_shaped(
        prompts=None, completions=["Av1"], gold=None, board_str=None, board_size=None
    )
    assert shaped_no_gold[0] == 0.0

    # Formatting bonus ramp with many think tokens exceeding cap
    big_think = rr._rush_formatting_bonus(
        "Av1 <think>" + ("w " * 50) + "</think>", think_min_tokens=0, think_full_tokens=10, think_bonus_cap=0.5
    )
    assert big_think == pytest.approx(0.01)


def test_shaped_reward_uses_board_and_weights(monkeypatch):
    gold_seq = ["A>2"]
    good_pred = ["A>2"]
    partial_pred = ["A>1"]

    shaped = rr.rush_solution_shaped(
        prompts=None,
        completions=[",".join(good_pred), ",".join(partial_pred)],
        gold=gold_seq,
        board_str=BOARD_4X4,
        board_size=4,
        gold_moves=2,
    )
    assert shaped[0] > shaped[1]  # solved path should score higher

    # Ensure target moves from gold_moves is used when provided
    shaped_with_target = rr.rush_solution_shaped(
        prompts=None,
        completions=[",".join(good_pred)],
        gold=gold_seq,
        board_str=None,
        board_size=None,
        gold_moves=1,
    )
    assert 0 < shaped_with_target[0] <= 1
