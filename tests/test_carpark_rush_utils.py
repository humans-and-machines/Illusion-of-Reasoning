import math

import src.inference.utils.carpark_rush_utils as rush


def test_canon_move_and_toklist_and_generic():
    assert rush._toklist("Bv2,A>1") == ["Bv2", "A>1"]
    assert rush._canon_move("bV2") == "Bv2"
    assert rush._canon_move("bad") is None
    assert rush._canon_rush_generic(["Bv2", "A>1"]) == "Bv2,A>1"
    assert rush._canon_rush_generic("Bv2 , A>1") == "Bv2,A>1"
    assert rush._canon_rush_generic(["Bv2", 3]) is None
    assert rush._toklist("") == []
    # split_token error path converts numeric suffix to int
    assert rush._split_token("A>12") == ("A", ">", 12)


def test_canon_rush_gold_collects_multiple_alternatives():
    gold = ["Bv2,A>1", ["Cv3", "Dv1"], None]
    canon_set = rush._canon_rush_gold(gold)
    assert canon_set == {"Bv2,A>1", "Cv3,Dv1"}


def test_canon_rush_string_and_generic_empty():
    assert rush._canon_rush_string("  ") is None
    assert rush._canon_rush_generic(None) is None


def test_multiset_overlap_and_piece_dir():
    # One token overlap out of three total
    assert math.isclose(rush._multiset_overlap_ratio(["A", "A"], ["A", "B"]), 1 / 3)
    assert rush._piece_dir("Bv2") == ("B", "v")
    match, step_score = rush._piece_dir_match_and_step_score("Bv2", "Bv3")
    assert match is True and math.isclose(step_score, 1 - 1 / 3)


def test_score_rush_pair_with_custom_weights():
    pred_tokens = rush._toklist("Bv2,A>1")
    gold = "Bv2,A>2"
    weights = {
        "prefix": 0.0,
        "pos_exact": 0.0,
        "piece_dir": 0.0,
        "step_close": 0.0,
        "lcs": 1.0,
        "bag_overlap": 0.0,
    }
    score, components = rush._score_rush_pair(
        pred_tokens,
        gold,
        weights=weights,
        length_penalty_base=1.0,
    )
    assert math.isclose(score, components["lcs"])
    assert math.isclose(score, 0.5)


def test_rush_soft_match_reward_exact_and_partial():
    score, detail = rush.rush_soft_match_reward("Bv2,A>1", "Bv2,A>1")
    assert score == 1.0
    assert detail["picked_gold"] == "Bv2,A>1"

    score2, detail2 = rush.rush_soft_match_reward("Bv2", ["Bv3", "Av1"])
    assert score2 < 1.0
    assert detail2["picked_gold"] == "Bv3,Av1"
    assert detail2["pred_canon"] == "Bv2"
    assert "gold_canon_options" in detail2


def test_rush_soft_match_reward_handles_invalid():
    score, detail = rush.rush_soft_match_reward("", None)
    assert score == 0.0
    assert detail["gold_canon_options"] == []


def test_score_rush_pair_length_penalty_and_components():
    pred_tokens = rush._toklist("Av1,Bv2")
    gold = "Av1,Bv3,Cv1"
    weights = {k: 0.0 for k in ["prefix", "pos_exact", "piece_dir", "step_close", "lcs", "bag_overlap"]}
    weights["prefix"] = 1.0
    score, components = rush._score_rush_pair(
        pred_tokens,
        gold,
        weights=weights,
        length_penalty_base=0.5,
    )
    assert components["length_delta"] == 1
    assert components["length_penalty"] == 0.5
    assert score <= 0.5


def test_canon_rush_generic_and_gold_invalid_types():
    # Non-str/list input hits fallback None.
    assert rush._canon_rush_generic(123) is None
    # Gold that is neither list nor string should return empty set (line 147).
    assert rush._canon_rush_gold({"not": "supported"}) == set()


def test_compute_pos_exact_zero_lengths():
    # gold_len or limit zero returns 0.0 (line 166).
    assert rush._compute_pos_exact([], ["A"], gold_len=0) == 0.0
    assert rush._compute_pos_exact(["A"], [], gold_len=1) == 0.0
