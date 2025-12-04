import pytest

from training import rewards_core as rc


def test_extract_content_variants():
    assert rc._extract_content("text") == "text"
    assert rc._extract_content([{"role": "assistant", "content": "hi"}]) == "hi"
    assert rc._extract_content([[{"content": "nested"}]]) == "nested"
    assert rc._extract_content(None) == ""
    # fallback to str() for unrecognized structures
    assert rc._extract_content(123) == "123"


def test_canon_crossword_letters_and_normalize():
    assert rc._canon_crossword_letters("café & crème") == "CAFEANDCREME"
    assert rc._canon_crossword_letters(None) == ""
    text = rc._normalize_for_word_match("“quote” & dash—")
    assert "quote" in text and "AND" in text
    assert rc._normalize_for_word_match(None) == ""


def test_has_gold_in_completion_modes():
    gold = rc._canon_crossword_letters("CAT")
    assert rc._has_gold_in_completion("", "anything", "any") is False
    # any mode
    assert rc._has_gold_in_completion(gold, "xxcatyy", "any")
    # word mode should require isolation
    assert rc._has_gold_in_completion(gold, "a cat!", "word")
    assert not rc._has_gold_in_completion(gold, "concatenate", "word")
    # contiguous allows inside token
    assert rc._has_gold_in_completion(gold, "concatenate", "contiguous")


def test_length_ramp_factor_edges():
    assert rc._length_ramp_factor("one two", min_tokens=5, max_full_len=10) == 0.0
    assert rc._length_ramp_factor(" ".join(["x"] * 10), min_tokens=5, max_full_len=10) == 1.0
    mid = rc._length_ramp_factor(" ".join(["x"] * 7), min_tokens=5, max_full_len=9)
    assert 0.0 < mid < 1.0


def test_score_single_crossword_scaled():
    comp = "<think>x</think><answer>cat</answer>"
    gold = "cat"
    config = rc._CrosswordScoreConfig(
        min_tokens=0,
        max_full_len=5,
        contains_mode="word",
        scaled=True,
    )
    score = rc._score_single_crossword(comp, gold, config=config)
    assert score > 0


def test_expected_length_and_score_single_pure(monkeypatch):
    spec = {"lengths": [2, 3]}
    assert rc._expected_length(spec) == 5
    comp = "<answer>abc</answer>"
    gold = "abc"
    # enumeration mismatch should drop to tiny bonus only
    out = rc._score_single_pure(comp, gold, contains_bonus=0.1, length_kwargs={"expected_len": 2})
    assert out == pytest.approx(0.05)
    # with tag and word match tiny bonus applies
    comp2 = "<think>x</think><answer>abc</answer> abc"
    out2 = rc._score_single_pure(comp2, "abc", contains_bonus=0.1, length_kwargs={})
    assert out2 > 0.0
    # No answer tag -> tiny bonus only if word match succeeds
    comp3 = "<think>hmm</think> abc"
    out3 = rc._score_single_pure(comp3, "abc", contains_bonus=0.1, length_kwargs={})
    assert 0.0 < out3 < 0.1


def test_pure_accuracy_reward_includes_shaping(monkeypatch):
    comp = "<answer>cat</answer>"
    gold = "cat"
    rewards = rc.pure_accuracy_reward([comp], [gold], contains_mode="any", min_tokens=1, max_full_len=1, scaled=False)
    # base 1 + 0.25 * shaping (shaping=1) clipped to 1
    assert rewards == [1.0]


def test_canon_math_and_pure_accuracy_reward_math():
    assert rc._canon_math(" { -0 } ") == "0"
    assert rc._canon_math("\\sqrt{4}") == "\\sqrt{4}".replace(" ", "")
    assert rc._canon_math("(2)") == "2"
    assert rc._canon_math("3.0") == "3"
    good = "<think>a</think><answer> 2+2 </answer>"
    bad = "<answer>3</answer>"
    out = rc.pure_accuracy_reward_math([good, bad], ["2+2", "2+2"])
    assert out == [1.0, 0.0]
    # If answer extraction fails, score should be 0
    missing_answer_pat = rc.re.compile(r"<nope>")
    old_pat = rc._answer_pat
    rc._answer_pat = missing_answer_pat
    try:
        out_missing = rc.pure_accuracy_reward_math([good], ["2+2"])
        assert out_missing == [0.0]
    finally:
        rc._answer_pat = old_pat


def test_crossword_format_and_length(monkeypatch):
    comp_correct = [[{"content": "<think>x</think><answer>CAT</answer>"}]]
    gold = ["DOG"]
    # Wrong but uppercase + tags => bonus
    assert rc.crossword_format_reward([comp_correct], gold, [None]) == [1.0]
    # Lowercase loses bonus
    assert rc.crossword_format_reward(["<answer>cat</answer>"], gold, [None]) == [0.0]
    # Length reward uses prompt enumeration
    prompt_chat = [{"role": "user", "content": "clue (3)"}]
    assert rc.crossword_length_reward([comp_correct], [prompt_chat]) == [1.0]
    assert rc._extract_answer_letters("no tags") == ""


def test_formating_helper():
    good = "<think>\nreason\n</think>\n<answer>\n42\n</answer>"
    bad = "<think>no</think><answer>42"
    assert rc.formating([good, bad]) == [1.0, 0.0]
