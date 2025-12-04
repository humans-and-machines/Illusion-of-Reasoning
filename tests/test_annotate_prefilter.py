import src.annotate.core.prefilter as pf


def test_extract_think_grabs_inner_text():
    txt = "<think> first idea </think> <answer>42</answer>"
    assert pf.extract_think(txt) == "first idea"
    assert pf.extract_think("no tags here") is None


def test_find_shift_cues_returns_hits_and_position():
    think = "I thought it was 5, but actually it's 7."
    hits, pos = pf.find_shift_cues(think)
    assert "actually" in hits
    # Position should land at or just before the token
    target = think.lower().find("actually")
    assert pos is not None and 0 <= target - pos <= 1


def test_find_cue_hits_shortcut():
    think = "wait, maybe wrong length"
    hits = pf.find_cue_hits(think)
    assert "wait" in hits
    assert "wrong length" in hits


def test_find_shift_cues_returns_empty_for_missing_text():
    hits, pos = pf.find_shift_cues("")
    assert hits == []
    assert pos is None
