from types import SimpleNamespace

import numpy as np
import pytest

import src.inference.utils.math_pass_utils as mp


def test_extract_blocks_and_valid_structure():
    text = "<think> t </think> middle <answer> a </answer>"
    think, ans = mp.extract_blocks(text)
    assert think == "t" and ans == "a"
    assert mp.valid_tag_structure(text) is True
    assert mp.valid_tag_structure("<think></think>") is False


def test_canon_math_normalizations():
    assert mp.canon_math(None) is None
    raw = r"\frac{ 1 }{ 2 } -- \pi , [a]"
    out = mp.canon_math(raw)
    assert out == "1/2-pia"
    assert mp.canon_math("+--1//2") == "-1/2"


def test_compute_math_reconsideration_info_marks_injected(monkeypatch):
    meta = mp.MathPassMeta(
        problem="prob",
        canon_gold=None,
        injected_cue=True,
        prev_output=None,
        cue_prefix_str="cue: ",
        stop_reason_think=None,
        stop_reason_answer=None,
    )
    monkeypatch.setattr(mp, "find_markers_and_context", lambda *a, **k: (["m1"], 1, "ctx", "ex"))
    info = mp._compute_math_reconsideration_info("think", meta, tokens_think=5)
    assert info.markers[0] == "injected_cue"
    assert info.t_cue == 0


def test_summarize_math_entropies_with_cue():
    stats = mp.MathTokenStats(tokens_think=3, tokens_answer=2, tokens_total=5)
    summary = mp._summarize_math_entropies(
        tok_ents_all=[0.1, 0.2, 0.3, 0.4, 0.5],
        ent_think=[0.1, 0.2, 0.3],
        ent_answer=[0.4, 0.5],
        stats=stats,
        t_cue=2,
    )
    assert summary.overall == np.mean([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary.reconsider_full == np.mean([0.3, 0.4, 0.5])
    assert summary.reconsider_think == np.mean([0.3])


def test_build_math_pass_meta_positional_and_kwargs():
    meta_kw = mp.build_math_pass_meta(
        problem="p",
        canon_gold="1",
        injected_cue=False,
        prev_output=None,
        cue_prefix_str="",
        stop_reason_think=None,
        stop_reason_answer=None,
    )
    meta_pos = mp.build_math_pass_meta("p", "1", False, None, "", None, None)
    assert meta_kw.problem == meta_pos.problem == "p"
    assert meta_pos.injected_cue is False
    try:
        mp.build_math_pass_meta("p", "1")
    except TypeError:
        pass
    else:
        assert False, "expected TypeError for wrong arg count"


def test_pack_math_pass_result_builds_fields(monkeypatch):
    meta = mp.MathPassMeta(
        problem="prob",
        canon_gold="1",
        injected_cue=False,
        prev_output=None,
        cue_prefix_str="",
        stop_reason_think=None,
        stop_reason_answer=None,
    )
    monkeypatch.setattr(mp, "find_markers_and_context", lambda *a, **k: (["m"], 0, "ctx", "ex"))
    res = mp.pack_math_pass_result(
        full_text="<think>x</think><answer>1</answer>",
        ent_think=[0.1],
        ent_answer=[0.2],
        meta=meta,
    )
    assert res["tokens_think"] == 1
    assert res["tokens_answer"] == 1
    assert res["entropy"] == np.mean([0.1, 0.2])


def test_valid_tag_structure_returns_false_on_missing_matches(monkeypatch):
    valid_text = "<think>t</think><answer>a</answer>"
    orig_re = mp.re

    def _run(skip_pattern: str):
        def fake_search(pattern, text):
            if skip_pattern in str(pattern):
                return None
            return orig_re.search(pattern, text)

        monkeypatch.setattr(
            mp,
            "re",
            SimpleNamespace(findall=orig_re.findall, search=fake_search),
        )
        return mp.valid_tag_structure(valid_text)

    for pattern in ("<think>", "</think>", "<answer>", "</answer>"):
        assert _run(pattern) is False


def test_build_math_pass_meta_rejects_mixed_args_kwargs():
    with pytest.raises(TypeError):
        mp.build_math_pass_meta(
            "prob",
            "canon",
            injected_cue=False,
            prev_output=None,
            cue_prefix_str="",
            stop_reason_think=None,
            stop_reason_answer=None,
        )
