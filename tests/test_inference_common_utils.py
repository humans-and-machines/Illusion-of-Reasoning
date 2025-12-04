#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
if not getattr(torch, "__file__", None):
    pytest.skip("torch stub detected; real torch required for these tests", allow_module_level=True)
# Require the tensor constructor and a usable SymFloat type; some minimal torch
# stubs expose a non-type SymFloat that breaks isinstance checks inside torch.
if not hasattr(torch, "tensor") or not isinstance(getattr(torch, "SymFloat", float), type):
    pytest.skip("torch stub lacks required tensor/SymFloat attributes", allow_module_level=True)
common = pytest.importorskip("src.inference.utils.common")


class _DummyTokenizerForGenerate:
    def __init__(self, pad_token_id=None, eos_token_id=7):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class _DummyTokenizerForDecode:
    def __init__(self):
        self.decoded = []

    def decode(self, ids, skip_special_tokens=True):
        # Represent the token ids as a simple joined string for assertions.
        text = ",".join(str(int(x)) for x in ids.tolist())
        self.decoded.append(text)
        return text


class _DummyTokenizerForStops:
    def encode(self, text, add_special_tokens=False):
        # Map each character to its ordinal for stable ids.
        return [ord(ch) for ch in text]


def test_build_generate_kwargs_greedy_and_sampling_modes():
    kwargs_greedy = common.build_generate_kwargs(
        cap=10,
        pad_token_id=3,
        eos_ids=[9],
        entropy_mode="none",
        temperature=0.0,
        top_p=0.95,
    )
    assert kwargs_greedy["max_new_tokens"] == 10
    assert kwargs_greedy["pad_token_id"] == 3
    assert kwargs_greedy["eos_token_id"] == [9]
    assert kwargs_greedy["do_sample"] is False
    assert kwargs_greedy["output_scores"] is False
    assert "temperature" not in kwargs_greedy
    assert "top_p" not in kwargs_greedy

    kwargs_sample = common.build_generate_kwargs(
        cap=5,
        pad_token_id=1,
        eos_ids=[2, 3],
        entropy_mode="reconsider",
        temperature=0.7,
        top_p=0.8,
    )
    assert kwargs_sample["do_sample"] is True
    assert kwargs_sample["output_scores"] is True
    assert kwargs_sample["temperature"] == pytest.approx(0.7)
    assert kwargs_sample["top_p"] == pytest.approx(0.8)


def test_make_generate_kwargs_for_cap_uses_tokenizer_pad_or_eos():
    tok = _DummyTokenizerForGenerate(pad_token_id=11, eos_token_id=22)
    kwargs = common.make_generate_kwargs_for_cap(
        cap=3,
        tokenizer=tok,
        eos_ids=[99],
        entropy_mode="none",
        temperature=None,
        top_p=None,
    )
    assert kwargs["pad_token_id"] == 11

    tok2 = _DummyTokenizerForGenerate(pad_token_id=None, eos_token_id=22)
    kwargs2 = common.make_generate_kwargs_for_cap(
        cap=3,
        tokenizer=tok2,
        eos_ids=[99],
        entropy_mode="none",
        temperature=None,
        top_p=None,
    )
    assert kwargs2["pad_token_id"] == 22


def test_build_second_pass_think_prefixes_aligns_examples_and_rows():
    base2_per_ex = ["BASE0 ", "BASE1 "]
    pre1_think = ["row0", "row1", "row2"]
    row_to_ex_idx = [0, 0, 1]
    cue_str = "CUE "

    result = common.build_second_pass_think_prefixes(
        base2_per_ex=base2_per_ex,
        pre1_think=pre1_think,
        row_to_ex_idx=row_to_ex_idx,
        cue_str=cue_str,
    )
    assert result == [
        "BASE0 <think>\nCUE ",
        "BASE0 <think>\nCUE ",
        "BASE1 <think>\nCUE ",
    ]


def test_build_extra_pass_results_for_cues_respects_flags_and_names():
    def pack_result(cue, outputs):
        return {"cue": cue, "marker": outputs}

    outputs = object()
    extra = [("c1", outputs), ("c2", outputs)]

    res = common.build_extra_pass_results_for_cues(
        two_pass=True,
        extra_passes=extra,
        pack_result_for_extra=pack_result,
        names=("a", "b"),
    )
    assert set(res.keys()) == {"a", "b"}
    assert res["a"]["cue"] == "c1"
    assert res["b"]["cue"] == "c2"

    res_disabled = common.build_extra_pass_results_for_cues(
        two_pass=False,
        extra_passes=extra,
        pack_result_for_extra=pack_result,
    )
    assert res_disabled == {}


def test_empty_pass_outputs_shapes_match_requested_rows():
    outputs = common.empty_pass_outputs(3)
    assert outputs.full_texts == ["", "", ""]
    assert len(outputs.ent_think) == 3
    assert all(isinstance(row, list) and not row for row in outputs.ent_think)
    assert len(outputs.ent_answer) == 3
    assert outputs.stop_reason_think == ["", "", ""]
    assert outputs.stop_reason_answer == ["", "", ""]


def test_decode_generated_row_uses_input_lengths_and_tokenizer():
    seqs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    input_lengths = torch.tensor([2, 3])
    tok = _DummyTokenizerForDecode()

    gen_ids, text, start_idx = common.decode_generated_row(
        tok,
        seqs,
        input_lengths,
        1,
        skip_special_tokens=True,
    )

    assert start_idx == 3
    assert gen_ids.tolist() == [8]
    assert text == "8"


@pytest.mark.parametrize(
    "found_stop,has_eos,hit_max,expected",
    [
        (True, True, True, "stop_token"),
        (False, True, True, "eos"),
        (False, False, True, "max_new_tokens"),
        (False, False, False, "other"),
    ],
)
def test_classify_stop_reason_priority(found_stop, has_eos, hit_max, expected):
    assert common.classify_stop_reason(found_stop, has_eos, hit_max) == expected


def test_stop_on_substrings_matches_suffix_and_reports_has_stops():
    tokenizer = _DummyTokenizerForStops()
    crit = common.StopOnSubstrings(tokenizer, ["Hi"])
    assert crit.has_stops() is True

    # ord('H') == 72, ord('i') == 105
    tokens = torch.tensor([[10, 72, 105], [1, 2, 3]], dtype=torch.long)
    logits = torch.zeros_like(tokens, dtype=torch.float)

    assert crit(tokens, logits) is True

    crit_empty = common.StopOnSubstrings(tokenizer, [])
    assert crit_empty.has_stops() is False
    assert crit_empty(tokens, logits) is False
