#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import types
from types import ModuleType, SimpleNamespace


# Minimal transformers/torch stubs for import-time wiring.
_ORIG_TRANSFORMERS_MOD = sys.modules.get("transformers")
transformers_stub = ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object
transformers_stub.AutoTokenizer = object
transformers_stub.StoppingCriteriaList = type("StoppingCriteriaList", (), {})
transformers_stub.StoppingCriteria = type("StoppingCriteria", (), {})
sys.modules["transformers"] = transformers_stub

import src.inference.utils.common as common_utils  # noqa: E402


_ORIG_REQUIRE_TORCH = common_utils.require_torch
_ORIG_REQUIRE_TRANSFORMERS = common_utils.require_transformers

common_utils.require_transformers = lambda caller: transformers_stub  # type: ignore[assignment]
common_utils.require_torch = lambda caller: types.SimpleNamespace(inference_mode=lambda: None)  # type: ignore[assignment]

# Reload to ensure the stubs are used.
sys.modules.pop("src.inference.domains.carpark.carpark_solver", None)

import src.inference.domains.carpark.carpark_solver as solver  # noqa: E402


def teardown_module(_module):
    """Restore shared require helpers to avoid leaking stubs to other tests."""
    common_utils.require_torch = _ORIG_REQUIRE_TORCH  # type: ignore[assignment]
    common_utils.require_transformers = _ORIG_REQUIRE_TRANSFORMERS  # type: ignore[assignment]
    if _ORIG_TRANSFORMERS_MOD is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = _ORIG_TRANSFORMERS_MOD


def test_chat_base_for_pass2_inserts_system():
    captured = {}

    class DummyTokenizer:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            captured["msgs"] = msgs
            return "rendered"

    tok = DummyTokenizer()
    out = solver.chat_base_for_pass2_from_messages(
        tok,
        messages=[{"role": "user", "content": "hi"}],
        prev_output="prev",
        cue=" cue ",
    )
    assert out == "rendered"
    roles = [m["role"] for m in captured["msgs"]]
    assert roles[0] == "system" and roles[-1] == "user"
    assert captured["msgs"][-1]["content"] == " cue "


def test_chat_base_for_pass2_respects_existing_system():
    captured = {}

    class DummyTokenizer:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            captured["msgs"] = msgs
            return "rendered"

    tok = DummyTokenizer()
    out = solver.chat_base_for_pass2_from_messages(
        tok,
        messages=[
            {"role": "system", "content": "keep-me"},
            {"role": "user", "content": "hi"},
        ],
        prev_output="prev-output",
        cue="cue text",
    )
    assert out == "rendered"
    assert [m["role"] for m in captured["msgs"]].count("system") == 1
    assert captured["msgs"][-2:] == [
        {"role": "assistant", "content": "prev-output"},
        {"role": "user", "content": "cue text"},
    ]


def test_find_reconsider_markers_and_injected():
    text = "prefix wait, reconsider"
    markers = solver._find_reconsider_markers(text, {"injected_cue": True, "cue_prefix_str": "prefix "})
    assert markers[0] == "injected_cue" and any("wait" in m for m in markers)


def test_find_reconsider_markers_skips_prefix_and_breaks():
    markers = solver._find_reconsider_markers(
        "prefix step-by-step then wait",
        {"injected_cue": True, "cue_prefix_str": "prefix "},
    )
    assert markers[:2] == ["injected_cue", "step_by_step"]
    assert len(markers) == 2  # first pattern match stops the loop


def test_build_second_pass_helpers(monkeypatch):
    prompts_seen = []

    def fake_chat_base(tok, msgs, prev_output, cue):
        prompts_seen.append((prev_output, cue))
        return f"prompt-{cue}"

    monkeypatch.setattr(solver, "chat_base_for_pass2_from_messages", fake_chat_base)
    ctx = SimpleNamespace(
        cue_str=" cue ",
        inference=SimpleNamespace(tokenizer="tok"),
        batch_items=[{"messages": ["m"]}],
        firstpass_choice=["prev"],
        num_samples=2,
    )
    prompts = solver._build_second_pass_prompts(ctx)
    assert prompts == ["prompt-cue"]

    prefixes = solver._build_second_pass_think_prefixes(prompts, ctx)
    assert prefixes == ["prompt-cue<think>\n cue ", "prompt-cue<think>\n cue "]


def test_build_second_pass_prompts_strip_and_prefixes(monkeypatch):
    seen = []

    def fake_chat_base(tok, messages, prev_output, cue):
        seen.append(cue)
        return f"prompt-{prev_output}-{cue}"

    monkeypatch.setattr(solver, "chat_base_for_pass2_from_messages", fake_chat_base)
    ctx = SimpleNamespace(
        cue_str="  cue  ",
        inference=SimpleNamespace(tokenizer="tok"),
        batch_items=[{"messages": ["m1"]}, {"messages": ["m2"]}],
        firstpass_choice=["p1", "p2"],
        num_samples=2,
    )
    prompts = solver._build_second_pass_prompts(ctx)
    assert prompts == ["prompt-p1-cue", "prompt-p2-cue"]
    assert seen == ["cue", "cue"]

    prefixes = solver._build_second_pass_think_prefixes(prompts, ctx)
    assert len(prefixes) == 4
    assert prefixes[0].startswith("prompt-p1-cue<think>\n  cue  ")
    assert prefixes[2].startswith("prompt-p2-cue<think>\n  cue  ")


def test_run_pass2_for_batch_uses_gen(monkeypatch):
    calls = {"gen": []}

    def fake_gen(prefixes, cap, stop_strings, context):
        calls["gen"].append((prefixes, stop_strings, cap))
        if len(calls["gen"]) == 1:
            return (["t"], [[0.1]], None, None, ["stop1"])
        return (["a"], [[0.2]], None, None, ["stop2"])

    monkeypatch.setattr(solver, "_gen_batch", fake_gen)
    ctx = SimpleNamespace(
        cue_str="cue ",
        inference=SimpleNamespace(
            config=SimpleNamespace(think_cap=1, answer_cap=1),
            tokenizer=SimpleNamespace(apply_chat_template=lambda msgs, **k: "prompt"),
        ),
        batch_items=[{"messages": []}],
        firstpass_choice=["prev"],
        num_samples=1,
    )
    out = solver._run_pass2_for_batch(ctx)
    assert out.full_texts[0].startswith("<think>cue t</think>")
    assert calls["gen"][0][1] == ["</think>"]
    assert calls["gen"][1][1] == ["</answer>"]


def test_run_pass2_for_batch_multi_sample(monkeypatch):
    calls = []

    def fake_gen(prefixes, cap, stop_strings, context):
        calls.append((prefixes, stop_strings, cap))
        if stop_strings == ["</think>"]:
            return (["t1", "t2"], [[0.1], [0.2]], None, None, ["st1", "st2"])
        return (["a1", "a2"], [[0.3], [0.4]], None, None, ["sa1", "sa2"])

    monkeypatch.setattr(solver, "_gen_batch", fake_gen)
    ctx = SimpleNamespace(
        cue_str="cue ",
        inference=SimpleNamespace(
            config=SimpleNamespace(think_cap=2, answer_cap=3),
            tokenizer=SimpleNamespace(apply_chat_template=lambda msgs, **k: "prompt"),
        ),
        batch_items=[{"messages": []}],
        firstpass_choice=["prev"],
        num_samples=2,
    )
    outputs = solver._run_pass2_for_batch(ctx)
    assert outputs.full_texts == [
        "<think>cue t1</think>\n<answer>a1</answer>",
        "<think>cue t2</think>\n<answer>a2</answer>",
    ]
    assert outputs.stop_reason_think == ["st1", "st2"]
    assert outputs.stop_reason_answer == ["sa1", "sa2"]
    assert calls[0][1] == ["</think>"] and calls[1][1] == ["</answer>"]


def test_compute_second_pass_outputs_branches(monkeypatch):
    monkeypatch.setattr(solver, "build_second_pass_cue_strings", lambda phrase: ["a ", "b "])
    fake_outputs = solver.PassOutputs(
        full_texts=["x"], ent_think=[[]], ent_answer=[[]], stop_reason_think=["s1"], stop_reason_answer=["s2"]
    )
    monkeypatch.setattr(solver, "_run_pass2_for_batch", lambda ctx: fake_outputs)
    ctx = SimpleNamespace(config=SimpleNamespace(two_pass=True, second_pass_phrase="p"))
    cue_strs, main_pass2, extra = solver._compute_second_pass_outputs_for_carpark(
        context=ctx,
        batch_items=[{"id": 1}],
        firstpass_choice=["p1"],
        num_samples=1,
    )
    assert cue_strs == ["a ", "b "]
    assert main_pass2 is fake_outputs
    assert extra == [("a ", fake_outputs)]

    ctx.config.two_pass = False
    cue_strs2, main2, extra2 = solver._compute_second_pass_outputs_for_carpark(
        context=ctx,
        batch_items=[{"id": 1}, {"id": 2}],
        firstpass_choice=["p1", "p2"],
        num_samples=2,
    )
    assert len(main2.full_texts) == 4  # empty_pass_outputs length
    assert extra2 == []


def test_build_row_for_sample_with_pass2_and_extras(monkeypatch):
    monkeypatch.setattr(
        solver,
        "_extract_blocks",
        lambda text: ("think", "CORRECT" if "pass2" in text else "WRONG" if "pass1" in text else "EXTRA"),
    )
    monkeypatch.setattr(solver, "_canon_rush_generic", lambda text: text.upper())
    monkeypatch.setattr(solver, "_is_valid_rush", lambda text: True)
    monkeypatch.setattr(solver, "rush_soft_match_reward", lambda pred, gold: (0.5, {"pred": pred, "gold": gold}))

    def fake_extra(two_pass, extra_passes, pack_result_for_extra):
        result = {}
        for idx, (cue, outputs) in enumerate(extra_passes or []):
            key = "pass2a" if idx == 0 else "pass2b"
            result[key] = pack_result_for_extra(cue, outputs)
        return result

    monkeypatch.setattr(solver, "build_extra_pass_results_for_cues", fake_extra)

    pass1 = solver.PassOutputs(
        full_texts=["pass1"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s1"],
        stop_reason_answer=["s2"],
    )
    pass2 = solver.PassOutputs(
        full_texts=["pass2"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s3"],
        stop_reason_answer=["s4"],
    )
    extra_outputs = solver.PassOutputs(
        full_texts=["extra"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s5"],
        stop_reason_answer=["s6"],
    )

    ctx = solver.SampleRowContext(
        example={"id": "ex1", "solution": "CORRECT", "messages": "problem"},
        results_ctx=solver.ResultsContext(
            outpath="out.jsonl",
            inference=solver.InferenceContext(
                tokenizer=None, model=None, config=SimpleNamespace(two_pass=True, step=1, split_name="spl")
            ),
            num_samples=1,
            cue_str="cue ",
            firstpass_choice=["pass1"],
            existing_by_example={"ex1": set()},
        ),
        batch_index=0,
        sample_idx=0,
        gold_set={"CORRECT"},
        passes=solver.SampleRowPasses(
            pass1=pass1,
            pass2=pass2,
            extra_passes=[("cueA", extra_outputs), ("cueB", extra_outputs)],
        ),
    )

    row = solver._build_row_for_sample(ctx)
    assert row["pass2"]["is_correct_pred"] is True
    assert row["pass2"]["improved_over_pass1"] is True
    assert "pass2c" in row  # added when extra passes exist
    assert row["problem"] == "problem"


def test_build_row_for_sample_with_real_extra_builder(monkeypatch):
    monkeypatch.setattr(
        solver, "_extract_blocks", lambda text: ("", "CORRECT" if "pass2" in text or "extraA" in text else "WRONG")
    )
    monkeypatch.setattr(solver, "_canon_rush_generic", lambda text: text.upper())
    monkeypatch.setattr(solver, "_is_valid_rush", lambda text: True)
    monkeypatch.setattr(solver, "rush_soft_match_reward", lambda pred, gold: (1.0 if pred == gold else 0.0, {}))

    pass1 = solver.PassOutputs(
        full_texts=["pass1"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s1"],
        stop_reason_answer=["s2"],
    )
    pass2 = solver.PassOutputs(
        full_texts=["pass2"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s3"],
        stop_reason_answer=["s4"],
    )
    extra_a = solver.PassOutputs(
        full_texts=["extraA"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s5"],
        stop_reason_answer=["s6"],
    )
    extra_b = solver.PassOutputs(
        full_texts=["extraB"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["s7"],
        stop_reason_answer=["s8"],
    )

    ctx = solver.SampleRowContext(
        example={"id": "ex1", "solution": "CORRECT", "messages": [{"content": "p1"}, {"content": "p2"}]},
        results_ctx=solver.ResultsContext(
            outpath="out.jsonl",
            inference=solver.InferenceContext(
                tokenizer=None,
                model=None,
                config=SimpleNamespace(two_pass=True, step=1, split_name="spl"),
            ),
            num_samples=1,
            cue_str="cue ",
            firstpass_choice=["pass1"],
            existing_by_example={"ex1": set()},
        ),
        batch_index=0,
        sample_idx=0,
        gold_set={"CORRECT"},
        passes=solver.SampleRowPasses(
            pass1=pass1,
            pass2=pass2,
            extra_passes=[("cueA", extra_a), ("cueB", extra_b)],
        ),
    )

    row = solver._build_row_for_sample(ctx)
    assert row["pass2"]["is_correct_pred"] is True and row["pass2"]["improved_over_pass1"] is True
    assert row["pass2a"]["is_correct_pred"] is True
    assert "pass2c" in row
    assert row["problem"] == "p1 p2"


def test_run_inference_on_split_core_skips_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(solver, "load_existing_example_index", lambda path: {})
    monkeypatch.setattr(solver, "build_batch_items_for_range", lambda **kwargs: [])
    called = {}
    monkeypatch.setattr(solver, "_run_pass1_for_batch", lambda *a, **k: called.setdefault("pass1", True))

    config = SimpleNamespace(
        output_dir=str(tmp_path),
        step=1,
        split_name="spl",
        batch_size=1,
        prompt_col="prompt",
        solution_col="solution",
        num_samples=1,
        two_pass=False,
    )
    context = solver.InferenceContext(tokenizer=None, model=None, config=config)
    solver._run_inference_on_split_core([], context)
    assert "pass1" not in called


def test_run_inference_on_split_core_continue_when_builder_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(solver, "load_existing_example_index", lambda path: {})
    build_calls = []
    monkeypatch.setattr(
        solver,
        "build_batch_items_for_range",
        lambda **kwargs: (build_calls.append(kwargs) or []),
    )
    called = {}
    monkeypatch.setattr(solver, "_run_pass1_for_batch", lambda *a, **k: called.setdefault("pass1", True))

    config = SimpleNamespace(
        output_dir=str(tmp_path),
        step=1,
        split_name="spl",
        batch_size=2,
        prompt_col="prompt",
        solution_col="solution",
        num_samples=1,
        two_pass=False,
    )
    context = solver.InferenceContext(tokenizer=None, model=None, config=config)

    solver._run_inference_on_split_core([{"id": "ex1"}], context)

    assert build_calls  # loop executed
    assert "pass1" not in called  # continued without processing batch


def test_run_inference_on_split_core_processes_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(solver, "load_existing_example_index", lambda path: {})
    build_calls = []

    def fake_build_batch_items_for_range(**kwargs):
        build_calls.append(kwargs)
        return [
            {"id": "ex1", "messages": [], "solution": "SOL", "missing_indices": [0]},
        ]

    monkeypatch.setattr(solver, "build_batch_items_for_range", fake_build_batch_items_for_range)
    pass1 = solver.PassOutputs(
        full_texts=["pass1"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["st"],
        stop_reason_answer=["sa"],
    )
    monkeypatch.setattr(solver, "_run_pass1_for_batch", lambda batch_items, context: (pass1, 1))
    monkeypatch.setattr(solver, "_build_first_pass_choice", lambda *a, **k: ["choice"])
    pass2 = solver.PassOutputs(
        full_texts=["pass2"],
        ent_think=[[]],
        ent_answer=[[]],
        stop_reason_think=["st2"],
        stop_reason_answer=["sa2"],
    )
    monkeypatch.setattr(solver, "_compute_second_pass_outputs_for_carpark", lambda **kwargs: (["cue "], pass2, []))

    captured = {}

    def fake_write_results_for_batch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(solver, "_write_results_for_batch", fake_write_results_for_batch)

    config = SimpleNamespace(
        output_dir=str(tmp_path),
        step=1,
        split_name="spl",
        batch_size=1,
        prompt_col="prompt",
        solution_col="solution",
        num_samples=1,
        two_pass=True,
    )
    context = solver.InferenceContext(tokenizer=None, model=None, config=config)
    solver._run_inference_on_split_core([{"id": "ex1"}], context)

    assert build_calls and captured.get("pass2") is pass2
    assert captured["results_ctx"].firstpass_choice == ["choice"]
