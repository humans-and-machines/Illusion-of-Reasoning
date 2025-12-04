#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
math_core = pytest.importorskip("src.inference.domains.math.math_core")


def _make_config(num_samples: int = 3) -> math_core.MathInferenceConfig:
    return math_core.MathInferenceConfig(
        split_name="test",
        output_dir=".",
        step=0,
        batch_size=1,
        num_samples=num_samples,
        temperature=0.0,
        top_p=0.95,
        entropy_mode="none",
        eos_ids=None,
        two_pass=True,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=1,
        think_cap=5,
        answer_cap=5,
    )


def test_build_work_items_for_slice_respects_existing_samples():
    examples = [
        {"problem": "p1", "answer": "a1"},
        {"problem": "p2", "answer": "a2"},
        {"question": "no-problem-field"},
    ]
    existing_samples = {"p1": {0, 2}}
    cfg = _make_config(num_samples=3)

    work_items = math_core._build_work_items_for_slice(  # type: ignore[attr-defined]
        examples_slice=examples,
        existing_samples=existing_samples,
        config=cfg,
    )

    assert len(work_items) == 2
    w0 = next(item for item in work_items if item["_normalized_problem"] == "p1")
    assert w0["_todo_samples"] == [1]
    w1 = next(item for item in work_items if item["_normalized_problem"] == "p2")
    assert w1["_todo_samples"] == [0, 1, 2]


def test_build_first_pass_choice_prefers_new_or_existing_samples():
    layout = math_core.BatchLayout(
        work_items=[
            {"_normalized_problem": "p1"},
            {"_normalized_problem": "p2"},
        ],
        row_to_ex_idx=[0, 0, 1],
        row_target_sample_idx=[0, 1, 0],
    )
    pass1_full_texts = ["p1-s0", "p1-s1", "p2-s0"]
    existing_state = math_core.ExistingPassState(
        existing_samples={"p1": {0}},
        existing_pass1={("p1", 0): "disk-prev"},
    )
    cfg = _make_config(num_samples=2)

    choice = math_core._build_first_pass_choice(  # type: ignore[attr-defined]
        layout=layout,
        pass1_full_texts=pass1_full_texts,
        existing_state=existing_state,
        config=cfg,
    )

    assert choice[0] == "p1-s1"
    assert choice[1] == "p2-s0"

    cfg_disabled = _make_config(num_samples=2)
    cfg_disabled.two_pass_cfg.enabled = False
    choice_disabled = math_core._build_first_pass_choice(  # type: ignore[attr-defined]
        layout=layout,
        pass1_full_texts=pass1_full_texts,
        existing_state=existing_state,
        config=cfg_disabled,
    )
    assert choice_disabled == ["", ""]


class _DummyTokenizerForPass2:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}:{m['content']}" for m in messages]
        text = "|".join(parts)
        if add_generation_prompt:
            text += "|GEN"
        return text


def test_build_second_pass_base_prompts_uses_chat_template():
    tokenizer = _DummyTokenizerForPass2()
    work_items = [
        {"_normalized_problem": "p1"},
        {"_normalized_problem": "p2"},
    ]
    firstpass = ["prev1", "prev2"]
    phrase = "cue"

    prompts = math_core._build_second_pass_base_prompts(  # type: ignore[attr-defined]
        tokenizer=tokenizer,
        work_items=work_items,
        firstpass_choice_text_per_ex=firstpass,
        phrase=phrase,
    )
    assert len(prompts) == 2
    assert "system" in prompts[0]
    assert "user:Problem: p1" in prompts[0]
    assert "assistant:prev1" in prompts[0]


def test_build_work_items_skips_when_no_missing_samples():
    class AlwaysContains:
        def __len__(self):
            return 0  # below num_samples but still claims to contain items

        def __contains__(self, item):
            return True

    cfg = _make_config(num_samples=1)
    examples = [{"problem": "p1", "answer": "a1"}]
    existing = {"p1": AlwaysContains()}
    work_items = math_core._build_work_items_for_slice(  # type: ignore[attr-defined]
        examples_slice=examples,
        existing_samples=existing,
        config=cfg,
    )
    assert work_items == []


def test_math_inference_config_properties_and_errors():
    cfg = math_core.MathInferenceConfig(
        split_name="dev",
        output_dir="out",
        step=7,
        batch_size=2,
        num_samples=3,
        temperature=0.9,
        top_p=0.8,
        entropy_mode="none",
        eos_ids=[1, 2],
        two_pass=True,
        second_pass_phrase="p",
        second_pass_use_sample_idx=2,
        think_cap=10,
        answer_cap=11,
    )
    assert cfg.batch_size == 2
    assert cfg.num_samples == 3
    assert cfg.temperature == 0.9
    assert cfg.top_p == 0.8
    assert cfg.entropy_mode == "none"
    assert cfg.eos_ids == [1, 2]
    assert cfg.two_pass is True
    assert cfg.second_pass_phrase == "p"
    assert cfg.second_pass_use_sample_idx == 2
    assert cfg.think_cap == 10
    assert cfg.answer_cap == 11

    with pytest.raises(TypeError):
        math_core.MathInferenceConfig(split_name="s", output_dir="o", step=0, extra="bad")  # type: ignore[arg-type]


def test_chat_base_prompts_build_expected_strings():
    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "|".join(f"{m['role']}:{m['content']}" for m in msgs) + ("|GEN" if add_generation_prompt else "")

    tok = Tok()
    p1 = math_core.chat_base_for_pass1(tok, "prob")
    assert "system:" in p1 and "user:Problem: prob" in p1 and p1.endswith("GEN")

    p2 = math_core.chat_base_for_pass2(tok, "prob", "prev", "cue")
    assert "assistant:prev" in p2 and "user:cue" in p2 and p2.endswith("GEN")


def test_pack_pass_result_uses_helpers(monkeypatch):
    captured = {}
    monkeypatch.setattr(math_core, "build_math_pass_meta", lambda **k: captured.setdefault("meta", k))
    monkeypatch.setattr(math_core, "_pack_math_pass_result", lambda **k: captured.setdefault("pack", k))

    out = math_core._pack_pass_result(
        full_text="full",
        ent_think=[0.1],
        ent_answer=[0.2],
        meta_args=math_core.MathPassMetaArgs(
            problem="p",
            canon_gold="g",
            injected_cue=True,
            prev_output=None,
            cue_prefix_str="cue",
            stop_reason_think="s1",
            stop_reason_answer="s2",
        ),
    )
    assert captured["meta"]["problem"] == "p"
    assert captured["pack"]["full_text"] == "full"
    assert out == captured["pack"]


def test_run_pass1_generations_and_for_batch(monkeypatch):
    pre1_prefixes = []

    def fake_gen(batch_spec, context):
        pre1_prefixes.append(batch_spec.prefixes)
        if batch_spec.stop_strs == ["</answer>"]:
            return ["ans"], [[0.0]], None, None, ["stop2"]
        return ["think"], [[0.1]], None, None, ["stop1"]

    monkeypatch.setattr(math_core, "_gen_batch", fake_gen)

    cfg = _make_config(num_samples=1)
    tok = _DummyTokenizerForPass2()
    ctx = math_core.MathInferenceContext(tokenizer=tok, model=None, config=cfg)
    work_items = [{"_normalized_problem": "p", "_todo_samples": [0]}]
    outputs = math_core._run_pass1_generations(["base<think>"], ctx)  # type: ignore[attr-defined]
    assert outputs.full_texts == ["<think>think</think>\n<answer>ans</answer>"]
    assert outputs.stop_reason_think == ["stop1"]
    assert outputs.stop_reason_answer == ["stop2"]

    monkeypatch.setattr(math_core, "_run_pass1_generations", lambda pre, ctx: outputs)
    outs, layout, pre = math_core._run_pass1_for_batch(work_items, ctx)  # type: ignore[attr-defined]
    assert outs is outputs
    assert layout.row_to_ex_idx == [0]
    assert pre and pre[0].endswith("<think>\n")


def test_select_first_pass_choice_branches(monkeypatch):
    cfg = _make_config(num_samples=3)
    cfg.two_pass_cfg.use_sample_idx = 5  # force fallback to available existing sample
    layout = math_core.BatchLayout(
        work_items=[{"_normalized_problem": "p"}], row_to_ex_idx=[0], row_target_sample_idx=[1]
    )
    inputs = math_core.FirstPassChoiceInputs(
        layout=layout,
        existing_state=math_core.ExistingPassState(existing_samples={"p": {2}}, existing_pass1={("p", 2): "prev2"}),
        new_pass1_by_ex_and_sample={(0, 1): "new1"},
        pass1_full_texts=["fallback"],
        config=cfg,
    )
    choice = math_core._select_first_pass_choice("p", 0, inputs)  # type: ignore[attr-defined]
    assert choice == "prev2"

    inputs.existing_state = math_core.ExistingPassState(existing_samples={}, existing_pass1={})
    choice2 = math_core._select_first_pass_choice("p", 0, inputs)  # type: ignore[attr-defined]
    assert choice2 == "new1"

    inputs.new_pass1_by_ex_and_sample = {}
    choice3 = math_core._select_first_pass_choice("p", 0, inputs)  # type: ignore[attr-defined]
    assert choice3 == "fallback"


def test_run_pass2_for_batch_disabled_and_enabled(monkeypatch):
    cfg = _make_config(num_samples=1)
    tok = _DummyTokenizerForPass2()
    ctx = math_core.MathInferenceContext(tokenizer=tok, model=None, config=cfg)
    layout = math_core.BatchLayout(
        work_items=[{"_normalized_problem": "p"}],
        row_to_ex_idx=[0],
        row_target_sample_idx=[0],
    )
    inputs = math_core.SecondPassInputs(
        layout=layout, pre1_think=["pre"], firstpass_choice_text_per_ex=["prev"], cue_str="CUE:"
    )

    cfg.two_pass_cfg.enabled = False
    out_disabled = math_core._run_pass2_for_batch(context=ctx, second_pass_inputs=inputs)  # type: ignore[attr-defined]
    assert out_disabled.full_texts == [""]

    cfg.two_pass_cfg.enabled = True
    monkeypatch.setattr(
        math_core,
        "_run_second_pass_generations",
        lambda context, pre2_think: (["t2"], [[0.1]], ["st"], ["a2"], [[0.2]], ["sa"]),
    )
    out_enabled = math_core._run_pass2_for_batch(context=ctx, second_pass_inputs=inputs)  # type: ignore[attr-defined]
    assert out_enabled.full_texts[0].startswith("<think>CUE:t2</think>")


def test_norm_fields_extracts_boxed_and_wrappers(monkeypatch):
    orig_extract = math_core.extract_problem_and_answer
    monkeypatch.setattr(math_core, "extract_problem_and_answer", lambda ex: ("p", "\\boxed{7}"))
    prob, gold = math_core._norm_fields({"other_key": "value"})
    assert gold == "7"
    monkeypatch.setattr(math_core, "extract_problem_and_answer", orig_extract)
    prob2, gold2 = math_core._norm_fields({"question": "q", "final_answer": "ans"})
    assert prob2 and gold2 == "ans"

    # Wrapper functions delegate through import_module.
    fake_mod = SimpleNamespace(
        build_extra_pass_results_for_row=lambda **k: {"ok": True},
        write_results_for_batch=lambda **k: k,
        compute_second_pass_outputs=lambda **k: ("p2", "extra", ["stop"]),
        run_inference_batch=lambda **k: k,
        run_inference_on_split=lambda **k: k,
        load_math500=lambda **k: {"loaded": True},
    )
    monkeypatch.setattr(math_core, "import_module", lambda name: fake_mod)

    ctx = math_core.BatchWriteContext(
        outpath="p",
        config=_make_config(),
        cue_strs=[],
        existing_state=math_core.ExistingPassState(defaultdict(set), {}),
        firstpass_choice_text_per_ex=[],
    )
    layout = math_core.BatchLayout(work_items=[], row_to_ex_idx=[], row_target_sample_idx=[])
    outputs = math_core.TwoPassBatchOutputs(pass1=math_core.empty_pass_outputs(0), pass2=None)

    assert math_core._build_extra_pass_results_for_row(
        row_index=0,
        row_context=math_core.ExtraPassRowContext(
            prob="p",
            canon_gold=None,
            layout=layout,
            context=ctx,
            extra_passes=None,
            pass1_result={},
        ),
    )["ok"]  # type: ignore[attr-defined]
    assert math_core._write_results_for_batch(layout=layout, outputs=outputs, context=ctx) is None  # type: ignore[attr-defined]
    assert math_core._compute_second_pass_outputs(
        context=None, layout=layout, pre1_think=[], firstpass_choice_text_per_ex=[]
    ) == ("p2", "extra", ["stop"])  # type: ignore[attr-defined]
    assert math_core._run_inference_batch(slice_ds=None, context=None, outpath="p", existing_state=None) is None  # type: ignore[attr-defined]
    assert math_core.run_inference_on_split(examples=None, tokenizer=None, model=None, config=None) is None
    assert math_core.load_math500(cache_dir="c", split="s", seed=0) == {"loaded": True}


def test_math_namespace_lazy_import(monkeypatch):
    import src.inference.domains.math as math_pkg

    captured = {}

    def fake_import(name):
        captured["name"] = name
        return "imported"

    monkeypatch.setattr(math_pkg, "import_module", fake_import)
    out = math_pkg.math_core_runner
    assert out == "imported"
    assert captured["name"].endswith("math_core_runner")
