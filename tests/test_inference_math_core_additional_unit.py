#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest


MODULE_PATH = "src.inference.domains.math.math_core"


@pytest.fixture
def math_core_module():
    sys.modules.pop(MODULE_PATH, None)
    return importlib.import_module(MODULE_PATH)


def test_import_survives_missing_torch(monkeypatch):
    import src.inference.utils.common as common

    def raise_import_error(_):
        raise ImportError("no torch for test")

    monkeypatch.setattr(common, "require_torch", raise_import_error)
    sys.modules.pop(MODULE_PATH, None)
    module = importlib.import_module(MODULE_PATH)
    assert module.SYSTEM_PROMPT  # import succeeded even when torch is unavailable
    sys.modules.pop(MODULE_PATH, None)


def test_math_inference_config_properties_and_error(math_core_module):
    cfg = math_core_module.MathInferenceConfig(
        split_name="dev",
        output_dir="out",
        step=3,
        batch_size=2,
        num_samples=2,
        temperature=0.5,
        top_p=0.4,
        entropy_mode="mode",
        eos_ids=[9],
        two_pass=True,
        second_pass_phrase=" phrase ",
        second_pass_use_sample_idx=3,
        think_cap=9,
        answer_cap=4,
    )
    assert cfg.batch_size == 2
    assert cfg.num_samples == 2
    assert cfg.temperature == 0.5
    assert cfg.top_p == 0.4
    assert cfg.entropy_mode == "mode"
    assert cfg.eos_ids == [9]
    assert cfg.two_pass is True
    assert cfg.second_pass_phrase == " phrase "
    assert cfg.second_pass_use_sample_idx == 3
    assert cfg.think_cap == 9
    assert cfg.answer_cap == 4

    with pytest.raises(TypeError):
        math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0, unexpected=True)  # type: ignore[arg-type]


def test_scan_existing_results_delegates(monkeypatch, math_core_module):
    captured = {}
    monkeypatch.setattr(
        math_core_module,
        "scan_existing_pass1_results",
        lambda path: captured.setdefault("path", path) or ("seen", {"p": "t"}),  # pragma: no cover - replaced below
    )

    def fake_scan(path):
        captured["path"] = path
        return "seen", {"p": "t"}

    monkeypatch.setattr(math_core_module, "scan_existing_pass1_results", fake_scan)

    existing_samples, existing_pass1 = math_core_module._scan_existing_results("results.jsonl")  # type: ignore[attr-defined]
    assert captured["path"] == "results.jsonl"
    assert existing_samples == "seen"
    assert existing_pass1 == {"p": "t"}


def test_build_work_items_for_slice_handles_existing_and_empty(math_core_module):
    cfg = math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0, num_samples=2)
    examples = [
        {"problem": "done", "answer": "a1"},
        {"problem": "todo", "answer": "a2"},
        {"question": "ignored-without-answer"},
    ]
    existing = {"done": {0, 1}}

    work_items = math_core_module._build_work_items_for_slice(examples, existing, cfg)  # type: ignore[attr-defined]

    assert len(work_items) == 1
    assert work_items[0]["_normalized_problem"] == "todo"
    assert work_items[0]["_todo_samples"] == [0, 1]


def test_run_pass1_generations_and_layout(monkeypatch, math_core_module):
    calls = []

    def fake_gen(batch_spec, context):
        calls.append(batch_spec)
        if batch_spec.stop_strs == ["</think>"]:
            return ["think"], [[0.1]], None, None, ["stop-think"]
        return ["answer"], [[0.2]], None, None, ["stop-answer"]

    monkeypatch.setattr(math_core_module, "_gen_batch", fake_gen)

    cfg = math_core_module.MathInferenceConfig(
        split_name="s", output_dir="o", step=0, num_samples=1, think_cap=3, answer_cap=2
    )
    tok = SimpleNamespace(apply_chat_template=lambda msgs, tokenize=False, add_generation_prompt=False: "base|")
    ctx = math_core_module.MathInferenceContext(tokenizer=tok, model=None, config=cfg)

    outputs = math_core_module._run_pass1_generations(["base<think>\n"], ctx)  # type: ignore[attr-defined]
    assert outputs.full_texts == ["<think>think</think>\n<answer>answer</answer>"]
    assert calls[0].stop_strs == ["</think>"]
    assert calls[1].stop_strs == ["</answer>"]

    monkeypatch.setattr(math_core_module, "_run_pass1_generations", lambda pre, ctx: outputs)
    work_items = [{"_normalized_problem": "prob", "_todo_samples": [0]}]
    outs, layout, pre1 = math_core_module._run_pass1_for_batch(work_items, ctx)  # type: ignore[attr-defined]
    assert outs.full_texts == outputs.full_texts
    assert layout.row_to_ex_idx == [0]
    assert pre1[0].endswith("<think>\n")


def test_select_first_pass_choice_paths(math_core_module):
    cfg = math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0, num_samples=3)
    layout = math_core_module.BatchLayout(
        work_items=[{"_normalized_problem": "p"}], row_to_ex_idx=[0], row_target_sample_idx=[0]
    )

    inputs_disk = math_core_module.FirstPassChoiceInputs(
        layout=layout,
        existing_state=math_core_module.ExistingPassState(
            existing_samples={"p": {2}}, existing_pass1={("p", 2): "from-disk"}
        ),
        new_pass1_by_ex_and_sample={},
        pass1_full_texts=["fallback0"],
        config=cfg,
    )
    assert math_core_module._select_first_pass_choice("p", 0, inputs_disk) == "from-disk"  # type: ignore[attr-defined]

    inputs_fresh = math_core_module.FirstPassChoiceInputs(
        layout=layout,
        existing_state=math_core_module.ExistingPassState(existing_samples={}, existing_pass1={}),
        new_pass1_by_ex_and_sample={},
        pass1_full_texts=["row0-text"],
        config=cfg,
    )
    assert math_core_module._select_first_pass_choice("p", 0, inputs_fresh) == "row0-text"  # type: ignore[attr-defined]

    layout_mismatch = math_core_module.BatchLayout(
        work_items=[{"_normalized_problem": "p"}], row_to_ex_idx=[1], row_target_sample_idx=[0]
    )
    inputs_empty = math_core_module.FirstPassChoiceInputs(
        layout=layout_mismatch,
        existing_state=math_core_module.ExistingPassState(existing_samples={}, existing_pass1={}),
        new_pass1_by_ex_and_sample={},
        pass1_full_texts=["unmatched"],
        config=cfg,
    )
    assert math_core_module._select_first_pass_choice("p", 0, inputs_empty) == ""  # type: ignore[attr-defined]


def test_run_second_pass_generations(monkeypatch, math_core_module):
    order = []

    def fake_gen(batch_spec, context):
        order.append(batch_spec.stop_strs[0])
        if batch_spec.stop_strs == ["</think>"]:
            return ["t2"], [[0.1]], None, None, ["stop2-think"]
        return ["a2"], [[0.2]], None, None, ["stop2-answer"]

    monkeypatch.setattr(math_core_module, "_gen_batch", fake_gen)

    cfg = math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0, think_cap=2, answer_cap=2)
    ctx = math_core_module.MathInferenceContext(tokenizer=None, model=None, config=cfg)
    outputs = math_core_module._run_second_pass_generations(context=ctx, pre2_think=["prefix"])  # type: ignore[attr-defined]

    assert order == ["</think>", "</answer>"]
    think_texts, think_ents, think_stop, answer_texts, answer_ents, answer_stop = outputs
    assert think_texts == ["t2"]
    assert answer_texts == ["a2"]
    assert think_stop == ["stop2-think"]
    assert answer_stop == ["stop2-answer"]


def test_run_pass2_for_batch_branches(monkeypatch, math_core_module):
    cfg = math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0, num_samples=1, two_pass=False)
    tok = SimpleNamespace(
        apply_chat_template=lambda msgs, tokenize=False, add_generation_prompt=False: "chat|" + str(len(msgs))
    )
    ctx = math_core_module.MathInferenceContext(tokenizer=tok, model=None, config=cfg)
    layout = math_core_module.BatchLayout(
        work_items=[{"_normalized_problem": "p"}], row_to_ex_idx=[0], row_target_sample_idx=[0]
    )
    inputs = math_core_module.SecondPassInputs(
        layout=layout, pre1_think=["pre"], firstpass_choice_text_per_ex=["chosen"], cue_str="CUE:"
    )

    disabled = math_core_module._run_pass2_for_batch(context=ctx, second_pass_inputs=inputs)  # type: ignore[attr-defined]
    assert disabled.full_texts == [""]

    cfg.two_pass_cfg.enabled = True
    monkeypatch.setattr(math_core_module, "build_second_pass_think_prefixes", lambda **_: ["filled"])
    monkeypatch.setattr(
        math_core_module,
        "_run_second_pass_generations",
        lambda context, pre2_think: (["t"], [[0.1]], ["st"], ["a"], [[0.2]], ["sa"]),
    )
    enabled = math_core_module._run_pass2_for_batch(context=ctx, second_pass_inputs=inputs)  # type: ignore[attr-defined]
    assert enabled.full_texts == ["<think>CUE:t</think>\n<answer>a</answer>"]
    assert enabled.stop_reason_think == ["st"]
    assert enabled.stop_reason_answer == ["sa"]


def test_norm_fields_unboxes_parentheses(monkeypatch, math_core_module):
    monkeypatch.setattr(math_core_module, "extract_problem_and_answer", lambda ex: ("p", "\\boxed(123)"))
    problem, gold = math_core_module._norm_fields({"other": "field"})
    assert problem == "p"
    assert gold == "123"


def test_wrapper_delegations(monkeypatch, math_core_module):
    captured = {}

    def record_write(**k):
        captured["write"] = k

    def record_batch(**k):
        captured["batch"] = k
        return {"batch": True, **k}

    def record_split(**k):
        captured["split"] = k
        return {"split": True, **k}

    fake_runner = SimpleNamespace(
        build_extra_pass_results_for_row=lambda **k: {"ok": True, **k},
        write_results_for_batch=record_write,
        compute_second_pass_outputs=lambda **k: ("pass2", "extra", ["stop"]),
        run_inference_batch=record_batch,
        run_inference_on_split=record_split,
        load_math500=lambda **k: {"loaded": k},
    )
    monkeypatch.setattr(math_core_module, "import_module", lambda name: fake_runner)

    cfg = math_core_module.MathInferenceConfig(split_name="s", output_dir="o", step=0)
    ctx = math_core_module.BatchWriteContext(
        outpath="out.jsonl",
        config=cfg,
        cue_strs=[],
        existing_state=math_core_module.ExistingPassState(defaultdict(set), {}),
        firstpass_choice_text_per_ex=[],
    )
    layout = math_core_module.BatchLayout(work_items=[], row_to_ex_idx=[], row_target_sample_idx=[])
    outputs = math_core_module.TwoPassBatchOutputs(pass1=math_core_module.empty_pass_outputs(0), pass2=None)

    assert math_core_module._build_extra_pass_results_for_row(  # type: ignore[attr-defined]
        row_index=0,
        row_context=math_core_module.ExtraPassRowContext(
            prob="p",
            canon_gold=None,
            layout=layout,
            context=ctx,
            extra_passes=None,
            pass1_result={},
        ),
    )["ok"]
    math_core_module._write_results_for_batch(layout=layout, outputs=outputs, context=ctx)  # type: ignore[attr-defined]
    assert captured["write"]["layout"] == layout
    assert math_core_module._compute_second_pass_outputs(
        context=None, layout=layout, pre1_think=[], firstpass_choice_text_per_ex=[]
    ) == ("pass2", "extra", ["stop"])  # type: ignore[attr-defined]
    math_core_module._run_inference_batch(slice_ds=None, context=None, outpath="p", existing_state=None)  # type: ignore[attr-defined]
    assert captured["batch"]["outpath"] == "p"
    math_core_module.run_inference_on_split(examples=None, tokenizer=None, model=None, config=None)
    assert captured["split"]["examples"] is None
    assert math_core_module.load_math500(cache_dir="c", split="s", seed=0) == {
        "loaded": {"cache_dir": "c", "split": "s", "seed": 0, "dataset_path": None}
    }
