#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

import src.inference.domains.math.math_core_runner as runner
from src.inference.domains.math import math_core


def _make_config(two_pass: bool = True, num_samples: int = 1, batch_size: int = 1):
    return math_core.MathInferenceConfig(
        split_name="test",
        output_dir=".",
        step=1,
        batch_size=batch_size,
        num_samples=num_samples,
        temperature=0.0,
        top_p=0.95,
        entropy_mode="none",
        eos_ids=None,
        two_pass=two_pass,
        second_pass_phrase="cue1|||cue2",
        second_pass_use_sample_idx=0,
        think_cap=5,
        answer_cap=5,
    )


def _pass_outputs(text: str):
    return runner.PassOutputs(
        full_texts=[text],
        ent_think=[[0.1]],
        ent_answer=[[0.2]],
        stop_reason_think=["stop_t"],
        stop_reason_answer=["stop_a"],
    )


def test_build_extra_pass_results_for_row_marks_improvement(monkeypatch):
    layout = math_core.BatchLayout(
        work_items=[{"_normalized_problem": "prob", "_normalized_gold": "GOLD", "_todo_samples": [0]}],
        row_to_ex_idx=[0],
        row_target_sample_idx=[0],
    )
    cfg = _make_config()
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    context = math_core.BatchWriteContext(
        outpath="out.jsonl",
        config=cfg,
        cue_strs=["cue "],
        existing_state=existing_state,
        firstpass_choice_text_per_ex=["prev"],
    )

    pass1_result = runner._pack_pass_result(  # type: ignore[attr-defined]
        full_text="<think>x</think><answer>BAD</answer>",
        ent_think=[0.1],
        ent_answer=[0.2],
        meta_args=math_core.MathPassMetaArgs(
            problem="prob",
            canon_gold="GOLD",
            injected_cue=False,
            prev_output=None,
            cue_prefix_str="",
            stop_reason_think="s",
            stop_reason_answer="s",
        ),
    )
    pass1_result["is_correct_pred"] = False
    extra_passes = [("cue ", _pass_outputs("<think>x</think><answer>GOLD</answer>"))]

    res = runner._build_extra_pass_results_for_row(
        row_index=0,
        row_context=math_core.ExtraPassRowContext(
            prob="prob",
            canon_gold="GOLD",
            layout=layout,
            context=context,
            extra_passes=extra_passes,
            pass1_result=pass1_result,
        ),
    )
    assert "pass2a" in res
    assert "improved_over_pass1" in res["pass2a"]


def test_write_results_for_batch_writes_pass2c(tmp_path):
    layout = math_core.BatchLayout(
        work_items=[{"_normalized_problem": "prob", "_normalized_gold": "GOLD", "_todo_samples": [0]}],
        row_to_ex_idx=[0],
        row_target_sample_idx=[0],
    )
    cfg = _make_config()
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    context = math_core.BatchWriteContext(
        outpath=str(tmp_path / "out.jsonl"),
        config=cfg,
        cue_strs=["a", "b", "c"],
        existing_state=existing_state,
        firstpass_choice_text_per_ex=["prev"],
    )
    outputs = math_core.TwoPassBatchOutputs(
        pass1=_pass_outputs("<think>t</think><answer>BAD</answer>"),
        pass2=_pass_outputs("<think>t</think><answer>GOLD</answer>"),
    )
    extra_passes = [("a", _pass_outputs("<think></think><answer>GOLD</answer>"))]

    runner._write_results_for_batch(
        layout=layout,
        outputs=outputs,
        context=context,
        extra_passes=extra_passes,
    )

    data = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert data[0]["pass2c"]["output"].startswith("<think>")
    assert data[0]["pass2"]["improved_over_pass1"] is True
    assert existing_state.existing_samples["prob"] == {0}


def test_compute_second_pass_outputs_handles_none(monkeypatch):
    cfg = _make_config(two_pass=False)
    context = SimpleNamespace(config=cfg)
    main, extra, cues = runner._compute_second_pass_outputs(
        context=context,
        layout=None,
        pre1_think=[],
        firstpass_choice_text_per_ex=[],
    )
    assert main is None and extra is None
    assert cues == ["cue1 ", "cue2 "]

    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["second_pass_inputs"].cue_str)
        return f"out-{kwargs['second_pass_inputs'].cue_str}"

    monkeypatch.setattr(runner, "_run_pass2_for_batch", fake_run)
    cfg2 = _make_config(two_pass=True)
    context2 = SimpleNamespace(config=cfg2)
    main2, extra2, cues2 = runner._compute_second_pass_outputs(
        context=context2,
        layout="layout",
        pre1_think=["p"],
        firstpass_choice_text_per_ex=["f"],
    )
    assert calls == ["cue1 ", "cue2 "]
    assert main2 == "out-cue2 "
    assert extra2 == [("cue1 ", "out-cue1 ")]
    assert cues2 == ["cue1 ", "cue2 "]


def test_run_inference_batch_calls_helpers(monkeypatch, tmp_path):
    cfg = _make_config()
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    slice_ds = ["row1"]
    calls = {"built": False, "pass1": False, "first": False, "second": False, "write": False}

    monkeypatch.setattr(
        runner,
        "_build_work_items_for_slice",
        lambda *_args, **_kwargs: [{"_normalized_problem": "p", "_normalized_gold": "G", "_todo_samples": [0]}],
    )

    def fake_pass1(work_items, context):
        calls["pass1"] = True
        layout = math_core.BatchLayout(work_items=work_items, row_to_ex_idx=[0], row_target_sample_idx=[0])
        return _pass_outputs("<think></think><answer>BAD</answer>"), layout, ["prefix"]

    monkeypatch.setattr(runner, "_run_pass1_for_batch", fake_pass1)
    monkeypatch.setattr(
        runner,
        "_build_first_pass_choice",
        lambda **_kwargs: calls.__setitem__("first", True) or ["choice"],
    )
    monkeypatch.setattr(
        runner,
        "_compute_second_pass_outputs",
        lambda **_kwargs: calls.__setitem__("second", True)
        or (_pass_outputs("p2"), [("a", _pass_outputs("p2a"))], ["a"]),
    )
    monkeypatch.setattr(
        runner,
        "_write_results_for_batch",
        lambda **_kwargs: calls.__setitem__("write", True),
    )

    runner._run_inference_batch(
        slice_ds=slice_ds,
        context=math_core.MathInferenceContext(tokenizer=None, model=None, config=cfg),
        outpath=str(tmp_path / "o.jsonl"),
        existing_state=existing_state,
    )

    assert calls["pass1"] and calls["first"] and calls["second"] and calls["write"]


def test_run_inference_batch_short_circuits(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_build_work_items_for_slice", lambda *_a, **_k: [])
    runner._run_inference_batch(
        slice_ds=[],
        context=math_core.MathInferenceContext(tokenizer=None, model=None, config=_make_config()),
        outpath=str(tmp_path / "o.jsonl"),
        existing_state=math_core.ExistingPassState(defaultdict(set), {}),
    )


def test_run_inference_on_split_batches_and_warns(monkeypatch, caplog):
    cfg = _make_config(num_samples=2, batch_size=2)
    caplog.set_level("WARNING")
    monkeypatch.setattr(runner, "scan_existing_pass1_results", lambda path: (defaultdict(set), {}))
    batches = []

    def fake_run_batch(slice_ds, context, outpath, existing_state):
        batches.append(list(slice_ds))

    monkeypatch.setattr(runner, "_run_inference_batch", fake_run_batch)

    class FakeDS(list):
        def select(self, rng):
            return [self[i] for i in rng]

    ds = FakeDS(["a", "b", "c"])
    runner.run_inference_on_split(examples=ds, tokenizer=None, model=None, config=cfg)
    assert "temperature=0" in caplog.text
    assert batches == [["a", "b"], ["c"]]


def test_load_math500_prefers_local(monkeypatch):
    monkeypatch.setattr(runner, "require_datasets", lambda: (None, None))
    monkeypatch.setattr(runner, "load_local_json_dataset", lambda path: {"local": path})
    out = runner.load_math500(cache_dir="cache", split="test", seed=0, dataset_path="local.json")
    assert out == {"local": "local.json"}


def test_load_math500_remote_and_fallback(monkeypatch):
    class FakeDS:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = ["problem", "answer"]

        def map(self, fn, remove_columns=None):
            mapped = [fn(row) for row in self.rows]
            return FakeDS(mapped)

        def filter(self, fn):
            return FakeDS([row for row in self.rows if fn(row)])

        def __len__(self):
            return len(self.rows)

        def shuffle(self, seed=None):
            return self

        def select(self, rng):
            return FakeDS(self.rows[: len(rng)])

    calls = {"repos": []}

    def fake_load_dataset(repo, split=None, cache_dir=None):
        calls["repos"].append(repo)
        if repo == "AI-MO/MATH-500":
            return FakeDS([{"problem": "p", "answer": "a"}, {"problem": None, "answer": None}])
        if repo.startswith("HuggingFaceH4"):
            raise OSError("fail")
        if repo == "hendrycks/competition_math":
            return FakeDS([{"problem": "p2", "answer": "a2"}])
        raise OSError("unknown")

    monkeypatch.setattr(runner, "require_datasets", lambda: (None, fake_load_dataset))
    out_remote = runner.load_math500(cache_dir="c", split="test", seed=0)
    assert isinstance(out_remote, FakeDS)
    assert out_remote.rows == [{"problem": "p", "answer": "a"}]
    assert calls["repos"][0].startswith("HuggingFaceH4")

    def load_dataset_fallback(repo, split=None, cache_dir=None):
        if repo.startswith("hendrycks/competition_math"):
            return FakeDS([{"problem": "pf", "answer": "af"}])
        raise OSError("fail")

    monkeypatch.setattr(runner, "require_datasets", lambda: (None, load_dataset_fallback))
    out_fallback = runner.load_math500(cache_dir="c", split="test", seed=0)
    assert isinstance(out_fallback, FakeDS)
    assert out_fallback.rows[0]["problem"] == "pf"


def test_load_math500_raises_on_empty_candidate(monkeypatch):
    class FakeDS:
        def __init__(self, rows):
            self.rows = rows
            self.column_names = ["problem", "answer"]

        def map(self, fn, remove_columns=None):
            return FakeDS([fn(row) for row in self.rows])

        def filter(self, fn):
            return FakeDS([row for row in self.rows if fn(row)])

        def __len__(self):
            return len(self.rows)

    calls = []

    def fake_load_dataset(repo, split=None, cache_dir=None):
        calls.append(repo)
        if repo.startswith("HuggingFaceH4"):
            return FakeDS([])  # triggers len == 0 -> ValueError path
        if repo == "AI-MO/MATH-500":
            return FakeDS([{"problem": "p", "answer": "a"}])
        raise OSError("unexpected")

    monkeypatch.setattr(runner, "require_datasets", lambda: (None, fake_load_dataset))
    out = runner.load_math500(cache_dir="c", split="test", seed=0)

    assert calls[:2] == ["HuggingFaceH4/MATH-500", "AI-MO/MATH-500"]
    assert isinstance(out, FakeDS)
    assert out.rows == [{"problem": "p", "answer": "a"}]


def test_build_extra_pass_results_for_row_no_improvement(monkeypatch):
    layout = math_core.BatchLayout(
        work_items=[
            {"_normalized_problem": "prob", "_normalized_gold": "GOLD", "_todo_samples": [1]},
        ],
        row_to_ex_idx=[0],
        row_target_sample_idx=[1],
    )
    cfg = _make_config(two_pass=True)
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    context = math_core.BatchWriteContext(
        outpath="out.jsonl",
        config=cfg,
        cue_strs=["cueX"],
        existing_state=existing_state,
        firstpass_choice_text_per_ex=["prev-choice"],
    )

    calls = []

    def fake_pack_pass_result(**kwargs):
        calls.append(kwargs)
        return {"output": kwargs["full_text"], "is_correct_pred": kwargs["full_text"].endswith("GOOD")}

    monkeypatch.setattr(runner, "_pack_pass_result", fake_pack_pass_result)
    extra_outputs = runner.PassOutputs(
        full_texts=["THINK GOOD"],
        ent_think=[[0.0]],
        ent_answer=[[0.0]],
        stop_reason_think=["stop_t"],
        stop_reason_answer=["stop_a"],
    )

    res = runner._build_extra_pass_results_for_row(
        row_index=0,
        row_context=math_core.ExtraPassRowContext(
            prob="prob",
            canon_gold="GOLD",
            layout=layout,
            context=context,
            extra_passes=[("cueX", extra_outputs)],
            pass1_result={"is_correct_pred": True},
        ),
    )

    assert res["pass2a"]["improved_over_pass1"] is False
    assert calls[0]["meta_args"].cue_prefix_str == "cueX"
    assert calls[0]["meta_args"].prev_output == "prev-choice"


def test_write_results_for_batch_single_pass(monkeypatch, tmp_path):
    layout = math_core.BatchLayout(
        work_items=[
            {"_normalized_problem": "prob", "_normalized_gold": "GOLD", "_todo_samples": [2]},
        ],
        row_to_ex_idx=[0],
        row_target_sample_idx=[2],
    )
    cfg = _make_config(two_pass=False)
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    context = math_core.BatchWriteContext(
        outpath=str(tmp_path / "out.jsonl"),
        config=cfg,
        cue_strs=[],
        existing_state=existing_state,
        firstpass_choice_text_per_ex=["prev"],
    )

    def fake_pack_pass_result(**kwargs):
        return {"output": kwargs["full_text"], "is_correct_pred": False}

    monkeypatch.setattr(runner, "_pack_pass_result", fake_pack_pass_result)
    outputs = math_core.TwoPassBatchOutputs(
        pass1=_pass_outputs("FIRST"),
        pass2=None,
    )

    runner._write_results_for_batch(
        layout=layout,
        outputs=outputs,
        context=context,
        extra_passes=None,
    )

    data = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert data[0]["pass2"] is None
    assert "pass2a" not in data[0]
    assert existing_state.existing_samples["prob"] == {2}
    assert existing_state.existing_pass1[("prob", 2)] == "FIRST"


def test_compute_second_pass_outputs_without_cues(monkeypatch):
    cfg = _make_config(two_pass=True)
    cfg.two_pass_cfg.phrase = ""
    context = SimpleNamespace(config=cfg)
    called = {"run": False}

    def fake_run_pass2_for_batch(**kwargs):
        called["run"] = True
        return "should-not-run"

    monkeypatch.setattr(runner, "_run_pass2_for_batch", fake_run_pass2_for_batch)
    main, extra, cues = runner._compute_second_pass_outputs(
        context=context,
        layout=None,
        pre1_think=[],
        firstpass_choice_text_per_ex=[],
    )
    assert main is None and extra is None
    assert cues == []
    assert called["run"] is False


def test_run_inference_batch_respects_two_pass_flag(monkeypatch):
    cfg = _make_config(two_pass=False)
    existing_state = math_core.ExistingPassState(defaultdict(set), {})
    work_items = [{"_normalized_problem": "p", "_normalized_gold": "G", "_todo_samples": [0]}]
    layout = math_core.BatchLayout(work_items=work_items, row_to_ex_idx=[0], row_target_sample_idx=[0])

    monkeypatch.setattr(runner, "_build_work_items_for_slice", lambda *_a, **_k: work_items)
    monkeypatch.setattr(runner, "_run_pass1_for_batch", lambda *_a, **_k: (_pass_outputs("p1"), layout, ["pre"]))
    monkeypatch.setattr(runner, "_build_first_pass_choice", lambda **_kwargs: ["first-choice"])

    def fake_compute_second_pass_outputs(**_kwargs):
        return _pass_outputs("main2"), [("cue1", _pass_outputs("extra2"))], ["cue1"]

    monkeypatch.setattr(runner, "_compute_second_pass_outputs", fake_compute_second_pass_outputs)

    captured = {}

    def fake_write_results_for_batch(**kwargs):
        captured["outputs"] = kwargs["outputs"]
        captured["context"] = kwargs["context"]

    monkeypatch.setattr(runner, "_write_results_for_batch", fake_write_results_for_batch)

    runner._run_inference_batch(
        slice_ds=["row"],
        context=math_core.MathInferenceContext(tokenizer=None, model=None, config=cfg),
        outpath="out.jsonl",
        existing_state=existing_state,
    )

    assert isinstance(captured["outputs"], math_core.TwoPassBatchOutputs)
    assert captured["outputs"].pass2 is None
    assert captured["context"].cue_strs == ["cue1"]


def test_run_inference_on_split_warns_and_batches(monkeypatch, caplog, tmp_path):
    cfg = _make_config(num_samples=3, batch_size=1)
    cfg.sampling.temperature = None
    cfg.output_dir = str(tmp_path)

    monkeypatch.setattr(runner, "scan_existing_pass1_results", lambda path: (defaultdict(set), {}))
    batches = []

    def fake_run_batch(slice_ds, context, outpath, existing_state):
        batches.append(list(slice_ds))

    monkeypatch.setattr(runner, "_run_inference_batch", fake_run_batch)

    class FakeDS(list):
        def select(self, rng):
            return [self[i] for i in rng]

    ds = FakeDS(["ex1", "ex2"])
    caplog.set_level("WARNING")
    runner.run_inference_on_split(examples=ds, tokenizer=None, model=None, config=cfg)
    assert "temperature=0" in caplog.text
    assert batches == [["ex1"], ["ex2"]]
    assert (tmp_path / "step0001_test.jsonl").exists() is False
    assert tmp_path.exists()


def test_load_math500_raises_if_all_sources_fail(monkeypatch):
    calls = []

    def failing_load_dataset(repo, split=None, cache_dir=None):
        calls.append(repo)
        raise OSError("network down")

    monkeypatch.setattr(runner, "require_datasets", lambda: (None, failing_load_dataset))

    with pytest.raises(RuntimeError) as excinfo:
        runner.load_math500(cache_dir="cache", split="test", seed=0)

    assert "Could not load MATH-500" in str(excinfo.value)
    assert calls[0].startswith("HuggingFaceH4")
    assert calls[-1] == "hendrycks/competition_math"
