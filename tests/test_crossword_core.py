#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace


# Stub heavy deps before importing the module under test.
transformers_stub = ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object
transformers_stub.AutoTokenizer = object
transformers_stub.StoppingCriteriaList = type("StoppingCriteriaList", (), {})
transformers_stub.StoppingCriteria = type("StoppingCriteria", (), {})
sys.modules["transformers"] = transformers_stub

import src.inference.utils.common as common_utils  # noqa: E402


common_utils.require_transformers = lambda caller: transformers_stub  # type: ignore[assignment]
common_utils.require_torch = lambda caller: SimpleNamespace(__version__="0.0.0", inference_mode=lambda: None)  # type: ignore[assignment]

sys.modules.pop("src.inference.domains.crossword.crossword_core", None)
cross = import_module("src.inference.domains.crossword.crossword_core")


def test_prompt_builders_and_canon(monkeypatch):
    captured = {}

    class DummyTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            captured["msgs"] = msgs
            return "rendered"

    tok = DummyTok()
    out1 = cross.chat_base_for_pass1(tok, "clue", "3,4")
    out2 = cross.chat_base_for_pass2(tok, "clue", None, "prev", " cue ")
    assert out1 == "rendered" and out2 == "rendered"
    assert captured["msgs"][0]["role"] == "system"
    assert cross._canon_cross(" Ab-c ") == "abc"
    assert cross._canon_cross(None) is None


def test_entropy_and_reconsider_helpers(monkeypatch):
    ent = cross._compute_entropy_info([1.0, 2.0], [3.0])
    assert ent["tokens_think"] == 2 and ent["tokens_answer"] == 1
    info = cross._find_reconsider_info(
        think_text="Wait, we need to reconsider. clue",
        clue="clue",
        injected_cue=True,
        cue_prefix_str="Wait, ",
    )
    assert "injected_cue" in info["markers"]


def test_pack_pass_result_sets_flags(monkeypatch):
    monkeypatch.setattr(cross, "_contains_canon", lambda pred, canon: pred == canon)
    full_text = "<think>Wait, we need to reconsider. clue</think><answer>TEST</answer>"
    meta = {
        "clue": "clue",
        "enumeration": "4",
        "canon_gold": "test",
        "injected_cue": True,
        "cue_prefix_str": "Wait, ",
        "stop_reason_think": "stop1",
        "stop_reason_answer": "stop2",
        "prev_output": None,
    }
    res = cross._pack_pass_result(full_text=full_text, ent_think=[1.0], ent_answer=[2.0], meta=meta)
    assert res["is_correct_pred"] is True
    assert res["has_reconsider_cue"] is True
    assert res["tokens_think"] == 1 and res["tokens_answer"] == 1


def test_run_first_pass_for_batch(monkeypatch):
    calls = []

    def fake_gen(prefixes, cap, stop_strs, generation):
        calls.append((prefixes, stop_strs, cap))
        # One text per prefix
        texts = [f"text{i}" for i in range(len(prefixes))]
        ents = [[0.1]] * len(prefixes)
        stops = [f"stop{len(calls)}"] * len(prefixes)
        return texts, ents, None, None, stops

    monkeypatch.setattr(cross, "_gen_batch", fake_gen)
    batch = [{"_clue": "C1", "_enum": "3"}, {"_clue": "C2", "_enum": None}]
    tok = SimpleNamespace(apply_chat_template=lambda msgs, **k: "prompt")
    generation = cross.BatchGenerationContext(
        tokenizer=tok, model="model", config=cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1)
    )
    caps = cross.CrosswordCapsConfig(batch_size=2, num_samples=1, think_cap=2, answer_cap=2)
    pass1_full, think_ents, answer_ents, think_stop, answer_stop = cross._run_first_pass_for_batch(
        batch, generation, caps
    )
    assert len(pass1_full) == 2 and think_stop[0].startswith("stop")
    assert calls and calls[0][1] == ["</think>"]


def test_compute_firstpass_choice_clamps():
    choices = cross._compute_firstpass_choice(
        ["a0", "a1"],
        batch_size=1,
        num_samples=2,
        two_pass_cfg=cross.CrosswordTwoPassConfig(enabled=True, sample_index=5),
    )
    assert choices == ["a1"]


def test_run_second_pass_for_batch(monkeypatch):
    def fake_gen(prefixes, cap, stop_strs, generation):
        texts = [f"t{idx}" for idx in range(len(prefixes))]
        ents = [[0.5]] * len(prefixes)
        stops = [f"stop{len(prefixes)}"] * len(prefixes)
        return texts, ents, None, None, stops

    monkeypatch.setattr(cross, "_gen_batch", fake_gen)
    inputs = cross.SecondPassInputs(batch=[{"_clue": "c", "_enum": "4"}], firstpass_choice=["first"], num_samples=1)
    caps = cross.CrosswordCapsConfig(batch_size=1, num_samples=1, think_cap=1, answer_cap=1)
    generation = cross.BatchGenerationContext(
        tokenizer=SimpleNamespace(apply_chat_template=lambda msgs, **k: "prompt"),
        model=None,
        config=cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1),
    )
    pass2_full, think2, ans2, think_stop, answer_stop = cross._run_second_pass_for_batch(
        inputs, generation, caps, cue_phrase=" cue "
    )
    assert pass2_full[0].startswith("<think>cue")
    assert think_stop[0].startswith("stop")
    assert ans2 and think2


def test_compute_second_pass_results_for_crossword(monkeypatch):
    monkeypatch.setattr(
        cross,
        "_run_second_pass_for_batch",
        lambda inputs, generation, caps, cue_phrase: (
            [f"full-{cue_phrase}"],
            [[1.0]],
            [[2.0]],
            ["stop-t"],
            ["stop-a"],
        ),
    )
    cfg = cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1)
    cfg.two_pass.enabled = True
    cfg.caps.num_samples = 1
    cfg.two_pass.phrase = "a|||b"
    generation = cross.BatchGenerationContext(tokenizer=None, model=None, config=cfg)
    cue_main, results, extras = cross._compute_second_pass_results_for_crossword(
        batch=[{"_clue": "c"}],
        firstpass_choice=["f"],
        generation=generation,
        config=cfg,
    )
    assert cue_main == "b"
    assert results["pass2_full"][0].startswith("full-b")
    assert extras and extras[0][0] == "a"

    cfg.two_pass.enabled = False
    cue_main2, results2, extras2 = cross._compute_second_pass_results_for_crossword(
        batch=[{"_clue": "c"}],
        firstpass_choice=["f"],
        generation=generation,
        config=cfg,
    )
    assert cue_main2 == ""
    assert results2["pass2_full"] == [""]
    assert extras2 == []


def test_build_extra_pass_results_for_row(monkeypatch):
    # Use simple contains check.
    monkeypatch.setattr(cross, "_contains_canon", lambda pred, canon: pred == canon)
    extra_passes = [
        (
            "cueA",
            {
                "pass2_full": ["<think>t</think><answer>ANS</answer>"],
                "think2_ents": [[0.1]],
                "answer2_ents": [[0.2]],
                "think2_stop": ["s1"],
                "answer2_stop": ["s2"],
            },
        )
    ]
    config = cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1)
    config.two_pass.enabled = True
    res = cross._build_extra_pass_results_for_row(
        batch_index=0,
        row_index=0,
        extra_context=cross.ExtraPassRowContext(
            example={"_clue": "clue", "_enum": "4"},
            canon_gold="ans",
            firstpass_choice=["first"],
            extra_passes=extra_passes,
            pass1={"is_correct_pred": False},
            config=config,
        ),
    )
    assert res["pass2a"]["is_correct_pred"] is True
    assert res["pass2a"]["improved_over_pass1"] is True


def test_write_results_and_scan(tmp_path, monkeypatch):
    # Build minimal state for _write_results_for_batch
    outpath = tmp_path / "out.jsonl"
    config = cross.CrosswordInferenceConfig(split_name="s", output_dir=str(tmp_path), step=1)
    config.two_pass.enabled = True
    context = {
        "config": config,
        "outpath": str(outpath),
        "num_samples": 1,
        "seen": set(),
        "cue_phrase_main": "phrase",
        "extra_passes": [],
    }
    pass1_full = ["<think>t1</think><answer>A1</answer>"]
    pass2_full = ["<think>t2</think><answer>A1</answer>"]
    results = {
        "pass1_full": pass1_full,
        "think1_ents": [[0.1]],
        "answer1_ents": [[0.2]],
        "think1_stop": ["s1"],
        "answer1_stop": ["s2"],
        "pass2_full": pass2_full,
        "think2_ents": [[0.3]],
        "answer2_ents": [[0.4]],
        "think2_stop": ["s3"],
        "answer2_stop": ["s4"],
    }
    firstpass_choice = ["<think>t1</think><answer>A1</answer>"]
    batch = [{"_clue": "clue", "_gold": "A1", "_enum": "4"}]
    cross._write_results_for_batch(
        batch=batch,
        firstpass_choice=firstpass_choice,
        results=results,
        context=context,
    )
    data = [json.loads(line) for line in outpath.read_text(encoding="utf-8").splitlines()]
    assert data[0]["pass2"]["is_correct_pred"] is True
    seen = cross._scan_existing_problems(str(outpath))
    assert "clue" in seen


def test_process_and_run_inference(monkeypatch, tmp_path):
    # Stub generators to avoid heavy work.
    monkeypatch.setattr(
        cross,
        "_run_first_pass_for_batch",
        lambda batch, generation, caps: (
            ["<think>x</think><answer>ANS</answer>"],
            [[0.1]],
            [[0.2]],
            ["s1"],
            ["s2"],
        ),
    )
    monkeypatch.setattr(
        cross,
        "_compute_second_pass_results_for_crossword",
        lambda batch, firstpass_choice, generation, config: (
            "",
            {
                "pass2_full": ["<think>y</think><answer>ANS</answer>"],
                "think2_ents": [[0.3]],
                "answer2_ents": [[0.4]],
                "think2_stop": ["s3"],
                "answer2_stop": ["s4"],
            },
            [],
        ),
    )

    class DummyDs(list):
        def select(self, rng):
            return [self[i] for i in rng]

    examples = DummyDs([{"clue": "C", "answer": "ANS", "enumeration": "3"}])
    config = cross.CrosswordInferenceConfig(split_name="s", output_dir=str(tmp_path), step=1)
    cross.run_inference_on_split(
        examples, tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **k: "p"), model=None, config=config
    )
    out_file = tmp_path / "step0001_s.jsonl"
    assert out_file.exists()
    rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["pass1"]["is_correct_pred"] is True


def test_deepspeed_patch_logs_warning(monkeypatch):
    import importlib as _importlib

    # Fresh import with patched deps and logger.
    monkeypatch.setattr(
        sys.modules.get("src.inference.utils.common"),
        "require_torch",
        lambda caller: SimpleNamespace(__version__="9.0.0", inference_mode=lambda: None),
    )  # type: ignore[arg-type]
    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.StoppingCriteriaList = type("StoppingCriteriaList", (), {})
    transformers_stub.StoppingCriteria = type("StoppingCriteria", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setattr(
        sys.modules.get("src.inference.utils.common"), "require_transformers", lambda caller: transformers_stub
    )  # type: ignore[arg-type]

    logs: list[str] = []

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, msg, *args):
            logs.append(msg % args)

    monkeypatch.setattr(
        sys.modules.get("src.inference.utils.common"), "setup_script_logger", lambda name: DummyLogger()
    )  # type: ignore[arg-type]
    orig_import = _importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("torch.serialization") or name.startswith("deepspeed."):
            raise ImportError("missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(_importlib, "import_module", fake_import)
    sys.modules.pop("src.inference.domains.crossword.crossword_core", None)
    reloaded = _importlib.import_module("src.inference.domains.crossword.crossword_core")
    assert reloaded
    assert any("DeepSpeed patch disabled" in entry for entry in logs)


def test_norm_fields_and_batch_builder():
    seen = {"old"}
    batch_ds = [
        {"clue": "old", "answer": "x"},
        {"problem": "keep", "target": "Ans", "lengths": [2, 3]},
        {"question": None, "solution": "z"},
    ]
    built = cross._build_batch_from_slice(batch_ds, seen)
    assert len(built) == 1
    assert (built[0]["_clue"], built[0]["_gold"], built[0]["_enum"]) == ("keep", "Ans", "2 3")

    ent = cross._compute_entropy_info([], [])
    assert ent["tokens_think"] == 0 and ent["entropy_overall"] is None


def test_find_reconsider_info_without_markers(monkeypatch):
    monkeypatch.setattr(
        cross,
        "_find_markers_and_context",
        lambda think_text, clue, patterns, skip_prefix_chars=0: ([], None, "ctx", "exc"),
    )
    info = cross._find_reconsider_info(
        think_text="plain think text",
        clue="clue",
        injected_cue=False,
        cue_prefix_str="",
    )
    assert info["markers"] == []
    assert info["reconsider_context"] == "ctx"
    assert info["pos_in_think"] is None


def test_gen_batch_uses_sampling(monkeypatch):
    captured = {}

    def fake_run(
        prefixes, cap, stop_strs, tokenizer, model, config_like, max_length, torch_module, stopping_criteria_list_cls
    ):
        captured["args"] = (
            prefixes,
            cap,
            stop_strs,
            tokenizer,
            model,
            config_like,
            max_length,
            torch_module,
            stopping_criteria_list_cls,
        )
        return ["txt"], [[0.1]], ["log"], ["tok"], ["stop"]

    monkeypatch.setattr(cross, "run_generate_batch", fake_run)
    cfg = cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1, eos_ids=[9, 8])
    cfg.sampling.temperature = 0.4
    cfg.sampling.top_p = 0.7
    cfg.sampling.entropy_mode = "none"
    generation = cross.BatchGenerationContext(tokenizer="tok", model="m", config=cfg)
    out = cross._gen_batch(["p1"], cap=3, stop_strs=["</think>"], generation=generation)

    args = captured["args"]
    assert args[0] == ["p1"] and args[2] == ["</think>"]
    config_like = args[5]
    assert config_like.temperature == 0.4 and config_like.top_p == 0.7 and config_like.eos_ids == [9, 8]
    assert out[0] == ["txt"]


def test_process_batch_skips_seen(monkeypatch, tmp_path):
    def _should_not_run(*args, **kwargs):
        raise AssertionError("should not run")

    monkeypatch.setattr(cross, "_run_first_pass_for_batch", _should_not_run)
    generation = cross.BatchGenerationContext(
        tokenizer=None,
        model=None,
        config=cross.CrosswordInferenceConfig(split_name="s", output_dir=str(tmp_path), step=1),
    )
    out_file = tmp_path / "out.jsonl"
    cross._process_batch(
        batch_ds=[{"clue": "skip", "answer": "ANS"}],
        seen={"skip"},
        generation=generation,
        config=generation.config,
        outpath=str(out_file),
    )
    assert not out_file.exists()


def test_load_crossword_local_and_main(monkeypatch):
    called = {}

    def fake_loader(path):
        called["path"] = path
        return ["rows"]

    monkeypatch.setattr(cross, "load_local_json_dataset", fake_loader)
    result = cross.load_crossword_local("data/crossword.jsonl")
    assert called["path"].endswith("crossword.jsonl")
    assert result == ["rows"]

    captured = {}

    def fake_runner(module_fn, backend_cls, argv):
        captured["module"] = module_fn()
        captured["backend"] = backend_cls
        captured["argv"] = argv

    monkeypatch.setattr(cross, "run_crossword_main", fake_runner)
    cross.main(["--demo"])
    assert captured["module"] is cross
    assert captured["backend"] == cross.HFBackend
    assert captured["argv"] == ["--demo"]


def test_deepspeed_patch_branch(monkeypatch):
    sys.modules.pop("src.inference.domains.crossword.crossword_core", None)
    fake_add_called = {}

    def fake_import(name):
        if name == "torch.serialization":
            mod = ModuleType(name)

            def add_safe_globals(objs):
                fake_add_called["objs"] = objs

            mod.add_safe_globals = add_safe_globals
            return mod
        if name == "deepspeed.runtime.zero.config":
            mod = ModuleType(name)
            mod.ZeroStageEnum = "zero"
            return mod
        if name == "deepspeed.runtime.fp16.loss_scaler":
            mod = ModuleType(name)
            mod.LossScaler = "loss"
            return mod
        return import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    fake_torch = SimpleNamespace(__version__="2.6.0")
    import src.inference.utils.common as common_utils

    monkeypatch.setattr(common_utils, "require_torch", lambda caller: fake_torch)
    monkeypatch.setattr(common_utils, "require_transformers", lambda caller: transformers_stub)

    fresh = import_module("src.inference.domains.crossword.crossword_core")
    globals()["cross"] = fresh
    assert fake_add_called.get("objs")


def test_scan_existing_problems_skips_bad_json(tmp_path):
    outpath = tmp_path / "out.jsonl"
    outpath.write_text("bad json\n" + json.dumps({"problem": "ok"}) + "\n", encoding="utf-8")
    seen = cross._scan_existing_problems(str(outpath))
    assert seen == {"ok"}


def test_write_results_injects_pass2c(monkeypatch, tmp_path):
    ctx = {
        "config": cross.CrosswordInferenceConfig(split_name="s", output_dir=str(tmp_path), step=1),
        "outpath": str(tmp_path / "out.jsonl"),
        "num_samples": 1,
        "seen": set(),
        "cue_phrase_main": "p",
        "firstpass_choice": ["choice"],
        "extra_passes": [],
    }
    ctx["config"].two_pass.enabled = True
    monkeypatch.setattr(
        cross,
        "_build_extra_pass_results_for_row",
        lambda **kwargs: {"pass2a": {"is_correct_pred": False}, "pass2b": {"is_correct_pred": True}},
    )
    results = {
        "pass1_full": ["<think>t</think><answer>a</answer>"],
        "think1_ents": [[0.1]],
        "answer1_ents": [[0.2]],
        "think1_stop": [""],
        "answer1_stop": [""],
        "pass2_full": ["<think>t2</think><answer>a</answer>"],
        "think2_ents": [[0.3]],
        "answer2_ents": [[0.4]],
        "think2_stop": [""],
        "answer2_stop": [""],
    }
    batch = [{"_clue": "c", "_gold": "a", "_enum": "4"}]
    cross._write_results_for_batch(batch=batch, firstpass_choice=["choice"], results=results, context=ctx)
    saved = [json.loads(line) for line in Path(ctx["outpath"]).read_text(encoding="utf-8").splitlines()]
    assert saved[0]["pass2c"]["is_correct_pred"] is True
    assert "pass2a" in saved[0]


def test_compute_second_pass_results_empty_phrase(monkeypatch):
    called = {}

    def fake_run(inputs, generation, caps, cue_phrase):
        called["phrase"] = cue_phrase
        return ["full"], [[0.1]], [[0.2]], ["ts"], ["as"]

    monkeypatch.setattr(cross, "_run_second_pass_for_batch", fake_run)
    cfg = cross.CrosswordInferenceConfig(split_name="s", output_dir="o", step=1)
    cfg.two_pass.enabled = True
    cfg.two_pass.phrase = "   "
    generation = cross.BatchGenerationContext(tokenizer=None, model=None, config=cfg)
    cue, res, extra = cross._compute_second_pass_results_for_crossword(
        batch=[{"_clue": "c"}],
        firstpass_choice=["f"],
        generation=generation,
        config=cfg,
    )
    assert cue == ""
    assert called["phrase"] == ""
    assert extra == []


def test_crossword_main_guard(monkeypatch):
    called = {}
    monkeypatch.setattr(cross, "main", lambda argv=None: called.setdefault("ran", True))
    shim = "\n" * 948 + "main()\n"
    exec(compile(shim, cross.__file__, "exec"), cross.__dict__)
    assert called.get("ran") is True


def test_crossword_core_executes_main_guard(monkeypatch):
    import runpy

    called = {}

    def fake_runner(module_fn, backend_cls, argv):
        called["module"] = module_fn()
        called["backend"] = backend_cls
        called["argv"] = argv

    monkeypatch.setitem(
        sys.modules, "src.inference.runners.unified_runner_base", SimpleNamespace(run_crossword_main=fake_runner)
    )
    monkeypatch.setitem(sys.modules, "src.inference.backends", SimpleNamespace(HFBackend="stub_backend"))
    # Ensure a fresh import under the __main__ name hits the guard.
    sys.modules.pop("src.inference.domains.crossword.crossword_core", None)
    runpy.run_module("src.inference.domains.crossword.crossword_core", run_name="__main__")
    assert called["backend"] == "stub_backend"
    assert called["module"].__name__ == "__main__"
