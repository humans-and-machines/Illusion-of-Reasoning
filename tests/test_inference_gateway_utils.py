#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from types import SimpleNamespace

import pytest


gw = pytest.importorskip("src.inference.utils.gateway_utils")


def test_append_iter_and_scan_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"

    rows = [
        {"problem": "p1", "sample_idx": 0, "pass1": {"output": "o1"}},
        {"problem": "p1", "sample_idx": 1, "pass1": {"output": "o2"}},
        {"problem": "p2", "sample_idx": 0, "pass1": {"output": "x"}},
    ]
    for row in rows:
        gw.append_jsonl_row(str(path), row)

    objs = list(gw.iter_jsonl_objects(str(path)))
    assert len(objs) == len(rows)

    existing_samples = gw.scan_existing_problem_samples(str(path))
    assert existing_samples["p1"] == {0, 1}
    assert existing_samples["p2"] == {0}

    existing_samples2, existing_pass1 = gw.scan_existing_pass1_results(str(path))
    assert existing_samples2["p1"] == {0, 1}
    assert existing_pass1[("p1", 0)] == "o1"
    assert existing_pass1[("p1", 1)] == "o2"


def test_extract_problem_and_answer_falls_back_across_keys():
    example = {
        "question": "Q?",
        "solution": "S1",
        "final_answer": "S2",
    }
    problem, answer = gw.extract_problem_and_answer(example)
    assert problem == "Q?"
    # solution should win over final_answer according to helper ordering
    assert answer == "S1"


def test_build_math_gateway_row_and_usage_dict():
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    usage_dict = gw.build_usage_dict(usage)
    assert usage_dict == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    usage_none = gw.build_usage_dict(object())
    assert usage_none == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    row = gw.build_math_gateway_row_base(
        problem="p",
        gold_answer="a",
        gold_answer_canon="a_c",
        split="test",
        step=7,
        sample_idx=3,
    )
    assert row["problem"] == "p"
    assert row["gold_answer_canon"] == "a_c"
    assert row["split"] == "test"
    assert row["step"] == 7
    assert row["sample_idx"] == 3


def test_build_two_pass_row_base_structure():
    base = gw.build_two_pass_row_base(
        step=1,
        split_name="val",
        sample_idx=0,
        pass1={"x": 1},
        pass2=None,
    )
    assert base == {
        "step": 1,
        "split": "val",
        "sample_idx": 0,
        "pass1": {"x": 1},
        "pass2": None,
    }


class _DummyDataset:
    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return _DummyDataset([self._records[i] for i in indices])

    def shuffle(self, seed=None):
        # deterministic "shuffle": reverse
        return _DummyDataset(list(reversed(self._records)))

    def __iter__(self):
        return iter(self._records)


def test_limit_dataset_examples_and_iter_math_gateway_samples():
    ds = _DummyDataset(
        [
            {"problem": "p1", "answer": "a1"},
            {"problem": "p2", "answer": "a2"},
        ],
    )
    limited = gw.limit_dataset_examples(ds, 1)
    assert len(limited._records) == 1

    # num_examples None returns dataset unchanged
    same = gw.limit_dataset_examples(ds, None)
    assert same is ds

    existing = {"p1": {0}}
    samples = list(gw.iter_math_gateway_samples(ds, num_samples=2, existing=existing))
    # p1 sample_idx=1, plus both samples for p2
    assert ("p1", "a1", 1) in samples
    assert ("p2", "a2", 0) in samples
    assert ("p2", "a2", 1) in samples


def test_limit_dataset_examples_from_end():
    ds = _DummyDataset(
        [
            {"problem": "p1", "answer": "a1"},
            {"problem": "p2", "answer": "a2"},
            {"problem": "p3", "answer": "a3"},
        ],
    )
    limited = gw.limit_dataset_examples(ds, 2, from_end=True)
    # _DummyDataset.select preserves order of chosen indices; from_end=True
    # should keep the last two records (p2, p3).
    assert [row["problem"] for row in limited._records] == ["p2", "p3"]


def test_limit_dataset_examples_with_start_offset():
    ds = _DummyDataset(
        [
            {"problem": "p1", "answer": "a1"},
            {"problem": "p2", "answer": "a2"},
            {"problem": "p3", "answer": "a3"},
            {"problem": "p4", "answer": "a4"},
        ],
    )
    limited = gw.limit_dataset_examples(ds, 2, start=1)
    # Starting at index 1, keep 2 examples → p2, p3.
    assert [row["problem"] for row in limited._records] == ["p2", "p3"]


def test_limit_dataset_for_args_wrapper_respects_defaults_and_flags():
    ds = _DummyDataset(
        [
            {"problem": "p1", "answer": "a1"},
            {"problem": "p2", "answer": "a2"},
            {"problem": "p3", "answer": "a3"},
        ],
    )
    # When args provide num_examples/from_end/start, they are passed through.
    args = SimpleNamespace(num_examples=2, examples_from_end=True, dataset_start=1)
    limited = gw.limit_dataset_for_args(ds, args)
    assert [row["problem"] for row in limited._records] == ["p2", "p3"]

    # Missing attributes should fall back to defaults (no slicing).
    args_missing = SimpleNamespace()
    same = gw.limit_dataset_for_args(ds, args_missing)
    assert same is ds


def test_parse_openai_chat_response_handles_minimal_shape():
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hello"),
            ),
        ],
        usage=SimpleNamespace(prompt_tokens=1),
    )
    text, finish, usage = gw.parse_openai_chat_response(resp)
    assert text == "hello"
    assert finish == "stop"
    assert usage.prompt_tokens == 1


class _DummyTokenizerForEos:
    def __init__(self):
        self.eos_token_id = 10
        self.pad_token_id = 0
        self.pad_token = None
        self.eos_token = "<eos>"

    def convert_tokens_to_ids(self, tok):
        mapping = {"<|im_end|>": 97, "<|endoftext|>": 0}
        return mapping.get(tok, None)


def test_build_eos_ids_and_configure_tokenizer():
    tok = _DummyTokenizerForEos()
    eos_ids = gw.build_eos_ids_from_tokenizer(tok, extra_tokens=("<|im_end|>", "<|endoftext|>"))
    assert sorted(eos_ids) == [10, 97]

    tok2 = _DummyTokenizerForEos()
    eos_ids2 = gw.configure_tokenizer_and_eos(tok2, extra_tokens=("<|im_end|>",))
    assert tok2.padding_side == "left"
    assert tok2.pad_token is not None
    assert eos_ids2 == [10, 97]


class _DummyBackend:
    def __init__(self):
        self.tokenizer = _DummyTokenizerForEos()


class _DummyBackendCls:
    called_kwargs = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        cls.called_kwargs = {"model_name_or_path": model_name_or_path, **kwargs}
        return _DummyBackend()


def test_init_unified_backend_and_eos_calls_backend_cls(tmp_path):
    backend, eos_ids = gw.init_unified_backend_and_eos(
        backend_cls=_DummyBackendCls,
        model_name_or_path="m",
        revision="rev",
        cache_dir=str(tmp_path),
        dtype="float16",
        device_map="auto",
        attn_implementation="sdpa",
        tokenizer_path=None,
    )
    assert isinstance(backend, _DummyBackend)
    assert sorted(eos_ids) == [10, 97]
    assert _DummyBackendCls.called_kwargs["model_name_or_path"] == "m"
    assert _DummyBackendCls.called_kwargs["revision"] == "rev"


def test_setup_hf_cache_dir_env_sets_environment(tmp_path, monkeypatch):
    base = tmp_path / "hf"
    path = gw.setup_hf_cache_dir_env(str(base))
    assert os.path.isabs(path)
    assert os.environ["HF_HOME"].startswith(path)
    assert os.environ["TRANSFORMERS_CACHE"].startswith(path)
    assert os.environ["HF_HUB_CACHE"].startswith(path)


class _DummyLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


def test_call_with_retries_retries_then_succeeds(monkeypatch):
    logger = _DummyLogger()
    attempts = {"n": 0}

    def func():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("boom")
        return "ok"

    # Avoid real sleeping in tests.
    monkeypatch.setattr(gw.time, "sleep", lambda *_args, **_kwargs: None)

    result = gw.call_with_retries(
        func,
        settings=gw.RetrySettings(max_retries=3, retry_backoff=0.1),
        context=gw.RetryContext(logger=logger, sample_idx=0, problem_snippet="prob"),
    )
    assert result == "ok"
    assert logger.warnings
    assert not logger.errors


def test_call_with_gateway_retries_forwards_args(monkeypatch):
    logger = _DummyLogger()

    called = {}

    def fake_call_with_retries(func, **kwargs):
        called.update(kwargs)
        return func()

    monkeypatch.setattr(gw, "call_with_retries", fake_call_with_retries)

    args = SimpleNamespace(max_retries=2, retry_backoff=0.5)

    def func():
        return "ok"

    result = gw.call_with_gateway_retries(
        func,
        args=args,
        context=gw.RetryContext(logger=logger, sample_idx=1, problem_snippet="snippet", min_sleep=0.0),
    )
    assert result == "ok"
    assert called["settings"].max_retries == 2
    assert called["settings"].retry_backoff == 0.5


def test_require_datasets_importerror(monkeypatch):
    def fake_import(_name):
        raise ImportError("missing")

    monkeypatch.setattr(gw, "import_module", fake_import)
    with pytest.raises(ImportError):
        gw.require_datasets()


def test_load_local_json_dataset_filters_invalid_lines(tmp_path, monkeypatch):
    captured_records = []

    class _FakeDataset:
        @classmethod
        def from_list(cls, records):
            captured_records.extend(records)
            return {"records": records}

    monkeypatch.setattr(gw, "require_datasets", lambda: (_FakeDataset, lambda *_args, **_kwargs: None))

    path = tmp_path / "data.jsonl"
    path.write_text('not json\n{"a": 1}\n  {"b": 2}\n[]\n', encoding="utf-8")
    result = gw.load_local_json_dataset(str(path))

    assert captured_records == [{"a": 1}, {"b": 2}]
    assert result["records"] == captured_records


def test_scan_existing_handles_missing_and_absent_files(tmp_path):
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert gw.scan_existing_problem_samples(str(missing_path)) == {}

    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    assert gw.scan_existing_pass1_results(str(empty_path)) == (defaultdict(set), {})

    populated_path = tmp_path / "rows.jsonl"
    populated_path.write_text('{"problem": "p1"}\n', encoding="utf-8")  # missing sample_idx -> continue branch
    samples = gw.scan_existing_problem_samples(str(populated_path))
    assert samples == {}

    # pass1 scanner should also skip incomplete rows
    samples2, pass1 = gw.scan_existing_pass1_results(str(populated_path))
    assert samples2 == defaultdict(set)
    assert pass1 == {}


def test_limit_dataset_examples_empty_and_clamped():
    empty = _DummyDataset([])
    # len(dataset) == 0 -> early return
    assert gw.limit_dataset_examples(empty, 5) is empty

    ds = _DummyDataset([{"problem": f"p{i}", "answer": i} for i in range(5)])
    # start near end with overflow triggers clamp branch
    limited = gw.limit_dataset_examples(ds, 3, start=4)
    assert [row["problem"] for row in limited._records] == ["p4"]


class _CapturingLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg % args if args else msg)


def test_prepare_math_gateway_dataset_config_and_legacy(tmp_path):
    logger = _CapturingLogger()

    def load_math(cache_dir, split, seed, dataset_path=None):
        assert cache_dir
        assert split == "test"
        assert dataset_path is None
        return _DummyDataset([{"problem": "p1", "answer": "a1"}, {"problem": "p2", "answer": "a2"}])

    def load_remote(dataset_id, split, cache_dir):
        assert dataset_id == "remote"
        assert split == "train"
        assert cache_dir
        return _DummyDataset([{"problem": "r1", "answer": "b"}])

    outpath = tmp_path / "out.jsonl"
    gw.append_jsonl_row(str(outpath), {"problem": "p1", "sample_idx": 0})

    config = gw.MathGatewayDatasetConfig(
        source=gw.MathGatewayDatasetSource("MATH-500", "test", None),
        limits=gw.MathGatewayDatasetLimits(seed=123, num_examples=1, cache_dir=None, from_end=False, start=0),
        outpath=str(outpath),
        logger=logger,
        load_math500_fn=load_math,
        load_remote_dataset_fn=load_remote,
    )
    ds, existing, name = gw.prepare_math_gateway_dataset(config=config)
    assert len(ds._records) == 1
    assert existing == {"p1": {0}}
    assert name == "MATH-500"
    assert any("Dataset:" in msg for msg in logger.infos)

    # Legacy kwarg path, exercising remote branch and defaults
    legacy_logger = _CapturingLogger()
    ds2, existing2, name2 = gw.prepare_math_gateway_dataset(
        dataset_id="remote",
        split="train",
        seed=7,
        num_examples=None,
        dataset_path=None,
        outpath=str(tmp_path / "missing.jsonl"),
        logger=legacy_logger,
        load_math500_fn=load_math,
        load_remote_dataset_fn=load_remote,
        cache_dir=str(tmp_path / "cache"),
    )
    assert len(ds2._records) == 1
    assert existing2 == {}
    assert name2 == "remote"

    # Missing required legacy kwargs should raise
    with pytest.raises(TypeError):
        gw.prepare_math_gateway_dataset(dataset_id="x", split="y")


def test_prepare_math_gateway_dataset_from_args_wrapper(tmp_path):
    logger = _CapturingLogger()
    args = SimpleNamespace(
        dataset_id="remote",
        split="train",
        seed=1,
        num_examples=2,
        dataset_path=None,
        examples_from_end=True,
        dataset_start=1,
    )

    def load_remote(dataset_id, split, cache_dir):
        return _DummyDataset(
            [{"problem": "a", "answer": 1}, {"problem": "b", "answer": 2}, {"problem": "c", "answer": 3}]
        )

    ds, existing, name = gw.prepare_math_gateway_dataset_from_args(
        args,
        outpath=str(tmp_path / "out.jsonl"),
        logger=logger,
        load_math500_fn=lambda *_a, **_k: _DummyDataset([]),
        load_remote_dataset_fn=load_remote,
        cache_dir=str(tmp_path / "cache"),
    )
    assert name == "remote"
    # examples_from_end True with num_examples=2 should keep last two records
    assert [row["problem"] for row in ds._records] == ["c", "b"]
    assert existing == {}


def test_load_remote_dataset_default_uses_require(monkeypatch):
    called = {}

    def fake_require():
        def load_dataset(dataset_id, split=None, cache_dir=None):
            called["args"] = (dataset_id, split, cache_dir)
            return "ds"

        return object, load_dataset

    monkeypatch.setattr(gw, "require_datasets", fake_require)
    result = gw.load_remote_dataset_default("id", "train", "cache")
    assert result == "ds"
    assert called["args"] == ("id", "train", "cache")


def test_cli_arg_helpers_cover_dataset_and_runner_options():
    parser = gw.build_math_gateway_arg_parser(default_temperature=0.7, description="desc")
    args = parser.parse_args(["--output_dir", "/tmp/out"])
    assert args.dataset_id == "MATH-500"
    assert args.temperature == 0.7
    assert args.max_output_tokens == 900
    assert args.request_timeout == 120

    runner_parser = argparse.ArgumentParser()
    gw.configure_unified_runner_common(runner_parser, default_dtype="float16")
    runner_args = runner_parser.parse_args([])
    assert runner_args.dtype == "float16"
    assert runner_args.entropy_mode == "reconsider"
    assert runner_args.two_pass is False

    model_parser = argparse.ArgumentParser()
    gw.add_model_and_output_args(model_parser)
    model_args = model_parser.parse_args(["--model_name_or_path", "m", "--output_dir", "o"])
    assert model_args.model_name_or_path == "m"
    assert model_args.output_dir == "o"


def test_iter_math_gateway_samples_skips_missing_problem_or_answer():
    dataset = [
        {"problem": "", "answer": "a"},
        {"question": None, "solution": "b"},
        {"problem": "p3", "answer": None},
    ]
    assert list(gw.iter_math_gateway_samples(dataset, num_samples=1, existing={})) == []


def test_call_with_retries_errors_after_max(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(gw.time, "sleep", lambda *_args, **_kwargs: None)
    attempts = {"n": 0}

    def func():
        attempts["n"] += 1
        raise ValueError("fail")

    with pytest.raises(ValueError):
        gw.call_with_retries(
            func,
            settings=gw.RetrySettings(max_retries=1, retry_backoff=0.1),
            context=gw.RetryContext(logger=logger, sample_idx=0, problem_snippet="prob", min_sleep=0.2),
        )
    # One warning, then an error when retries are exhausted.
    assert len(logger.warnings) == 1
    assert len(logger.errors) == 1
