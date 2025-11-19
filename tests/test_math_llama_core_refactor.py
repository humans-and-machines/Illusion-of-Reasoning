#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton tests for math_llama_core refactor.

These tests wire a fake tokenizer/model/dataset through
math_llama_core.run_inference_on_split so you can safely refactor its
two-pass loop and helpers.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.inference import math_llama_core  # noqa: E402


class FakeTokenizer:
    """Minimal tokenizer compatible with math_llama_core expectations."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 99
        self._decode_calls = 0

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)
        batch_size = len(batch)
        seq_len = 4
        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn}

    def decode(self, ids, skip_special_tokens=True):
        self._decode_calls += 1
        # Produce simple tag-structured text alternating for think/answer.
        return f"<think>t{self._decode_calls}</think><answer>a{self._decode_calls}</answer>"

    def convert_tokens_to_ids(self, tok):
        mapping = {"<|eot_id|>": 97, "<|end_of_text|>": 98}
        return mapping.get(tok, None)


class FakeGenerateOutput:
    """Container mimicking transformers.GenerationMixin output."""

    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    """Minimal model exposing generate used inside math_llama_core."""

    def __init__(self):
        # mimic a device attribute for _model_device
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        new_tokens = torch.full((batch_size, 2), 5, dtype=torch.long)
        sequences = torch.cat([input_ids, new_tokens], dim=1)
        vocab = 10
        score_step = torch.zeros((batch_size, vocab))
        scores = [score_step.clone(), score_step.clone()]
        return FakeGenerateOutput(sequences=sequences, scores=scores)


class FakeMathDataset:
    """Tiny Dataset-like wrapper for math records."""

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return FakeMathDataset([self._records[i] for i in indices])


def test_run_inference_on_split_math_llama_smoke(tmp_path):
    """Smoke test that math_llama_core.run_inference_on_split writes JSONL."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    dataset = FakeMathDataset([{"problem": "1+1?", "answer": "2"}])

    outdir = tmp_path / "math-llama"
    outdir.mkdir()

    math_llama_core.run_inference_on_split(
        split_name="test",
        examples=dataset,
        tokenizer=tokenizer,
        model=model,
        step=0,
        outdir=str(outdir),
        batch_size=1,
        num_samples=1,
        temperature=0.0,
        top_p=0.95,
        entropy_mode="none",
        eos_ids=[tokenizer.eos_token_id],
        two_pass=False,
        second_pass_phrase="cue",
        second_pass_use_sample_idx=0,
        think_cap=5,
        answer_cap=5,
    )

    outpath = outdir / "step0000_test.jsonl"
    assert outpath.exists()
    rows = [
        json.loads(line)
        for line in outpath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["problem"] == "1+1?"
    assert "pass1" in row
    assert row["pass2"] is None

