#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton tests for carpark_core refactor.

These tests use fake tokenizer/model/dataset to exercise the Rush Hour
inference wrapper in a controlled way. They are intended as a safety net
while refactoring carpark_core.run_inference_on_split and helpers.
"""

from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")
carpark_core = pytest.importorskip("src.inference.domains.carpark.carpark_core")


class FakeTokenizer:
    """Minimal tokenizer compatible with carpark_core expectations."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 99
        self._decode_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Concatenate user+system content; append a marker if requested."""
        parts = [m["content"] for m in messages if "content" in m]
        text = " ".join(parts)
        if add_generation_prompt:
            text += " <GEN>"
        return text

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        """Return fixed-size tensor inputs for any batch of strings."""
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
        """Return deterministic text alternating between think/answer-ish."""
        self._decode_calls += 1
        return f"rush_think{self._decode_calls}" if self._decode_calls % 2 else f"rush_answer{self._decode_calls}"

    def convert_tokens_to_ids(self, tok):
        """Map special tokens used for EOS set construction."""
        mapping = {"<|im_end|>": 97, "<|endoftext|>": 98}
        return mapping.get(tok, None)


class FakeGenerateOutput:
    """Container mimicking transformers.GenerationMixin output."""

    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    """Minimal model exposing generate used by carpark_core._gen_batch."""

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        # Append two dummy tokens to each sequence.
        new_tokens = torch.full((batch_size, 2), 5, dtype=torch.long)
        sequences = torch.cat([input_ids, new_tokens], dim=1)
        vocab = 10
        score_step = torch.zeros((batch_size, vocab))
        scores = [score_step.clone(), score_step.clone()]
        return FakeGenerateOutput(sequences=sequences, scores=scores)


class FakeRushDataset:
    """Tiny Dataset-like wrapper for Rush Hour entries."""

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return FakeRushDataset([self._records[i] for i in indices])


def test_run_inference_on_split_carpark_smoke(tmp_path):
    """Smoke test that carpark_core.run_inference_on_split writes a JSONL row."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    # Minimal Rush Hour-style record with messages+solution.
    examples = FakeRushDataset(
        [
            {
                "messages": json.dumps(
                    [
                        {"role": "user", "content": "Board: AABB....; moves=3"},
                    ],
                ),
                "solution": "A>2,B<1",
            },
        ],
    )

    outdir = tmp_path / "rush"
    outdir.mkdir()

    carpark_core.run_inference_on_split(
        split_name="test",
        examples=examples,
        tokenizer=tokenizer,
        model=model,
        step=0,
        outdir=str(outdir),
        prompt_col="messages",
        solution_col="solution",
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
    rows = [json.loads(line) for line in outpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    # Basic structural checks.
    assert "problem" in row
    assert "gold_answer" in row
    assert "pass1" in row
    assert row["pass2"] is None
