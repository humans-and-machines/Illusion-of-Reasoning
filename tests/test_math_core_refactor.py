#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton tests for math_core refactor.

These tests exercise normalization helpers, per-pass packing, and a basic
run_inference_on_split path using fake tokenizer/model/dataset. They are
intended to guard behaviour while refactoring math_core.
"""

from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")
if not all(hasattr(torch, attr) for attr in ("tensor", "zeros", "full")):
    pytest.skip("torch stub lacks required tensor helpers", allow_module_level=True)
# Some minimal stubs expose no inference_mode; fall back to no_grad.
if not hasattr(torch, "inference_mode"):
    torch.inference_mode = torch.no_grad  # type: ignore[attr-defined]
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "AutoTokenizer") or not hasattr(transformers, "AutoModelForCausalLM"):
    pytest.skip("transformers stub lacks required classes", allow_module_level=True)
math_core = pytest.importorskip("src.inference.domains.math.math_core")


class FakeTokenizer:
    """Minimal tokenizer compatible with math_core expectations."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 99
        self._decode_calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Concatenate user content; ignore system/assistant roles."""
        user_bits = [m["content"] for m in messages if m.get("role") == "user"]
        base = " ".join(user_bits)
        if add_generation_prompt:
            base += " <GEN>"
        return base

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
        """Return a simple, deterministic string for each decode call."""
        self._decode_calls += 1
        return f"think{self._decode_calls}" if self._decode_calls % 2 else f"answer{self._decode_calls}"

    def convert_tokens_to_ids(self, tok):
        """Map a couple of special tokens to ids for EOS building."""
        mapping = {"<|im_end|>": 97, "<|endoftext|>": 98}
        return mapping.get(tok, None)


class FakeGenerateOutput:
    """Container mimicking transformers.GenerationMixin output."""

    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeModel:
    """Minimal model exposing generate used by _gen_batch."""

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        # Append two dummy tokens to each sequence.
        new_tokens = torch.full((batch_size, 2), 5, dtype=torch.long)
        sequences = torch.cat([input_ids, new_tokens], dim=1)
        # Build two score steps with simple, finite logits.
        vocab = 10
        score_step = torch.zeros((batch_size, vocab))
        scores = [score_step.clone(), score_step.clone()]
        return FakeGenerateOutput(sequences=sequences, scores=scores)


class FakeDataset:
    """Tiny Dataset-like wrapper providing select/shuffle/len for tests."""

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return FakeDataset([self._records[i] for i in indices])

    def shuffle(self, seed=None):
        import random

        rng = random.Random(seed)
        records = list(self._records)
        rng.shuffle(records)
        return FakeDataset(records)


def test_norm_fields_basic_roundtrip():
    """_norm_fields should recover simple (problem, answer) pairs."""
    example = {"problem": "p(x)=?", "answer": "42"}
    problem, gold = math_core._norm_fields(example)  # type: ignore[attr-defined]
    assert problem == "p(x)=?"
    assert gold == "42"


def test_pack_pass_result_extracts_answer_and_tags():
    """_pack_pass_result should parse tags and fill core fields."""
    full_text = "<think>some reasoning</think>\n<answer>7</answer>"
    ent_think = [1.0]
    ent_answer = [2.0]

    row = math_core._pack_pass_result(  # type: ignore[attr-defined]
        full_text=full_text,
        ent_think=ent_think,
        ent_answer=ent_answer,
        meta_args=math_core.MathPassMetaArgs(
            problem="1+6",
            canon_gold="7",
            injected_cue=False,
            prev_output=None,
            cue_prefix_str="",
            stop_reason_think="eos",
            stop_reason_answer="eos",
        ),
    )

    assert row["pred_answer"] == "7"
    assert row["valid_tag_structure"] is True
    assert row["entropy_think"] is not None
    assert row["entropy_answer"] is not None


def test_run_inference_on_split_smoke(tmp_path):
    """Smoke test: run_inference_on_split writes a JSONL row."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    dataset = FakeDataset([{"problem": "2+2?", "answer": "4"}])

    config = math_core.MathInferenceConfig(
        split_name="test",
        output_dir=str(tmp_path),
        step=0,
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

    math_core.run_inference_on_split(
        examples=dataset,
        tokenizer=tokenizer,
        model=model,
        config=config,
    )

    outpath = tmp_path / "step0000_test.jsonl"
    assert outpath.exists()
    rows = [json.loads(line) for line in outpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["problem"] == "2+2?"
    assert row["gold_answer"] == "4"
    assert "pass1" in row
    assert row["pass2"] is None
