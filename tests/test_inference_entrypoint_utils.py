#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
if not getattr(torch, "__file__", None):
    pytest.skip("torch stub detected; real torch required for these tests", allow_module_level=True)
required_attrs = ("zeros", "tensor", "__version__")
if not all(hasattr(torch, attr) for attr in required_attrs):
    pytest.skip("torch stub lacks required tensor helpers", allow_module_level=True)
if not isinstance(getattr(torch, "SymFloat", float), type):
    pytest.skip("torch stub provides non-type SymFloat; real torch required", allow_module_level=True)
# Ensure functional module is importable; otherwise skip early to avoid attr errors.
try:
    pass  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch.nn.functional unavailable: {exc}", allow_module_level=True)

inference_mod = pytest.importorskip("src.inference.runners.openr1_math_runner")


def test_append_jsonl_and_load_seen_problems(tmp_path):
    path = tmp_path / "out.jsonl"
    row1 = {"problem": "p1"}
    row2 = {"problem": "p2"}
    inference_mod.append_jsonl(str(path), row1)
    inference_mod.append_jsonl(str(path), row2)

    seen = inference_mod._load_seen_problems(str(path))
    assert seen == {"p1", "p2"}


def test_build_generation_kwargs_respects_num_samples_and_temperature():
    class Tok:
        eos_token_id = 9

    kw_single = inference_mod._build_generation_kwargs(Tok(), num_samples=1, temperature=0.7)
    assert kw_single["do_sample"] is False
    assert kw_single["temperature"] == 0.0

    kw_multi = inference_mod._build_generation_kwargs(Tok(), num_samples=2, temperature=0.3)
    assert kw_multi["do_sample"] is True
    assert kw_multi["temperature"] == 0.3


def test_compute_entropies_returns_per_sequence_values():
    torch_module, functional = inference_mod._require_torch_modules()
    scores = [
        torch_module.zeros((2, 4)),
        torch_module.zeros((2, 4)),
    ]
    entropies = inference_mod._compute_entropies(scores, functional)
    assert len(entropies) == 2
    assert all(math.isfinite(e) for e in entropies)


def test_decode_new_tokens_slices_after_prompt():
    sequences = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    input_ids = torch.tensor([[10, 11], [12, 13]])

    class Tok:
        def batch_decode(self, ids, skip_special_tokens=False):
            return ["-".join(str(int(x)) for x in row.tolist()) for row in ids]

    out = inference_mod._decode_new_tokens(Tok(), sequences, input_ids)
    assert out == ["3-4", "7-8"]
