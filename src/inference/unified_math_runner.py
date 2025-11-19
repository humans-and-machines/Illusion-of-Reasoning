#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified math inference runner that reuses math_core.
Thin wrapper over src.inference.unified_runner_base.
"""
from __future__ import annotations

from src.inference.backends import HFBackend
from src.inference.unified_runner_base import (
    MathTestConfig,
    MathTestLimits,
    MathTestSampling,
    run_math_inference as _run_math_inference,
    run_math_main,
)


def run_math_inference(
    *,
    backend,
    dataset,
    output_dir: str,
    step: int,
    batch_size: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    think_cap: int,
    answer_cap: int,
    two_pass: bool,
    second_pass_phrase: str,
    second_pass_use_sample_idx: int,
    eos_ids,
) -> None:
    """
    Test helper for running math inference with a simple backend + dataset.

    This thin wrapper builds a MathTestConfig and delegates to the shared
    run_math_inference implementation in unified_runner_base.
    """
    config = MathTestConfig(
        dataset=dataset,
        output_dir=output_dir,
        step=step,
        limits=MathTestLimits(
            batch_size=batch_size,
            num_samples=num_samples,
            think_cap=think_cap,
            answer_cap=answer_cap,
        ),
        sampling=MathTestSampling(
            temperature=temperature,
            top_p=top_p,
            two_pass=two_pass,
            second_pass_phrase=second_pass_phrase,
            second_pass_use_sample_idx=second_pass_use_sample_idx,
        ),
        eos_ids=eos_ids,
    )
    _run_math_inference(backend=backend, config=config)


def main() -> None:
    """Entry point for unified math inference using math_core."""
    run_math_main(backend_cls=HFBackend)


__all__ = ["run_math_inference", "main"]


if __name__ == "__main__":
    main()
