"""
Library helpers for the unified math runner.

CLI wiring lives in :mod:`src.inference.cli.unified_math`.
"""

from __future__ import annotations

from src.inference.runners.unified_runner_base import (
    MathTestConfig,
    run_math_inference as _run_math_inference,
)

__all__ = ["run_math_inference"]


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

    Builds a :class:`MathTestConfig` and delegates to the shared
    :func:`run_math_inference` implementation in :mod:`unified_runner_base`.
    """
    config = MathTestConfig(
        dataset=dataset,
        output_dir=output_dir,
        step=step,
        batch_size=batch_size,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        think_cap=think_cap,
        answer_cap=answer_cap,
        two_pass=two_pass,
        second_pass_phrase=second_pass_phrase,
        second_pass_use_sample_idx=second_pass_use_sample_idx,
        eos_ids=eos_ids,
    )
