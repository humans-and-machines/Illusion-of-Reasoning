"""
Library helpers for the unified math runner.

CLI wiring lives in :mod:`src.inference.cli.unified_math`.
"""

from __future__ import annotations

from src.inference.runners.unified_runner_base import MathTestConfig
from src.inference.runners.unified_runner_base import run_math_inference as _run_math_inference


__all__ = ["run_math_inference"]


def run_math_inference(
    *,
    backend,
    config: MathTestConfig | None = None,
    **config_overrides,
) -> None:
    """
    Test helper for running math inference with a simple backend + dataset.

    Builds a :class:`MathTestConfig` and delegates to the shared
    :func:`run_math_inference` implementation in :mod:`unified_runner_base`.
    """
    if config is None:
        config = _build_math_test_config_from_overrides(config_overrides)
    elif config_overrides:
        unexpected = ", ".join(sorted(config_overrides.keys()))
        raise TypeError(f"Cannot pass config_overrides with an explicit config: {unexpected}")
    _run_math_inference(backend=backend, config=config)


def _build_math_test_config_from_overrides(
    config_overrides: dict,
) -> MathTestConfig:
    """
    Build a :class:`MathTestConfig` from keyword overrides to keep the public
    :func:`run_math_inference` signature small for linting.
    """
    required_fields = ["dataset", "output_dir", "step"]
    missing = [field for field in required_fields if config_overrides.get(field) is None]
    if missing:
        missing_fields = ", ".join(missing)
        raise TypeError(
            f"Provide a MathTestConfig or dataset/output_dir/step to build one (missing: {missing_fields}).",
        )

    allowed_fields = set(
        required_fields
        + [
            "batch_size",
            "num_samples",
            "temperature",
            "top_p",
            "think_cap",
            "answer_cap",
            "two_pass",
            "second_pass_phrase",
            "second_pass_use_sample_idx",
            "eos_ids",
        ],
    )
    unexpected = set(config_overrides).difference(allowed_fields)
    if unexpected:
        unexpected_fields = ", ".join(sorted(unexpected))
        raise TypeError(f"Unexpected config overrides: {unexpected_fields}")

    return MathTestConfig(
        dataset=config_overrides["dataset"],
        output_dir=config_overrides["output_dir"],
        step=config_overrides["step"],
        batch_size=config_overrides.get("batch_size") or 8,
        num_samples=config_overrides.get("num_samples") or 1,
        temperature=config_overrides.get("temperature") or 0.0,
        top_p=config_overrides.get("top_p") or 0.95,
        think_cap=config_overrides.get("think_cap") or 750,
        answer_cap=config_overrides.get("answer_cap") or 50,
        two_pass=bool(config_overrides.get("two_pass")),
        second_pass_phrase=config_overrides.get("second_pass_phrase") or "cue",
        second_pass_use_sample_idx=config_overrides.get("second_pass_use_sample_idx") or 0,
        eos_ids=config_overrides.get("eos_ids"),
    )
