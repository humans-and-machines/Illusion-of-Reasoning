"""Thin CLI entrypoint for GRPO training.

This module parses command-line arguments with ``trl.TrlParser`` and delegates
to :func:`src.training.grpo_impl.main`, which contains the full training logic.
"""

from __future__ import annotations

import importlib

from ..configs import (
    ChatBenchmarkConfig,
    GRPOConfig,
    GRPOCosineRewardConfig,
    GRPODatasetColumnsConfig,
    GRPOOnlyTrainingConfig,
    GRPORewardConfig,
    GRPOScriptArguments,
    GRPOSpanKLConfig,
    HubRevisionConfig,
    WandbRunConfig,
    merge_dataclass_attributes,
)


_MAIN_IMPL = None  # filled lazily to keep import-time deps light


def _load_trl_parser():
    """Dynamically import TrlParser/ModelConfig to avoid hard dependency at import time."""
    trl_mod = importlib.import_module("trl")
    model_config = getattr(trl_mod, "ModelConfig")
    parser_cls = getattr(trl_mod, "TrlParser")
    return model_config, parser_cls


def _resolve_main():
    """Return the underlying grpo_impl.main, importing only when needed."""
    global _MAIN_IMPL  # pylint: disable=global-statement
    if _MAIN_IMPL is None:
        grpo_mod = importlib.import_module("src.training.grpo_impl")
        _MAIN_IMPL = getattr(grpo_mod, "main")
    return _MAIN_IMPL


def _main(*args, **kwargs):
    """Indirection layer so tests can monkeypatch the training entrypoint."""
    return _resolve_main()(*args, **kwargs)


def main(merge_fn=merge_dataclass_attributes) -> None:
    """Parse CLI arguments and launch GRPO training."""
    model_config_cls, trl_parser_cls = _load_trl_parser()
    parser = trl_parser_cls(
        (
            GRPOScriptArguments,
            GRPORewardConfig,
            GRPOCosineRewardConfig,
            GRPODatasetColumnsConfig,
            GRPOSpanKLConfig,
            ChatBenchmarkConfig,
            HubRevisionConfig,
            WandbRunConfig,
            GRPOOnlyTrainingConfig,
            GRPOConfig,
            model_config_cls,
        )
    )
    (
        script_args,
        reward_cfg,
        cosine_cfg,
        dataset_cfg,
        span_kl_cfg,
        chat_cfg,
        hub_cfg,
        wandb_cfg,
        grpo_only_cfg,
        training_args,
        model_args,
    ) = parser.parse_args_and_config()

    # Flatten the auxiliary config dataclasses onto the main script/training args
    merge_fn(
        script_args,
        reward_cfg,
        cosine_cfg,
        dataset_cfg,
        span_kl_cfg,
    )
    merge_fn(
        training_args,
        chat_cfg,
        hub_cfg,
        wandb_cfg,
        grpo_only_cfg,
    )

    _main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
