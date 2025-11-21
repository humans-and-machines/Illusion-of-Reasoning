"""Thin CLI entrypoint for GRPO training.

This module parses command-line arguments with ``trl.TrlParser`` and delegates
to :func:`src.training.grpo_impl.main`, which contains the full training logic.
"""

from __future__ import annotations

import importlib

from .configs import (
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
from .grpo_impl import main as _main


def _load_trl_parser():
    """Dynamically import TrlParser/ModelConfig to avoid hard dependency at import time."""
    trl_mod = importlib.import_module("trl")
    model_config = getattr(trl_mod, "ModelConfig")
    parser_cls = getattr(trl_mod, "TrlParser")
    return model_config, parser_cls


def main() -> None:
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
    merge_dataclass_attributes(
        script_args,
        reward_cfg,
        cosine_cfg,
        dataset_cfg,
        span_kl_cfg,
    )
    merge_dataclass_attributes(
        training_args,
        chat_cfg,
        hub_cfg,
        wandb_cfg,
        grpo_only_cfg,
    )

    _main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
