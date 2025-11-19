"""Thin CLI entrypoint for GRPO training.

This module parses command-line arguments with ``trl.TrlParser`` and delegates
to :func:`src.training.grpo_impl.main`, which contains the full training logic.
"""

from __future__ import annotations

import importlib

from .configs import GRPOConfig, GRPOScriptArguments
from .grpo_impl import main as _main


def _load_trl_parser():
    """Dynamically import TrlParser/ModelConfig to avoid hard dependency at import time."""
    trl_mod = importlib.import_module("trl")
    model_config = getattr(trl_mod, "ModelConfig")
    parser_cls = getattr(trl_mod, "TrlParser")
    return model_config, parser_cls


def main() -> None:
    """Parse CLI arguments and launch GRPO training."""
    ModelConfig, TrlParser = _load_trl_parser()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    _main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
