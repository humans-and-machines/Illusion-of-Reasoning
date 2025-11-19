"""Helpers for constructing models and tokenizers for training."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ..configs import GRPOConfig, SFTConfig

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
    from trl import ModelConfig


def get_tokenizer(
    model_args: "ModelConfig",
    training_args: SFTConfig | GRPOConfig,
) -> "PreTrainedTokenizer":
    """Return a tokenizer configured for the given model and training args."""
    transformers_mod = import_module("transformers")
    auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")

    tokenizer = auto_tokenizer_cls.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def _resolve_torch_dtype(model_args: "ModelConfig") -> "torch.dtype | str | None":
    """Map the string dtype in ModelConfig to torch dtype or 'auto'/None."""
    if model_args.torch_dtype in ("auto", None):
        return model_args.torch_dtype
    torch_mod = import_module("torch")
    return getattr(torch_mod, model_args.torch_dtype)


def _build_model_kwargs(
    model_args: "ModelConfig",
    training_args: SFTConfig | GRPOConfig,
) -> dict:
    """Build keyword arguments for `from_pretrained` calls."""
    torch_dtype = _resolve_torch_dtype(model_args)
    trl_mod = import_module("trl")
    get_quantization_config = getattr(trl_mod, "get_quantization_config")
    get_kbit_device_map = getattr(trl_mod, "get_kbit_device_map")

    quantization_config = get_quantization_config(model_args)
    device_map = (
        get_kbit_device_map() if quantization_config is not None else None
    )

    return {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }


def get_model(
    model_args: "ModelConfig",
    training_args: SFTConfig | GRPOConfig,
) -> "AutoModelForCausalLM":
    """Instantiate and return the causal LM model."""
    transformers_mod = import_module("transformers")
    auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")

    model_kwargs = _build_model_kwargs(model_args, training_args)
    model = auto_model_cls.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
