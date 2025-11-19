#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable inference backends.

- HFBackend wraps transformers AutoModelForCausalLM generate.
- AzureBackend wraps the Azure OpenAI Responses / Chat Completions client.

These keep model plumbing out of the task/routing logic so scripts can share
one uniform interface for "give me generations for these prompts".
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import (
    StopOnSubstrings,
    classify_stop_reason,
    decode_generated_row,
    move_inputs_to_device,
)

try:  # pragma: no cover - optional Azure dependency
    from src.annotate.config import load_azure_config as _load_azure_config
    from src.annotate.llm_client import build_preferred_client as _build_preferred_client
except ImportError:  # pragma: no cover - environments without Azure support
    _load_azure_config = None
    _build_preferred_client = None

load_azure_config = _load_azure_config
build_preferred_client = _build_preferred_client


def _load_torch_and_transformers() -> Tuple[Any, Any, Any, Any]:
    """
    Lazily import torch and key transformers classes used by the HF backend.

    This keeps heavy optional dependencies out of module import time so that
    code paths that do not require HF models can still import this module.
    """
    torch_mod = import_module("torch")
    transformers_mod = import_module("transformers")
    auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")
    auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")
    stopping_criteria_list_cls = getattr(transformers_mod, "StoppingCriteriaList")
    return torch_mod, auto_tokenizer_cls, auto_model_cls, stopping_criteria_list_cls


def _load_hf_tokenizer(
    auto_tokenizer_cls,
    tok_src: str,
    revision: Optional[str],
    trust_remote_code: bool,
    cache_dir: Optional[str],
):
    """Load and configure a HuggingFace tokenizer for causal LM inference."""
    tokenizer = auto_tokenizer_cls.from_pretrained(
        tok_src,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer


def _load_hf_model(
    auto_model_cls,
    torch_mod,
    model_name_or_path: str,
    *,
    revision: Optional[str],
    trust_remote_code: bool,
    cache_dir: Optional[str],
    dtype: str,
    device_map: str,
    attn_implementation: Optional[str],
):
    """Load a HuggingFace causal LM with the requested dtype and device map."""
    torch_dtype = torch_mod.bfloat16 if dtype == "bfloat16" else torch_mod.float16
    model = auto_model_cls.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    return model.eval()


@dataclass
class HFDecodeContext:
    """Bundle decode-and-classify inputs to keep helper signature small."""

    tokenizer: Any
    model: Any
    sequences: Any
    input_lengths: Any
    stop_strings: List[str]
    max_new_tokens: Optional[int]


def _prepare_hf_inputs(
    tokenizer,
    prompts: List[str],
    max_length: Optional[int],
):
    """
    Tokenize prompts and move them (and attention-mask lengths) to the right device.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return move_inputs_to_device(inputs)


def _build_hf_stopping_criteria(
    stopping_criteria_list_cls,
    tokenizer,
    stop_strings: Optional[List[str]],
):
    """Construct a StoppingCriteriaList for the provided stop strings, if any."""
    if not stop_strings:
        return None
    return stopping_criteria_list_cls([StopOnSubstrings(tokenizer, stop_strings)])


# ----------------------- HF backend -----------------------
@dataclass
class HFGenerateResult:
    """Container for HuggingFace generate() outputs."""

    texts: List[str]
    stop_reasons: List[str]
    sequences: Any
    output: Any  # the raw generate output


class HFBackend:
    """Backend that wraps a transformers AutoModelForCausalLM + tokenizer."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        dtype: str = "float16",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        tokenizer_path: Optional[str] = None,
    ) -> "HFBackend":
        """
        Load a tokenizer + causal LM from HuggingFace and wrap them in HFBackend.
        """
        env = _load_torch_and_transformers()
        tok_src = tokenizer_path or model_name_or_path
        tokenizer = _load_hf_tokenizer(
            env[1],
            tok_src,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        model = _load_hf_model(
            env[2],
            env[0],
            model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        return cls(tokenizer, model)

    def generate(
        self,
        prompts: List[str],
        *,
        stop_strings: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        **gen_kwargs,
    ) -> HFGenerateResult:
        """
        Generate continuations for a batch of string prompts.
        """
        env = _load_torch_and_transformers()
        inputs, input_lengths = _prepare_hf_inputs(self.tokenizer, prompts, max_length)
        stopping_criteria = _build_hf_stopping_criteria(
            env[3],
            self.tokenizer,
            stop_strings,
        )
        with env[0].inference_mode():
            out = self.model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                **gen_kwargs,
            )

        sequences = out.sequences if hasattr(out, "sequences") else out
        decode_ctx = HFDecodeContext(
            tokenizer=self.tokenizer,
            model=self.model,
            sequences=sequences,
            input_lengths=input_lengths,
            stop_strings=stop_strings or [],
            max_new_tokens=(
                int(gen_kwargs["max_new_tokens"])
                if gen_kwargs.get("max_new_tokens")
                else None
            ),
        )
        texts, stop_reasons = self._decode_and_classify(decode_ctx)

        return HFGenerateResult(
            texts=texts,
            stop_reasons=stop_reasons,
            sequences=sequences,
            output=out,
        )

    @staticmethod
    def _decode_and_classify(ctx: HFDecodeContext) -> Tuple[List[str], List[str]]:
        """
        Decode generated sequences and compute stop reasons for each row.
        """
        total_rows = ctx.sequences.shape[0]
        texts: List[str] = []
        stop_reasons: List[str] = []

        def _sequence_has_eos(gen_ids, eos_token_id) -> bool:
            if eos_token_id is None:
                return False
            if isinstance(eos_token_id, Iterable):
                return any((gen_ids == int(e)).any() for e in eos_token_id)
            return (gen_ids == int(eos_token_id)).any()

        for row_i in range(total_rows):
            gen_ids, raw_txt, _ = decode_generated_row(
                ctx.tokenizer,
                ctx.sequences,
                ctx.input_lengths,
                row_i,
                skip_special_tokens=True,
            )
            texts.append(raw_txt)

            found_stop = any(stop in raw_txt for stop in ctx.stop_strings)
            hit_max = ctx.max_new_tokens is not None and len(gen_ids) >= ctx.max_new_tokens
            has_eos = _sequence_has_eos(gen_ids, ctx.model.config.eos_token_id)
            stop_reasons.append(classify_stop_reason(found_stop, has_eos, hit_max))

        return texts, stop_reasons


# ----------------------- Azure backend -----------------------
@dataclass
class AzureGenerateResult:
    """Container for batched Azure generation outputs."""

    texts: List[str]
    finish_reasons: List[str]
    raw: List[Any]


class AzureBackend:
    """Backend that wraps an Azure/OpenAI client for chat-style generation."""

    def __init__(self, client: Any, deployment: str, uses_v1: bool) -> None:
        if client is None:
            raise ValueError("Azure client is required")
        self.client = client
        self.deployment = deployment
        self.uses_v1 = uses_v1

    @classmethod
    def from_env(cls, deployment: Optional[str] = None) -> "AzureBackend":
        """
        Construct an AzureBackend using environment / YAML configuration.

        The optional `deployment` parameter overrides the configured deployment
        name if provided.
        """
        if load_azure_config is None or build_preferred_client is None:
            raise RuntimeError(
                "Azure backend requires src.annotate.config and src.annotate.llm_client; "
                "ensure optional Azure dependencies are installed.",
            )

        cfg = load_azure_config()
        deployment_name = deployment or cfg["deployment"]
        client, uses_v1 = build_preferred_client(
            endpoint=cfg["endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            use_v1=cfg.get("use_v1", True),
        )
        return cls(client=client, deployment=deployment_name, uses_v1=uses_v1)

    def _call(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_output_tokens: int = 900,
    ) -> tuple[str, str, Any]:
        """
        Call the underlying Azure/OpenAI client for a single message batch.
        """
        if hasattr(self.client, "responses"):
            return self._call_responses_api(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
        return self._call_chat_completions(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

    def _call_responses_api(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: Optional[float],
        max_output_tokens: int,
    ) -> tuple[str, str, Any]:
        """
        Call the Azure Responses-style API (if available on the client).
        """
        resp = self.client.responses.create(
            model=self.deployment,
            input=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = getattr(resp, "output_text", None)
        if not text and getattr(resp, "output", None):
            # Some SDKs expose output/content instead of output_text
            out = resp.output
            if hasattr(out, "message") and getattr(out.message, "content", None):
                parts = getattr(out.message, "content", []) or []
                texts = [getattr(part, "text", "") for part in parts]
                text = "".join(texts) or None
            elif hasattr(out, "content") and out.content:
                # content is a list of parts
                parts = getattr(out, "content", [])
                texts = [getattr(part, "text", "") for part in parts]
                text = "".join(texts) or None
        text = text or ""
        finish = getattr(getattr(resp, "output", None), "finish_reason", None) or getattr(
            resp,
            "finish_reason",
            None,
        )
        return text, finish, resp

    def _call_chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: Optional[float],
        max_output_tokens: int,
    ) -> tuple[str, str, Any]:
        """
        Call the Chat Completions API as a fallback.
        """
        chat = getattr(self.client, "chat", None)
        if chat is None or not hasattr(chat, "completions"):
            raise RuntimeError("Azure/OpenAI client does not expose chat.completions")
        completions = getattr(chat, "completions")
        resp = completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        choice = resp.choices[0]
        text = choice.message.content
        finish = getattr(choice, "finish_reason", None)
        return text, finish, resp

    def generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_output_tokens: int = 900,
    ) -> AzureGenerateResult:
        """
        Generate responses for a batch of message lists.
        """
        texts: List[str] = []
        finish_reasons: List[str] = []
        raws: List[Any] = []
        for msgs in batch_messages:
            txt, fin, raw = self._call(
                msgs,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            texts.append(txt)
            finish_reasons.append(fin)
            raws.append(raw)
        return AzureGenerateResult(texts=texts, finish_reasons=finish_reasons, raw=raws)
