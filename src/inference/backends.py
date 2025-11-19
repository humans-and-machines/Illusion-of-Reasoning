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
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .common import move_inputs_to_device

try:
    from src.annotate.config import load_azure_config
    from src.annotate.llm_client import build_preferred_client
except Exception:  # noqa: BLE001
    load_azure_config = None
    build_preferred_client = None


# ----------------------- Shared helpers -----------------------
class StopOnSubstrings(StoppingCriteria):
    """Stop generation when any of the provided substrings is seen."""

    def __init__(self, tokenizer, stops: List[str]):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]

    @staticmethod
    def _endswith(a: torch.Tensor, b: List[int]) -> bool:
        return len(a) >= len(b) and a[-len(b) :].tolist() == b

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for row in input_ids:
            for s in self.stop_ids:
                if s and self._endswith(row, s):
                    return True
        return False


# ----------------------- HF backend -----------------------
@dataclass
class HFGenerateResult:
    texts: List[str]
    stop_reasons: List[str]
    sequences: torch.Tensor
    output: Any  # the raw generate output


class HFBackend:
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
        tok_src = tokenizer_path or model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tok_src,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.truncation_side = "left"

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()
        return cls(tokenizer, model)

    def generate(
        self,
        prompts: List[str],
        *,
        stop_strings: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        **gen_kwargs,
    ) -> HFGenerateResult:
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs, input_lengths = move_inputs_to_device(inputs)

        stop = (
            StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strings)])
            if stop_strings
            else None
        )
        with torch.inference_mode():
            out = model.generate(**inputs, stopping_criteria=stop, **gen_kwargs)

        seqs = out.sequences if hasattr(out, "sequences") else out
        total_rows = seqs.shape[0]
        texts: List[str] = []
        stop_reasons: List[str] = []

        for row_i in range(total_rows):
            start_tok_idx = int(input_lengths[row_i].item())
            gen_ids = seqs[row_i, start_tok_idx:]
            raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            texts.append(raw_txt)

            found_stop = any(s in raw_txt for s in (stop_strings or []))
            hit_max = bool(gen_kwargs.get("max_new_tokens")) and len(gen_ids) >= int(
                gen_kwargs["max_new_tokens"]
            )
            has_eos = False
            if model.config.eos_token_id is not None:
                eos_id = model.config.eos_token_id
                if isinstance(eos_id, Iterable):
                    has_eos = any((gen_ids == int(e)).any() for e in eos_id)
                else:
                    has_eos = (gen_ids == int(eos_id)).any()
            if found_stop:
                stop_reasons.append("stop_token")
            elif has_eos:
                stop_reasons.append("eos")
            elif hit_max:
                stop_reasons.append("max_new_tokens")
            else:
                stop_reasons.append("other")

        return HFGenerateResult(texts=texts, stop_reasons=stop_reasons, sequences=seqs, output=out)


# ----------------------- Azure backend -----------------------
@dataclass
class AzureGenerateResult:
    texts: List[str]
    finish_reasons: List[str]
    raw: List[Any]


class AzureBackend:
    def __init__(self, client):
        if client is None:
            raise ValueError("Azure client is required")
        self.client = client

    @classmethod
    def from_env(cls, deployment: Optional[str] = None) -> "AzureBackend":
        if load_azure_config is None or build_preferred_client is None:
            raise RuntimeError("Azure dependencies are unavailable")
        cfg = load_azure_config(deployment_override=deployment)
        client = build_preferred_client(
            endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            azure_ad_token=cfg.token,
            api_version=cfg.api_version,
            deployment_name=cfg.deployment,
            azure_ad_token_provider=lambda: cfg.token,  # type: ignore[arg-type]
        )
        return cls(client)

    def _call(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_output_tokens: int = 900,
    ) -> tuple[str, str, Any]:
        client = self.client
        # Prefer Responses API if available
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=getattr(client, "deployment", None),  # type: ignore[attr-defined]
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
                    texts = [getattr(p, "text", "") for p in parts]
                    text = "".join(texts) or None
                elif hasattr(out, "content") and out.content:
                    # content is a list of parts
                    parts = getattr(out, "content", [])
                    texts = [getattr(p, "text", "") for p in parts]
                    text = "".join(texts) or None
            text = text or ""
            finish = getattr(getattr(resp, "output", None), "finish_reason", None) or getattr(
                resp, "finish_reason", None
            )
            return text, finish, resp
        # Fallback: Chat Completions
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=getattr(client, "deployment", None),  # type: ignore[attr-defined]
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
