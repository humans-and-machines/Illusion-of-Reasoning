"""Generation utilities for hierarchical GRPO training."""
# pylint: disable=too-few-public-methods

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Union

from .hierarchical_grpo_dependencies import (
    FSDP,
    PaddingStrategy,
    broadcast_object_list,
    canonicalize_device,
    gather_object,
    is_conversational,
    is_flash_attn_2_available,
    maybe_apply_chat_template,
    pad,
    profiling_context,
    resolve_dependency,
    torch,
    unwrap_model_for_generation,
)


@dataclass
class GenerationBatch:  # pylint: disable=too-few-public-methods
    """Container for generation inputs shared across helper methods."""

    prompts: list[Any]
    prompts_text: list[str]
    prompt_ids: Any
    prompt_mask: Any
    device: Any


class HierarchicalGenerationMixin:
    """Generation helpers mixed into the hierarchical GRPO trainer."""

    def _prepare_prompts(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        device: torch.device,
    ) -> tuple[list[Any], list[str], torch.Tensor, torch.Tensor]:
        """Tokenize prompts and optionally trim to max_prompt_length."""
        device = canonicalize_device(device)
        prompts = [example["prompt"] for example in inputs]
        template_fn = resolve_dependency(
            self,
            "maybe_apply_chat_template",
            maybe_apply_chat_template,
        )
        prompts_text = [template_fn(example, self.processing_class)["prompt"] for example in inputs]

        padding_strategy = resolve_dependency(
            self,
            "PaddingStrategy",
            PaddingStrategy,
        )
        padding_value = getattr(padding_strategy, "LONGEST", "longest")

        encoding = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=padding_value,
            truncation=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = encoding["input_ids"]
        prompt_mask = encoding["attention_mask"]
        if hasattr(prompt_ids, "to"):
            prompt_ids = prompt_ids.to(device)
        if hasattr(prompt_mask, "to"):
            prompt_mask = prompt_mask.to(device)

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts_text = [
                re.sub(
                    rf"^({re.escape(self.processing_class.pad_token)})+",
                    "",
                    text,
                )
                for text in prompts_text
            ]

        return prompts, prompts_text, prompt_ids, prompt_mask

    def _run_generation_backends(
        self,
        batch: GenerationBatch,
    ) -> torch.Tensor:
        """Run the appropriate generation backend and return completion ids."""

        def _extract_completion_ids(result):
            if isinstance(result, tuple) and len(result) >= 2:
                return result[1]
            return result

        if self.rollout_fn is not None and not self.use_vllm and not self.use_transformers_paged:
            completion_ids = _extract_completion_ids(self._generate_two_stage(batch.prompt_ids))
        elif self.use_vllm:
            completion_ids = _extract_completion_ids(self._generate_with_vllm(batch))
        elif self.use_transformers_paged:
            completion_ids = self._generate_with_paged(batch)
        else:
            completion_ids = _extract_completion_ids(
                self._generate_with_hf(
                    batch.prompt_ids,
                    batch.prompt_mask,
                )
            )

        return completion_ids

    def _generate_two_stage(
        self,
        prompt_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use an external rollout function for two-stage generation."""
        _, full_ids = self.rollout_fn(
            prompt_ids,
            max_new_tokens=self.max_completion_length,
        )
        completion_ids = full_ids[:, prompt_ids.size(1) :]
        return full_ids, completion_ids

    def _generate_with_vllm(
        self,
        batch: GenerationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using a vLLM client."""
        device = canonicalize_device(batch.device)
        completion_ids: list[Optional[list[int]]]
        broadcast_fn = resolve_dependency(
            self,
            "broadcast_object_list",
            broadcast_object_list,
        )
        gather_fn = resolve_dependency(
            self,
            "gather_object",
            gather_object,
        )
        all_prompts = broadcast_fn(  # pylint: disable=assignment-from-no-return
            gather_fn(batch.prompts_text),
            from_process=0,
        )
        if self.accelerator.is_main_process:
            ordered = all_prompts[:: self.num_generations]
            with resolve_dependency(
                self,
                "profiling_context",
                profiling_context,
            )(
                self,
                "vLLM.generate",
            ):
                completion_ids = self.vllm_client.generate(
                    prompts=ordered,
                    n=self.num_generations,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding_regex=self.guided_decoding_regex,
                )
        else:
            completion_ids = [None] * len(batch.prompts_text)

        completion_ids = broadcast_fn(  # pylint: disable=assignment-from-no-return
            completion_ids,
            from_process=0,
        )
        completion_slice = slice(
            self.accelerator.process_index * len(batch.prompts),
            (self.accelerator.process_index + 1) * len(batch.prompts),
        )
        completion_ids = completion_ids[completion_slice]
        if not completion_ids or all(ids in (None, []) for ids in completion_ids):
            batch_size = batch.prompt_ids.size(0)
            completion_padded = torch.zeros(
                batch_size,
                0,
                dtype=torch.long,
                device=device,
            )
        else:
            completion_tensors = [torch.tensor(ids or [], device=device) for ids in completion_ids]

            pad_fn = resolve_dependency(self, "pad", pad)
            completion_padded = pad_fn(
                completion_tensors,
                padding_value=self.processing_class.pad_token_id,
            )

        prompt_completion_ids = torch.cat(
            [batch.prompt_ids, completion_padded],
            dim=1,
        )
        return prompt_completion_ids, completion_padded

    def _generate_with_paged(
        self,
        batch: GenerationBatch,
    ) -> torch.Tensor:
        """Generate completions using transformers paged attention."""
        device = canonicalize_device(batch.device)
        prompt_inputs = self.processing_class(text=batch.prompts_text)
        previous_attn = getattr(
            self.model_wrapped.config,
            "_attn_implementation",
            None,
        )

        pad_fn = resolve_dependency(self, "pad", pad)
        attn_available_fn = resolve_dependency(
            self,
            "is_flash_attn_2_available",
            is_flash_attn_2_available,
        )
        if attn_available_fn():
            setattr(
                self.model_wrapped.config,
                "_attn_implementation",
                "paged_attention",
            )
        else:
            setattr(
                self.model_wrapped.config,
                "_attn_implementation",
                "sdpa_paged",
            )
        with (
            resolve_dependency(
                self,
                "profiling_context",
                profiling_context,
            )(
                self,
                "transformers.generate_batch",
            ),
            resolve_dependency(
                self,
                "unwrap_model_for_generation",
                unwrap_model_for_generation,
            )(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            (
                resolve_dependency(
                    self,
                    "FSDP",
                    FSDP,
                ).summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ),
        ):
            if self.args.bf16:
                unwrapped_model.to(torch.bfloat16)
            elif self.args.fp16:
                unwrapped_model.to(torch.float16)
            with torch.inference_mode():
                all_outputs = unwrapped_model.generate_batch(
                    prompt_inputs.input_ids,
                    generation_config=self.generation_config,
                    progress_bar=False,
                )

        completion_ids = [output.generated_tokens for output in all_outputs.values()]

        def _normalize_ids_to_tensor(seq, target_device):
            """
            Convert a sequence or tensor-like object to a tensor on target_device.

            When a real torch Tensor is provided, this keeps the existing storage
            and just moves it to the requested device to avoid copy-construct
            warnings. Lightweight stubs used in tests typically do not implement
            ``to``; in those cases we fall back to ``torch.tensor``.
            """
            if hasattr(seq, "to"):
                return seq.to(target_device)
            return torch.tensor(seq, device=target_device)

        completion_padded = pad_fn(
            [_normalize_ids_to_tensor(ids, device) for ids in completion_ids],
            padding_value=self.processing_class.pad_token_id,
            padding_side="right",
        )

        prompt_padded = pad_fn(
            [_normalize_ids_to_tensor(ids, device) for ids in prompt_inputs.input_ids],
            padding_value=self.processing_class.pad_token_id,
            padding_side="left",
        )

        batch.prompt_ids = prompt_padded
        try:
            mask = prompt_padded.ne(
                self.processing_class.pad_token_id,
            )  # type: ignore[attr-defined]
        except AttributeError:
            try:
                mask = ~(
                    prompt_padded == self.processing_class.pad_token_id  # type: ignore[operator]
                )
            except (RuntimeError, TypeError, ValueError):
                mask = prompt_padded
        try:
            mask = mask.long()  # type: ignore[attr-defined]
        except AttributeError:
            try:
                mask = torch.tensor(
                    mask,
                    dtype=getattr(torch, "long", None),
                    device=device,
                )
            except (RuntimeError, TypeError, ValueError):
                pass
        batch.prompt_mask = mask

        if previous_attn is not None:
            setattr(
                self.model_wrapped.config,
                "_attn_implementation",
                previous_attn,
            )

        return completion_padded

    def _generate_with_hf(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using standard transformers.generate."""
        with (
            resolve_dependency(
                self,
                "profiling_context",
                profiling_context,
            )(
                self,
                "transformers.generate",
            ),
            resolve_dependency(
                self,
                "unwrap_model_for_generation",
                unwrap_model_for_generation,
            )(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )

        prompt_len = prompt_ids.size(1)
        completion_ids_slice = prompt_completion_ids[:, prompt_len:]
        return prompt_completion_ids, completion_ids_slice

    def _build_completion_mask(
        self,
        completion_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor]:
        """Build EOS-based completion mask and lengths."""
        device = canonicalize_device(device)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            device=device,
            dtype=torch.long,
        )
        eos_rows = is_eos.any(dim=1)
        eos_first_idx = is_eos.int().argmax(dim=1)
        try:
            eos_rows_list = eos_rows.tolist()
        except AttributeError:
            eos_rows_list = eos_rows
        for row_idx, has_eos in enumerate(eos_rows_list):
            if has_eos:
                eos_idx[row_idx] = eos_first_idx[row_idx]
        seq_idx = torch.arange(
            is_eos.size(1),
            device=device,
        ).expand_as(is_eos)
        comp_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            comp_mask *= (~truncated).unsqueeze(1).int()

        comp_ids_list = [
            [int(token_id) for token_id, mask in zip(row, mask_row) if mask]
            for row, mask_row in zip(completion_ids, comp_mask)
        ]
        completion_lengths = comp_mask.sum(1)

        return comp_mask, comp_ids_list, completion_lengths, is_eos

    def _decode_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompts: list[Any],
        completion_ids: torch.Tensor,
    ) -> tuple[list[str], list[Union[str, list[dict[str, str]]]]]:
        """Decode completion ids into text and conversational structures."""
        completions_text = self.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        is_conv_fn = resolve_dependency(
            self,
            "is_conversational",
            is_conversational,
        )
        if is_conv_fn(inputs[0]):
            completions: list[Union[str, list[dict[str, str]]]] = []
            for prompt_messages, completion_text in zip(prompts, completions_text):
                bootstrap = prompt_messages.pop()["content"] if prompt_messages[-1]["role"] == "assistant" else ""
                completions.append(
                    [
                        {
                            "role": "assistant",
                            "content": bootstrap + completion_text,
                        }
                    ]
                )
        else:
            completions = completions_text

        return completions_text, completions
