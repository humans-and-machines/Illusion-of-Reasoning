"""Hierarchical GRPO utilities and rollout helpers."""

from __future__ import annotations

import re
from contextlib import nullcontext
from typing import Any, Optional, Tuple, Union

try:  # pragma: no cover - optional dependency
    import torch
    from accelerate.utils import broadcast_object_list, gather_object
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.utils.rnn import pad_sequence

    from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback
    from transformers.generation.utils import GenerationMixin
    from transformers.tokenization_utils_base import PaddingStrategy
    from transformers.utils import is_flash_attn_2_available
except ImportError:  # pragma: no cover - type-check / lint environment
    torch = None  # type: ignore[assignment]
    FSDP = None  # type: ignore[assignment]

    def broadcast_object_list(*_args: Any, **_kwargs: Any) -> Any:
        """Stub broadcast_object_list that raises when accelerate is missing."""
        msg = "accelerate is required for hierarchical GRPO utilities."
        raise ImportError(msg)

    def gather_object(*_args: Any, **_kwargs: Any) -> Any:
        """Stub gather_object that raises when accelerate is missing."""
        msg = "accelerate is required for hierarchical GRPO utilities."
        raise ImportError(msg)

    def pad_sequence(*_args: Any, **_kwargs: Any) -> Any:
        """Stub pad_sequence that raises when torch is missing."""
        msg = "torch is required for hierarchical GRPO utilities (pad_sequence)."
        raise ImportError(msg)

    PreTrainedTokenizerBase = object  # type: ignore[assignment]
    GenerationMixin = object  # type: ignore[assignment]
    Trainer = object  # type: ignore[assignment]
    TrainerCallback = object  # type: ignore[assignment]
    PaddingStrategy = object  # type: ignore[assignment]

    def is_flash_attn_2_available() -> bool:
        """Stub flash-attn-2 check that always returns False."""
        return False

try:  # pragma: no cover - optional dependency
    from trl import GRPOTrainer
    from trl.data_utils import is_conversational, maybe_apply_chat_template
    from trl.extras.profiling import profiling_context
    from trl.trainer.grpo_trainer import pad, unwrap_model_for_generation
except ImportError:  # pragma: no cover - type-check / lint environment
    GRPOTrainer = Trainer  # type: ignore[assignment]

    def is_conversational(*_args: Any, **_kwargs: Any) -> bool:
        """Stub conversational check that always returns False."""
        return False

    def maybe_apply_chat_template(example: Any, _processor: Any) -> dict:
        """Minimal passthrough: expect dicts with a 'prompt' key."""
        if isinstance(example, dict) and "prompt" in example:
            return {"prompt": example["prompt"]}
        return {"prompt": example}

    def profiling_context(*_args: Any, **_kwargs: Any):
        """Stub profiling context that acts as a no-op context manager."""
        return nullcontext()

    def pad(*_args: Any, **_kwargs: Any) -> Any:
        """Stub pad that raises when trl is missing."""
        msg = "trl is required for hierarchical GRPO training (pad)."
        raise ImportError(msg)

    def unwrap_model_for_generation(*_args: Any, **_kwargs: Any) -> Any:
        """Stub unwrap_model_for_generation that raises when trl is missing."""
        msg = "trl is required for hierarchical GRPO training (unwrap_model_for_generation)."
        raise ImportError(msg)


class HierarchicalGRPOTrainer(GRPOTrainer):
    """GRPO trainer that supports hierarchical two-stage rollouts and rich logging."""
    def __init__(
        self,
        *args,
        rollout_fn: Any = None,
        tokenizer: Any = None,
        return_reason: bool = True,
        callbacks: list[Any] = None,
        **kwargs,
    ):
        self.rollout_fn = rollout_fn
        self.tokenizer = tokenizer
        self.return_reason = return_reason

        # split callbacks into instances vs factories
        callback_instances, callback_factories = [], []
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, TrainerCallback):
                    callback_instances.append(callback)
                else:
                    callback_factories.append(callback)

        super().__init__(*args, callbacks=callback_instances, **kwargs)
        self.mask_truncated_completions = True

        for factory in callback_factories:
            self.add_callback(factory(self))

    @property
    def textual_logs(self) -> dict:
        """Expose textual logs via a public attribute for callbacks."""
        return self._textual_logs

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts, prompts_text, prompt_ids, prompt_mask = self._prepare_prompts(
            inputs,
            device,
        )

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            prompt_completion_ids,
        ) = self._run_generation_backends(
            prompts,
            prompts_text,
            prompt_ids,
            prompt_mask,
            device,
        )

        return self._postprocess_and_score(
            inputs,
            prompts,
            prompts_text,
            prompt_ids,
            prompt_mask,
            completion_ids,
            prompt_completion_ids,
            device,
            mode,
        )

    def _prepare_prompts(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        device: torch.device,
    ) -> tuple[list[Any], list[str], torch.Tensor, torch.Tensor]:
        """Tokenize prompts and optionally trim to max_prompt_length."""
        prompts = [example["prompt"] for example in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        encoding = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = encoding["input_ids"].to(device)
        prompt_mask = encoding["attention_mask"].to(device)

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
        prompts: list[Any],
        prompts_text: list[str],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the appropriate generation backend and return ids and masks."""
        if (
            self.rollout_fn is not None
            and not self.use_vllm
            and not self.use_transformers_paged
        ):
            prompt_completion_ids, completion_ids = self._generate_two_stage(prompt_ids)
        elif self.use_vllm:
            prompt_completion_ids, completion_ids = self._generate_with_vllm(
                prompts,
                prompts_text,
                prompt_ids,
                device,
            )
        elif self.use_transformers_paged:
            prompt_ids, prompt_completion_ids, completion_ids = self._generate_with_paged(
                prompts_text,
                prompt_ids,
                device,
            )
        else:
            prompt_completion_ids, completion_ids = self._generate_with_hf(
                prompt_ids,
                prompt_mask,
            )

        return prompt_ids, prompt_mask, completion_ids, prompt_completion_ids

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
        prompts: list[Any],
        prompts_text: list[str],
        prompt_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using a vLLM client."""
        completion_ids: list[Optional[list[int]]]
        all_prompts = broadcast_object_list(
            gather_object(prompts_text),
            from_process=0,
        )
        if self.accelerator.is_main_process:
            ordered = all_prompts[:: self.num_generations]
            with profiling_context(self, "vLLM.generate"):
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
            completion_ids = [None] * len(prompts_text)

        completion_ids = broadcast_object_list(
            completion_ids,
            from_process=0,
        )
        completion_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        completion_ids = completion_ids[completion_slice]
        completion_tensors = [
            torch.tensor(ids, device=device) for ids in completion_ids
        ]

        if completion_tensors:
            completion_padded = pad(
                completion_tensors,
                padding_value=self.processing_class.pad_token_id,
            )
        else:
            batch_size = prompt_ids.size(0)
            completion_padded = torch.zeros(
                batch_size,
                0,
                dtype=torch.long,
                device=device,
            )

        prompt_completion_ids = torch.cat([prompt_ids, completion_padded], dim=1)
        return prompt_completion_ids, completion_padded

    def _generate_with_paged(
        self,
        prompts_text: list[str],
        prompt_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate completions using transformers paged attention."""
        prompt_inputs = self.processing_class(text=prompts_text)
        previous_attn = getattr(
            self.model_wrapped.config,
            "_attn_implementation",
            None,
        )

        if is_flash_attn_2_available():
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
            profiling_context(self, "transformers.generate_batch"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
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

        completion_ids = [
            output.generated_tokens for output in all_outputs.values()
        ]
        completion_tensors = [
            torch.tensor(ids, device=device) for ids in completion_ids
        ]
        completion_padded = pad(
            completion_tensors,
            padding_value=self.processing_class.pad_token_id,
            padding_side="right",
        )

        prompt_tensors = [
            torch.tensor(ids, device=device)
            for ids in prompt_inputs.input_ids
        ]
        prompt_padded = pad(
            prompt_tensors,
            padding_value=self.processing_class.pad_token_id,
            padding_side="left",
        )
        prompt_completion_ids = torch.cat(
            [prompt_padded, completion_padded],
            dim=1,
        )

        if previous_attn is not None:
            setattr(
                self.model_wrapped.config,
                "_attn_implementation",
                previous_attn,
            )

        return prompt_padded, prompt_completion_ids, completion_padded

    def _generate_with_hf(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using standard transformers.generate."""
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
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
        prompt_ids_slice = prompt_completion_ids[:, :prompt_len]
        completion_ids_slice = prompt_completion_ids[:, prompt_len:]
        return prompt_completion_ids, completion_ids_slice

    def _build_completion_mask(
        self,
        completion_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor]:
        """Build EOS-based completion mask and lengths."""
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            device=device,
            dtype=torch.long,
        )
        eos_rows = is_eos.any(dim=1)
        eos_idx[eos_rows] = is_eos.int().argmax(dim=1)[eos_rows]
        seq_idx = torch.arange(
            is_eos.size(1),
            device=device,
        ).expand_as(is_eos)
        comp_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        comp_ids_list = [
            [int(token_id) for token_id, mask in zip(row, mask_row) if mask]
            for row, mask_row in zip(completion_ids, comp_mask)
        ]
        completion_lengths = comp_mask.sum(1)
        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            comp_mask *= (~truncated).unsqueeze(1).int()

        return comp_mask, comp_ids_list, completion_lengths, is_eos

    def _compute_logps(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute old and reference per-token log-probabilities."""
        with torch.no_grad():
            gen_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % gen_every != 0:
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model,
                    torch.cat([prompt_ids, completion_ids], 1),
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )["logps"]
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        torch.cat([prompt_ids, completion_ids], 1),
                        attention_mask,
                        logits_to_keep,
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model,
                            torch.cat([prompt_ids, completion_ids], 1),
                            attention_mask,
                            logits_to_keep,
                        )["logps"]
            else:
                ref_per_token_logps = None

        return old_per_token_logps, ref_per_token_logps

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
        if is_conversational(inputs[0]):
            completions: list[Union[str, list[dict[str, str]]]] = []
            for prompt_messages, completion_text in zip(prompts, completions_text):
                bootstrap = (
                    prompt_messages.pop()["content"]
                    if prompt_messages[-1]["role"] == "assistant"
                    else ""
                )
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

    def _compute_rewards(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompts: list[Any],
        completions: list[Union[str, list[dict[str, str]]]],
        comp_ids_list: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-function rewards and aggregate per-sequence rewards."""
        rewards_per_func = self._calculate_rewards(
            inputs,
            prompts,
            completions,
            comp_ids_list,
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        return rewards_per_func, rewards

    def _normalize_rewards_and_compute_advantages(
        self,
        rewards: torch.Tensor,
        prompts: list[Any],
        attention_mask: torch.Tensor,
        completion_lengths: torch.Tensor,
        is_eos: torch.Tensor,
        device: torch.device,
        mode: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Normalize rewards, compute advantages, and update token metrics."""
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards,
            torch.zeros_like(std_grouped_rewards),
        )

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            tokens_seen = self.accelerator.gather(attention_mask.sum()).sum().item()
            self.state.num_input_tokens_seen += tokens_seen
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        aggregated_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            aggregated_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            aggregated_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            aggregated_completion_lengths.float().max().item()
        )

        aggregated_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        terminated_completion_lengths = aggregated_completion_lengths[
            aggregated_terminated_with_eos
        ]
        clipped_completions_ratio = 1 - (
            len(terminated_completion_lengths) / len(aggregated_completion_lengths)
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(terminated_completion_lengths) == 0:
            terminated_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            terminated_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            terminated_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            terminated_completion_lengths.float().max().item()
        )

        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(
            std_grouped_rewards.mean().item()
        )
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        return (
            advantages,
            all_process_advantages,
            mean_grouped_rewards,
            std_grouped_rewards,
            is_std_zero,
        )

    def _log_text_and_rewards(
        self,
        mode: str,
        prompts_text: list[str],
        completions_text: list[str],
        rewards_per_func: torch.Tensor,
        all_process_advantages: torch.Tensor,
    ) -> None:
        """Log textual prompts/completions and per-function rewards."""
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for index, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(
                rewards_per_func[:, index].tolist()
            )
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

    def _postprocess_and_score(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
        prompts: list[Any],
        prompts_text: list[str],
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_completion_ids: torch.Tensor,
        device: torch.device,
        mode: str,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Post-process completions, compute rewards, metrics, and outputs."""
        comp_mask, comp_ids_list, completion_lengths, is_eos = (
            self._build_completion_mask(
                completion_ids,
                device,
            )
        )

        attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_ids,
            completion_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
        )

        completions_text, completions = self._decode_completions(
            inputs,
            prompts,
            completion_ids,
        )

        rewards_per_func, rewards = self._compute_rewards(
            inputs,
            prompts,
            completions,
            comp_ids_list,
            device,
        )

        (
            advantages,
            all_process_advantages,
            mean_grouped_rewards,
            std_grouped_rewards,
            is_std_zero,
        ) = self._normalize_rewards_and_compute_advantages(
            rewards,
            prompts,
            attention_mask,
            completion_lengths,
            is_eos,
            device,
            mode,
        )

        self._log_text_and_rewards(
            mode,
            prompts_text,
            completions_text,
            rewards_per_func,
            all_process_advantages,
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": comp_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }


# your two‐stage rollout helper
class HierarchicalRollout:
    """
    Two-stage generation:
      1) Generate until </think> (or first <answer>), append the tags.
      2) Feed that full sequence back in to finish the answer.
    """

    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        vllm_client: Optional[Any] = None,
        max_reason_tokens: int = 800,
    ):
        self.model = model
        self.tok = tokenizer
        self.vllm_client = vllm_client
        self.max_reason_tokens = max_reason_tokens

        # your tags
        self.think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.answer_tag_ids = tokenizer.encode("<answer>", add_special_tokens=False)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using the two-stage hierarchical rollout."""
        return self(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    def get_tag_ids(self) -> Tuple[list[int], list[int]]:
        """Return the token ids used to detect </think> and <answer> tags."""
        return self.think_close_ids, self.answer_tag_ids

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        reason_ids = self._run_stage1_reasoning(
            input_ids,
            device,
            **gen_kwargs,
        )
        full_ids = self._run_stage2_answer(
            reason_ids,
            device,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        return reason_ids, full_ids

    def _run_stage1_reasoning(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the first-stage reasoning step and return padded reason ids."""
        if self.vllm_client:
            prompts = self.tok.batch_decode(
                input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            reason_lists = self.vllm_client.generate(
                prompts=prompts,
                n=1,
                max_tokens=self.max_reason_tokens,
                **gen_kwargs,
            )
        else:
            reason_tensor = self.model.generate(
                input_ids,
                max_new_tokens=self.max_reason_tokens,
                eos_token_id=self.think_close_ids[-1],
                do_sample=True,
                **gen_kwargs,
            )
            reason_lists = [tensor.tolist() for tensor in reason_tensor]

        padded_reason_ids = []
        for sequence in reason_lists:
            if sequence[-len(self.think_close_ids) :] != self.think_close_ids:
                sequence = sequence + self.think_close_ids + self.answer_tag_ids
            else:
                sequence = sequence + self.answer_tag_ids
            padded_reason_ids.append(torch.tensor(sequence, device=device))

        return pad_sequence(
            padded_reason_ids,
            batch_first=True,
            padding_value=self.tok.pad_token_id,
        )

    def _run_stage2_answer(
        self,
        reason_ids: torch.Tensor,
        device: torch.device,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the second-stage answer generation step."""
        if self.vllm_client:
            reason_texts = self.tok.batch_decode(
                reason_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer_lists = self.vllm_client.generate(
                prompts=reason_texts,
                n=1,
                max_tokens=max_new_tokens or 100,
                **gen_kwargs,
            )
            full_lists = [
                reason.tolist() + answer
                for reason, answer in zip(reason_ids, answer_lists)
            ]
        else:
            full_tensor = self.model.generate(
                reason_ids,
                max_new_tokens=max_new_tokens or 100,
                eos_token_id=self.tok.eos_token_id,
                do_sample=True,
                **gen_kwargs,
            )
            full_lists = [tensor.tolist() for tensor in full_tensor]

        return pad_sequence(
            [torch.tensor(sequence, device=device) for sequence in full_lists],
            batch_first=True,
            padding_value=self.tok.pad_token_id,
        )
