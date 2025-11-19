"""Hierarchical GRPO utilities and rollout helpers."""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

try:  # pragma: no cover - optional dependencies at runtime
    import torch
    from accelerate.utils import broadcast_object_list, gather_object
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.utils.rnn import pad_sequence

    from transformers import PreTrainedTokenizerBase, Trainer, TrainerCallback
    from transformers.generation.utils import GenerationMixin
    from transformers.tokenization_utils_base import PaddingStrategy
    from transformers.utils import is_flash_attn_2_available
except ImportError:  # pragma: no cover - type-check / lint environment
    torch = None
    FSDP = None

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

    PreTrainedTokenizerBase = object
    GenerationMixin = object
    Trainer = object
    TrainerCallback = object
    PaddingStrategy = object

    def is_flash_attn_2_available() -> bool:
        """Stub flash-attn-2 check that always returns False."""
        return False

try:  # pragma: no cover - optional dependencies at runtime
    from trl import GRPOTrainer
    from trl.data_utils import is_conversational, maybe_apply_chat_template
    from trl.extras.profiling import profiling_context
    from trl.trainer.grpo_trainer import pad, unwrap_model_for_generation
except ImportError:  # pragma: no cover - type-check / lint environment
    GRPOTrainer = Trainer

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


@dataclass
class GenerationBatch:
    """Container for generation inputs shared across helper methods."""

    prompts: list[Any]
    prompts_text: list[str]
    prompt_ids: Any
    prompt_mask: Any
    device: Any


@dataclass
class RewardStatistics:
    """Grouped reward statistics and advantages."""

    advantages: Any
    mean_grouped_rewards: Any
    std_grouped_rewards: Any
    is_std_zero: Any


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

    def reset_textual_logs(self) -> None:
        """Clear accumulated textual logs."""
        if hasattr(self, "_textual_logs"):
            self._textual_logs.clear()

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts, prompts_text, prompt_ids, prompt_mask = self._prepare_prompts(
            inputs,
            device,
        )

        batch = GenerationBatch(
            prompts=prompts,
            prompts_text=prompts_text,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            device=device,
        )

        completion_ids = self._run_generation_backends(batch)

        return self._postprocess_and_score(
            inputs,
            batch,
            completion_ids,
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
        batch: GenerationBatch,
    ) -> torch.Tensor:
        """Run the appropriate generation backend and return completion ids."""
        if (
            self.rollout_fn is not None
            and not self.use_vllm
            and not self.use_transformers_paged
        ):
            _, completion_ids = self._generate_two_stage(batch.prompt_ids)
        elif self.use_vllm:
            _, completion_ids = self._generate_with_vllm(batch)
        elif self.use_transformers_paged:
            completion_ids = self._generate_with_paged(batch)
        else:
            _, completion_ids = self._generate_with_hf(
                batch.prompt_ids,
                batch.prompt_mask,
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
        completion_ids: list[Optional[list[int]]]
        all_prompts = broadcast_object_list(
            gather_object(batch.prompts_text),
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
            completion_ids = [None] * len(batch.prompts_text)

        completion_ids = broadcast_object_list(
            completion_ids,
            from_process=0,
        )
        completion_slice = slice(
            self.accelerator.process_index * len(batch.prompts),
            (self.accelerator.process_index + 1) * len(batch.prompts),
        )
        completion_ids = completion_ids[completion_slice]
        completion_tensors = [
            torch.tensor(ids, device=batch.device) for ids in completion_ids
        ]

        if completion_tensors:
            completion_padded = pad(
                completion_tensors,
                padding_value=self.processing_class.pad_token_id,
            )
        else:
            batch_size = batch.prompt_ids.size(0)
            completion_padded = torch.zeros(
                batch_size,
                0,
                dtype=torch.long,
                device=batch.device,
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
        prompt_inputs = self.processing_class(text=batch.prompts_text)
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
            torch.tensor(ids, device=batch.device) for ids in completion_ids
        ]
        completion_padded = pad(
            completion_tensors,
            padding_value=self.processing_class.pad_token_id,
            padding_side="right",
        )

        prompt_tensors = [
            torch.tensor(ids, device=batch.device)
            for ids in prompt_inputs.input_ids
        ]
        prompt_padded = pad(
            prompt_tensors,
            padding_value=self.processing_class.pad_token_id,
            padding_side="left",
        )

        batch.prompt_ids = prompt_padded
        batch.prompt_mask = (prompt_padded != self.processing_class.pad_token_id).int()

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
        batch: GenerationBatch,
        state: dict[str, Any],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute old and reference per-token log-probabilities."""
        completion_ids = state["completion_ids"]
        completion_mask = state["completion_mask"]
        attention_mask = torch.cat(
            [batch.prompt_mask, completion_mask],
            dim=1,
        )
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            gen_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % gen_every != 0:
                if self.model.training:
                    batch_size = self.args.per_device_train_batch_size
                else:
                    batch_size = self.args.per_device_eval_batch_size
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model,
                    torch.cat([batch.prompt_ids, completion_ids], 1),
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
                        torch.cat([batch.prompt_ids, completion_ids], 1),
                        attention_mask,
                        logits_to_keep,
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model,
                            torch.cat([batch.prompt_ids, completion_ids], 1),
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
        batch: GenerationBatch,
        completions: list[Union[str, list[dict[str, str]]]],
        comp_ids_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-function rewards and aggregate per-sequence rewards."""
        rewards_per_func = self._calculate_rewards(
            inputs,
            batch.prompts,
            completions,
            comp_ids_list,
        )
        rewards = (
            rewards_per_func
            * self.reward_weights.to(batch.device).unsqueeze(0)
        ).nansum(dim=1)
        return rewards_per_func, rewards

    def _normalize_rewards_and_compute_advantages(
        self,
        state: dict[str, Any],
        batch: GenerationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize rewards, compute advantages, and update token metrics."""
        rewards: torch.Tensor = state["rewards"]
        stats = self._normalize_rewards(rewards)
        advantages, all_process_advantages = self._slice_advantages_for_process(
            stats,
            batch,
        )
        self._update_reward_metrics(
            batch,
            stats,
            state,
        )
        return advantages, all_process_advantages

    def _normalize_rewards(self, rewards: torch.Tensor) -> RewardStatistics:
        """Compute grouped reward statistics and un-sliced advantages."""
        rewards_view = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = rewards_view.mean(dim=1)
        std_grouped_rewards = rewards_view.std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards,
            torch.zeros_like(std_grouped_rewards),
        )
        repeated_means = mean_grouped_rewards.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        repeated_stds = std_grouped_rewards.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        advantages = rewards - repeated_means
        if self.scale_rewards:
            advantages = advantages / (repeated_stds + 1e-4)
        return RewardStatistics(
            advantages=advantages,
            mean_grouped_rewards=mean_grouped_rewards,
            std_grouped_rewards=std_grouped_rewards,
            is_std_zero=is_std_zero,
        )

    def _slice_advantages_for_process(
        self,
        stats: RewardStatistics,
        batch: GenerationBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice advantages for the local process."""
        all_process_advantages = stats.advantages.clone()
        batch_per_process = len(batch.prompts)
        process_index = self.accelerator.process_index
        start = process_index * batch_per_process
        end = (process_index + 1) * batch_per_process
        advantages = stats.advantages[start:end]
        return advantages, all_process_advantages

    def _update_reward_metrics(
        self,
        batch: GenerationBatch,
        stats: RewardStatistics,
        state: dict[str, Any],
    ) -> None:
        """Update scalar metrics derived from rewards and completion lengths."""
        completion_lengths: torch.Tensor = state["completion_lengths"]
        is_eos: torch.Tensor = state["is_eos"]
        mode_str = "train" if self.model.training else "eval"
        if self.model.training:
            tokens_seen = (
                self.accelerator.gather(
                    (batch.prompt_mask.sum(dim=1) + completion_lengths).sum()
                )
                .sum()
                .item()
            )
            self.state.num_input_tokens_seen += tokens_seen
        self._metrics[mode_str]["num_tokens"] = [self.state.num_input_tokens_seen]

        aggregated_completion_lengths = self.accelerator.gather(completion_lengths)
        lengths_float = aggregated_completion_lengths.float()
        self._metrics[mode_str]["completions/mean_length"].append(
            lengths_float.mean().item()
        )
        self._metrics[mode_str]["completions/min_length"].append(
            lengths_float.min().item()
        )
        self._metrics[mode_str]["completions/max_length"].append(
            lengths_float.max().item()
        )

        terminated_mask = self.accelerator.gather(is_eos.any(dim=1))
        terminated_completion_lengths = aggregated_completion_lengths[terminated_mask]
        if len(terminated_completion_lengths) == 0:
            terminated_completion_lengths = torch.zeros(1, device=batch.device)
        terminated_float = terminated_completion_lengths.float()
        self._metrics[mode_str]["completions/mean_terminated_length"].append(
            terminated_float.mean().item()
        )
        self._metrics[mode_str]["completions/min_terminated_length"].append(
            terminated_float.min().item()
        )
        self._metrics[mode_str]["completions/max_terminated_length"].append(
            terminated_float.max().item()
        )
        clipped_completions_ratio = 1 - (
            len(terminated_completion_lengths) / len(aggregated_completion_lengths)
        )
        self._metrics[mode_str]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )

        self._metrics[mode_str]["reward"].append(
            stats.mean_grouped_rewards.mean().item()
        )
        self._metrics[mode_str]["reward_std"].append(
            stats.std_grouped_rewards.mean().item()
        )
        self._metrics[mode_str]["frac_reward_zero_std"].append(
            stats.is_std_zero.float().mean().item()
        )

    def _log_text_and_rewards(
        self,
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
        batch: GenerationBatch,
        completion_ids: torch.Tensor,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Post-process completions, compute rewards, metrics, and outputs."""
        state: dict[str, Any] = {}

        state["completion_ids"] = completion_ids
        (
            state["completion_mask"],
            state["comp_ids_list"],
            state["completion_lengths"],
            state["is_eos"],
        ) = self._build_completion_mask(
            completion_ids,
            batch.device,
        )

        (
            state["old_per_token_logps"],
            state["ref_per_token_logps"],
        ) = self._compute_logps(
            batch,
            state,
        )

        (
            state["completions_text"],
            state["completions"],
        ) = self._decode_completions(
            inputs,
            batch.prompts,
            completion_ids,
        )

        (
            state["rewards_per_func"],
            state["rewards"],
        ) = self._compute_rewards(
            inputs,
            batch,
            state["completions"],
            state["comp_ids_list"],
        )

        (
            state["advantages"],
            state["all_process_advantages"],
        ) = self._normalize_rewards_and_compute_advantages(
            state,
            batch,
        )

        self._log_text_and_rewards(
            batch.prompts_text,
            state["completions_text"],
            state["rewards_per_func"],
            state["all_process_advantages"],
        )

        return {
            "prompt_ids": batch.prompt_ids,
            "prompt_mask": batch.prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": state["completion_mask"],
            "advantages": state["advantages"],
            "old_per_token_logps": state["old_per_token_logps"],
            "ref_per_token_logps": state["ref_per_token_logps"],
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
