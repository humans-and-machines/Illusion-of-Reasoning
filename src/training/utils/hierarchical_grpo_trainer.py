"""Hierarchical GRPO trainer utilities."""

from __future__ import annotations

from typing import Any, Union

from . import hierarchical_grpo_dependencies as _deps
from .hierarchical_grpo_generation import GenerationBatch, HierarchicalGenerationMixin
from .hierarchical_grpo_rewards import HierarchicalRewardMixin
from .hierarchical_grpo_rewards import RewardStatistics as _RewardStatistics


# Re-export dependency handles so callers can keep importing from this module.
for _name in _deps.__all__:
    globals()[_name] = getattr(_deps, _name)
RewardStatistics = _RewardStatistics

# Explicit bindings for static analyzers (mirrors dynamic re-exports above).
torch = _deps.torch
FSDP = _deps.FSDP
broadcast_object_list = _deps.broadcast_object_list
gather_object = _deps.gather_object
Trainer = _deps.Trainer
TrainerCallback = _deps.TrainerCallback
PaddingStrategy = _deps.PaddingStrategy
is_flash_attn_2_available = _deps.is_flash_attn_2_available
GRPOTrainer = _deps.GRPOTrainer
is_conversational = _deps.is_conversational
maybe_apply_chat_template = _deps.maybe_apply_chat_template
profiling_context = _deps.profiling_context
pad = _deps.pad
unwrap_model_for_generation = _deps.unwrap_model_for_generation
canonicalize_device = _deps.canonicalize_device

__all__ = [
    "GenerationBatch",
    "RewardStatistics",
    "HierarchicalGRPOTrainer",
    *_deps.__all__,
]


class HierarchicalGRPOTrainer(
    HierarchicalGenerationMixin,
    HierarchicalRewardMixin,
    GRPOTrainer,
):
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
        device = _deps.canonicalize_device(self.accelerator.device)

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
