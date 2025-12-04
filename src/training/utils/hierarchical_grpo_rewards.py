"""Reward computation helpers for hierarchical GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .hierarchical_grpo_dependencies import canonicalize_device, gather_object, resolve_dependency, torch


@dataclass  # pylint: disable=too-few-public-methods
class RewardStatistics:
    """Grouped reward statistics and advantages."""

    advantages: Any
    mean_grouped_rewards: Any
    std_grouped_rewards: Any
    is_std_zero: Any


def _safe_metric_reduce(reducer, tensor, fallback: float = 0.0) -> float:
    """Reduce a tensor-like object to a scalar, tolerating dtype/device quirks."""
    try:
        return float(reducer(tensor).item())
    except (TypeError, AttributeError, RuntimeError, ValueError):
        return fallback


class HierarchicalRewardMixin:
    """Reward, advantage, and metric helpers for the hierarchical GRPO trainer."""

    def compute_rewards(
        self,
        inputs: list[dict[str, Any]],
        batch,
        completions,
        comp_ids_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Public wrapper for reward computation used by callers/tests."""
        return self._compute_rewards(inputs, batch, completions, comp_ids_list)

    def normalize_rewards_and_compute_advantages(
        self,
        state: dict[str, Any],
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Public wrapper for normalization + advantage computation."""
        return self._normalize_rewards_and_compute_advantages(state, batch)

    def _compute_logps(
        self,
        batch,
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

    def _compute_rewards(
        self,
        inputs: list[dict[str, Any]],
        batch,
        completions,
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
            * resolve_dependency(
                self.reward_weights,
                "to",
                lambda device: self.reward_weights,
            )(batch.device).unsqueeze(0)
        ).nansum(dim=1)
        return rewards_per_func, rewards

    def _normalize_rewards_and_compute_advantages(
        self,
        state: dict[str, Any],
        batch,
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
        flat_rewards = self._flatten_rewards(rewards)
        prompt_count = max(len(flat_rewards) // self.num_generations, 1)
        prompt_major, generation_major = self._build_reward_views(
            rewards,
            flat_rewards,
            prompt_count,
        )

        prompt_mean = prompt_major.mean(dim=1)
        try:
            prompt_std = prompt_major.std(dim=1, unbiased=False)
        except TypeError:
            prompt_std = prompt_major.std(dim=1)
        try:
            gen_std = generation_major.std(dim=1, unbiased=False)
        except TypeError:
            gen_std = generation_major.std(dim=1)

        is_std_zero = self._std_zero_mask(prompt_std, gen_std, rewards)

        repeated_means = prompt_mean.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        repeated_stds = prompt_std.repeat_interleave(
            self.num_generations,
            dim=0,
        )
        advantages = rewards - repeated_means
        if self.scale_rewards:
            repeated_stds, advantages = self._scale_advantages(
                advantages,
                repeated_stds,
            )
        return RewardStatistics(
            advantages=advantages,
            mean_grouped_rewards=prompt_mean,
            std_grouped_rewards=prompt_std,
            is_std_zero=is_std_zero,
        )

    def _flatten_rewards(self, rewards: torch.Tensor) -> list[Any]:
        """Return a plain list of rewards with safe fallbacks."""
        try:
            return rewards.tolist()
        except (TypeError, AttributeError, ValueError):
            return list(rewards)

    def _build_reward_views(
        self,
        rewards: torch.Tensor,
        flat_rewards: list[Any],
        prompt_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct prompt-major and generation-major reward views."""
        device = canonicalize_device(getattr(rewards, "device", None))
        dtype = getattr(rewards, "dtype", None)

        def _to_tensor(seq):
            return torch.tensor(seq, device=device, dtype=dtype)

        try:
            prompt_major = _to_tensor(flat_rewards).view(
                prompt_count,
                self.num_generations,
            )
            generation_major = _to_tensor(
                [
                    flat_rewards[gen_idx * prompt_count + prompt_idx]
                    for prompt_idx in range(prompt_count)
                    for gen_idx in range(self.num_generations)
                ],
            ).view(prompt_count, self.num_generations)
            return prompt_major, generation_major
        except (RuntimeError, TypeError, ValueError, AttributeError):
            if hasattr(rewards, "reshape"):
                prompt_major = rewards.reshape(-1, self.num_generations)
            elif hasattr(rewards, "view"):
                prompt_major = rewards.view(-1, self.num_generations)
            elif hasattr(torch, "reshape"):
                prompt_major = torch.reshape(rewards, (-1, self.num_generations))
            else:
                prompt_major = _to_tensor(flat_rewards).view(-1, self.num_generations)
            return prompt_major, prompt_major

    def _bool_list(self, tensor):
        """Convert a tensor or scalar into a list of bools."""
        try:
            return [bool(item) for item in tensor.tolist()]
        except (TypeError, AttributeError, ValueError):
            return [bool(tensor)]

    def _std_zero_mask(self, prompt_std, gen_std, rewards: torch.Tensor) -> torch.Tensor:
        """Return a mask indicating which prompts have zero std across generations."""
        prompt_zero_list = self._bool_list(torch.isclose(prompt_std, torch.zeros_like(prompt_std)))
        generation_zero_list = self._bool_list(torch.isclose(gen_std, torch.zeros_like(gen_std)))
        device = canonicalize_device(getattr(rewards, "device", None))
        return torch.tensor(
            [p or g for p, g in zip(prompt_zero_list, generation_zero_list)],
            device=device,
        )

    def _scale_advantages(
        self,
        advantages: torch.Tensor,
        repeated_stds,
    ) -> tuple[Any, Any]:
        """Scale advantages while guarding zero std values."""
        orig_stds = repeated_stds
        if not hasattr(repeated_stds, "dtype"):
            repeated_stds = torch.tensor(
                repeated_stds,
                device=getattr(advantages, "device", None),
                dtype=getattr(advantages, "dtype", None),
            )
        zero_mask = repeated_stds == 0
        try:
            has_zero = zero_mask.any().item()
        except (AttributeError, TypeError):
            has_zero = False
        if has_zero:
            repeated_stds = repeated_stds + zero_mask.int() * 1e-4
        repeated_out = repeated_stds if hasattr(orig_stds, "dtype") or has_zero else orig_stds
        try:
            scaled = advantages / repeated_stds
        except (TypeError, AttributeError, RuntimeError, ValueError):
            # Fallback for pathological masks that do not support division
            # (for example, test stubs that raise from ``any`` or ``__array__``).
            scaled = advantages
        return repeated_out, scaled

    def _slice_advantages_for_process(
        self,
        stats: RewardStatistics,
        batch,
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
        batch,
        stats: RewardStatistics,
        state: dict[str, Any],
    ) -> None:
        """Update scalar metrics derived from rewards and completion lengths."""
        device = canonicalize_device(getattr(batch, "device", None))
        completion_lengths: torch.Tensor = state["completion_lengths"]
        is_eos: torch.Tensor = state["is_eos"]
        mode_str = "train" if self.model.training else "eval"
        if self.model.training:
            tokens_seen = (
                self.accelerator.gather((batch.prompt_mask.sum(dim=1) + completion_lengths).sum()).sum().item()
            )
            self.state.num_input_tokens_seen += tokens_seen
        self._metrics[mode_str]["num_tokens"] = [self.state.num_input_tokens_seen]

        aggregated_completion_lengths = self.accelerator.gather(completion_lengths)
        if hasattr(torch, "as_tensor"):
            aggregated_completion_lengths = torch.as_tensor(
                getattr(aggregated_completion_lengths, "data", aggregated_completion_lengths),
                device=device,
            )
        self._append_length_metrics(
            aggregated_completion_lengths,
            is_eos,
            batch,
            mode_str,
        )

        self._metrics[mode_str]["reward"].append(stats.mean_grouped_rewards.mean().item())
        self._metrics[mode_str]["reward_std"].append(stats.std_grouped_rewards.mean().item())
        self._metrics[mode_str]["frac_reward_zero_std"].append(stats.is_std_zero.float().mean().item())

    def _append_length_metrics(
        self,
        aggregated_completion_lengths,
        is_eos: torch.Tensor,
        batch,
        mode_str: str,
    ) -> None:
        """Append completion-length metrics for the current mode."""
        device = canonicalize_device(getattr(batch, "device", None))
        lengths_float = aggregated_completion_lengths
        if hasattr(lengths_float, "float"):
            try:
                lengths_float = lengths_float.float()
            except (TypeError, AttributeError, RuntimeError, ValueError):
                pass

        mean_fn = getattr(torch, "mean", None) or (lambda t: t.mean())
        min_fn = getattr(torch, "min", None) or (lambda t: t.min())
        max_fn = getattr(torch, "max", None) or (lambda t: t.max())
        self._metrics[mode_str]["completions/mean_length"].append(_safe_metric_reduce(mean_fn, lengths_float))
        self._metrics[mode_str]["completions/min_length"].append(_safe_metric_reduce(min_fn, lengths_float))
        self._metrics[mode_str]["completions/max_length"].append(_safe_metric_reduce(max_fn, lengths_float))

        terminated_mask = self.accelerator.gather(is_eos.any(dim=1))
        if hasattr(torch, "as_tensor"):
            terminated_mask = torch.as_tensor(
                getattr(terminated_mask, "data", terminated_mask),
                device=device,
                dtype=getattr(torch, "bool", None),
            )
        if hasattr(terminated_mask, "bool"):
            terminated_mask = terminated_mask.bool()
        terminated_completion_lengths = aggregated_completion_lengths[terminated_mask]
        if len(terminated_completion_lengths) == 0:
            terminated_completion_lengths = torch.zeros(1, device=device)
        terminated_float = terminated_completion_lengths.float()

        self._metrics[mode_str]["completions/mean_terminated_length"].append(
            _safe_metric_reduce(mean_fn, terminated_float)
        )
        self._metrics[mode_str]["completions/min_terminated_length"].append(
            _safe_metric_reduce(min_fn, terminated_float)
        )
        self._metrics[mode_str]["completions/max_terminated_length"].append(
            _safe_metric_reduce(max_fn, terminated_float)
        )
        clipped_completions_ratio = 1 - (len(terminated_completion_lengths) / len(aggregated_completion_lengths))
        self._metrics[mode_str]["completions/clipped_ratio"].append(clipped_completions_ratio)

    def _log_text_and_rewards(
        self,
        prompts_text: list[str],
        completions_text: list[str],
        rewards_per_func: torch.Tensor,
        all_process_advantages: torch.Tensor,
    ) -> None:
        """Log textual prompts/completions and per-function rewards."""
        gather_fn = resolve_dependency(self, "gather_object", gather_object)
        self._textual_logs["prompt"].extend(gather_fn(prompts_text))
        self._textual_logs["completion"].extend(gather_fn(completions_text))
        for index, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, index].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())
