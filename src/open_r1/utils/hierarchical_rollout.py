import re
import torch
from torch.nn.utils.rnn import pad_sequence

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from transformers.utils import is_flash_attn_2_available

import torch
from typing import Optional, Any, Tuple, List, Union

from transformers.generation.utils import GenerationMixin
from transformers import PreTrainedTokenizerBase

# TRL trainer subclass
from trl import GRPOTrainer
from transformers.integrations import WandbCallback
from transformers import TrainerCallback

# needed inside _generate_and_score_completions
from accelerate.utils import broadcast_object_list, gather_object
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
from trl.extras.profiling import profiling_context
from trl.trainer.grpo_trainer import unwrap_model_for_generation, pad
from trl.data_utils import maybe_apply_chat_template
from trl.data_utils import is_conversational
from transformers import Trainer
from transformers.tokenization_utils_base import PaddingStrategy
import re



class HierarchicalGRPOTrainer(GRPOTrainer):
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

        # split callbacks
        cb_instances, cb_factories = [], []
        if callbacks:
            for cb in callbacks:
                (cb_instances if isinstance(cb, TrainerCallback) else cb_factories).append(cb)

        super().__init__(*args, callbacks=cb_instances, **kwargs)
        self.mask_truncated_completions = True

        for factory in cb_factories:
            self.add_callback(factory(self))

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Union[torch.Tensor, Any]]],
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ─── collect & tokenize prompts ──────────────────────────────────
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"]
            for ex in inputs
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

        # optional prompt‐trimming
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.processing_class.pad_token)})+", "", t)
                for t in prompts_text
            ]

        # ─── two‐stage rollout if provided ──────────────────────────────
        if self.rollout_fn is not None and not self.use_vllm and not self.use_transformers_paged:
            _, full_ids = self.rollout_fn(
                prompt_ids,
                max_new_tokens=self.max_completion_length,
            )
            completion_ids = full_ids[:, prompt_ids.size(1) :]
            prompt_completion_ids = full_ids

        else:
            # ─── vLLM branch ────────────────────────────────────────────
            if self.use_vllm:
                all_prompts = broadcast_object_list(gather_object(prompts_text), from_process=0)
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
                            # note: drop generation_kwargs=self.args.generation_kwargs
                        )
                else:
                    completion_ids = [None] * len(prompts_text)

                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                sl = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[sl]
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                # guard against empty list (numpy pad() will error on [])
                if completion_ids:
                    completion_ids = pad(
                        completion_ids,
                        padding_value=self.processing_class.pad_token_id,
                    )
                else:
                    # create a [batch_size x 0] tensor instead of calling pad([])
                    batch_size = prompt_ids.size(0)
                    completion_ids = torch.zeros(
                        batch_size, 0,
                        dtype=torch.long,
                        device=device,
                    )


                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            # ─ paged‐attention branch ────────────────────────────────────
            elif self.use_transformers_paged:
                prompt_inputs = self.processing_class(text=prompts_text)
                previous_attn = self.model_wrapped.config._attn_implementation

                if is_flash_attn_2_available():
                    self.model_wrapped.config._attn_implementation = "paged_attention"
                else:
                    self.model_wrapped.config._attn_implementation = "sdpa_paged"
                with (
                    profiling_context(self, "transformers.generate_batch"),
                    unwrap_model_for_generation(
                        self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                    ) as unwrapped_model,
                    torch.no_grad(),
                    FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
                ):
                    # Cast to the appropriate dtype based on training configuration
                    if self.args.bf16:
                        unwrapped_model.to(torch.bfloat16)
                    elif self.args.fp16:
                        unwrapped_model.to(torch.float16)
                    with torch.inference_mode():
                        all_outputs = unwrapped_model.generate_batch(
                            prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                        )
                completion_ids = [output.generated_tokens for output in all_outputs.values()]
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(
                    completion_ids, padding_value=self.processing_class.pad_token_id, padding_side="right"
                )
                prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_inputs.input_ids]
                prompt_ids = pad(prompt_ids, padding_value=self.processing_class.pad_token_id, padding_side="left")
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                # Restore the original attention implementation, training mode
                self.model_wrapped.config._attn_implementation = previous_attn
                
            # ─ plain HF generate ───────────────────────────────────────
            else:
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

                prompt_len     = prompt_ids.size(1)
                prompt_ids     = prompt_completion_ids[:, :prompt_len]
                completion_ids = prompt_completion_ids[:, prompt_len:]

        # ─── everything after here is identical to stock TRL post‐processing ─

        # mask past first EOS
        is_eos   = completion_ids == self.processing_class.eos_token_id
        eos_idx  = torch.full((is_eos.size(0),), is_eos.size(1), device=device, dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx  = torch.arange(is_eos.size(1), device=device).expand_as(is_eos)
        comp_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        comp_ids_list = [
            [int(tok) for tok, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, comp_mask)
        ]
        comp_lens = comp_mask.sum(1)
        if self.mask_truncated_completions:
            truncated = ~is_eos.any(dim=1)
            comp_mask *= (~truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        bs = self.args.per_device_train_batch_size if mode=="train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            gen_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % gen_every != 0:
                old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model, torch.cat([prompt_ids, completion_ids],1),
                    attention_mask, logits_to_keep, bs
                )["logps"]
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model, torch.cat([prompt_ids, completion_ids],1),
                        attention_mask, logits_to_keep
                    )["logps"]
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps_and_entropies(
                            self.model, torch.cat([prompt_ids, completion_ids],1),
                            attention_mask, logits_to_keep
                        )["logps"]
            else:
                ref_per_token_logps = None

        # decode, reward, normalize, log… (all unchanged)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for pr, ct in zip(prompts, completions_text):
                bootstrap = pr.pop()["content"] if pr[-1]["role"]=="assistant" else ""
                completions.append([{"role":"assistant","content":bootstrap+ct}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, comp_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
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

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device

        # Stage 1: reasoning …
        if self.vllm_client:
            prompts = self.tok.batch_decode(
                input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
            reason_lists = [rt.tolist() for rt in reason_tensor]

        # pad & tack on <answer>
        padded = []
        for seq in reason_lists:
            if seq[-len(self.think_close_ids):] != self.think_close_ids:
                seq = seq + self.think_close_ids + self.answer_tag_ids
            else:
                seq = seq + self.answer_tag_ids
            padded.append(torch.tensor(seq, device=device))
        reason_ids = pad_sequence(padded, batch_first=True, padding_value=self.tok.pad_token_id)

        # Stage 2: finish answer …
        if self.vllm_client:
            reason_texts = self.tok.batch_decode(
                reason_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            answer_lists = self.vllm_client.generate(
                prompts=reason_texts,
                n=1,
                max_tokens=max_new_tokens or 100,
                **gen_kwargs,
            )
            full_lists = [r.tolist() + ans for r, ans in zip(reason_ids, answer_lists)]
        else:
            full_tensor = self.model.generate(
                reason_ids,
                max_new_tokens=max_new_tokens or 100,
                eos_token_id=self.tok.eos_token_id,
                do_sample=True,
                **gen_kwargs,
            )
            full_lists = [ft.tolist() for ft in full_tensor]

        full_ids = pad_sequence(
            [torch.tensor(l, device=device) for l in full_lists],
            batch_first=True,
            padding_value=self.tok.pad_token_id,
        )

        return reason_ids, full_ids