"""Replay-enabled GRPO trainer implementation.

This module hosts the core :class:`GRPOTrainerReplay` class extracted from
:mod:`training.grpo_runtime_impl_full`.
"""

from __future__ import annotations

import copy
import logging
import random
import re
import time
from typing import Any, Dict, Optional

import numpy as np

from .grpo_runtime_env import torch, dist, GRPOTrainer
from .grpo_rewards_router import reward_router
from .rewards_core import pure_accuracy_reward
from .grpo_trainer_replay_support import (
    ReplaySettings,
    TemperatureSchedule,
    MixSettings,
    RuntimeState,
    EASY_MIX_SCHEDULE,
    TEXT_COMPLETION_KEYS,
    _is_rank0,
    _to_env_schema,
    _extract_rewards_for_logging,
    _extract_text_completions,
    _pad_to_batch_size,
    _normalize_for_trl,
    _label_easy_copy,
)

logger = logging.getLogger(__name__)


class GRPOTrainerReplay(GRPOTrainer):
    """GRPO trainer with replay buffer and EASY/CRYPTIC mixing."""
    def __init__(self, *args, **kwargs):
        # Pull subclass-specific knobs out of kwargs so they are not
        # forwarded to the base GRPOTrainer constructor.
        replay_settings = ReplaySettings(
            buffer=kwargs.pop("replay_buffer", None),
            warmup_steps=kwargs.pop("replay_warmup", 500),
            mix_exploit_ratio=kwargs.pop("mix_exploit_ratio", 0.9),
            constant_test_reward=kwargs.pop("constant_test_reward", None),
        )
        temp_start = kwargs.pop("temp_start", 1.0)
        temp_end = kwargs.pop("temp_end", 0.3)
        anneal_steps = kwargs.pop("anneal_steps", 3_000)
        high_temp_period = kwargs.pop("high_temp_period", 2_000)
        easy_pool = kwargs.pop("easy_pool", None)
        mix_schedule = kwargs.pop("mix_schedule", None)
        inject_every_batch = bool(kwargs.pop("inject_every_batch", False))

        my_tok = kwargs.get("tokenizer", None)
        my_proc = kwargs.get("processing_class", None) or my_tok

        super().__init__(*args, **kwargs)

        # Force the objects we want after parent init (parent may have set its own)
        if my_tok is not None:
            self.tokenizer = my_tok
        if my_proc is not None:
            self.processing_class = my_proc
        self._ensure_pad_token_on(self.processing_class, self.model)

        # Replay / temperature / mixing configuration
        self.replay_settings = replay_settings
        self.temperature_schedule = TemperatureSchedule(
            start_temperature=temp_start,
            end_temperature=temp_end,
            anneal_steps=max(1, anneal_steps),
            high_temperature_period=max(1, high_temp_period),
        )
        self.mix_settings = MixSettings(
            easy_pool=easy_pool or [],
            schedule=mix_schedule or EASY_MIX_SCHEDULE,
        )

        # Ephemeral runtime state
        self.runtime_state = RuntimeState()

        if inject_every_batch:
            logger.warning(
                "inject_every_batch flag is currently unused; "
                "it is accepted for backward compatibility only.",
            )

    @staticmethod
    def _ensure_pad_token_on(tok, model):
        if tok is None:
            return
        if getattr(tok, "pad_token_id", None) is None:
            if getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(model, "resize_token_embeddings"):
                    model.resize_token_embeddings(len(tok))
        if (
            getattr(model.config, "pad_token_id", None) is None
            and getattr(tok, "pad_token_id", None) is not None
        ):
            model.config.pad_token_id = tok.pad_token_id
        try:
            tok.padding_side = "left"
        except AttributeError:
            pass


    def _p_easy(self, step: int) -> float:
        if not self.mix_settings.schedule:
            return 0.0
        prob_easy = 0.10
        for step_threshold, prob_easy_value in self.mix_settings.schedule:
            if step >= int(step_threshold):
                prob_easy = float(prob_easy_value)
            else:
                break
        return max(0.0, min(1.0, prob_easy))

    def p_easy(self, step: int) -> float:
        """Return EASY-mix probability for a given global step."""
        return self._p_easy(step)

    def _update_generation_temperature(self, step: int) -> float:
        """Internal helper to update generation temperature for a step."""
        schedule = self.temperature_schedule
        if step < schedule.anneal_steps:  # linear decay
            frac = step / float(schedule.anneal_steps)
            new_temperature = (
                schedule.start_temperature
                + frac * (schedule.end_temperature - schedule.start_temperature)
            )
        else:  # sprinkle hi-T
            new_temperature = (
                schedule.start_temperature
                if (step - schedule.anneal_steps) % schedule.high_temperature_period == 0
                else schedule.end_temperature
            )
        self.model.generation_config.temperature = new_temperature
        if _is_rank0(self.accelerator):
            print(f"[Anneal] step={step}  T={new_temperature:.3f}")
        return new_temperature

    def update_generation_temperature(self, step: int) -> float:
        """Public wrapper exposing the temperature schedule."""
        return self._update_generation_temperature(step)

    def _maybe_mix_easy_batch(self, generation_batch, step: int) -> Any:
        if not (isinstance(generation_batch, list) and self.mix_settings.easy_pool):
            return generation_batch

        batch_size = len(generation_batch)
        num_replicas = 4
        prob_easy = self._p_easy(step)

        if _is_rank0(self.accelerator):
            print(
                f"[MixDBG] step={step} "
                f"bs={batch_size} p_easy={prob_easy:.2f}"
            )

        mix_uniform = random.random()
        do_mix = (
            (batch_size >= num_replicas)
            and (prob_easy > 0.0)
            and (mix_uniform < prob_easy)
        )
        print("Are we mixing?", do_mix, prob_easy, mix_uniform)

        if not do_mix:
            if _is_rank0(self.accelerator):
                print(
                    f"[Mix] step={step} skipped "
                    f"(bs={batch_size}, p_easy={prob_easy:.2f})"
                )
            return generation_batch

        start_index = 0  # always the first block
        base_example = copy.deepcopy(random.choice(self.mix_settings.easy_pool))
        base_example["task"] = "EASY"
        base_example.setdefault("is_replay", False)

        mix_group_counter = self.runtime_state.mix_group_counter
        self.runtime_state.mix_group_counter = mix_group_counter + 1

        for replica_index in range(num_replicas):
            generation_batch[start_index + replica_index] = _label_easy_copy(
                copy.deepcopy(base_example),
                mix_group_counter,
                replica_index + 1,
                total=num_replicas,
            )

        if _is_rank0(self.accelerator):
            print(
                f"[Mix] step={step} "
                f"p_easy={prob_easy:.2f} → inserted 1 group of "
                f"{num_replicas} at [0:4)"
            )
        return generation_batch

    def _compute_inject_now_flag(
        self,
        step: int,
        new_temperature: float,
    ) -> tuple[int, bool, bool]:
        steps_per_gen = getattr(self.args, "steps_per_generation", 8)
        gen_round = step // steps_per_gen
        new_round = gen_round != self.runtime_state.gen_round
        self.runtime_state.gen_round = gen_round
        inject_now = new_round and (new_temperature <= self.temp_end + 1e-4)
        is_rank0 = _is_rank0(self.accelerator)
        return gen_round, inject_now, is_rank0

    def _maybe_inject_replay_group(
        self,
        generation_batch,
        step: int,
        inject_now: bool,
        is_rank0: bool,
    ):
        if not (
            inject_now
            and is_rank0
            and len(self.replay_settings.buffer) >= self.replay_settings.warmup_steps
        ):
            return generation_batch

        uid = group = None
        try:
            uid = self.replay_settings.buffer.sample_uid(
                mix_exploit_ratio=self.replay_settings.mix_exploit_ratio
            )
            if uid is not None and uid >= 0:
                group = self.replay_settings.buffer.get_group(uid)
        except (KeyError, ValueError, RuntimeError) as error:
            print(f"[ReplayPrep][ERR] sample/get_group failed: {error!r}")

        if uid is not None and group:
            generation_batch = self._inject_group(generation_batch, uid, group)
            print(f"[ReplayPrep] step={step} injected uid={uid} size={len(group)}")
        return generation_batch
    # ─────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────
    def _dump_out_once(self, out: dict):
        if self.runtime_state.printed_out_keys_once:
            return
        self.runtime_state.printed_out_keys_once = True
        print("[Replay][DEBUG] out keys:", list(out.keys()))

    # ─────────────────────────────────────────────────────────────────────
    # main hook
    # ─────────────────────────────────────────────────────────────────────
    def _prepare_inputs(self, generation_batch):
        # skip when not training or no buffer
        if not self.model.training or self.replay_settings.buffer is None:
            return super()._prepare_inputs(generation_batch)

        step = self.state.global_step
        # ── 1. temperature schedule ────────────────────────────────
        new_temperature = self._update_generation_temperature(step)

        # ── EASY↔️CRYPTIC mixture gate ─────────────────────────────
        generation_batch = self._maybe_mix_easy_batch(generation_batch, step)

        # ── 2. decide whether this batch starts a new “round” ──────
        _, inject_now, is_rank0 = self._compute_inject_now_flag(
            step,
            new_temperature,
        )

        # ── 3. replay injection when conditions met ─────────────────
        generation_batch = self._maybe_inject_replay_group(
            generation_batch,
            step,
            inject_now,
            is_rank0,
        )

        # ---- normalize keys across the whole batch (avoids KeyError in TRL) ----
        if isinstance(generation_batch, list):
            for ex in generation_batch:
                if isinstance(ex, dict):
                    ex.setdefault("task", "MATH")
                    ex.setdefault("mix_group_id", -1)
                    ex.setdefault("mix_copy_idx", -1)
                    ex.setdefault("is_replay", False)

        # continue with vanilla GRPO flow
        return super()._prepare_inputs(generation_batch)

    def _maybe_credit_injected_uids(self, out: Dict[str, Any]) -> None:
        if not _is_rank0(self.accelerator):  # <- guard
            self.runtime_state.latest_injected_uids.clear()
            return
        if not self.runtime_state.latest_injected_uids:
            return
        if self.replay_settings.buffer is None:
            self.runtime_state.latest_injected_uids.clear()
            return

        mean_reward = self._compute_mean_reward_for_credit(out)

        for uid in self.runtime_state.latest_injected_uids:
            try:
                self.replay_settings.buffer.update_priority_by_uid(uid, mean_reward)
            except (KeyError, ValueError, RuntimeError) as error:
                if _is_rank0(self.accelerator):
                    print(
                        "[Replay][WARN] update_priority_by_uid failed "
                        f"(uid={uid}): {error}"
                    )

        self.runtime_state.latest_injected_uids.clear()

    def _compute_mean_reward_for_credit(self, out: Dict[str, Any]) -> float:
        reward_keys = ("rewards", "reward", "scores", "advantages")
        rewards = None
        for key in reward_keys:
            if key in out:
                rewards = out[key]
                break

        mean_reward = float(self.replay_settings.constant_test_reward or 0.0)
        if rewards is None:
            return mean_reward

        try:
            if isinstance(rewards, torch.Tensor):
                rewards_array = rewards.detach().cpu().float().numpy()
            elif isinstance(rewards, np.ndarray):
                rewards_array = rewards.astype(np.float32)
            else:
                rewards_array = np.asarray(rewards, dtype=np.float32)
            mean_reward = float(np.mean(rewards_array))
        except (TypeError, ValueError, RuntimeError) as error:
            if _is_rank0(self.accelerator):
                print(
                    "[Replay][WARN] couldn't aggregate rewards for crediting: "
                    f"{error}",
                )
        return mean_reward

    # ------------------ hook 2: generate & score (we add to buffer here) ------------------
    _ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        self._ensure_pad_token_on(self.processing_class, self.model)
        self._maybe_throttle_vllm()

        batch_info, clean_inputs = self._build_replay_batch_info(inputs)
        clean_inputs = self._normalize_batch_for_trl(clean_inputs)

        out = super()._generate_and_score_completions(clean_inputs)
        self._dump_out_once(out)

        # rank / world info
        dist_ok = dist.is_initialized()
        rank = dist.get_rank() if dist_ok else 0
        world = dist.get_world_size() if dist_ok else 1
        is_rank0 = rank == 0

        tokenizer = (
            getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        self._ensure_pad_token_on(tokenizer, self.model)

        if not isinstance(batch_info["orig_inputs"], list):
            if is_rank0:
                print(
                    "[Replay][DEBUG] collated batch without per-example "
                    "gold/prompt; skipping replay push."
                )
            return out

        completions_txt = self._decode_replay_completions(
            out, batch_info, tokenizer, is_rank0
        )
        if completions_txt is None:
            return out

        completions_txt, batch_info = self._align_replay_metadata(
            inputs, out, completions_txt, batch_info
        )

        self._attach_rewards_from_completions(out, completions_txt, batch_info)

        winners_all = self._select_replay_winners(
            completions_txt,
            batch_info,
            {"rank": rank, "world": world, "is_rank0": is_rank0},
        )

        self._push_replay_winners_to_buffer(out, winners_all, world, is_rank0)

        return out

    def _maybe_throttle_vllm(self) -> None:
        if (
            not self.accelerator.is_main_process
            or self.runtime_state.vllm_cooldown <= 0
        ):
            return
        now = time.time()
        last = self.runtime_state.last_vllm_upload_ts
        if last is None:
            return
        sleep_for = self.runtime_state.vllm_cooldown - (now - last)
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _build_replay_batch_info(
        self, inputs: Any
    ) -> tuple[dict[str, Any], Any]:
        if isinstance(inputs, list):
            clean_inputs = [
                {
                    key: value
                    for key, value in example.items()
                    if not str(key).startswith("_")
                }
                for example in inputs
            ]
            batch_size = len(clean_inputs)
        else:
            clean_inputs = inputs
            batch_size = 0

        injected = isinstance(inputs, list) and any(
            "_buf_uid" in example for example in inputs
        )

        batch_info: dict[str, Any] = {
            "orig_inputs": inputs,
            "clean_inputs": clean_inputs,
            "batch_size": batch_size,
            "injected": injected,
        }

        if isinstance(clean_inputs, list) and batch_size > 0:
            batch_info["gold_answers"] = [
                example.get("answer") for example in clean_inputs
            ]
            batch_info["prompts_list"] = [
                example.get("prompt") for example in clean_inputs
            ]
            batch_info["tasks_list"] = [
                example.get("task", "MATH") for example in clean_inputs
            ]
            batch_info["boards_list"] = [
                example.get("board") or example.get("board_str")
                for example in clean_inputs
            ]
            batch_info["sizes_list"] = [
                example.get("size") or example.get("N")
                for example in clean_inputs
            ]
            batch_info["moves_list"] = [
                example.get("moves") or example.get("gold_moves")
                for example in clean_inputs
            ]

        if (
            injected
            and isinstance(inputs, list)
            and batch_size > 0
            and batch_info.get("gold_answers") is not None
            and batch_info.get("prompts_list") is not None
        ):
            anchor_index = next(
                idx for idx, example in enumerate(inputs) if "_buf_uid" in example
            )
            gold_answers = batch_info["gold_answers"]
            prompts_list = batch_info["prompts_list"]
            anchor_answer = gold_answers[anchor_index]
            anchor_prompt = prompts_list[anchor_index]

            batch_info["gold_answers"] = [anchor_answer] * batch_size
            batch_info["prompts_list"] = [anchor_prompt] * batch_size

            for example in clean_inputs:
                example["answer"] = anchor_answer
                example["prompt"] = anchor_prompt

        return batch_info, clean_inputs

    def _normalize_batch_for_trl(self, clean_inputs: Any) -> Any:
        proc = getattr(self, "processing_class", None) or self.tokenizer
        if isinstance(clean_inputs, list):
            return [_normalize_for_trl(example, proc) for example in clean_inputs]
        if isinstance(clean_inputs, dict):
            return _normalize_for_trl(clean_inputs, proc)
        return clean_inputs

    def _decode_replay_completions(
        self,
        out: Dict[str, Any],
        batch_info: Dict[str, Any],
        tokenizer: Any,
        is_rank0: bool,
    ) -> Optional[list[list[str]]]:
        batch_size = int(batch_info.get("batch_size") or 0)
        completions_txt: Optional[list[list[str]]] = None

        completion_ids = out.get("completion_ids")
        if isinstance(completion_ids, torch.Tensor) and tokenizer is not None:
            completions_txt = self._decode_replay_from_ids(
                completion_ids,
                out,
                batch_size,
                tokenizer,
                is_rank0,
            )
        else:
            completions_txt = self._decode_replay_from_text_keys(out, batch_size)

        if self.accelerator.is_main_process:
            self.runtime_state.last_vllm_upload_ts = time.time()

        if completions_txt is None:
            if is_rank0:
                print(
                    "[Replay][DEBUG] no decodable completions; "
                    "skipping replay push.",
                )
            return None

        if isinstance(completions_txt, list) and completions_txt:
            batch_info["batch_size"] = len(completions_txt)
            batch_info["num_candidates"] = max(
                1,
                len(completions_txt[0]),
            )
            if not self.runtime_state.paired_debug_once:
                self.runtime_state.paired_debug_once = True
                print(
                    "[LogPairing] B=%d K=%d (nested) — leaving TRL shapes untouched",
                    batch_info["batch_size"],
                    batch_info["num_candidates"],
                )

        return completions_txt

    @staticmethod
    def _decode_replay_from_ids(
        completion_ids: torch.Tensor,
        out: Dict[str, Any],
        batch_size: int,
        tokenizer: Any,
        is_rank0: bool,
    ) -> Optional[list[list[str]]]:
        if batch_size <= 0:
            return None

        ids = completion_ids
        total, seq_len = ids.shape
        if total % batch_size != 0:
            if is_rank0:
                print(
                    "[Replay][DEBUG] unexpected shapes: "
                    f"total={total}, batch_size={batch_size}",
                )
            return None

        num_candidates = total // batch_size
        ids = ids.view(batch_size, num_candidates, seq_len)

        mask_tensor = out.get("completion_mask")
        if isinstance(mask_tensor, torch.Tensor):
            mask_tensor = mask_tensor.view(batch_size, num_candidates, seq_len).bool()
        else:
            mask_tensor = None

        completions_txt: list[list[str]] = []
        for prompt_index in range(batch_size):
            per_prompt: list[str] = []
            for candidate_index in range(num_candidates):
                seq = ids[prompt_index, candidate_index]
                if mask_tensor is not None:
                    seq = seq[mask_tensor[prompt_index, candidate_index]]
                per_prompt.append(
                    tokenizer.decode(
                        seq.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            completions_txt.append(per_prompt)

        return completions_txt

    @staticmethod
    def _decode_replay_from_text_keys(
        out: Dict[str, Any],
        batch_size: int,
    ) -> Optional[list[list[str]]]:
        if batch_size <= 0:
            return None

        for key in TEXT_COMPLETION_KEYS:
            if key in out and isinstance(out[key], list) and out[key]:
                texts = out[key]
                num_candidates = max(1, len(texts) // max(1, batch_size))
                return [
                    texts[index * num_candidates : (index + 1) * num_candidates]
                    for index in range(batch_size)
                ]
        return None

    def _align_replay_metadata(
        self,
        inputs: Any,
        out: Dict[str, Any],
        completions_txt: list[list[str]],
        batch_info: Dict[str, Any],
    ) -> tuple[list[list[str]], Dict[str, Any]]:
        unsort_idx = out.get("unsort_idx")
        if isinstance(unsort_idx, torch.Tensor):
            order = unsort_idx.tolist()
        else:
            order = None

        if order is not None:
            completions_txt = [completions_txt[i] for i in order]
            for key in (
                "gold_answers",
                "prompts_list",
                "tasks_list",
                "boards_list",
                "sizes_list",
                "moves_list",
            ):
                values = batch_info.get(key)
                if values is not None:
                    batch_info[key] = [values[i] for i in order]

        if isinstance(inputs, list) and any(
            "_buf_uid" in example for example in inputs
        ):
            anchor_index = next(
                idx
                for idx, example in enumerate(inputs)
                if "_buf_uid" in example
            )
            gold_answers = batch_info.get("gold_answers") or []
            prompts_list = batch_info.get("prompts_list") or []
            if gold_answers and prompts_list:
                gold_anchor = gold_answers[anchor_index]
                prompt_anchor = prompts_list[anchor_index]
                batch_length = len(gold_answers)
                batch_info["gold_answers"] = [gold_anchor] * batch_length
                batch_info["prompts_list"] = [prompt_anchor] * batch_length
                self._patch_out_for_injected(out, batch_info)

        return completions_txt, batch_info

    @staticmethod
    def _patch_out_for_injected(
        out: Dict[str, Any], batch_info: Dict[str, Any]
    ) -> None:
        gold_answers = batch_info.get("gold_answers")
        prompts_list = batch_info.get("prompts_list")
        tasks_list = batch_info.get("tasks_list")

        def _set(dest: dict, keys: tuple[str, ...], value: Any) -> None:
            for key in keys:
                dest[key] = value

        if gold_answers is not None:
            _set(out, ("gold_answers", "answers", "gold"), gold_answers)
        if prompts_list is not None:
            _set(out, ("prompts_list", "prompts", "prompt"), prompts_list)
        if tasks_list is not None:
            _set(out, ("tasks_list", "tasks"), tasks_list)

        if "reward_kwargs" in out and isinstance(out["reward_kwargs"], dict):
            reward_kwargs = out["reward_kwargs"]
            if gold_answers is not None:
                if "answer" in reward_kwargs:
                    reward_kwargs["answer"] = gold_answers
                if "gold" in reward_kwargs:
                    reward_kwargs["gold"] = gold_answers
            if tasks_list is not None:
                reward_kwargs["tasks"] = tasks_list

    def _attach_rewards_from_completions(
        self,
        out: Dict[str, Any],
        completions_txt: list[list[str]],
        batch_info: Dict[str, Any],
    ) -> None:
        gold_answers = batch_info.get("gold_answers")
        prompts_list = batch_info.get("prompts_list")
        tasks_list = batch_info.get("tasks_list")
        boards_list = batch_info.get("boards_list")
        sizes_list = batch_info.get("sizes_list")
        moves_list = batch_info.get("moves_list")

        reward_kwargs = out.setdefault("reward_kwargs", {})
        reward_kwargs["answer"] = gold_answers
        reward_kwargs["gold"] = gold_answers
        reward_kwargs["prompts"] = prompts_list
        reward_kwargs["tasks"] = tasks_list
        reward_kwargs["board_str"] = boards_list
        reward_kwargs["N"] = sizes_list
        reward_kwargs["gold_moves"] = moves_list

        scores_nested = reward_router(
            completions=completions_txt,
            tasks=tasks_list,
            answer=gold_answers,
            gold=gold_answers,
            board_str=boards_list,
            N=sizes_list,
            gold_moves=moves_list,
            prompts=prompts_list,
            proc=self.processing_class or self.tokenizer,
            script_args=self.args,
        )

        if (
            isinstance(scores_nested, list)
            and scores_nested
            and isinstance(scores_nested[0], list)
        ):
            flat_scores = [score for row in scores_nested for score in row]
        else:
            flat_scores = scores_nested

        out["rewards"] = flat_scores
        advantages = out.get("advantages")
        if isinstance(advantages, torch.Tensor):
            if isinstance(flat_scores, list):
                out["rewards"] = torch.tensor(
                    flat_scores,
                    device=advantages.device,
                    dtype=advantages.dtype,
                )
            elif isinstance(flat_scores, torch.Tensor):
                out["rewards"] = flat_scores.to(
                    device=advantages.device,
                    dtype=advantages.dtype,
                )

    def _build_local_replay_winners(
        self,
        completions_txt: list[list[str]],
        gold_answers: list[Any],
        prompts_list: list[Any],
        dist_info: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        batch_size = len(completions_txt)
        max_candidates = 4
        limited_completions = [
            completions[:max_candidates] for completions in completions_txt
        ]

        winners_local: list[dict[str, Any]] = []
        for index in range(batch_size):
            gold = gold_answers[index]
            prompt = prompts_list[index]
            if not isinstance(gold, str) or not gold.strip() or prompt is None:
                continue
            preds_for_prompt = limited_completions[index]
            accuracies = pure_accuracy_reward(
                preds_for_prompt, [gold] * len(preds_for_prompt)
            )
            hit_index = next(
                (
                    idx
                    for idx, accuracy in enumerate(accuracies)
                    if accuracy == 1.0
                ),
                None,
            )
            if hit_index is None:
                continue

            if dist_info.get("is_rank0", False):
                print(
                    f"[Replay][HIT] rank={dist_info.get('rank', 0)} "
                    f"i={index} gold='{gold}'",
                )
            winners_local.append(
                _to_env_schema(
                    {
                        "prompt": prompt,
                        "answer": gold,
                        "reward": 1.0,
                        "_last_success_pred": preds_for_prompt[hit_index],
                    }
                )
            )
        return winners_local

    def _gather_replay_winners(
        self,
        winners_local: list[dict[str, Any]],
        dist_info: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        world_size = dist_info.get("world", 1)
        if world_size > 1 and dist.is_initialized():
            gathered: list[Optional[list[dict[str, Any]]]] = [None] * world_size
            dist.all_gather_object(gathered, winners_local)
            winners_all = [
                example
                for group in gathered
                for example in (group or [])
            ]
            if dist_info.get("is_rank0", False):
                counts = [len(group or []) for group in gathered]
                print(
                    "[Replay][GATHER] world=%s per-rank winners=%s total=%s",
                    world_size,
                    counts,
                    sum(counts),
                )
            return winners_all

        if dist_info.get("is_rank0", False):
            print(
                "[Replay][GATHER] single-process winners=%s",
                len(winners_local),
            )
        return winners_local

    def _select_replay_winners(
        self,
        completions_txt: list[list[str]],
        batch_info: Dict[str, Any],
        dist_info: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        gold_answers = batch_info.get("gold_answers") or []
        prompts_list = batch_info.get("prompts_list") or []

        winners_local = self._build_local_replay_winners(
            completions_txt,
            gold_answers,
            prompts_list,
            dist_info,
        )
        return self._gather_replay_winners(winners_local, dist_info)

    def _push_replay_winners_to_buffer(
        self,
        out: Dict[str, Any],
        winners_all: list[dict[str, Any]],
        world: int,
        is_rank0: bool,
    ) -> None:
        if not (
            self.model.training
            and is_rank0
            and self.replay_buffer is not None
        ):
            return

        if not winners_all:
            print("[Replay][DEBUG] gathered no unique winners to add.")
            return

        def _signature(example: dict[str, Any]) -> tuple[str, str]:
            messages_value = example.get("prompt", [])
            if isinstance(messages_value, str):
                user_text = messages_value.strip()
            else:
                user_text = " ".join(
                    message.get("content", "").strip()
                    for message in messages_value
                    if message.get("role") == "user"
                )
            gold_text = (example.get("answer") or "").strip().lower()
            return (user_text, gold_text)

        seen: set[tuple[str, str]] = set()
        unique_examples: list[dict[str, Any]] = []
        for example in winners_all:
            sig = _signature(example)
            if sig in seen:
                continue
            seen.add(sig)
            unique_examples.append(example)

        if not unique_examples:
            print("[Replay][DEBUG] gathered no unique winners to add.")
            return

        result = self.replay_buffer.add_group(unique_examples, reward=1.0)
        if isinstance(result, tuple) and len(result) == 2:
            success, uid = result
        else:
            success, uid = True, result

        if success and int(uid) >= 0:
            print(
                "[ReplayAdd] uid=%s size=%s from %s ranks | len(buf)=%s",
                uid,
                len(unique_examples),
                world,
                len(self.replay_buffer),
            )
        else:
            debug_state = getattr(
                self.replay_buffer, "debug_state", lambda: {}
            )()
            print(
                "[Replay][WARN] add_group failed (uid=%s). state=%s",
                uid,
                debug_state,
            )

        try:
            self._maybe_credit_injected_uids(out)
        except (RuntimeError, ValueError, TypeError) as error:
            print(f"[Replay][WARN] credit failed: {error}")
    # ------------------ helpers ------------------

    def _inject_group(self, generation_batch, uid: int, group: list[dict[str, Any]]):
        if not group:
            return generation_batch

        rank = getattr(self.accelerator, "process_index", 0)
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        tagged = []
        for ex in group:
            ex = _normalize_for_trl(dict(ex), tok)
            ex["_buf_uid"] = uid
            ex["_buf_rank"] = rank
            ex["is_replay"] = True
            # ↓ add defaults so union-of-keys won’t break
            ex.setdefault("task", "MATH")
            ex.setdefault("mix_group_id", -1)
            ex.setdefault("mix_copy_idx", -1)
            tagged.append(ex)

        batch_length = len(generation_batch)
        inject_count = min(len(tagged), batch_length // 2)  # inject at most half the batch
        # replace first inject_count items (don’t prepend)
        new_batch = copy.deepcopy(tagged[:inject_count]) + list(generation_batch)[inject_count:]

        if _is_rank0(self.accelerator):

            def _head_user(batch_example):
                try:
                    msgs = (
                        batch_example.get("messages")
                        or batch_example.get("prompt")
                        or []
                    )
                    if isinstance(msgs, list):
                        return next(
                            (
                                message.get("content", "")
                                for message in msgs
                                if message.get("role") == "user"
                            ),
                            "<no-user>",
                        )
                    return str(msgs)[:60]
                except (TypeError, KeyError, AttributeError):  # pragma: no cover - defensive
                    return "<err>"
            print(f"[ReplayInject] replaced {inject_count}/{batch_length} (uid={uid})")
            if batch_length > 0:
                print(f"[ReplayInject] head user after : «{_head_user(new_batch[0])}»")

        self.runtime_state.latest_injected_uids = [uid]
        return new_batch

    def _extract_completions_and_rewards(
        self,
        out: Dict[str, Any],
        batch_size: int,
    ):
        tokenizer = getattr(self, "tokenizer", None) or getattr(
            self,
            "processing_class",
            None,
        )

        rewards = _extract_rewards_for_logging(out)
        completions = _extract_text_completions(out, tokenizer)

        completions = _pad_to_batch_size(completions, None, batch_size)
        rewards = _pad_to_batch_size(rewards, 0.0, batch_size)

        return completions, rewards
