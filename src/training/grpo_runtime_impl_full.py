"""
Entry-point script for GRPO training with the two-stage reasoning ➜ answer
hierarchical rollout.

This module contains the full runtime implementation and is intentionally
large; a thin wrapper (:mod:`training.grpo_runtime_impl`) re-exports
``main`` to satisfy stricter linting limits.
"""

from __future__ import annotations

import copy
import functools
import importlib
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import numpy as np

from .rewards import get_reward_funcs
from .rewards_core import (
    pure_accuracy_reward,
    pure_accuracy_reward_math,
    rush_solution_shaped,
)
from .utils import get_dataset, get_model, get_tokenizer
from .utils.callbacks import get_callbacks
from .utils.replay_buffer import ReplayBuffer
from .utils.replay_dataset import ReplayMixDataset
from .grpo_dataset import _make_conversation, _load_easy_pool
from .grpo_rewards_router import reward_router, _wrap_reward_for_nested

datasets = importlib.import_module("datasets")
torch = importlib.import_module("torch")
dist = importlib.import_module("torch.distributed")
torch_serialization = importlib.import_module("torch.serialization")
setattr(torch, "serialization", torch_serialization)
transformers = importlib.import_module("transformers")
wandb = importlib.import_module("wandb")
torch_utils_data = importlib.import_module("torch.utils.data")
DataLoader = getattr(torch_utils_data, "DataLoader")
RandomSampler = getattr(torch_utils_data, "RandomSampler")
accelerate_state = importlib.import_module("accelerate.state")
AcceleratorState = getattr(accelerate_state, "AcceleratorState")
deepspeed_zero_config = importlib.import_module("deepspeed.runtime.zero.config")
ZeroStageEnum = getattr(deepspeed_zero_config, "ZeroStageEnum")
deepspeed_zero_pp = importlib.import_module("deepspeed.runtime.zero.partition_parameters")
ZeroParamStatus = getattr(deepspeed_zero_pp, "ZeroParamStatus")
transformers_trainer_utils = importlib.import_module("transformers.trainer_utils")
get_last_checkpoint = getattr(transformers_trainer_utils, "get_last_checkpoint")
trl_module = importlib.import_module("trl")
get_peft_config = getattr(trl_module, "get_peft_config")
trl_grpo_trainer_module = importlib.import_module("trl.trainer.grpo_trainer")
GRPOTrainer = getattr(trl_grpo_trainer_module, "GRPOTrainer")
AutoTokenizer = getattr(transformers, "AutoTokenizer")
TrainerCallback = getattr(transformers, "TrainerCallback")
set_seed = getattr(transformers, "set_seed")

# ────────────────── ZeRO pickle patch (torch-2.6) ─────────────────────────

_TORCH_DEFAULT_WEIGHTS_ONLY_ATTR = "_default_weights_only"
setattr(torch.serialization, _TORCH_DEFAULT_WEIGHTS_ONLY_ATTR, False)

torch.serialization.add_safe_globals(
    {
        ("deepspeed.runtime.zero.config", "ZeroStageEnum"): ZeroStageEnum,
        (
            "deepspeed.runtime.zero.partition_parameters",
            "ZeroParamStatus",
        ): ZeroParamStatus,
    }
)

_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)  # type: ignore[arg-type]

# ─────────────────── NLTK data path (WordNet) ───────────────────────────
NLTK_DATA_DEFAULT = Path(__file__).resolve().parents[2] / ".nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DEFAULT))
# os.environ.setdefault("EASY_DATASET_NAME", "od2961/mini-crosswords")


logger = logging.getLogger(__name__)


@dataclass
class ReplaySettings:
    """Configuration and state for replay sampling."""

    buffer: Optional[ReplayBuffer]
    warmup_steps: int
    mix_exploit_ratio: float
    constant_test_reward: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dict representation of the replay settings."""
        return {
            "buffer": self.buffer,
            "warmup_steps": self.warmup_steps,
            "mix_exploit_ratio": self.mix_exploit_ratio,
            "constant_test_reward": self.constant_test_reward,
        }

    def is_warmed_up(self, step: int) -> bool:
        """Return True if the replay buffer has passed its warmup window."""
        return self.buffer is not None and step >= self.warmup_steps


@dataclass
class TemperatureSchedule:
    """Annealing schedule for generation temperature."""

    start_temperature: float
    end_temperature: float
    anneal_steps: int
    high_temperature_period: int

    def as_dict(self) -> Dict[str, float]:
        """Return a shallow dict representation of the schedule."""
        return {
            "start_temperature": self.start_temperature,
            "end_temperature": self.end_temperature,
            "anneal_steps": float(self.anneal_steps),
            "high_temperature_period": float(self.high_temperature_period),
        }

    def fraction_complete(self, step: int) -> float:
        """Return the [0, 1] fraction of the anneal completed at a given step."""
        if self.anneal_steps <= 0:
            return 1.0
        return max(0.0, min(1.0, step / float(self.anneal_steps)))


@dataclass
class MixSettings:
    """Configuration for EASY/CRYPTIC mixing."""

    easy_pool: List[dict]
    schedule: List[Tuple[int, float]]

    def is_enabled(self) -> bool:
        """Return True if EASY/CRYPTIC mixing is active."""
        return bool(self.easy_pool and self.schedule)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dict representation of mix settings."""
        return {
            "easy_pool_size": len(self.easy_pool),
            "schedule": list(self.schedule),
        }


@dataclass
class RuntimeState:
    """Ephemeral state updated during training and replay."""

    gen_round: int = -1
    latest_injected_uids: List[int] = field(default_factory=list)
    printed_out_keys_once: bool = False
    vllm_cooldown: int = 3
    last_vllm_upload_ts: Optional[float] = None
    mix_group_counter: int = 0
    paired_debug_once: bool = False

    def reset_latest_uids(self) -> None:
        """Clear the latest-injected UID list."""
        self.latest_injected_uids.clear()

    def should_throttle_vllm(self) -> bool:
        """Return True if vLLM uploads should currently be throttled."""
        return self.vllm_cooldown > 0 and self.last_vllm_upload_ts is not None

class LossLoggingCallback(TrainerCallback):
    """Callback that logs selected loss metrics to W&B and a CSV."""
    # map keys in `logs` → names you prefer
    MAP = {
        "loss/policy_loss":  "policy_loss",
        "loss/value_loss":   "value_loss",
        "loss/kl":           "kl",
        "beta":              "beta",
    }

    def __init__(self, output_dir: str):
        super().__init__()
        self.csv_path = os.path.join(output_dir, "loss_history.csv")
        self._csv_initialized = False

    # helper – create header once
    def _init_csv(self, payload: Dict[str, Any]) -> None:
        if self._csv_initialized:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["step"] + list(payload))
            writer.writeheader()
        self._csv_initialized = True

    def on_log(self, args, state, _control, logs=None, **_kwargs) -> None:
        """Log selected loss metrics to Weights & Biases and a CSV file."""
        if args.local_rank not in (-1, 0) or not logs:
            return

        # pick out the keys we care about
        payload = {v: logs[k] for k, v in self.MAP.items() if k in logs}
        if not payload:
            return

        step = int(state.global_step)

        # 1) W&B
        wandb.log(payload, step=step)

        # 2) CSV (append)
        self._init_csv(payload)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["step"] + list(payload))
            writer.writerow({"step": step, **payload})

    def on_train_begin(self, *_, **__):
        """No-op hook kept to satisfy callback interface and linting."""
        return None

def _is_rank0(accelerator) -> bool:
    # accelerate sets .is_main_process on Accelerator
    return getattr(accelerator, "is_main_process", True)

# ───────── EASY/CRYPTIC mixture schedule ─────────────────────────
# warm → blend → taper
EASY_MIX_SCHEDULE = [
    (   0,  1.00),
    (50, 0.9),
    ( 100,  0.8),
    (150, 0.7),
    ( 200,  0.6),
    (250, 0.55),
    ( 300,  0.53),
    ( 400,  0.51),
    ( 500,  0.49),
    ( 600,  0.47),
    ( 700,  0.44),
    ( 800,  0.42),
    ( 900,  0.40),
    (1000,  0.38),
    (1100,  0.36),
    (1200,  0.34),
    (1300,  0.32),
    (1400,  0.30),
    (1500,  0.28),
    (1600,  0.26),
    (1700,  0.24),
    (1800,  0.22),
    (1900,  0.20),
    (2000,  0.18),
    (2100,  0.16),
    (2200,  0.14),
    (2300,  0.12),
    (2400,  0.10),
    (2500,  0.08),
    (2600,  0.06),
    (2700,  0.04),
    (2800,  0.03),
    (2900,  0.02),
    (3000,  0.01),
]

def _p_easy_for_step(step: int) -> float:
    """Return the EASY-mix probability for a given global step."""
    prob_easy = 0.10
    for step_threshold, prob_easy_value in EASY_MIX_SCHEDULE:
        if step >= int(step_threshold):
            prob_easy = float(prob_easy_value)
        else:
            break
    return prob_easy

def _to_env_schema(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve the entire prompt (system + user + assistant) so that when a
    replay entry is injected the model sees the exact same context it was
    trained on.
    """
    out: Dict[str, Any] = {
        "prompt": example.get("prompt", []),          # ← full history
    }

    # carry these through if present
    for k in ("answer", "gold", "label", "solution", "metadata", "clue_id"):
        if k in example:
            out[k] = example[k]

    return out

def _shorten_for_log(prompt: List[Dict[str, str]], max_chars: int = 60) -> str:
    txts = [m.get("content", "") for m in prompt if m.get("role") == "user"]
    if not txts:
        return ""
    user_text = " ".join(txts).strip().replace("\n", " ")
    if len(user_text) > max_chars:
        user_text = user_text[: max_chars - 1] + "…"
    return user_text

def _to_float_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().tolist()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def _pad_to_batch_size(
    values: Optional[List[Any]],
    pad_value: Any,
    batch_size: int,
) -> Optional[List[Any]]:
    """Pad or trim a list-like container to exactly batch_size elements."""
    if values is None:
        return None
    if len(values) < batch_size:
        values = list(values) + [pad_value] * (batch_size - len(values))
    return values[:batch_size]


def _extract_rewards_for_logging(out: Dict[str, Any]) -> Optional[List[float]]:
    """Extract a flat list of float rewards from a TRL output dict."""
    reward_keys = ["rewards", "reward", "scores", "advantages"]
    rewards_raw = None
    for key in reward_keys:
        if key in out:
            rewards_raw = out[key]
            break
    return _to_float_list(rewards_raw)


def _decode_batch_sequences(sequences: Any, tokenizer: Any) -> Optional[List[str]]:
    """Decode a batch of token id sequences into strings using the tokenizer."""
    if tokenizer is None or sequences is None:
        return None

    seqs = sequences
    try:
        # torch / numpy tensors
        if hasattr(seqs, "detach"):
            seqs = seqs.detach().cpu().tolist()
        elif hasattr(seqs, "cpu"):
            seqs = seqs.cpu().tolist()
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None

    try:
        return tokenizer.batch_decode(seqs, skip_special_tokens=True)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None


def _extract_text_completions(
    out: Dict[str, Any],
    tokenizer: Any,
) -> Optional[List[str]]:
    """Best-effort extraction of flat text completions from TRL outputs."""
    text_keys = [
        "completions_text",
        "completions",
        "generated_responses",
        "responses",
        "texts",
    ]
    for key in text_keys:
        value = out.get(key)
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value

    # TRL variant that only returns token ids
    if "completion_ids" in out:
        decoded = _decode_batch_sequences(out.get("completion_ids"), tokenizer)
        if decoded is not None:
            return decoded

    # Newer TRL: generation_outputs.sequences
    if "generation_outputs" in out:
        seqs = out["generation_outputs"]
        if isinstance(seqs, (list, tuple)) and seqs:
            seqs = seqs[0]
        seqs = getattr(seqs, "sequences", None)
        decoded = _decode_batch_sequences(seqs, tokenizer)
        if decoded is not None:
            return decoded

    return None


def _label_easy_copy(ex: dict, group_id: int, copy_idx: int, total: int = 4) -> dict:
    """Prefix the user turn with [G:<id>] [COPY:i/total] and set bookkeeping keys."""
    tag = f"[G:{group_id}] [COPY:{copy_idx}/{total}]"

    def _prefix(text: str) -> str:
        # strip any old tags to avoid duplication on retries
        cleaned = re.sub(
            r"^\s*\[G:\d+\]\s*\[COPY:\d+/\d+\]\s*\n?",
            "",
            text,
        )
        return f"{tag}\n{cleaned}"

    if isinstance(ex.get("prompt"), list):
        for message in ex["prompt"]:
            if message.get("role") == "user":
                message["content"] = _prefix(message.get("content", ""))
                break
    elif isinstance(ex.get("prompt"), str):
        ex["prompt"] = _prefix(ex["prompt"])
    elif isinstance(ex.get("messages"), list):
        # older schema support
        for message in ex["messages"]:
            if message.get("role") == "user":
                message["content"] = _prefix(message.get("content", ""))
                break

    # uniform keys so TRL's union-of-keys logic never KeyErrors
    ex["task"] = "EASY"
    ex.setdefault("is_replay", False)
    ex["mix_group_id"] = group_id
    ex["mix_copy_idx"] = copy_idx
    return ex

def _summarize_val(value: Any) -> str:
    """Return a short string summary for logging/debugging values."""
    if isinstance(value, torch.Tensor):
        return f"Tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, list):
        element_type = type(value[0]).__name__ if value else "empty"
        return f"list[{element_type}] len={len(value)}"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:8]} ...)"
    return type(value).__name__


# put this at module scope (or as a method) – not inside another function
def _default_join_messages(messages):
    # very simple fallback if no chat template is available
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts) + "\nASSISTANT:"

def _normalize_for_trl(example, proc=None, add_gen_prompt=True):
    example_copy = dict(example)  # copy
    # Prefer 'messages' if present
    messages = example_copy.pop("messages", None)
    prompt_value = example_copy.get("prompt", None)

    if isinstance(messages, list):
        if proc is not None and hasattr(proc, "apply_chat_template"):
            example_copy["prompt"] = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen_prompt
            )
        else:
            example_copy["prompt"] = _default_join_messages(messages)
    elif isinstance(prompt_value, list):  # 'prompt' provided as chat messages
        if proc is not None and hasattr(proc, "apply_chat_template"):
            example_copy["prompt"] = proc.apply_chat_template(
                prompt_value, tokenize=False, add_generation_prompt=add_gen_prompt
            )
        else:
            example_copy["prompt"] = _default_join_messages(prompt_value)
    elif isinstance(prompt_value, str):
        pass  # already fine
    else:
        raise ValueError(
            f"Example is missing a usable prompt; keys={list(example_copy.keys())}"
        )

    # ensure only string prompt is kept
    example_copy.pop("messages", None)
    return example_copy

class GRPOTrainerReplay(GRPOTrainer):
    """GRPO trainer variant with replay buffer and EASY mixing."""
    def __init__(
        self,
        *args,
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_warmup: int = 500,
        mix_exploit_ratio: float = 0.9,
        constant_test_reward: Optional[float] = None,
        inject_every_batch: bool = False,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        anneal_steps: int = 3_000,
        high_temp_period: int = 2_000,
        easy_pool: Optional[list] = None,                # ← NEW
        mix_schedule: Optional[list] = None,            # ← NEW
        **kwargs,
    ):
        my_tok = kwargs.get("tokenizer", None)
        my_proc = kwargs.get("processing_class", None) or my_tok

        super().__init__(*args, **kwargs)
        # force the objects we want after parent init (parent may have set its own)
        if my_tok is not None:
            self.tokenizer = my_tok
        if my_proc is not None:
            self.processing_class = my_proc
        self._ensure_pad_token_on(self.processing_class, self.model)
        # replay
        self.replay_settings = ReplaySettings(
            buffer=replay_buffer,
            warmup_steps=replay_warmup,
            mix_exploit_ratio=mix_exploit_ratio,
            constant_test_reward=constant_test_reward,
        )

        # temperature schedule
        self.temperature_schedule = TemperatureSchedule(
            start_temperature=temp_start,
            end_temperature=temp_end,
            anneal_steps=max(1, anneal_steps),
            high_temperature_period=max(1, high_temp_period),
        )

        # NEW: EASY mixing
        self.mix_settings = MixSettings(
            easy_pool=easy_pool or [],
            schedule=mix_schedule or EASY_MIX_SCHEDULE,
        )

        # state
        self.runtime_state = RuntimeState()

        if inject_every_batch:
            logger.warning(
                "inject_every_batch flag is currently unused; "
                "it is accepted for backward compatibility only."
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
        """Public wrapper exposing the EASY/CRYPTIC mixing schedule."""
        return self._p_easy(step)

    def _update_generation_temperature(self, step: int) -> float:
        """Update and return the generation temperature for the given step."""
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
        """Public wrapper exposing the temperature schedule for a step."""
        return self._update_generation_temperature(step)

    def _maybe_mix_easy_batch(self, generation_batch, step: int) -> Any:
        """
        Optionally inject EASY-clue copies into the current batch according to
        the mix schedule. Returns the (possibly modified) batch.
        """
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
        """
        Compute generation-round bookkeeping and whether we should inject
        replay at this step. Returns (gen_round, new_round, inject_now).
        """
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
        """Optionally inject a replay group into generation_batch."""
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
        """
        Give a single scalar credit to every uid we injected in this window.
        We take the mean over whatever reward-like key exists in `out`.
        """
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
        """
        Find a reward-like key in `out` and return its mean value, falling back
        to constant_test_reward when aggregation fails.
        """
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
        """Generate completions with TRL, compute rewards, and update replay."""
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
        """Throttle vLLM calls on rank-0 according to cooldown."""
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
        """Build metadata and cleaned inputs for replay processing."""
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
        """Normalize prompts for TRL using the chat template."""
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
        """Decode TRL completions into nested text form."""
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
        """Decode nested completions from token id tensor."""
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
        """Decode nested completions from various text-key layouts."""
        if batch_size <= 0:
            return None

        for key in (
            "completions_text",
            "completions",
            "generated_responses",
            "responses",
            "texts",
        ):
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
        """Apply unsort indices and injected alignment to metadata and outputs."""
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
        """Patch the TRL output dict for injected replay alignment."""
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
        """Compute task-aware rewards and attach them to TRL outputs."""
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
        """Select successful completions per prompt on the current rank."""
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
        """Gather winners across ranks (if distributed) and log a short summary."""
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
        """Select successful completions per prompt and gather across ranks."""
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
        """Deduplicate winners and push them into the replay buffer."""
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
        """
        Extract flat-text completions and corresponding rewards for logging/metrics.

        This method is intentionally tolerant of different TRL output schemas and
        falls back gracefully when fields are missing or malformed.
        """
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



# ────────────────────────────── main ────────────────────────────────────
def main(script_args, training_args, model_args):
    """Entry point for GRPO training with replay buffer and EASY/CRYPTIC mixing."""
    set_seed(training_args.seed)
    training_args.return_reward = True        #  ← THE ONE-LINE SWITCH
    training_args.steps_per_generation = 8
    training_args.num_iterations       = 5

    # -------- Logging --------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank %s — device %s — n_gpu %s — distributed %s — bf16 %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.bf16,
    )

    # -------- Dataset --------
    dataset = get_dataset(script_args)

    # -------- Tokenizer & model --------
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    # ---- PAD TOKEN GUARD (run on every rank, immediately) ----
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token    # reuse EOS
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    # ----------------------------------------------------------
    model.generation_config.sort_inputs = False
    model.generation_config.return_dict_in_generate = True
    model.generation_config.output_scores = False
    model.generation_config.do_sample = True
    model.config.return_dict_in_generate = True
    model.config.output_scores = False

    # -------- Optional EASY dataset → pool --------
    easy_pool = None
    want_easy = bool(
        getattr(script_args, "easy_dataset_name", None)
        or os.environ.get("EASY_DATASET_NAME")
        or os.environ.get("EASY_DATASET")
    )
    if want_easy:
        easy_pool = _load_easy_pool(script_args, tokenizer, training_args)

    # -------- Rewards --------
    ref_model = get_model(model_args, training_args).eval().requires_grad_(False)
    reward_funcs = get_reward_funcs(script_args, ref_model=ref_model, tokenizer=tokenizer)

    # Make reward fns tolerant to nested B×K completions (no changes to TRL logs)
    if isinstance(reward_funcs, dict):
        reward_funcs = {
            key: _wrap_reward_for_nested(fn)
            for key, fn in dict(reward_funcs).items()
        }
    elif isinstance(reward_funcs, (list, tuple)):
        reward_funcs = [_wrap_reward_for_nested(fn) for fn in reward_funcs]
    else:
        reward_funcs = _wrap_reward_for_nested(reward_funcs)

    # -------- Dataset --------
    dataset = dataset.map(
        lambda ex: _make_conversation(
            ex,
            script_args.dataset_prompt_column,
            script_args.dataset_solution_column,
            tokenizer,
            training_args.system_prompt,
        )
    ).filter(lambda x: x is not None)

    # 1) make sure we have a pad token (use EOS for Llama-style models)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # ultra-rare fallback: create a real PAD token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))

    # 2) mirror onto model/config
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 3) left padding is best for causal LMs
    tokenizer.padding_side = "left"

    # Tag CRYPTIC if the column is missing
    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
        if "task" not in dataset[split].column_names:
            dataset[split] = dataset[split].add_column("task", ["MATH"] * len(dataset[split]))
        if "is_replay" not in dataset[split].column_names:
            dataset[split] = dataset[split].add_column("is_replay", [False] * len(dataset[split]))

    # ------ Subsample 10% of validation for eval ------
    if training_args.do_eval and script_args.dataset_test_split in dataset:
        full_eval = dataset[script_args.dataset_test_split]
        n_total = len(full_eval)
        n_keep = max(1, int(n_total * 0.1))
        eval_ds = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))
    else:
        eval_ds = None

    # wrapped dataset
    train_ds = ReplayMixDataset(
        base_dataset=dataset[script_args.dataset_train_split],
        tokenizer=tokenizer,
    )

    replay_buffer = ReplayBuffer(capacity=4000, C=1.0, debug_steps=3)

    # build callbacks
    callback_objects = get_callbacks(
        training_args,
        model_args,
        replay_buffer=replay_buffer,
        tokenizer=tokenizer,          # pass it through
    )

    callback_objects.append(LossLoggingCallback(training_args.output_dir))
    trainer = GRPOTrainerReplay(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        callbacks=callback_objects,
        replay_buffer=replay_buffer,
        mix_exploit_ratio=0.7,
        constant_test_reward=4.0,
        processing_class=tokenizer,
        temp_start=1.0,
        temp_end=0.3,
        anneal_steps=3000,
        high_temp_period=5000,
        easy_pool=easy_pool,
        mix_schedule=None,
    )
    trainer.data_collator.sort_by_length = False

    # -------- Train (FORCED resume path) --------
    last_checkpoint = training_args.resume_from_checkpoint
    if last_checkpoint is None and os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # -------- Save model & card --------
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["open-r1"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # -------- Evaluate --------
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # -------- Hub upload --------
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])
