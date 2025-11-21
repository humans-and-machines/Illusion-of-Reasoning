"""Helper types and utilities for the GRPO replay trainer.

This module was extracted from :mod:`training.grpo_runtime_impl_full` to
keep the main runtime and trainer implementation small and focused.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import csv
import numpy as np

from .utils.replay_buffer import ReplayBuffer
from .grpo_runtime_env import torch, wandb, TrainerCallback


TEXT_COMPLETION_KEYS = (
    "completions_text",
    "completions",
    "generated_responses",
    "responses",
    "texts",
)


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
    for key in TEXT_COMPLETION_KEYS:
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
