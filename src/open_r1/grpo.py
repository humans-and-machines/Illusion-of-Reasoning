# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entry-point script for GRPO training with the two-stage reasoning ➜ answer
hierarchical rollout.

Changes (2025-07-10)
────────────────────
* Switched to **trl.TrlParser / trl.ModelConfig** — fixes AttributeError
  (`transformers` has no submodule `trl`).
* Import `get_peft_config` from **trl** rather than `transformers.peft`.
* Added optional `return_reason` flag passthrough so you can inspect Stage-1
  chains-of-thought in logged completions.
"""

from __future__ import annotations
from open_r1.utils.replay_buffer import ReplayBuffer
import copy
import math
import numpy as np
import torch.distributed as dist

from trl.trainer.grpo_trainer import GRPOTrainer
from accelerate.state import AcceleratorState
import functools
import logging
import os
import sys
from typing import Any
import hashlib, json
import random   # ← add copy here
import torch
import torch.serialization  # ensure sub-module is imported early

from open_r1.utils.replay_buffer   import ReplayBuffer
from open_r1.utils.replay_dataset  import ReplayMixDataset
from functools import partial

# ────────────────── ZeRO pickle patch (torch-2.6) ─────────────────────────

torch.serialization._default_weights_only = False  # type: ignore[attr-defined]
from transformers import DataCollatorWithPadding

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from trl import GRPOTrainer

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
import os
os.environ["NLTK_DATA"] = "/n/fs/similarity/open-r1/openr1/nltk_data"
#os.environ.setdefault("EASY_DATASET_NAME", "od2961/mini-crosswords")  # ← add this


# ───────────────────────── Library imports ───────────────────────────────

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

from trl import TrlParser, ModelConfig, get_peft_config

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.utils.hierarchical_rollout import (
    HierarchicalGRPOTrainer,
    HierarchicalRollout
)

from transformers import DataCollatorWithPadding

import re

logger = logging.getLogger(__name__)


from trl import GRPOTrainer
from open_r1.utils.replay_buffer import ReplayBuffer
import torch.distributed as dist
from open_r1.rewards_core import crossword_accuracy_reward, pure_accuracy_reward          # ← import once

# ──────────────────────────────────────────────────────────────────────────────

from transformers import TrainerCallback
import wandb


import csv, os, wandb
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
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
    def _init_csv(self, payload):
        if self._csv_initialized:
            return
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + list(payload))
            writer.writeheader()
        self._csv_initialized = True

    def on_log(self, args, state, control, logs=None, **kwargs):
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
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + list(payload))
            writer.writerow({"step": step, **payload})

from typing import Any, Dict, List, Optional, Tuple

def _make_conversation(
    example: dict,
    prompt_column: str,
    solution_column: str,
    tokenizer,
    system_prompt: str | None,
    max_prompt_tokens: int = 2048,
    **kwargs,
):
    """
    Build a chat-style sample for RL/GRPO.

    Fixes:
    - Do NOT strip <think>/<answer> from the system prompt (only sanitize USER text).
    - Ensure system_prompt is injected if missing.
    - Token length guard uses tokenizer.apply_chat_template when available.

    Returns a dict or None (if over-length).
    """
    import re, logging
    logger = logging.getLogger(__name__)

    def _strip_answer_blocks(s: str) -> str:
        # Remove any embedded solutions/thoughts if they accidentally show up in user/system text
        # NOTE: We will only apply this to USER messages, not to the system prompt.
        s = re.sub(r"(?is)<think>.*?</think>\s*", "", s)
        s = re.sub(r"(?is)<answer>.*?</answer>\s*", "", s)
        return s

    # ---- unpack messages ----
    raw_prompt = example.get(prompt_column, None)
    messages: list[dict[str, str]] = []
    dropped_assistants = 0

    # Case A: dict with 'role' and 'content' (can be lists or scalars)
    if isinstance(raw_prompt, dict) and ("role" in raw_prompt and "content" in raw_prompt):
        roles = raw_prompt.get("role")
        conts = raw_prompt.get("content")

        # dict-of-arrays
        if isinstance(roles, (list, tuple)) and isinstance(conts, (list, tuple)):
            for r, c in zip(roles, conts):
                role = str(r).strip().lower()
                content = str(c)
                if role == "assistant":
                    dropped_assistants += 1
                    continue
                if role not in ("system", "user"):
                    role = "user"
                if role == "user":
                    content = _strip_answer_blocks(content)
                messages.append({"role": role, "content": content})
        else:
            # single message dict
            role = str(roles).strip().lower()
            content = str(conts)
            if role != "assistant":
                if role not in ("system", "user"):
                    role = "user"
                if role == "user":
                    content = _strip_answer_blocks(content)
                messages.append({"role": role, "content": content})
            else:
                dropped_assistants += 1

    # Case B: list of {"role","content"} dicts — drop assistant entries
    elif isinstance(raw_prompt, list):
        for m in raw_prompt:
            role = str(m.get("role", "user")).strip().lower()
            content = str(m.get("content", ""))
            if role == "assistant":
                dropped_assistants += 1
                continue
            if role not in ("system", "user"):
                role = "user"
            if role == "user":
                content = _strip_answer_blocks(content)
            messages.append({"role": role, "content": content})

    # Case C: plain string → treat as single user message (sanitized)
    elif isinstance(raw_prompt, str):
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _strip_answer_blocks(raw_prompt.strip())})

    # Case D: fallback to 'board' field if present (legacy)
    else:
        board = str(example.get("board", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": board})

    if dropped_assistants:
        logger.debug("Dropped %d assistant message(s) from '%s'", dropped_assistants, prompt_column)

    # If there is no user message left, fall back to 'board'
    if not any(m.get("role") == "user" for m in messages):
        board = str(example.get("board", "")).strip()
        if not board:
            raise ValueError("No user content after filtering and no 'board' field present")
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": board})

    # Ensure a system message is present if system_prompt provided
    if system_prompt and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Final safety scrub: ONLY sanitize USER text (leave system untouched to preserve tag template)
    for m in messages:
        if m.get("role") == "user":
            m["content"] = _strip_answer_blocks(m["content"])

    # ---- optional augmentation (legacy fields) ----
    size_val = example.get("size", None)
    moves_val = example.get("moves", None)
    if size_val is not None or moves_val is not None:
        size_str = f"Board size: {size_val}x{size_val}" if size_val is not None else ""
        moves_str = f"Minimal moves to solve: {moves_val}" if moves_val is not None else ""
        augment = "\n".join(s for s in (size_str, moves_str) if s)
        for m in messages:
            if m.get("role") == "user":
                m["content"] = f"{m['content']}\n{augment}"
                break

    # ---- extract solution (from solution_column ONLY) ----
    raw_sol = example.get(solution_column, None)
    if raw_sol is None:
        raise ValueError(f"Dataset row missing '{solution_column}'")

    if isinstance(raw_sol, (list, tuple)):
        sol_core = ",".join(str(t).strip() for t in raw_sol if str(t).strip())
    else:
        raw_sol = str(raw_sol)
        m = re.search(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>", raw_sol)
        sol_core = (m.group(1) if m else raw_sol).strip()

    # ---- length guard (prefer templated token count) ----
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        n_tokens = int(templated.input_ids.shape[-1])
    except Exception:
        flat = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        n_tokens = int(tokenizer(flat, return_tensors="pt").input_ids.shape[-1])

    if n_tokens >= max_prompt_tokens:
        logger.warning("Skipping over-length prompt (%s tokens)", n_tokens)
        return None

    return {
        "prompt": messages,           # chat-format; call apply_chat_template downstream
        "answer": sol_core,
        "accuracy": 0.0,
        "is_replay": False,
        "task": str(example.get("task", "MATH")),
        "mix_group_id": -1,
        "mix_copy_idx": -1,
    }
    
from torch.utils.data import DataLoader, RandomSampler

def _load_easy_pool(script_args, tokenizer, training_args):
    """
    If EASY_DATASET_NAME is defined (or script_args.easy_dataset_name exists),
    load it and format to the same schema as the cryptic set, tagging task=EASY.
    Returns: List[dict] or None
    """
    easy_name = getattr(script_args, "easy_dataset_name", None) \
                or os.environ.get("EASY_DATASET_NAME") \
                or os.environ.get("EASY_DATASET")
    if not easy_name:
        return None

    try:
        easy_raw = datasets.load_dataset(easy_name)
    except Exception as e:
        print(f"[EASY] failed to load '{easy_name}': {e}")
        return None

    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")

    def _fmt(ex):
        row = _make_conversation(
            ex, pc, sc, tokenizer, training_args.system_prompt, prefix=""
        )
        if row is not None:
            row["task"] = "EASY"
        return row

    easy = easy_raw.map(_fmt).filter(lambda x: x is not None)

    for split in list(easy.keys()):
        if "messages" in easy[split].column_names:
            easy[split] = easy[split].remove_columns("messages")

    # pick a split for training
    train_key = script_args.dataset_train_split \
        if script_args.dataset_train_split in easy \
        else ("train" if "train" in easy else list(easy.keys())[0])

    # convert to a python list of dicts (fast random access)
    pool = list(easy[train_key])
    print(f"[EASY] loaded {len(pool)} items from '{easy_name}' ({train_key})")
    return pool

def _wrap_reward_for_nested(fn):
    import functools
    @functools.wraps(fn)
    def wrapped(*, prompts, completions, **kwargs):
        is_nested = (
            isinstance(completions, list) and
            len(completions) > 0 and
            isinstance(completions[0], (list, tuple))
        )
        if not is_nested:
            return fn(prompts=prompts, completions=completions, **kwargs)

        B = len(completions)
        sizes = [len(row) for row in completions]

        # Flatten inputs
        flat_comps, flat_prompts = [], []
        for i, row in enumerate(completions):
            p_i = prompts[i] if isinstance(prompts, list) and len(prompts) == B else prompts
            flat_comps.extend(row)
            flat_prompts.extend([p_i] * len(row))

        # Expand per-example kwargs from len=B → len=B*K
        def _expand(v):
            if isinstance(v, list) and len(v) == B:
                return [v[i] for i in range(B) for _ in range(sizes[i])]
            return v
        kwargs = {k: _expand(v) for k, v in kwargs.items()}

        flat_scores = fn(prompts=flat_prompts, completions=flat_comps, **kwargs)

        # Repack to B×K
        out, idx = [], 0
        for k in sizes:
            out.append(list(flat_scores[idx:idx + k]))
            idx += k
        return out
    return wrapped

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
    p = 0.10
    for t, pe in EASY_MIX_SCHEDULE:
        if step >= int(t):
            p = float(pe)
        else:
            break
    return p

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
    s = " ".join(txts).strip().replace("\n", " ")
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s

def _to_float_list(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
    except Exception:
        pass
    if isinstance(x, list):
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


# --- put these near the other helpers in your file ---
from typing import Dict, List, Optional, Tuple, Any

def _label_easy_copy(ex: dict, group_id: int, copy_idx: int, total: int = 4) -> dict:
    """Prefix the user turn with [G:<id>] [COPY:i/total] and set bookkeeping keys."""
    tag = f"[G:{group_id}] [COPY:{copy_idx}/{total}]"

    def _prefix(s: str) -> str:
        # strip any old tags to avoid duplication on retries
        s = re.sub(r"^\s*\[G:\d+\]\s*\[COPY:\d+/\d+\]\s*\n?", "", s)
        return f"{tag}\n{s}"

    if isinstance(ex.get("prompt"), list):
        for m in ex["prompt"]:
            if m.get("role") == "user":
                m["content"] = _prefix(m.get("content", ""))
                break
    elif isinstance(ex.get("prompt"), str):
            ex["prompt"] = _prefix(ex["prompt"])
    elif isinstance(ex.get("messages"), list):
            # older schema support
            for m in ex["messages"]:
                if m.get("role") == "user":
                    m["content"] = _prefix(m.get("content", ""))
                    break

    # uniform keys so TRL's union-of-keys logic never KeyErrors
    ex["task"] = "EASY"
    ex.setdefault("is_replay", False)
    ex["mix_group_id"] = group_id
    ex["mix_copy_idx"] = copy_idx
    return ex

def _summarize_val(v: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(v, torch.Tensor):
        return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, list):
        t = type(v[0]).__name__ if v else "empty"
        return f"list[{t}] len={len(v)}"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())[:8]} ...)"
    return type(v).__name__


# put this at module scope (or as a method) – not inside another function
def _default_join_messages(msgs):
    # very simple fallback if no chat template is available
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts) + "\nASSISTANT:"

def _normalize_for_trl(ex, proc=None, add_gen_prompt=True):
    e = dict(ex)  # copy
    # Prefer 'messages' if present
    msgs = e.pop("messages", None)
    p = e.get("prompt", None)

    if isinstance(msgs, list):
        if proc is not None and hasattr(proc, "apply_chat_template"):
            e["prompt"] = proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_gen_prompt
            )
        else:
            e["prompt"] = _default_join_messages(msgs)
    elif isinstance(p, list):  # 'prompt' provided as chat messages
        if proc is not None and hasattr(proc, "apply_chat_template"):
            e["prompt"] = proc.apply_chat_template(
                p, tokenize=False, add_generation_prompt=add_gen_prompt
            )
        else:
            e["prompt"] = _default_join_messages(p)
    elif isinstance(p, str):
        pass  # already fine
    else:
        raise ValueError(f"Example is missing a usable prompt; keys={list(e.keys())}")

    # ensure only string prompt is kept
    e.pop("messages", None)
    return e

import time


class GRPOTrainerReplay(GRPOTrainer):
    def __init__(
        self,
        *args,
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_warmup: int = 500,
        mix_exploit_ratio: float = 0.9,
        constant_test_reward: Optional[float] = None,
        inject_every_batch: bool = False,
        T0: float = 1.0,
        T1: float = 0.3,
        anneal_steps: int = 3_000,
        hiT_period: int = 2_000,
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
        self.replay_buffer        = replay_buffer
        self.replay_warmup        = replay_warmup
        self.mix_exploit_ratio    = mix_exploit_ratio
        self.constant_test_reward = constant_test_reward

        # temperature schedule
        self.T0, self.T1 = T0, T1
        self.anneal_steps = max(1, anneal_steps)
        self.hiT_period   = max(1, hiT_period)
        self._easy_group_seq = 0

        # NEW: EASY mixing
        self.easy_pool     = easy_pool or []
        self.mix_schedule  = mix_schedule or EASY_MIX_SCHEDULE

        # state
        self._gen_round = -1
        self._latest_injected_uids = []
        self._printed_out_keys_once = False
        self.vllm_cooldown = 3
        self._last_vllm_upload_ts = None

    @staticmethod
    def _ensure_pad_token_on(tok, model):
        if tok is None:
            return
        if getattr(tok, "pad_token_id", None) is None:
            if getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})
                if hasattr(model, "resize_token_embeddings"):
                    model.resize_token_embeddings(len(tok))
        if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
            model.config.pad_token_id = tok.pad_token_id
        try:
            tok.padding_side = "left"
        except Exception:
            pass


    def _p_easy(self, step: int) -> float:
        if not self.mix_schedule:
            return 0.0
        p = 0.10
        for t, pe in self.mix_schedule:
            if step >= int(t):
                p = float(pe)
            else:
                break
        return max(0.0, min(1.0, p))
    # ─────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────
    def _dump_out_once(self, out: dict):
        if getattr(self, "_dumped_out_once", False):
            return
        self._dumped_out_once = True
        print("[Replay][DEBUG] out keys:", list(out.keys()))

    # ─────────────────────────────────────────────────────────────────────
    # main hook
    # ─────────────────────────────────────────────────────────────────────
    def _prepare_inputs(self, generation_batch):
        # skip when not training or no buffer
        if not self.model.training or self.replay_buffer is None:
            return super()._prepare_inputs(generation_batch)

        # ── 1. temperature schedule ────────────────────────────────
        step = self.state.global_step
        if step < self.anneal_steps:                      # linear decay
            frac = step / float(self.anneal_steps)
            new_T = self.T0 + frac * (self.T1 - self.T0)
        else:                                             # sprinkle hi-T
            new_T = self.T0 if (step - self.anneal_steps) % self.hiT_period == 0 else self.T1
        self.model.generation_config.temperature = new_T
        if _is_rank0(self.accelerator):
            print(f"[Anneal] step={step}  T={new_T:.3f}")

        # ── EASY↔️CRYPTIC mixture: per-step gate; insert 1 EASY clue ×4 at positions [0:4) ──
        if isinstance(generation_batch, list) and self.easy_pool:
            bs = len(generation_batch)
            R  = 4
            pe = self._p_easy(self.state.global_step)   # schedule-driven value for this step

            if _is_rank0(self.accelerator):
                print(f"[MixDBG] step={self.state.global_step} bs={bs} p_easy={pe:.2f}")

            # Gate the mixing once per step.
            # Option A (probabilistic): fire with probability = pe
            mix = random.random()
            do_mix = (bs >= R) and (pe > 0.0) and ( mix < pe)
            print("Are we mixing?", do_mix, pe, mix)

            # Option B (thresholded by schedule): uncomment the next line instead
            # do_mix = (bs >= R) and (pe > 0.30)

            if do_mix:
                start = 0  # always the first block
                base = copy.deepcopy(random.choice(self.easy_pool))
                base["task"] = "EASY"
                base.setdefault("is_replay", False)

                gid = getattr(self, "_mix_group_counter", 0)
                self._mix_group_counter = gid + 1

                for j in range(R):
                    generation_batch[start + j] = _label_easy_copy(
                        copy.deepcopy(base), gid, j + 1, total=R
                    )

                if _is_rank0(self.accelerator):
                    print(f"[Mix] step={self.state.global_step} p_easy={pe:.2f} → inserted 1 group of {R} at [0:4)")

            else:
                if _is_rank0(self.accelerator):
                    print(f"[Mix] step={self.state.global_step} skipped (bs={bs}, p_easy={pe:.2f})")

        # ── 2. decide whether this batch starts a new “round” ──────
        #     (steps_per_generation hard-coded to 2 → every other batch)
        steps_per_gen = getattr(self.args, "steps_per_generation", 8)
        gen_round = step // steps_per_gen
        new_round = (gen_round != self._gen_round)
        self._gen_round = gen_round

        # inject only on even-numbered batches (= new_round True)
        inject_now = new_round and (new_T <= self.T1 + 1e-4)

        # ── 3. replay injection when conditions met ─────────────────
        is_rank0 = _is_rank0(self.accelerator)
        if (
            inject_now and is_rank0
            and len(self.replay_buffer) >= self.replay_warmup
        ):
            uid = group = None
            try:
                uid = self.replay_buffer.sample_uid(
                    mix_exploit_ratio=self.mix_exploit_ratio
                )
                if uid is not None and uid >= 0:
                    group = self.replay_buffer.get_group(uid)
            except Exception as e:
                print(f"[ReplayPrep][ERR] sample/get_group failed: {e!r}")

            if uid is not None and group:
                generation_batch = self._inject_group(generation_batch, uid, group)
                print(f"[ReplayPrep] step={step} injected uid={uid} size={len(group)}")

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
        if not _is_rank0(self.accelerator):       # <- guard
            self._latest_injected_uids.clear()
            return
        if not getattr(self, "_latest_injected_uids", None):
            return
        if self.replay_buffer is None:
            self._latest_injected_uids.clear()
            return

        # find something we can treat as reward
        reward_keys = ("rewards", "reward", "scores", "advantages")
        rewards = None
        for k in reward_keys:
            if k in out:
                rewards = out[k]
                break

        # default fallback
        r = float(self.constant_test_reward or 0.0)

        try:
            import numpy as np
            import torch

            if rewards is not None:
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.detach().cpu().float().numpy()
                elif not isinstance(rewards, np.ndarray):
                    rewards = np.asarray(rewards, dtype=np.float32)
                r = float(np.mean(rewards))
        except Exception as e:
            if _is_rank0(self.accelerator):
                print(f"[Replay][WARN] couldn't aggregate rewards for crediting: {e}")

        for uid in self._latest_injected_uids:
            try:
                self.replay_buffer.update_priority_by_uid(uid, r)
            except Exception as e:
                if _is_rank0(self.accelerator):
                    print(f"[Replay][WARN] update_priority_by_uid failed (uid={uid}): {e}")

        self._latest_injected_uids.clear()

    # ------------------ hook 2: generate & score (we add to buffer here) ------------------
    _ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        self._ensure_pad_token_on(self.processing_class, self.model)
        # Only rank-0 actually contacts the vLLM HTTP endpoint
        if self.accelerator.is_main_process and self.vllm_cooldown > 0:
            now  = time.time()
            last = self._last_vllm_upload_ts
            if last is not None:
                sleep_for = self.vllm_cooldown - (now - last)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        # ----------------------------------------------------------
        # 0)  detect whether we injected a replay item
        # ----------------------------------------------------------
        injected = (
            isinstance(inputs, list)
            and any("_buf_uid" in ex for ex in inputs)
        )

        # ---------- original sanitise / normalise block -----------
        orig_inputs = inputs
        if isinstance(inputs, list):
            clean_inputs = [{k: v for k, v in ex.items()
                            if not str(k).startswith("_")}
                            for ex in inputs]
            B = len(clean_inputs)
            gold_answers = [ex.get("answer")  for ex in clean_inputs]
            prompts_list = [ex.get("prompt")  for ex in clean_inputs]
            tasks_list   = [ex.get("task", "MATH") for ex in clean_inputs]   # ← NEW
        else:
            clean_inputs  = inputs
            gold_answers  = None
            prompts_list  = None
            B             = 0
            tasks_list = None

        # ----------------------------------------------------------
        # **NEW**  – broadcast answer / prompt if we injected
        # ----------------------------------------------------------
        if injected:
            anchor_idx      = next(
                i for i, ex in enumerate(inputs) if "_buf_uid" in ex
            )
            anchor_answer   = gold_answers[anchor_idx]
            anchor_prompt   = prompts_list[anchor_idx]

            #   – patch the lists we just built …
            gold_answers[:] = [anchor_answer] * B
            prompts_list[:] = [anchor_prompt] * B

            #   – and also patch the dicts that TRL will see
            for ex in clean_inputs:
                ex["answer"] = anchor_answer
                ex["prompt"] = anchor_prompt

        # ----------------------------------------------------------
        # continue as before – clean_inputs is now self-consistent
        # ----------------------------------------------------------
        proc = getattr(self, "processing_class", None) or self.tokenizer
        clean_inputs = [_normalize_for_trl(ex, proc) for ex in clean_inputs]

        out = super()._generate_and_score_completions(clean_inputs)

        self._dump_out_once(out)

        # rank / world info
        dist_ok = dist.is_initialized()
        rank    = dist.get_rank()       if dist_ok else 0
        world   = dist.get_world_size() if dist_ok else 1
        is_r0   = (rank == 0)
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is not None:
            if getattr(tok, "pad_token_id", None) is None:
                eos_tok = getattr(tok, "eos_token", None)
                if eos_tok is not None:
                    tok.pad_token = eos_tok
                else:
                    try:
                        tok.add_special_tokens({'pad_token': '[PAD]'})
                        if hasattr(self.model, "resize_token_embeddings"):
                            self.model.resize_token_embeddings(len(tok))
                    except Exception:
                        pass
            if getattr(self.model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
                self.model.config.pad_token_id = tok.pad_token_id
            try:
                tok.padding_side = "left"
            except Exception:
                pass
                
        # If batch was a collated dict without lists of gold/prompts, skip replay this step
        if not isinstance(orig_inputs, list):
            if is_r0:
                print("[Replay][DEBUG] collated batch without per-example gold/prompt; skipping replay push.")
            return out

        # --- 3) decode completions as B×K (no sorting/unsorting) ---
        completions_txt = None
        if isinstance(out.get("completion_ids"), torch.Tensor):
            ids = out["completion_ids"]                 # [B*K, T]
            total, T = ids.shape
            if B == 0 or total % B != 0:
                if is_r0:
                    print(f"[Replay][DEBUG] unexpected shapes: total={total}, B={B}")
                return out
            K = total // B
            ids = ids.view(B, K, T)

            mask = out.get("completion_mask", None)
            mask = mask.view(B, K, T).bool() if isinstance(mask, torch.Tensor) else None

            completions_txt = []
            for i in range(B):
                per_prompt = []
                for j in range(K):
                    seq = ids[i, j] if mask is None else ids[i, j][mask[i, j]]
                    txt = tok.decode(seq.detach().cpu().tolist(),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                    per_prompt.append(txt)
                completions_txt.append(per_prompt)
        else:
            for key in ("completions_text", "completions", "generated_responses", "responses", "texts"):
                if key in out and isinstance(out[key], list) and out[key]:
                    texts = out[key]
                    K = max(1, len(texts) // max(1, B))
                    completions_txt = [texts[i*K:(i+1)*K] for i in range(B)]
                    break

        # Record when that upload finished (approx.)
        if self.accelerator.is_main_process:
            self._last_vllm_upload_ts = time.time()


        if completions_txt is None:
            if is_r0:
                print("[Replay][DEBUG] no decodable completions; skipping replay push.")
            return out
        
        # ------------------------------------------------------------------
        # 2)  Keep nested B×K completions (no flatten, no repetition)
        # ------------------------------------------------------------------
        if isinstance(completions_txt, list) and completions_txt:
            B = len(completions_txt)
            K = max(1, len(completions_txt[0]))
            # (Optional) tiny debug once
            if not getattr(self, "_paired_debug_once", False):
                self._paired_debug_once = True
                print(f"[LogPairing] B={B} K={K} (nested) — leaving TRL shapes untouched")
                
        unsort_idx = out.get("unsort_idx")
        if unsort_idx is not None:
            order = unsort_idx.tolist()
            completions_txt = [completions_txt[i] for i in order]
            if gold_answers is not None:
                gold_answers = [gold_answers[i] for i in order]
            if prompts_list is not None:
                prompts_list = [prompts_list[i] for i in order]
            if tasks_list is not None:
                tasks_list = [tasks_list[i] for i in order]

        if isinstance(inputs, list) and any("_buf_uid" in ex for ex in inputs):
            print("FIXING INJECTED!")
            # ── pick whatever index we injected (usually 0) ────────────────
            inj_idx      = next(i for i, ex in enumerate(inputs) if "_buf_uid" in ex)
            gold_anchor  = gold_answers[inj_idx]
            prompt_anchor = prompts_list[inj_idx]

            # ── force-broadcast to the rest of the batch ───────────────────
            gold_answers   = [gold_anchor]   * len(gold_answers)
            prompts_list   = [prompt_anchor] * len(prompts_list)

            # ------------------------------------------------------------------
            # 7) ***NEW***  patch the returned `out` so downstream code
            #               receives the same (fixed) alignment
            # ------------------------------------------------------------------
            def _set(dest: dict, keys, value):
                for k in keys:
                    dest[k] = value

            _set(out, ("gold_answers", "answers", "gold"),      gold_answers)
            _set(out, ("prompts_list", "prompts", "prompt"),    prompts_list)
            _set(out, ("tasks_list", "tasks"),                  tasks_list)   # ← NEW

            if "reward_kwargs" in out and isinstance(out["reward_kwargs"], dict):
                rk = out["reward_kwargs"]
                if "answer" in rk: rk["answer"] = gold_answers
                if "gold"   in rk: rk["gold"]   = gold_answers
                rk["tasks"] = tasks_list     
            
        # ------------------------------------------------------------------
        # 3)  (optional) keep only the first 8 completions from each prompt
        # ------------------------------------------------------------------
        MAX_K = 4
        completions_txt = [c[:MAX_K] for c in completions_txt]

        winners_local = []
        for i in range(B):
            gold   = gold_answers[i]
            prompt = prompts_list[i]
            if not isinstance(gold, str) or not gold.strip() or prompt is None:
                continue
            preds_i = completions_txt[i]          # <- still nested
            accs_i  = pure_accuracy_reward(preds_i, [gold] * len(preds_i))
            j_hit   = next((jj for jj, a in enumerate(accs_i) if a == 1.0), None)
            if j_hit is None:
                continue

            print(f"[Replay][HIT] rank={rank} i={i} gold='{gold}'")
            winners_local.append(_to_env_schema({
                "prompt": prompt,
                "answer": gold,
                "reward": 1.0,
                "_last_success_pred": preds_i[j_hit],
            }))

        # --- 5) gather winners from all ranks ---
        if world > 1 and dist_ok:
            gathered = [None] * world
            dist.all_gather_object(gathered, winners_local)
            winners_all = [ex for sub in gathered for ex in (sub or [])]
            if is_r0:
                counts = [len(sub or []) for sub in gathered]
                print(f"[Replay][GATHER] world={world} per-rank winners={counts} total={sum(counts)}")
        else:
            winners_all = winners_local
            if is_r0:
                print(f"[Replay][GATHER] single-process winners={len(winners_all)}")

        # --- 6) rank-0: dedup & push (buffer may be empty → still truthy check!) ---
        if self.model.training and is_r0 and (self.replay_buffer is not None):
            def _sig(ex):
                msgs = ex.get("prompt", [])

                # prompt is already flattened to a string → just use it
                if isinstance(msgs, str):
                    user = msgs.strip()

                # prompt is still a list[dict] → extract the user lines
                else:
                    user = " ".join(
                        m.get("content", "").strip()
                        for m in msgs
                        if m.get("role") == "user"
                    )

                gold = (ex.get("answer") or "").strip().lower()
                return (user, gold)

            seen, unique = set(), []
            for ex in winners_all:
                s = _sig(ex)
                if s in seen:
                    continue
                seen.add(s)
                unique.append(ex)

            if unique:
                ret = self.replay_buffer.add_group(unique, reward=1.0)
                if isinstance(ret, tuple) and len(ret) == 2:
                    ok, uid = ret
                else:
                    ok, uid = True, ret

                if ok and int(uid) >= 0:
                    print(f"[ReplayAdd] uid={uid} size={len(unique)} from {world} ranks | len(buf)={len(self.replay_buffer)}")
                else:
                    dbg = getattr(self.replay_buffer, "debug_state", lambda: {})()
                    print(f"[Replay][WARN] add_group failed (uid={uid}). state={dbg}")
            else:
                print("[Replay][DEBUG] gathered no unique winners to add.")

            # optional credit of injected UIDs
            try:
                self._maybe_credit_injected_uids(out)
            except Exception as e:
                print(f"[Replay][WARN] credit failed: {e}")

        return out
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

        nb = len(generation_batch)
        gk = min(len(tagged), nb // 2)  # inject at most half the batch
        # replace first gk items (don’t prepend)
        new_batch = copy.deepcopy(tagged[:gk]) + list(generation_batch)[gk:]

        if _is_rank0(self.accelerator):
            def _head_user(b):
                try:
                    msgs = b.get("messages") or b.get("prompt") or []
                    if isinstance(msgs, list):
                        return next((m.get("content","") for m in msgs if m.get("role")=="user"), "<no-user>")
                    return str(msgs)[:60]
                except Exception:
                    return "<err>"
            print(f"[ReplayInject] replaced {gk}/{nb} (uid={uid})")
            if nb > 0:
                print(f"[ReplayInject] head user after : «{_head_user(new_batch[0])}»")

        self._latest_injected_uids = [uid]
        return new_batch

    def _extract_completions_and_rewards(self, out: Dict[str, Any], batch_size: int):
        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)

        # ---------------- rewards ----------------
        reward_keys = ["rewards", "reward", "scores", "advantages"]
        rewards = None
        for k in reward_keys:
            if k in out:
                rewards = out[k]
                break
        rewards = _to_float_list(rewards)

        # ---------------- completions ----------------
        text_keys = ["completions_text", "completions", "generated_responses", "responses", "texts"]
        completions = None
        for k in text_keys:
            if k in out and isinstance(out[k], list) and out[k] and isinstance(out[k][0], str):
                completions = out[k]
                break

        # TRL variant that only returns token ids
        if completions is None and "completion_ids" in out and tok is not None:
            seqs = out["completion_ids"]
            try:
                if hasattr(seqs, "detach"):
                    seqs = seqs.detach().cpu().tolist()
                completions = tok.batch_decode(seqs, skip_special_tokens=True)
            except Exception:
                completions = None

        # Newer TRL: generation_outputs.sequences
        if completions is None and "generation_outputs" in out and tok is not None:
            seqs = out["generation_outputs"]
            if isinstance(seqs, (list, tuple)) and seqs:
                seqs = seqs[0]
            seqs = getattr(seqs, "sequences", None)
            if seqs is not None:
                try:
                    if hasattr(seqs, "detach"):
                        seqs = seqs.detach().cpu().tolist()
                    completions = tok.batch_decode(seqs, skip_special_tokens=True)
                except Exception:
                    completions = None

        # ---------------- size-normalise ----------------
        def pad(x, pad_val, n):
            if x is None:
                return None
            if len(x) < n:
                x = list(x) + [pad_val] * (n - len(x))
            return x[:n]

        completions = pad(completions, None, batch_size)
        rewards     = pad(rewards,     0.0,  batch_size)

        return completions, rewards



# ────────────────────────────── main ────────────────────────────────────
def main(script_args, training_args, model_args):
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
        reward_funcs = {k: _wrap_reward_for_nested(v) for k, v in reward_funcs.items()}
    elif isinstance(reward_funcs, (list, tuple)):
        reward_funcs = [_wrap_reward_for_nested(v) for v in reward_funcs]
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
        n_total   = len(full_eval)
        n_keep    = max(1, int(n_total * 0.1))
        eval_ds   = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))
    else:
        eval_ds = None

    # wrapped dataset
    train_ds = ReplayMixDataset(
        base_ds         =  dataset[script_args.dataset_train_split],
        tok     = tokenizer,
    )

    replay_buffer = ReplayBuffer(capacity=4000, C=1.0, debug_steps=3)

    print(
        "[RB] impl:", type(replay_buffer),
        "module:", replay_buffer.__class__.__module__,
        "capacity:", getattr(replay_buffer, "capacity", None),
        "has last_error:", hasattr(replay_buffer, "last_error"),
    )
    import inspect, open_r1.utils.replay_buffer as rb
    print("add_group source head:\n", inspect.getsource(rb.ReplayBuffer.add_group)[:400])

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
        train_dataset=ReplayMixDataset(
            base_ds=dataset[script_args.dataset_train_split],
            tok=tokenizer,
        ),
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        callbacks=callback_objects,
        replay_buffer=replay_buffer,
        mix_exploit_ratio=0.7,
        constant_test_reward=4.0,
        processing_class=tokenizer,
        T0=1.0, T1=0.3, anneal_steps=3000, hiT_period=5000,
        easy_pool=None,
        mix_schedule=None,
    )
    trainer.data_collator.sort_by_length = False

    # -------- Train (FORCED resume path) --------
    last_ckpt = (
        training_args.resume_from_checkpoint
        or (get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None)
    )
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
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


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
