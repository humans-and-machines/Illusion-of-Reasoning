# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
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

# ───────────────────────── stdlib & typing ─────────────────────────
import copy
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ───────────────────────── third-party ────────────────────────────
import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.serialization  # ensure sub-module is imported early
import transformers
from accelerate.state import AcceleratorState
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

# ───────────────────────── open_r1 imports ────────────────────────
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs  # (you may bypass with reward_router below)
from open_r1.rewards_core import (
    crossword_accuracy_reward,
    pure_accuracy_reward,
    rush_solution_shaped, 
    rush_solution_exact,        # crossword exact/shaping wrapper
    pure_accuracy_reward_math
)

from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.hierarchical_rollout import (
    HierarchicalGRPOTrainer,
    HierarchicalRollout,
)
from open_r1.utils.replay_buffer import ReplayBuffer
from open_r1.utils.replay_dataset import ReplayMixDataset


# ────────────────── ZeRO pickle patch (torch-2.6) ─────────────────────────
torch.serialization._default_weights_only = False  # type: ignore[attr-defined]
torch.serialization.add_safe_globals(
    {
        ("deepspeed.runtime.zero.config", "ZeroStageEnum"): ZeroStageEnum,
        ("deepspeed.runtime.zero.partition_parameters", "ZeroParamStatus"): ZeroParamStatus,
    }
)
_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)  # type: ignore[arg-type]

# ─────────────────── NLTK data path (WordNet) ───────────────────────────
os.environ.setdefault("NLTK_DATA", "/n/fs/similarity/open-r1/openr1/nltk_data")
# os.environ.setdefault("EASY_DATASET_NAME", "od2961/mini-crosswords")

logger = logging.getLogger(__name__)

# ───────────────────────── helpers: default task ─────────────────────────

def _task_from_script_args(script_args) -> Optional[str]:
    """
    Try to infer a single task from the YAML/CLI `reward_funcs` field.
    Returns "RUSH" | "CROSSWORD" | "MATH" | None.
    """
    if script_args is None:
        return None

    rf = getattr(script_args, "reward_funcs", None)
    if rf is None:
        return None

    # Normalize to a flat list of lowercase strings (handles str | list | tuple | dict)
    if isinstance(rf, str):
        names = [rf.lower()]
    elif isinstance(rf, dict):
        names = [str(k).lower() for k in rf.keys()]
    elif isinstance(rf, (list, tuple)):
        names = [str(x).lower() for x in rf]
    else:
        try:
            names = [str(rf).lower()]
        except Exception:
            names = []

    # Common patterns, including module-qualified forms like "rush_reward:rush_solution_exact"
    joined = " ".join(names)
    if ("rush_solution_exact" in joined) or ("rush_solution_shaped" in joined) or ("rush" in joined):
        return "RUSH"
    if ("pure_accuracy_reward_math" in joined) or ("math" in joined and "reward" in joined):
        return "MATH"
    if ("pure_accuracy_reward" in joined) or ("cross" in joined) or ("crypt" in joined):
        return "CROSSWORD"
    return None


def _default_task(args=None, *, system_prompt: Optional[str] = None,
                  dataset_name_hint: Optional[str] = None,
                  prompt_hint: Optional[str] = None) -> str:
    """
    Heuristically decide the task label when examples/batches don't carry one.
    Priority: explicit args.dataset_name → dataset_name_hint → system/prompt hints.
    """

    t_from_rf = _task_from_script_args(args)
    if t_from_rf:
        return t_from_rf

    name = ""
    if args is not None:
        # try common fields where a dataset name might live
        for fld in ("dataset_name", "dataset", "dataset_path", "output_dir", "hub_model_id"):
            val = getattr(args, fld, None)
            if val:
                name = str(val)
                break
    if not name and dataset_name_hint:
        name = str(dataset_name_hint)

    blob = " ".join(
        s for s in (
            name,
            system_prompt or "",
            prompt_hint or "",
            os.environ.get("DEFAULT_TASK_HINT", "")
        ) if s
    ).lower()

    if any(k in blob for k in ("rush", "carpark", "car_parking", "parking")):
        return "RUSH"
    if any(k in blob for k in ("cross", "crypt")):
        return "CROSSWORD"
    if any(k in blob for k in ("math", "algebra", "calculus")):
        return "MATH"

    # Last resort: lean CROSSWORD to avoid the previous bad default to "MATH"
    # that caused crossword runs to get zero rewards.
    return "CROSSWORD"

# ───────────────────────── reward router (task-aware) ────────────────────
def _adapt_gold(seq_or_list, **kw):
    gold = (kw.get("answer") or kw.get("answers") or
            kw.get("gold")   or kw.get("references") or
            kw.get("labels"))
    if isinstance(gold, str):
        n = len(seq_or_list) if isinstance(seq_or_list, list) else 1
        return [gold] * n
    return list(gold or [])

def _flatten_nested(comps):
    if not (isinstance(comps, list) and comps and isinstance(comps[0], (list, tuple))):
        return comps, 1
    flat = [y for x in comps for y in x]
    return flat, len(comps[0])

def _to_text_list(items, proc=None):
    """
    Coerce a list of completions that may be str | list[int] | torch.Tensor
    into a list[str], decoding with `proc` if available.
    """
    try:
        import torch
    except Exception:
        torch = None

    if not isinstance(items, list):
        items = [items]

    out = []
    for x in items:
        if isinstance(x, str):
            out.append(x)
            continue
        # torch tensor of token ids
        if torch is not None and hasattr(x, "detach"):
            ids = x.detach().cpu().tolist()
            if proc is not None and hasattr(proc, "decode"):
                out.append(proc.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            else:
                out.append(str(ids))
            continue
        # list/tuple of ints = token ids
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], int):
            if proc is not None and hasattr(proc, "decode"):
                out.append(proc.decode(list(x), skip_special_tokens=True, clean_up_tokenization_spaces=False))
            else:
                out.append(" ".join(map(str, x)))
            continue
        # fallback
        out.append(str(x))
    return out


def reward_router(*, prompts=None, completions=None, tasks=None, proc=None, **kw):
    """
    Task-aware reward router.
    For RUSH: calls rush_solution_shaped per-sample, passing gold-only fields (gold_moves)
    and optional board fields (board_str, N) IF provided in reward_kwargs.
    """
    comps, K = _flatten_nested(completions)
    comps = _to_text_list(comps, proc=proc)
    gold  = _adapt_gold(comps, **kw)

    def _kw_no_answer(d):
        d = dict(d)
        for k in ("answer", "answers", "gold", "labels", "references"):
            d.pop(k, None)
        return d
    kw_clean = _kw_no_answer(kw)

    # Normalize prompts to per-sample list
    if isinstance(prompts, list) and len(prompts) == len(comps):
        p_list = prompts
    else:
        p_list = [prompts] * len(comps)

    # Optional per-sample extras (can be scalar or list aligned to samples)
    gm = kw.get("gold_moves")      # minimal moves
    bs = kw.get("board_str") or kw.get("board")
    nn = kw.get("N") or kw.get("size")

    def ith(v, i):
        if isinstance(v, list) and len(v) == len(comps): return v[i]
        if isinstance(v, (list, tuple)) and v:           return v[0]
        return v

    script_args = kw.get("script_args", None)
    t = _task_from_script_args(script_args)
    if not t:
        # fall back to batch tasks or heuristics
        prompt_hint = None
        if isinstance(prompts, list) and prompts:
            p0 = prompts[0]
            if isinstance(p0, str):
                prompt_hint = p0
            elif isinstance(p0, list):
                for m in p0:
                    if isinstance(m, dict) and m.get("role") == "user":
                        prompt_hint = m.get("content", "")
                        break
        t = (tasks[0] if isinstance(tasks, list) and tasks else _default_task(script_args, prompt_hint=prompt_hint))
    t = t.upper()

    if "RUSH" in t:
        # Per-sample scoring so each completion uses its own prompt/gold/extras
        scores = []
        for i, (p, c, g) in enumerate(zip(p_list, comps, gold)):
            s = rush_solution_shaped(
                prompts=p,
                completions=[c],
                gold=[g],
                # Gold-only shaping and/or board shaping:
                gold_moves=ith(gm, i),
                board_str=ith(bs, i),
                N=ith(nn, i),
                # Slightly stronger prefix in boardless bring-up:
                w_exact=0.5, w_solve=0.2, w_prefix=0.2, w_phi=0.1,
            )[0]
            scores.append(s)
        return scores

    if "MATH" in t:
        return pure_accuracy_reward_math(comps, gold, **kw_clean)

    # default → CROSSWORD
    return pure_accuracy_reward(comps, gold, **kw_clean)


def _wrap_reward_for_nested(fn):
    """Make any reward fn tolerant to B×K completions by flattening/repacking."""
    import functools as _ft
    @_ft.wraps(fn)
    def wrapped(*, prompts, completions, **kwargs):
        is_nested = (
            isinstance(completions, list)
            and len(completions) > 0
            and isinstance(completions[0], (list, tuple))
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

# ───────────────────────── logging callback ──────────────────────────────
import csv
import wandb
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

# ───────────────────────── conversation builder ──────────────────────────
def _make_conversation(
    example: dict,
    prompt_column: str,
    solution_column: str,
    tokenizer,
    system_prompt: str | None,
    max_prompt_tokens: int = 1500,
    **kwargs,
):
    """
    Build a chat-style sample for RL/GRPO.

    Fixes:
    - Do NOT strip <think>/<answer> from the system prompt (only sanitize USER text).
    - Ensure system_prompt is injected if missing.
    - Token length guard uses tokenizer.apply_chat_template when available.
    - **Task default** now inferred via `_default_task(...)` instead of hard "MATH".
    """
    import logging as _logging
    logger = _logging.getLogger(__name__)

    def _strip_answer_blocks(s: str) -> str:
        s = re.sub(r"(?is)<think>.*?</think>\s*", "", s)
        s = re.sub(r"(?is)<answer>.*?</answer>\s*", "", s)
        return s

    raw_prompt = example.get(prompt_column, None)
    messages: list[dict[str, str]] = []
    dropped_assistants = 0

    # Case A: dict with 'role' and 'content' (can be lists or scalars)
    if isinstance(raw_prompt, dict) and ("role" in raw_prompt and "content" in raw_prompt):
        roles = raw_prompt.get("role")
        conts = raw_prompt.get("content")

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

    # After building `messages` in _make_conversation, replace any system turn:
    if system_prompt:
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": system_prompt})

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

    # ---- task default (fixed) ----
    script_args = kwargs.get("script_args", None)
    # Try to use the first user message as a hint for task detection
    user_hint = None
    for m in messages:
        if m.get("role") == "user":
            user_hint = m.get("content", "")
            break
    task_default = _default_task(script_args, system_prompt=system_prompt, prompt_hint=user_hint)

    return {
        "prompt": messages,
        "answer": sol_core,            # already comma-joined if list
        "accuracy": 0.0,
        "is_replay": False,
        "task": str(example.get("task", task_default)),
        "mix_group_id": -1,
        "mix_copy_idx": -1,
        # NEW: pass dataset fields straight through (for Rush rewards)
        "board": example.get("board") or example.get("Board"),
        "size": example.get("size") or example.get("N"),
        "moves": example.get("moves"),
    }

# ───────────────────────── EASY pool loader (optional) ───────────────────
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
            ex, pc, sc, tokenizer, training_args.system_prompt, script_args=script_args
        )
        if row is not None:
            row["task"] = "EASY"
        return row

    easy = easy_raw.map(_fmt).filter(lambda x: x is not None)

    for split in list(easy.keys()):
        if "messages" in easy[split].column_names:
            easy[split] = easy[split].remove_columns("messages")

    train_key = script_args.dataset_train_split \
        if script_args.dataset_train_split in easy \
        else ("train" if "train" in easy else list(easy.keys())[0])

    pool = list(easy[train_key])
    print(f"[EASY] loaded {len(pool)} items from '{easy_name}' ({train_key})")
    return pool

# ───────────────────────── small utils ───────────────────────────────────
def _is_rank0(accelerator) -> bool:
    return getattr(accelerator, "is_main_process", True)

def _to_env_schema(example: Dict[str, Any]) -> Dict[str, Any]:
    """Keep full prompt history when adding to replay buffer."""
    out: Dict[str, Any] = {"prompt": example.get("prompt", [])}
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
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().tolist()
    except Exception:
        pass
    try:
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

# ───────────────────────── EASY mixing helpers ───────────────────────────
EASY_MIX_SCHEDULE = [
    (   0, 1.00), (  50, 0.90), ( 100, 0.80), (150, 0.70), (200, 0.60),
    ( 250, 0.55), ( 300, 0.53), ( 400, 0.51), (500, 0.49), (600, 0.47),
    ( 700, 0.44), ( 800, 0.42), ( 900, 0.40), (1000, 0.38), (1100, 0.36),
    (1200, 0.34), (1300, 0.32), (1400, 0.30), (1500, 0.28), (1600, 0.26),
    (1700, 0.24), (1800, 0.22), (1900, 0.20), (2000, 0.18), (2100, 0.16),
    (2200, 0.14), (2300, 0.12), (2400, 0.10), (2500, 0.08), (2600, 0.06),
    (2700, 0.04), (2800, 0.03), (2900, 0.02), (3000, 0.01),
]
def _p_easy_for_step(step: int) -> float:
    p = 0.10
    for t, pe in EASY_MIX_SCHEDULE:
        if step >= int(t):
            p = float(pe)
        else:
            break
    return p

# ───────────────────────── EASY batch tagging ────────────────────────────
def _label_easy_copy(ex: dict, group_id: int, copy_idx: int, total: int = 4) -> dict:
    """Prefix the user turn with [G:<id>] [COPY:i/total] and set bookkeeping keys."""
    tag = f"[G:{group_id}] [COPY:{copy_idx}/{total}]"

    def _prefix(s: str) -> str:
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
        for m in ex["messages"]:
            if m.get("role") == "user":
                m["content"] = _prefix(m.get("content", ""))
                break

    ex["task"] = "EASY"
    ex.setdefault("is_replay", False)
    ex["mix_group_id"] = group_id
    ex["mix_copy_idx"] = copy_idx
    return ex

def _summarize_val(v: Any) -> str:
    try:
        import torch as _t
    except Exception:
        _t = None
    if _t is not None and isinstance(v, _t.Tensor):
        return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, list):
        t = type(v[0]).__name__ if v else "empty"
        return f"list[{t}] len={len(v)}"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())[:8]} ...)"
    return type(v).__name__

# ───────────────────────── TRL normalization helpers ─────────────────────
def _default_join_messages(msgs):
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{(role.upper() if hasattr(role,'upper') else str(role).upper())}: {content}")
    return "\n".join(parts) + "\nASSISTANT:"

def _normalize_for_trl(ex, proc=None, add_gen_prompt=True):
    e = dict(ex)  # copy
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

    e.pop("messages", None)
    return e

# ───────────────────────── Trainer subclass (replay) ─────────────────────
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
        easy_pool: Optional[list] = None,
        mix_schedule: Optional[list] = None,
        **kwargs,
    ):
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.processing_class = kwargs.pop("processing_class", None) or self.tokenizer

        super().__init__(*args, **kwargs)

        # --- Ensure PAD on the exact tokenizer the trainer will use ---
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            tok.padding_side = "right"
        if hasattr(self, "model") and getattr(self.model, "config", None) is not None:
            self.model.config.pad_token_id = getattr(tok, "pad_token_id", self.model.config.eos_token_id)


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

        # EASY mixing
        self.easy_pool     = easy_pool or []
        self.mix_schedule  = mix_schedule or EASY_MIX_SCHEDULE

        # state
        self._gen_round = -1
        self._latest_injected_uids = []
        self._printed_out_keys_once = False
        self.vllm_cooldown = 3
        self._last_vllm_upload_ts = None

        try:
            self.args.return_reward = True
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

    def _dump_out_once(self, out: dict):
        if getattr(self, "_dumped_out_once", False):
            return
        self._dumped_out_once = True
        print("[Replay][DEBUG] out keys:", list(out.keys()))

    # ---------------- main hook ----------------
    def _prepare_inputs(self, generation_batch):
        if not self.model.training or self.replay_buffer is None:
            return super()._prepare_inputs(generation_batch)

        # 1) temperature schedule
        step = self.state.global_step
        if step < self.anneal_steps:
            frac = step / float(self.anneal_steps)
            new_T = self.T0 + frac * (self.T1 - self.T0)
        else:
            new_T = self.T0 if (step - self.anneal_steps) % self.hiT_period == 0 else self.T1
        self.model.generation_config.temperature = new_T
        if _is_rank0(self.accelerator):
            print(f"[Anneal] step={step}  T={new_T:.3f}")

        # EASY mixing (optional)
        if isinstance(generation_batch, list) and self.easy_pool:
            bs = len(generation_batch)
            R  = 4
            pe = self._p_easy(self.state.global_step)
            if _is_rank0(self.accelerator):
                print(f"[MixDBG] step={self.state.global_step} bs={bs} p_easy={pe:.2f}")
            mix = random.random()
            do_mix = (bs >= R) and (pe > 0.0) and (mix < pe)
            if do_mix:
                start = 0
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

        # 2) decide whether this batch starts a new “round”
        steps_per_gen = getattr(self.args, "steps_per_generation", 8)
        gen_round = step // steps_per_gen
        new_round = (gen_round != self._gen_round)
        self._gen_round = gen_round
        inject_now = new_round and (new_T <= self.T1 + 1e-4)

        # 3) replay injection when conditions met
        is_rank0 = _is_rank0(self.accelerator)
        if (
            inject_now and is_rank0
            and len(self.replay_buffer) >= self.replay_warmup
        ):
            uid = group = None
            try:
                uid = self.replay_buffer.sample_uid(mix_exploit_ratio=self.mix_exploit_ratio)
                if uid is not None and uid >= 0:
                    group = self.replay_buffer.get_group(uid)
            except Exception as e:
                print(f"[ReplayPrep][ERR] sample/get_group failed: {e!r}")

            if uid is not None and group:
                generation_batch = self._inject_group(generation_batch, uid, group)
                print(f"[ReplayPrep] step={step} injected uid={uid} size={len(group)}")

        # normalize keys across batch (avoid KeyError in TRL)
        if isinstance(generation_batch, list):
            for ex in generation_batch:
                if isinstance(ex, dict):
                    # ↓↓↓ FIX: no hard "MATH"
                    ex.setdefault("task", _default_task(self.args))
                    ex.setdefault("mix_group_id", -1)
                    ex.setdefault("mix_copy_idx", -1)
                    ex.setdefault("is_replay", False)

        return super()._prepare_inputs(generation_batch)

    def _maybe_credit_injected_uids(self, out: Dict[str, Any]) -> None:
        if not _is_rank0(self.accelerator):
            self._latest_injected_uids.clear()
            return
        if not getattr(self, "_latest_injected_uids", None):
            return
        if self.replay_buffer is None:
            self._latest_injected_uids.clear()
            return

        reward_keys = ("rewards", "reward", "scores", "advantages")
        rewards = None
        for k in reward_keys:
            if k in out:
                rewards = out[k]
                break

        r = float(self.constant_test_reward or 0.0)
        try:
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

    _ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        # --- Re-assert PAD right before TRL tokenizes (handles spawn/reload edge cases) ---
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            tok.padding_side = "right"
        if hasattr(self, "model") and getattr(self.model, "config", None) is not None:
            self.model.config.pad_token_id = getattr(tok, "pad_token_id", self.model.config.eos_token_id)



        # vLLM cooldown on rank-0
        if self.accelerator.is_main_process and self.vllm_cooldown > 0:
            now  = time.time()
            last = self._last_vllm_upload_ts
            if last is not None:
                sleep_for = self.vllm_cooldown - (now - last)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        injected = (isinstance(inputs, list) and any("_buf_uid" in ex for ex in inputs))

        # sanitize/normalize
        orig_inputs = inputs
        if isinstance(inputs, list):
            clean_inputs = [{k: v for k, v in ex.items() if not str(k).startswith("_")} for ex in inputs]
            B = len(clean_inputs)
            gold_answers = [ex.get("answer")  for ex in clean_inputs]
            prompts_list = [ex.get("prompt")  for ex in clean_inputs]
            tasks_list   = [ex.get("task", _default_task(self.args)) for ex in clean_inputs]
            # NEW: carry Rush extras for reward
            boards_list  = [ex.get("board") for ex in clean_inputs]
            sizes_list   = [ex.get("size")  for ex in clean_inputs]
            moves_list   = [ex.get("moves") for ex in clean_inputs]
        else:
            clean_inputs  = inputs
            gold_answers  = None
            prompts_list  = None
            tasks_list    = None
            boards_list   = None
            sizes_list    = None
            moves_list    = None
            B             = 0

        # broadcast answer/prompt if injected
        if injected:
            anchor_idx      = next(i for i, ex in enumerate(inputs) if "_buf_uid" in ex)
            anchor_answer   = gold_answers[anchor_idx]
            anchor_prompt   = prompts_list[anchor_idx]
            gold_answers[:] = [anchor_answer] * B
            prompts_list[:] = [anchor_prompt] * B
            for ex in clean_inputs:
                ex["answer"] = anchor_answer
                ex["prompt"] = anchor_prompt

        proc = getattr(self, "processing_class", None) or self.tokenizer
        clean_inputs = [_normalize_for_trl(ex, proc) for ex in clean_inputs]

        out = super()._generate_and_score_completions(clean_inputs)
        self._dump_out_once(out)

        dist_ok = dist.is_initialized()
        rank    = dist.get_rank()       if dist_ok else 0
        world   = dist.get_world_size() if dist_ok else 1
        is_r0   = (rank == 0)

        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tok is None:
            if is_r0:
                print("[Replay][DEBUG] no tokenizer; cannot decode completions.")
            return out

        if not isinstance(orig_inputs, list):
            if is_r0:
                print("[Replay][DEBUG] collated batch without per-example gold/prompt; skipping replay push.")
            return out

        # decode completions as B×K
        completions_txt = None
        if isinstance(out.get("completion_ids"), torch.Tensor):
            ids = out["completion_ids"]
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
                    txt = tok.decode(
                        seq.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    per_prompt.append(txt)
                completions_txt.append(per_prompt)
        else:
            for key in ("completions_text", "completions", "generated_responses", "responses", "texts"):
                if key in out and isinstance(out[key], list) and out[key]:
                    texts = out[key]
                    K = max(1, len(texts) // max(1, B))
                    completions_txt = [texts[i*K:(i+1)*K] for i in range(B)]
                    break

        if self.accelerator.is_main_process:
            self._last_vllm_upload_ts = time.time()

        if completions_txt is None:
            if is_r0:
                print("[Replay][DEBUG] no decodable completions; skipping replay push.")
            return out
        if is_r0 and completions_txt:
            sizes = [len(c) for c in completions_txt[:8]]
            print("[DBG] per-prompt K sizes:", sizes)

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
            if boards_list is not None:
                boards_list = [boards_list[i] for i in order]
            if sizes_list is not None:
                sizes_list  = [sizes_list[i]  for i in order]
            if moves_list is not None:
                moves_list  = [moves_list[i]  for i in order]

        # Always patch reward_kwargs so router gets per-sample gold & extras
        rk = out.setdefault("reward_kwargs", {})
        rk["answer"]     = gold_answers
        rk["gold"]       = gold_answers
        rk["prompts"]    = prompts_list
        rk["tasks"]      = tasks_list
        rk["board_str"]  = boards_list
        rk["N"]          = sizes_list
        rk["gold_moves"] = moves_list

        scores_nested = reward_router(
            completions=completions_txt,     # B×K
            tasks=tasks_list,                # avoid prompt-based inference
            answer=gold_answers,
            gold=gold_answers,
            board_str=boards_list,
            N=sizes_list,
            gold_moves=moves_list,
            proc=self.processing_class or self.tokenizer,
            script_args=self.args,
        )

        # Flatten if the trainer expects a flat list
        flat = [s for row in scores_nested for s in row] if isinstance(scores_nested[0], list) else scores_nested
        out["rewards"] = flat
        # --- make trainer actually use our rewards ---

        # Put rewards onto the same device/dtype as advantages if present
        if isinstance(out.get("advantages"), torch.Tensor):
            device = out["advantages"].device
            dtype  = out["advantages"].dtype
        else:
            device = self.accelerator.device
            dtype  = torch.float32

        adv = torch.as_tensor(flat, device=device, dtype=dtype)
        out["advantages"] = adv                     # <-- critical (trainer reads this)
        out["scores"]     = [float(x) for x in flat]  # pretty table / logging

        if self.accelerator.is_main_process:
            print("[DBG] rew sample:", out["scores"][:8])

        # adjust injected block's kwargs if needed
        if isinstance(inputs, list) and any("_buf_uid" in ex for ex in inputs):
            print("FIXING INJECTED!")
            inj_idx       = next(i for i, ex in enumerate(inputs) if "_buf_uid" in ex)
            gold_anchor   = gold_answers[inj_idx]
            prompt_anchor = prompts_list[inj_idx]
            gold_answers  = [gold_anchor]   * len(gold_answers)
            prompts_list  = [prompt_anchor] * len(prompts_list)

            def _set(dest: dict, keys, value):
                for k in keys:
                    dest[k] = value

            _set(out, ("gold_answers", "answers", "gold"),      gold_answers)
            _set(out, ("prompts_list", "prompts", "prompt"),    prompts_list)
            _set(out, ("tasks_list", "tasks"),                  tasks_list)

            if "reward_kwargs" in out and isinstance(out["reward_kwargs"], dict):
                rk = out["reward_kwargs"]
                rk["answer"]     = gold_answers
                rk["gold"]       = gold_answers
                rk["prompts"]    = prompts_list
                rk["tasks"]      = tasks_list
                rk["board_str"]  = boards_list
                rk["N"]          = sizes_list
                rk["gold_moves"] = moves_list

        MAX_K = 8
        completions_txt = [c[:MAX_K] for c in completions_txt]

        # --- Replay HIT detection (keep strict exact for cleanliness) ---
        winners_local = []
        for i in range(B):
            gold   = gold_answers[i]
            prompt = prompts_list[i]
            if not isinstance(gold, str) or not gold.strip() or prompt is None:
                continue
            preds_i  = completions_txt[i]
            task_i = str(tasks_list[i]).upper() if (tasks_list and i < len(tasks_list)) else _default_task(self.args)

            if "RUSH" in task_i:
                accs_i = rush_solution_exact(
                    prompts=[prompt]*len(preds_i),
                    completions=preds_i,
                    gold=[gold]*len(preds_i),
                )
            elif "MATH" in task_i:
                accs_i = pure_accuracy_reward_math(preds_i, [gold]*len(preds_i))
            else:  # default CROSSWORD
                accs_i = pure_accuracy_reward(preds_i, [gold]*len(preds_i))

            j_hit = next((jj for jj, a in enumerate(accs_i) if a == 1.0), None)

            if j_hit is None:
                continue

            print(f"[Replay][HIT] rank={rank} i={i} gold='{gold}'")
            winners_local.append(_to_env_schema({
                "prompt": prompt,
                "answer": gold,
                "reward": 1.0,
                "_last_success_pred": preds_i[j_hit],
            }))

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

        if self.model.training and is_r0 and (self.replay_buffer is not None):
            def _sig(ex):
                msgs = ex.get("prompt", [])
                if isinstance(msgs, str):
                    user = msgs.strip()
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

            try:
                self._maybe_credit_injected_uids(out)
            except Exception as e:
                print(f"[Replay][WARN] credit failed: {e}")

        return out

    # ---------------- inject helper ----------------
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
            # ↓↓↓ FIX: no hard "MATH"
            ex.setdefault("task", _default_task(self.args))
            ex.setdefault("mix_group_id", -1)
            ex.setdefault("mix_copy_idx", -1)
            tagged.append(ex)

        nb = len(generation_batch)
        gk = min(len(tagged), nb // 2)  # inject at most half the batch
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

    # ---------------- extract helper ----------------
    def _extract_completions_and_rewards(self, out: Dict[str, Any], batch_size: int):
        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)

        # rewards
        reward_keys = ["rewards", "reward", "scores", "advantages"]
        rewards = None
        for k in reward_keys:
            if k in out:
                rewards = out[k]
                break
        rewards = _to_float_list(rewards)

        # completions
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

        # size-normalise
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
    # Ensure GRPO returns per-sample rewards to us
    training_args.return_reward = True
    # Reasonable defaults; override in YAML if you prefer
    training_args.steps_per_generation = getattr(training_args, "steps_per_generation", 8)
    training_args.num_iterations       = getattr(training_args, "num_iterations", 5)

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"              # causal LM friendly
    model.config.pad_token_id = tokenizer.pad_token_id

    # Generation settings for GRPO rollouts
    model.generation_config.sort_inputs = False
    model.generation_config.return_dict_in_generate = True
    model.generation_config.output_scores = False
    model.generation_config.do_sample = True
    # Some model configs also use these flags:
    model.config.return_dict_in_generate = True
    model.config.output_scores = False

    # -------- Rewards (router, nested-safe, token decoding aware) --------
    from functools import partial
    reward_funcs = _wrap_reward_for_nested(
        partial(
            reward_router,
            script_args=script_args,  # for task inference
            proc=tokenizer,           # to decode token ids -> text
        )
    )

    # -------- Map dataset into conversation schema (and carry board/size/moves) --------
    dataset = dataset.map(
        lambda ex: _make_conversation(
            ex,
            script_args.dataset_prompt_column,
            script_args.dataset_solution_column,
            tokenizer,
            training_args.system_prompt,
            script_args=script_args,   # for task default heuristic
        )
    ).filter(lambda x: x is not None)

    # Drop raw "messages" if present (we now carry a 'prompt' string)
    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
        if "task" not in dataset[split].column_names:
            default_t = _default_task(script_args)
            dataset[split] = dataset[split].add_column("task", [default_t] * len(dataset[split]))
        if "is_replay" not in dataset[split].column_names:
            dataset[split] = dataset[split].add_column("is_replay", [False] * len(dataset[split]))

    # ------ Optional: subsample eval for speed ------
    if training_args.do_eval and script_args.dataset_test_split in dataset:
        full_eval = dataset[script_args.dataset_test_split]
        n_total   = len(full_eval)
        n_keep    = max(1, int(n_total * 0.1))
        eval_ds   = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))
    else:
        eval_ds = None

    # -------- Wrap train split for GRPO --------
    train_ds = ReplayMixDataset(
        base_ds=dataset[script_args.dataset_train_split],
        tok=tokenizer,
    )

    # -------- Simple replay buffer --------
    replay_buffer = ReplayBuffer(capacity=4000, C=1.0, debug_steps=3)
    print(
        "[RB] impl:", type(replay_buffer),
        "module:", replay_buffer.__class__.__module__,
        "capacity:", getattr(replay_buffer, "capacity", None),
        "has last_error:", hasattr(replay_buffer, "last_error"),
    )
    import inspect, open_r1.utils.replay_buffer as rb
    print("add_group source head:\n", inspect.getsource(rb.ReplayBuffer.add_group)[:400])

    # -------- Callbacks --------
    callback_objects = get_callbacks(
        training_args,
        model_args,
        replay_buffer=replay_buffer,
        tokenizer=tokenizer,          # pass tokenizer so callbacks can decode if needed
    )
    callback_objects.append(LossLoggingCallback(training_args.output_dir))

    # -------- Trainer --------
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
        constant_test_reward=4.0,          # only used if replay credit fallback is needed
        processing_class=tokenizer,
        T0=1.0, T1=0.3, anneal_steps=3000, hiT_period=5000,
        # disable EASY mixing by default for Rush training
        easy_pool=None,
        mix_schedule=None,
    )

    if trainer.accelerator.is_main_process and not getattr(trainer, "_printed_ng_once", False):
        print("[DBG] num_generations =", getattr(trainer.args, "num_generations", None))
        trainer._printed_ng_once = True

    # Avoid length-based re-sorting (we want per-sample fields to align)
    trainer.data_collator.sort_by_length = False

    # -------- Train --------
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
