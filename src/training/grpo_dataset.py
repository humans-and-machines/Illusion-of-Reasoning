from __future__ import annotations

import importlib
import logging
import os
import re
from typing import Any, Dict, Optional

datasets = importlib.import_module("datasets")

logger = logging.getLogger(__name__)


def _make_conversation(
    example: dict,
    prompt_column: str,
    solution_column: str,
    tokenizer,
    system_prompt: str | None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Build a chat-style sample for RL/GRPO.

    Fixes:
    - Do NOT strip <think>/<answer> from the system prompt (only sanitize USER text).
    - Ensure system_prompt is injected if missing.
    - Token length guard uses tokenizer.apply_chat_template when available.

    Returns a dict or None (if over-length).
    """

    def _strip_answer_blocks(text: str) -> str:
        # Remove any embedded solutions/thoughts if they accidentally show up in user/system text
        # NOTE: We will only apply this to USER messages, not to the system prompt.
        text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
        text = re.sub(r"(?is)<answer>.*?</answer>\s*", "", text)
        return text

    # ---- unpack messages ----
    raw_prompt = example.get(prompt_column, None)
    messages: list[dict[str, str]] = []
    dropped_assistants = 0

    # Case A: dict with 'role' and 'content' (can be lists or scalars)
    if isinstance(raw_prompt, dict) and ("role" in raw_prompt and "content" in raw_prompt):
        roles = raw_prompt.get("role")
        contents = raw_prompt.get("content")

        # dict-of-arrays
        if isinstance(roles, (list, tuple)) and isinstance(contents, (list, tuple)):
            for role_raw, content_raw in zip(roles, contents):
                role = str(role_raw).strip().lower()
                content = str(content_raw)
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
            content = str(contents)
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
        for message in raw_prompt:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", ""))
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
    for message in messages:
        if message.get("role") == "user":
            message["content"] = _strip_answer_blocks(message["content"])

    # ---- optional augmentation (legacy fields) ----
    size_val = example.get("size", None)
    moves_val = example.get("moves", None)
    if size_val is not None or moves_val is not None:
        size_str = f"Board size: {size_val}x{size_val}" if size_val is not None else ""
        moves_str = f"Minimal moves to solve: {moves_val}" if moves_val is not None else ""
        augment = "\n".join(line for line in (size_str, moves_str) if line)
        for message in messages:
            if message.get("role") == "user":
                message["content"] = f"{message['content']}\n{augment}"
                break

    # ---- extract solution (from solution_column ONLY) ----
    raw_sol = example.get(solution_column, None)
    if raw_sol is None:
        raise ValueError(f"Dataset row missing '{solution_column}'")

    if isinstance(raw_sol, (list, tuple)):
        sol_core = ",".join(
            str(token).strip() for token in raw_sol if str(token).strip()
        )
    else:
        raw_sol_str = str(raw_sol)
        answer_match = re.search(
            r"(?si)<answer>\s*([^<\n]+?)\s*</answer>",
            raw_sol_str,
        )
        sol_core = (answer_match.group(1) if answer_match else raw_sol_str).strip()

    # ---- length guard (prefer templated token count) ----
    total_tokens = 0
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        total_tokens = int(templated.input_ids.shape[-1])
    except (TypeError, AttributeError, ValueError):
        flat_prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )
        total_tokens = int(
            tokenizer(flat_prompt, return_tensors="pt").input_ids.shape[-1]
        )

    max_prompt_tokens = kwargs.get("max_prompt_tokens", 2048)
    if total_tokens >= max_prompt_tokens:
        logger.warning("Skipping over-length prompt (%s tokens)", total_tokens)
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
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[EASY] failed to load '{easy_name}': {error}")
        return None

    prompt_column_name = getattr(script_args, "dataset_prompt_column", "problem")
    solution_column_name = getattr(script_args, "dataset_solution_column", "answer")

    def _fmt(ex):
        row = _make_conversation(
            ex,
            prompt_column_name,
            solution_column_name,
            tokenizer,
            training_args.system_prompt,
            prefix="",
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


__all__ = ["_make_conversation", "_load_easy_pool"]

