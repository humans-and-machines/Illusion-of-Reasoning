"""Dataset helpers for GRPO training (chat formatting and easy-pool loading)."""

from __future__ import annotations

import importlib
import logging
import os
import re
from typing import Any, Dict, Optional


datasets = importlib.import_module("datasets")

logger = logging.getLogger(__name__)


def _strip_answer_blocks(text: str) -> str:
    """Strip any accidental <think>/<answer> blocks from user text."""
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"(?is)<answer>.*?</answer>\s*", "", text)
    return text


def _messages_from_dict_prompt(raw_prompt: dict) -> tuple[list[dict[str, str]], int]:
    """Normalize a dict `{'role': ..., 'content': ...}` prompt into messages."""
    roles = raw_prompt.get("role")
    contents = raw_prompt.get("content")

    messages: list[dict[str, str]] = []
    dropped_assistants = 0

    if isinstance(roles, (list, tuple)) and isinstance(contents, (list, tuple)):
        # dict-of-arrays
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

    return messages, dropped_assistants


def _messages_from_list_prompt(raw_prompt: list) -> tuple[list[dict[str, str]], int]:
    """Normalize a list of {'role','content'} dicts into messages."""
    messages: list[dict[str, str]] = []
    dropped_assistants = 0

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

    return messages, dropped_assistants


def _build_base_messages(
    example: dict,
    prompt_column: str,
    system_prompt: str | None,
) -> tuple[list[dict[str, str]], int]:
    """Construct initial message list from the raw dataset prompt."""
    raw_prompt = example.get(prompt_column, None)
    messages: list[dict[str, str]] = []
    dropped_assistants = 0

    if isinstance(raw_prompt, dict) and ("role" in raw_prompt and "content" in raw_prompt):
        messages, dropped_assistants = _messages_from_dict_prompt(raw_prompt)
    elif isinstance(raw_prompt, list):
        messages, dropped_assistants = _messages_from_list_prompt(raw_prompt)
    elif isinstance(raw_prompt, str):
        # Plain string → treat as single user message (sanitized)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _strip_answer_blocks(raw_prompt.strip())})
    else:
        # Fallback to 'board' field if present (legacy)
        board = str(example.get("board", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": board})

    return messages, dropped_assistants


def _ensure_user_and_system_messages(
    messages: list[dict[str, str]],
    example: dict,
    system_prompt: str | None,
) -> None:
    """Guarantee at least one user message and inject system prompt when required."""
    has_user = any(message.get("role") == "user" for message in messages)
    if not has_user:
        board = str(example.get("board", "")).strip()
        if not board:
            raise ValueError("No user content after filtering and no 'board' field present")
        if system_prompt and not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": board})

    if system_prompt and not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Final safety scrub: ONLY sanitize USER text (leave system untouched to preserve tag template)
    for message in messages:
        if message.get("role") == "user":
            message["content"] = _strip_answer_blocks(message["content"])


def _augment_with_legacy_metadata(messages: list[dict[str, str]], example: dict) -> None:
    """Optionally append board size / moves metadata to the first user message."""
    size_val = example.get("size", None)
    moves_val = example.get("moves", None)
    if size_val is None and moves_val is None:
        return

    size_str = f"Board size: {size_val}x{size_val}" if size_val is not None else ""
    moves_str = f"Minimal moves to solve: {moves_val}" if moves_val is not None else ""
    augment = "\n".join(line for line in (size_str, moves_str) if line)

    for message in messages:
        if message.get("role") == "user":
            message["content"] = f"{message['content']}\n{augment}"
            break


def _extract_solution(example: dict, solution_column: str) -> str:
    """Extract the gold solution/answer from the configured column."""
    raw_sol = example.get(solution_column, None)
    if raw_sol is None:
        raise ValueError(f"Dataset row missing '{solution_column}'")

    if isinstance(raw_sol, (list, tuple)):
        return ",".join(str(token).strip() for token in raw_sol if str(token).strip())

    raw_sol_str = str(raw_sol)
    answer_match = re.search(
        r"(?si)<answer>\s*([^<\n]+?)\s*</answer>",
        raw_sol_str,
    )
    return (answer_match.group(1) if answer_match else raw_sol_str).strip()


def _estimate_prompt_tokens(messages: list[dict[str, str]], tokenizer) -> int:
    """Estimate prompt length in tokens using the chat template when available."""
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return int(templated.input_ids.shape[-1])
    except (TypeError, AttributeError, ValueError):
        flat_prompt = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return int(tokenizer(flat_prompt, return_tensors="pt").input_ids.shape[-1])


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

    - Do NOT strip <think>/<answer> from the system prompt (only sanitize USER text).
    - Ensure system_prompt is injected if missing.
    - Token length guard uses tokenizer.apply_chat_template when available.

    Returns a dict or None (if over-length).
    """
    messages, dropped_assistants = _build_base_messages(
        example,
        prompt_column,
        system_prompt,
    )

    if dropped_assistants:
        logger.debug(
            "Dropped %d assistant message(s) from '%s'",
            dropped_assistants,
            prompt_column,
        )

    _ensure_user_and_system_messages(messages, example, system_prompt)
    _augment_with_legacy_metadata(messages, example)

    sol_core = _extract_solution(example, solution_column)
    total_tokens = _estimate_prompt_tokens(messages, tokenizer)
    max_prompt_tokens = kwargs.get("max_prompt_tokens", 2048)

    if total_tokens >= max_prompt_tokens:
        logger.warning("Skipping over-length prompt (%s tokens)", total_tokens)
        return None

    return {
        "prompt": messages,  # chat-format; call apply_chat_template downstream
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
    easy_name = (
        getattr(script_args, "easy_dataset_name", None)
        or os.environ.get("EASY_DATASET_NAME")
        or os.environ.get("EASY_DATASET")
    )
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
    train_key = (
        script_args.dataset_train_split
        if script_args.dataset_train_split in easy
        else ("train" if "train" in easy else list(easy.keys())[0])
    )

    # convert to a python list of dicts (fast random access)
    pool = list(easy[train_key])
    print(f"[EASY] loaded {len(pool)} items from '{easy_name}' ({train_key})")
    return pool


__all__ = ["_make_conversation", "_load_easy_pool"]
