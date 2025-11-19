from __future__ import annotations

import functools
import importlib
import os
from typing import Any, Dict, List, Optional, Tuple

torch = importlib.import_module("torch")

from .rewards_core import (
    pure_accuracy_reward,
    pure_accuracy_reward_math,
    rush_solution_shaped,
)


def _flatten_nested_for_rewards(prompts, completions):
    """Flatten B×K nested completions and align prompts."""
    batch_size = len(completions)
    sizes = [len(row) for row in completions]
    flat_completions: list[Any] = []
    flat_prompts: list[Any] = []

    for index, row in enumerate(completions):
        prompt_for_row = (
            prompts[index]
            if isinstance(prompts, list) and len(prompts) == batch_size
            else prompts
        )
        flat_completions.extend(row)
        flat_prompts.extend([prompt_for_row] * len(row))

    return flat_prompts, flat_completions, sizes, batch_size


def _wrap_reward_for_nested(reward_fn):
    @functools.wraps(reward_fn)
    def wrapped(*, prompts, completions, **kwargs):
        is_nested = (
            isinstance(completions, list)
            and completions
            and isinstance(completions[0], (list, tuple))
        )
        if not is_nested:
            return reward_fn(prompts=prompts, completions=completions, **kwargs)

        flat_prompts, flat_completions, sizes, batch_size = _flatten_nested_for_rewards(
            prompts,
            completions,
        )

        def _expand_value(value):
            if isinstance(value, list) and len(value) == batch_size:
                return [
                    value[index]
                    for index in range(batch_size)
                    for _ in range(sizes[index])
                ]
            return value

        nested_kwargs = {key: _expand_value(value) for key, value in kwargs.items()}

        flat_scores = reward_fn(
            prompts=flat_prompts,
            completions=flat_completions,
            **nested_kwargs,
        )

        # Repack to B×K
        nested_scores: list[list[float]] = []
        flat_index = 0
        for size in sizes:
            nested_scores.append(list(flat_scores[flat_index : flat_index + size]))
            flat_index += size
        return nested_scores

    return wrapped


def _task_from_script_args(script_args) -> Optional[str]:
    """
    Try to infer a single task from the YAML/CLI `reward_funcs` field.
    Returns "RUSH" | "CROSSWORD" | "MATH" | None.
    """
    if script_args is None:
        return None

    reward_funcs = getattr(script_args, "reward_funcs", None)
    if reward_funcs is None:
        return None

    # Normalize to a flat list of lowercase strings (handles str | list | tuple | dict)
    if isinstance(reward_funcs, str):
        names = [reward_funcs.lower()]
    elif isinstance(reward_funcs, dict):
        names = [str(key).lower() for key in reward_funcs.keys()]
    elif isinstance(reward_funcs, (list, tuple)):
        names = [str(value).lower() for value in reward_funcs]
    else:
        try:
            names = [str(reward_funcs).lower()]
        except (TypeError, ValueError):
            names = []

    joined = " ".join(names)
    if any(
        token in joined
        for token in ("rush_solution_exact", "rush_solution_shaped", "rush")
    ):
        return "RUSH"
    if any(
        token in joined for token in ("pure_accuracy_reward_math", "pure_accuracy_math")
    ) or ("math" in joined and "reward" in joined):
        return "MATH"
    if any(token in joined for token in ("pure_accuracy_reward", "cross", "crypt")):
        return "CROSSWORD"
    return None


def _default_task(
    args=None,
    *,
    system_prompt: Optional[str] = None,
    dataset_name_hint: Optional[str] = None,
    prompt_hint: Optional[str] = None,
) -> str:
    """
    Heuristically decide the task label when examples/batches don't carry one.
    Priority: explicit args.dataset_name → dataset_name_hint → system/prompt hints.
    """
    t_from_rf = _task_from_script_args(args)
    if t_from_rf:
        return t_from_rf

    name = ""
    if args is not None:
        for fld in ("dataset_name", "dataset", "dataset_path", "output_dir", "hub_model_id"):
            val = getattr(args, fld, None)
            if val:
                name = str(val)
                break
    if not name and dataset_name_hint:
        name = str(dataset_name_hint)

    blob = " ".join(
        s
        for s in (
            name,
            system_prompt or "",
            prompt_hint or "",
            os.environ.get("DEFAULT_TASK_HINT", ""),
        )
        if s
    ).lower()

    if any(k in blob for k in ("rush", "carpark", "car_parking", "parking")):
        return "RUSH"
    if any(k in blob for k in ("cross", "crypt")):
        return "CROSSWORD"
    if any(k in blob for k in ("math", "algebra", "calculus")):
        return "MATH"

    # Last resort: lean CROSSWORD to avoid misrouting crossword runs to math.
    return "CROSSWORD"


def _adapt_gold(seq_or_list, **kwargs):
    gold = (
        kwargs.get("answer")
        or kwargs.get("answers")
        or kwargs.get("gold")
        or kwargs.get("references")
        or kwargs.get("labels")
    )
    if isinstance(gold, str):
        num_examples = len(seq_or_list) if isinstance(seq_or_list, list) else 1
        return [gold] * num_examples
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

    if not isinstance(items, list):
        items = [items]

    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, torch.Tensor):
            ids = item.detach().cpu().tolist()
            if proc is not None and hasattr(proc, "decode"):
                out.append(
                    proc.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            else:
                out.append(str(ids))
            continue
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], int):
            if proc is not None and hasattr(proc, "decode"):
                out.append(
                    proc.decode(
                        list(item),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            else:
                out.append(" ".join(map(str, item)))
            continue
        out.append(str(item))
    return out


def _strip_answer_like_keys(kwargs_in: dict) -> dict:
    """Return a copy of kwargs with answer/gold-like keys removed."""
    keys_to_drop = ("answer", "answers", "gold", "labels", "references")
    return {key: value for key, value in kwargs_in.items() if key not in keys_to_drop}


def _normalize_prompt_list(prompts, num_samples: int):
    """Return a per-sample prompt list given `prompts` and desired length."""
    if isinstance(prompts, list) and len(prompts) == num_samples:
        return prompts
    return [prompts] * num_samples


def _extract_prompt_hint(prompts) -> Optional[str]:
    """Best-effort extraction of a user-facing text hint from prompts."""
    if not isinstance(prompts, list) or not prompts:
        return None
    first_prompt = prompts[0]
    if isinstance(first_prompt, str):
        return first_prompt
    if isinstance(first_prompt, list):
        for message in first_prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", "")
    return None


def _infer_task_label(script_args, tasks, prompts) -> str:
    """Infer an upper-cased task label from script args, tasks or prompts."""
    label = _task_from_script_args(script_args)
    if not label:
        if isinstance(tasks, list) and tasks:
            label = tasks[0]
        else:
            prompt_hint = _extract_prompt_hint(prompts)
            label = _default_task(script_args, prompt_hint=prompt_hint)
    return str(label).upper()


def _ith(values, index: int, num_samples: int):
    """Return the index-th value when list-like, or a broadcast scalar."""
    if isinstance(values, list) and len(values) == num_samples:
        return values[index]
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return values


def _rush_scores(
    prompt_list,
    completion_texts: list[str],
    gold_answers: list[str],
    kwargs: dict,
) -> list[float]:
    """Compute per-sample Rush Hour rewards with optional board/move hints."""
    gold_moves = kwargs.get("gold_moves")
    board_string = kwargs.get("board_str") or kwargs.get("board")
    board_size = kwargs.get("N") or kwargs.get("size")
    num_samples = len(completion_texts)

    scores: list[float] = []
    for index, (prompt_value, completion_text, gold_answer) in enumerate(
        zip(prompt_list, completion_texts, gold_answers),
    ):
        scores.append(
            rush_solution_shaped(
                prompts=prompt_value,
                completions=[completion_text],
                gold=[gold_answer],
                gold_moves=_ith(gold_moves, index, num_samples),
                board_str=_ith(board_string, index, num_samples),
                N=_ith(board_size, index, num_samples),
                w_exact=0.5,
                w_solve=0.2,
                w_prefix=0.2,
                w_phi=0.1,
            )[0]
        )
    return scores


def reward_router(*, prompts=None, completions=None, tasks=None, proc=None, **kwargs):
    """
    Task-aware reward router.
    For RUSH: per-sample rush_solution_shaped with optional board/moves hints.
    For MATH: pure_accuracy_reward_math.
    Default: crossword pure_accuracy_reward (with shaping).
    """
    completions_flat, _nested_factor = _flatten_nested(completions)
    completion_texts = _to_text_list(completions_flat, proc=proc)
    gold_answers = _adapt_gold(completion_texts, **kwargs)

    kwargs_clean = _strip_answer_like_keys(kwargs)

    num_samples = len(completion_texts)
    prompt_list = _normalize_prompt_list(prompts, num_samples)

    script_args = kwargs.get("script_args", None)
    task_label = _infer_task_label(script_args, tasks, prompts)

    if "RUSH" in task_label:
        return _rush_scores(prompt_list, completion_texts, gold_answers, kwargs)

    if "MATH" in task_label:
        return pure_accuracy_reward_math(completion_texts, gold_answers, **kwargs_clean)

    return pure_accuracy_reward(completion_texts, gold_answers, **kwargs_clean)


__all__ = ["reward_router", "_wrap_reward_for_nested"]

