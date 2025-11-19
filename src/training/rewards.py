# coding=utf-8
"""
Thin wrapper around rewards_core to keep legacy imports working.

We only use the crossword/math/rush rewards exposed in rewards_core; the older
math_verify/code rewards are intentionally omitted.
"""

from typing import Callable, List

from .rewards_core import (
    crossword_accuracy_reward,
    pure_accuracy_reward,
    pure_accuracy_reward_math,
    rush_solution_exact,
    rush_solution_shaped,
)


def get_reward_funcs(
    script_args,
    ref_model=None,  # kept for backwards-compatible signature
    tokenizer=None,  # kept for backwards-compatible signature
) -> List[Callable]:
    """
    Build the list of reward functions requested via script_args.reward_funcs.

    The ref_model and tokenizer parameters are accepted for API compatibility
    with older callers but are not used in the current implementation.
    """
    # Mark legacy-only parameters as used to satisfy linters.
    _ = (ref_model, tokenizer)

    registry = {
        "crossword_accuracy": crossword_accuracy_reward,
        "pure_accuracy": pure_accuracy_reward,
        "pure_accuracy_math": pure_accuracy_reward_math,
        "rush_solution_exact": rush_solution_exact,
        "rush_solution_shaped": rush_solution_shaped,
    }

    names = getattr(script_args, "reward_funcs", None) or []
    if isinstance(names, str):
        names = [names]

    funcs = []
    for name in names:
        key = str(name)
        if key not in registry:
            raise KeyError(f"Unknown reward function '{key}'. Allowed: {sorted(registry)}")
        funcs.append(registry[key])
    return funcs


__all__ = ["get_reward_funcs"]
