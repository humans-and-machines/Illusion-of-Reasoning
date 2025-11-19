#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core aggregation helpers for crossword inference results.

This module holds the bulk of the data structures and aggregation logic used by
``summarize_inference`` so that the CLI wrapper can stay small and focused.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def mean_safe(values: List[Optional[float]]) -> Optional[float]:
    """Return the mean of non-None numeric values, or None if empty."""
    numeric_values: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def pct(numerator: int, denominator: int) -> str:
    """Format a numerator/denominator pair as a percentage string."""
    if denominator == 0:
        return "-"
    return f"{100.0 * numerator / denominator:5.1f}%"


def fmt_float(value: Optional[float]) -> str:
    """Format an optional float with fixed width, or '-' for missing."""
    return "-" if value is None else f"{value:6.3f}"


def nat_step_from_path(path: str) -> Optional[int]:
    """Extract the integer step#### from a results file path, if present."""
    match = re.search(r"step(\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def scan_files(root: str, split: Optional[str]) -> List[str]:
    """Return sorted JSONL result paths under a root, optionally filtered by split substring."""
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                continue
            if split and split not in filename:
                continue
            matches.append(os.path.join(dirpath, filename))
    matches.sort(key=lambda path: (nat_step_from_path(path) or 0, path))
    return matches


DEFAULT_PROMPT_KEYS = [
    "prompt",
    "prompt_text",
    "input_prompt",
    "input",
    "question_prompt",
    "fmt_prompt",
    "aug_prompt",
]


def _get_nested(mapping: Dict[str, Any], dotpath: str) -> Optional[Any]:
    """Return nested value from a dict via a dotted path like 'a.b.c', or None if missing."""
    cur: Any = mapping
    for part in dotpath.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def extract_prompt(rec: Dict[str, Any], preferred_key: str, strict: bool) -> Optional[str]:
    """Extract a prompt string from a record using a preferred key and fallbacks."""
    value = _get_nested(rec, preferred_key) if preferred_key else None
    if value is not None or strict:
        return str(value) if value is not None else None
    for key in DEFAULT_PROMPT_KEYS:
        value = _get_nested(rec, key)
        if value is not None:
            return str(value)
    return None


def extract_group(rec: Dict[str, Any], key: str) -> str:
    """
    Grouping key for per-group capping.
    For crosswords, the default 'problem' is the clue text (good grouping).
    """
    if key == "problem":
        return str(rec.get("problem", ""))
    val = _get_nested(rec, key)
    if val is None and key == "prompt":
        return str(rec.get("problem", ""))
    return "" if val is None else str(val)


def should_drop_group(
    record: Dict[str, Any],
    drop_groups: set[str],
    filter_scope: str,
) -> bool:
    """Return True if a record belongs to a group that should be dropped."""
    if not drop_groups:
        return False
    if filter_scope == "per_problem":
        group_name = str(record.get("problem", ""))
    else:
        group_name = "__GLOBAL__"
    return group_name in drop_groups


def _substr(hay: Optional[str], needle: Optional[str]) -> bool:
    """Return True if needle is a substring of hay, handling Nones."""
    if not hay or not needle:
        return False
    return needle in hay


def _exact(left: Optional[str], right: Optional[str]) -> bool:
    """Return True iff both strings are non-None and exactly equal."""
    if left is None or right is None:
        return False
    return left == right


def maybe_recompute_correctness(
    pass_data: Dict[str, Any],
    gold_canon: Optional[str],
    mode: str,
) -> Optional[bool]:
    """
    Optionally recompute correctness based on canonical fields.

    The CLI passes one of:
      - ``\"none\"`` / ``\"original\"`` → keep the recorded correctness
      - ``\"substring\"`` → treat correct if gold is a substring of pred
      - ``\"exact\"`` → treat correct only if strings match exactly

    Returns:
      - ``True`` or ``False`` if an override should be used
      - ``None`` to keep the original correctness flag.
    """
    # Backwards compatibility: both "none" and "original" mean "no override".
    if mode in ("none", "original"):
        return None
    if gold_canon is None:
        return None

    pred_canon = pass_data.get("pred_answer_canon")
    if not isinstance(pred_canon, str):
        return None

    gold_canon_str = str(gold_canon)
    if mode == "substring":
        return _substr(pred_canon, gold_canon_str)
    if mode == "exact":
        return _exact(pred_canon, gold_canon_str)

    # Unrecognized modes fall back to the recorded correctness.
    return None


@dataclass
class PassAggregate:
    """Per-pass counters and distributions for a single step."""

    _state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._state.setdefault(
            "correct_by_problem",
            {},
        )
        self._state.setdefault(
            "counts",
            {
                "n_samples": 0,
                "sample_correct": 0,
                "sample_improved": 0,
                "tag_ok": 0,
                "reconsider_numer": 0,
            },
        )
        self._state.setdefault(
            "entropies",
            {
                "all": [],
                "think": [],
                "answer": [],
            },
        )
        self._state.setdefault(
            "tokens",
            {
                "think": [],
                "answer": [],
            },
        )
        self._state.setdefault("soft_values", [])
        self._state.setdefault("soft_by_problem", defaultdict(list))
        self._state.setdefault(
            "stop_counts",
            {
                "think": Counter(),
                "answer": Counter(),
            },
        )

    @property
    def correct_by_problem(self) -> Dict[str, bool]:
        """Mapping from problem text to any-correct flag for this pass."""
        return self._state["correct_by_problem"]

    @property
    def soft_values(self) -> List[Optional[float]]:
        """Flat list of per-sample soft rewards."""
        return self._state["soft_values"]

    @property
    def soft_by_problem(self) -> Dict[str, List[Optional[float]]]:
        """Mapping from problem text to list of soft rewards."""
        return self._state["soft_by_problem"]

    @property
    def stop_counts(self) -> Dict[str, Counter]:
        """Stop-reason counters keyed by phase ('think' / 'answer')."""
        return self._state["stop_counts"]

    @property
    def _counts(self) -> Dict[str, int]:
        return self._state["counts"]

    @property
    def _entropies(self) -> Dict[str, List[Optional[float]]]:
        return self._state["entropies"]

    @property
    def _tokens(self) -> Dict[str, List[Optional[int]]]:
        return self._state["tokens"]

    @property
    def n_samples(self) -> int:
        """Total number of samples accumulated for this pass."""
        return self._counts["n_samples"]

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        self._counts["n_samples"] = value

    @property
    def sample_correct(self) -> int:
        """Number of correctly predicted samples for this pass."""
        return self._counts["sample_correct"]

    @sample_correct.setter
    def sample_correct(self, value: int) -> None:
        self._counts["sample_correct"] = value

    @property
    def sample_improved(self) -> int:
        """Number of samples marked as improved over pass-1."""
        return self._counts["sample_improved"]

    @sample_improved.setter
    def sample_improved(self, value: int) -> None:
        self._counts["sample_improved"] = value

    @property
    def tag_ok(self) -> int:
        """Number of samples with valid tag structure."""
        return self._counts["tag_ok"]

    @tag_ok.setter
    def tag_ok(self, value: int) -> None:
        self._counts["tag_ok"] = value

    @property
    def reconsider_numer(self) -> int:
        """Number of samples containing reconsideration markers."""
        return self._counts["reconsider_numer"]

    @reconsider_numer.setter
    def reconsider_numer(self, value: int) -> None:
        self._counts["reconsider_numer"] = value

    @property
    def entropy_all(self) -> List[Optional[float]]:
        """Entropy values over full pass tokens."""
        return self._entropies["all"]

    @property
    def entropy_think(self) -> List[Optional[float]]:
        """Entropy values over think-block tokens."""
        return self._entropies["think"]

    @property
    def entropy_answer(self) -> List[Optional[float]]:
        """Entropy values over answer-block tokens."""
        return self._entropies["answer"]

    @property
    def tokens_think(self) -> List[Optional[int]]:
        """Token counts for think blocks."""
        return self._tokens["think"]

    @property
    def tokens_answer(self) -> List[Optional[int]]:
        """Token counts for answer blocks."""
        return self._tokens["answer"]

    @property
    def stop_think(self) -> Counter:
        """Stop-reason counts for think blocks."""
        return self.stop_counts["think"]

    @property
    def stop_answer(self) -> Counter:
        """Stop-reason counts for answer blocks."""
        return self.stop_counts["answer"]


@dataclass
class StepAgg:
    """Aggregate per-step metrics across all examples and samples."""

    step: int
    pass1: PassAggregate = field(default_factory=PassAggregate)
    pass2: PassAggregate = field(default_factory=PassAggregate)
    examples: set[str] = field(default_factory=set)
    ex_improved_p2: int = 0

    def add_record(self, record: Dict[str, Any], recompute_mode: str) -> None:
        """Add a single JSONL record to the aggregate metrics."""
        problem = record.get("problem", "")
        self.examples.add(problem)
        gold_canon = record.get("gold_answer_canon")

        pass1 = record.get("pass1") or {}
        if pass1:
            self._add_pass1(pass1, problem, gold_canon, recompute_mode)

        pass2 = record.get("pass2") or {}
        if pass2:
            self._add_pass2(pass2, problem, gold_canon, recompute_mode)

    def _add_pass1(
        self,
        pass_data: Dict[str, Any],
        problem: str,
        gold_canon: Optional[str],
        recompute_mode: str,
    ) -> None:
        """Update aggregates for pass-1 metrics."""
        agg = self.pass1
        agg.n_samples += 1
        correct_original = bool(pass_data.get("is_correct_pred"))
        correct_override = maybe_recompute_correctness(pass_data, gold_canon, recompute_mode)
        is_correct = correct_original if correct_override is None else bool(correct_override)
        agg.sample_correct += int(is_correct)
        previous_correct = agg.correct_by_problem.get(problem, False)
        agg.correct_by_problem[problem] = previous_correct or is_correct

        agg.entropy_all.append(pass_data.get("entropy"))
        agg.entropy_think.append(pass_data.get("entropy_think"))
        agg.entropy_answer.append(pass_data.get("entropy_answer"))
        agg.tokens_think.append(pass_data.get("tokens_think"))
        agg.tokens_answer.append(pass_data.get("tokens_answer"))
        agg.stop_think[pass_data.get("stop_reason_think") or "unknown"] += 1
        agg.stop_answer[pass_data.get("stop_reason_answer") or "unknown"] += 1
        if pass_data.get("valid_tag_structure"):
            agg.tag_ok += 1
        if pass_data.get("has_reconsider_cue"):
            agg.reconsider_numer += 1

        soft_reward = pass_data.get("soft_reward")
        if soft_reward is None and ("soft reward" in pass_data):
            soft_reward = pass_data.get("soft reward")
        agg.soft_values.append(soft_reward)
        agg.soft_by_problem[problem].append(soft_reward)

    def _add_pass2(
        self,
        pass_data: Dict[str, Any],
        problem: str,
        gold_canon: Optional[str],
        recompute_mode: str,
    ) -> None:
        """Update aggregates for pass-2 metrics."""
        agg = self.pass2
        agg.n_samples += 1
        correct_original = bool(pass_data.get("is_correct_pred"))
        correct_override = maybe_recompute_correctness(pass_data, gold_canon, recompute_mode)
        is_correct = correct_original if correct_override is None else bool(correct_override)
        agg.sample_correct += int(is_correct)
        previous_correct = agg.correct_by_problem.get(problem, False)
        agg.correct_by_problem[problem] = previous_correct or is_correct

        agg.entropy_all.append(pass_data.get("entropy"))
        agg.entropy_think.append(pass_data.get("entropy_think"))
        agg.entropy_answer.append(pass_data.get("entropy_answer"))
        agg.tokens_think.append(pass_data.get("tokens_think"))
        agg.tokens_answer.append(pass_data.get("tokens_answer"))
        agg.stop_think[pass_data.get("stop_reason_think") or "unknown"] += 1
        agg.stop_answer[pass_data.get("stop_reason_answer") or "unknown"] += 1
        if pass_data.get("valid_tag_structure"):
            agg.tag_ok += 1
        if pass_data.get("has_reconsider_cue"):
            agg.reconsider_numer += 1
        if pass_data.get("improved_over_pass1"):
            agg.sample_improved += 1

        soft_reward = pass_data.get("soft_reward")
        if soft_reward is None and ("soft reward" in pass_data):
            soft_reward = pass_data.get("soft reward")
        agg.soft_values.append(soft_reward)
        agg.soft_by_problem[problem].append(soft_reward)

    def finalize(self) -> None:
        """Compute per-example improvement counts after all records are added."""
        for prob in self.examples:
            p1_ok = self.pass1.correct_by_problem.get(prob, False)
            p2_ok = self.pass2.correct_by_problem.get(prob, False)
            if p2_ok and not p1_ok:
                self.ex_improved_p2 += 1

    def example_level_soft_mean(
        self,
        by_problem: Dict[str, List[Optional[float]]],
    ) -> Optional[float]:
        """Reduce sample-level soft rewards to per-example means via max-over-samples."""
        per_example = []
        for prob in self.examples:
            vals = [float(v) for v in by_problem.get(prob, []) if v is not None]
            if vals:
                per_example.append(max(vals))
        return mean_safe(per_example)

    def row_text(self) -> str:
        """Return a single formatted summary row for this step."""
        metrics = self._row_metrics()
        return (
            f"{self.step:6d} "
            f"{self.pass1.n_samples:6d} {metrics['acc1_sample']:>6} {metrics['acc1_example']:>6} "
            f"{metrics['ent1']:>8} {metrics['think1']:>7} {metrics['answer1']:>7} "
            f"{metrics['soft1_sample']:>7} {metrics['soft1_example']:>7} "
            f"{self.pass2.n_samples:6d} {metrics['acc2_sample']:>6} {metrics['acc2_example']:>6} "
            f"{metrics['ent2']:>8} {metrics['think2']:>7} {metrics['answer2']:>7} "
            f"{metrics['soft2_sample']:>7} {metrics['soft2_example']:>7} "
            f"{metrics['improvement_sample']:>6} {metrics['improvement_example']:>6} "
            f"{metrics['tag1']:>6} {metrics['tag2']:>6}"
        )

    def footer_text(self) -> str:
        """Return a multi-line human-readable summary for this step."""
        lines: List[str] = [f"   • examples: {len(self.examples)}"]
        self._append_stop_reason_lines(lines)
        self._append_reconsider_lines(lines)
        self._append_token_stats_lines(lines)
        self._append_soft_reward_lines(lines)
        return "\n".join(lines)

    def _row_metrics(self) -> Dict[str, str]:
        """Compute formatted metric strings used in row_text()."""
        num_examples = len(self.examples) if self.examples else 0
        metrics: Dict[str, str] = {}

        metrics["acc1_sample"] = pct(self.pass1.sample_correct, self.pass1.n_samples)
        metrics["acc2_sample"] = pct(self.pass2.sample_correct, self.pass2.n_samples)

        if num_examples:
            total_correct = sum(1 for value in self.pass1.correct_by_problem.values() if value)
            metrics["acc1_example"] = pct(total_correct, num_examples)
            total_correct = sum(1 for value in self.pass2.correct_by_problem.values() if value)
            metrics["acc2_example"] = pct(total_correct, num_examples)
            metrics["improvement_example"] = pct(self.ex_improved_p2, num_examples)
        else:
            metrics["acc1_example"] = "-"
            metrics["acc2_example"] = "-"
            metrics["improvement_example"] = "-"

        metrics["ent1"] = fmt_float(mean_safe(self.pass1.entropy_all))
        metrics["ent2"] = fmt_float(mean_safe(self.pass2.entropy_all))
        metrics["think1"] = fmt_float(mean_safe(self.pass1.entropy_think))
        metrics["answer1"] = fmt_float(mean_safe(self.pass1.entropy_answer))
        metrics["think2"] = fmt_float(mean_safe(self.pass2.entropy_think))
        metrics["answer2"] = fmt_float(mean_safe(self.pass2.entropy_answer))

        metrics["soft1_sample"] = fmt_float(mean_safe(self.pass1.soft_values))
        metrics["soft2_sample"] = fmt_float(mean_safe(self.pass2.soft_values))
        metrics["soft1_example"] = fmt_float(
            self.example_level_soft_mean(self.pass1.soft_by_problem),
        )
        metrics["soft2_example"] = fmt_float(
            self.example_level_soft_mean(self.pass2.soft_by_problem),
        )

        metrics["improvement_sample"] = (
            pct(self.pass2.sample_improved, self.pass2.n_samples) if self.pass2.n_samples else "-"
        )
        metrics["tag1"] = (
            pct(self.pass1.tag_ok, self.pass1.n_samples) if self.pass1.n_samples else "-"
        )
        metrics["tag2"] = (
            pct(self.pass2.tag_ok, self.pass2.n_samples) if self.pass2.n_samples else "-"
        )
        return metrics

    @staticmethod
    def _format_stop_counter(counter: Counter, denominator: int) -> str:
        """Format a stop-reason counter as 'name=XX.X%'."""
        if denominator == 0:
            return "—"
        keys = ["stop_token", "eos", "max_new_tokens", "other", "unknown"]
        return ", ".join(f"{key}={pct(counter.get(key, 0), denominator)}" for key in keys)

    def _append_stop_reason_lines(self, lines: List[str]) -> None:
        """Append per-pass stop reason breakdown lines."""
        if self.pass1.n_samples:
            lines.append(
                "   • p1 think stops: "
                f"{self._format_stop_counter(self.pass1.stop_think, self.pass1.n_samples)}",
            )
            lines.append(
                "   • p1 answer stops: "
                f"{self._format_stop_counter(self.pass1.stop_answer, self.pass1.n_samples)}",
            )
        if self.pass2.n_samples:
            lines.append(
                "   • p2 think stops: "
                f"{self._format_stop_counter(self.pass2.stop_think, self.pass2.n_samples)}",
            )
            lines.append(
                "   • p2 answer stops: "
                f"{self._format_stop_counter(self.pass2.stop_answer, self.pass2.n_samples)}",
            )

    def _append_reconsider_lines(self, lines: List[str]) -> None:
        """Append reconsideration marker rate lines."""
        if self.pass1.n_samples:
            lines.append(
                "   • p1 reconsider-markers rate: "
                f"{pct(self.pass1.reconsider_numer, self.pass1.n_samples)}",
            )
        if self.pass2.n_samples:
            lines.append(
                "   • p2 reconsider-markers rate: "
                f"{pct(self.pass2.reconsider_numer, self.pass2.n_samples)}",
            )

    def _append_token_stats_lines(self, lines: List[str]) -> None:
        """Append mean token count lines if token data is present."""
        if not any(
            value is not None for value in self.pass1.tokens_think + self.pass2.tokens_think
        ):
            return

        mean_think1 = mean_safe(
            [
                value
                for value in self.pass1.tokens_think
                if isinstance(value, (int, float))
            ],
        )
        mean_answer1 = mean_safe(
            [
                value
                for value in self.pass1.tokens_answer
                if isinstance(value, (int, float))
            ],
        )
        mean_think2 = mean_safe(
            [
                value
                for value in self.pass2.tokens_think
                if isinstance(value, (int, float))
            ],
        )
        mean_answer2 = mean_safe(
            [
                value
                for value in self.pass2.tokens_answer
                if isinstance(value, (int, float))
            ],
        )

        mean_think1_str = "-" if mean_think1 is None else f"{mean_think1:.1f}"
        mean_answer1_str = "-" if mean_answer1 is None else f"{mean_answer1:.1f}"
        mean_think2_str = "-" if mean_think2 is None else f"{mean_think2:.1f}"
        mean_answer2_str = "-" if mean_answer2 is None else f"{mean_answer2:.1f}"

        lines.append(
            "   • mean tokens — "
            f"p1: think={mean_think1_str} answer={mean_answer1_str}; "
            f"p2: think={mean_think2_str} answer={mean_answer2_str}",
        )

    def _append_soft_reward_lines(self, lines: List[str]) -> None:
        """Append mean soft-reward lines for pass-1 and pass-2."""
        mean_soft1_samples = mean_safe(self.pass1.soft_values)
        mean_soft1_examples = self.example_level_soft_mean(self.pass1.soft_by_problem)
        if mean_soft1_samples is not None or mean_soft1_examples is not None:
            mean_soft1_samples_str = (
                "-" if mean_soft1_samples is None else f"{mean_soft1_samples:.3f}"
            )
            mean_soft1_examples_str = (
                "-" if mean_soft1_examples is None else f"{mean_soft1_examples:.3f}"
            )
            lines.append(
                "   • mean soft (p1): "
                f"samples={mean_soft1_samples_str}; examples[max]={mean_soft1_examples_str}",
            )

        mean_soft2_samples = mean_safe(self.pass2.soft_values)
        mean_soft2_examples = self.example_level_soft_mean(self.pass2.soft_by_problem)
        if mean_soft2_samples is not None or mean_soft2_examples is not None:
            mean_soft2_samples_str = (
                "-" if mean_soft2_samples is None else f"{mean_soft2_samples:.3f}"
            )
            mean_soft2_examples_str = (
                "-" if mean_soft2_examples is None else f"{mean_soft2_examples:.3f}"
            )
            lines.append(
                "   • mean soft (p2): "
                f"samples={mean_soft2_samples_str}; examples[max]={mean_soft2_examples_str}",
            )


def group_name_for_record(
    record: Dict[str, Any],
    filter_scope: str,
) -> str:
    """Return the group key (problem or global) for a record."""
    if filter_scope == "per_problem":
        return str(record.get("problem", ""))
    return "__GLOBAL__"


def accumulate_prompt_variants(
    record: Dict[str, Any],
    args,
    variants_by_group: Dict[str, set],
    counts: Dict[str, int],
) -> None:
    """Update prompt-variant tracking structures from a single JSON record."""
    counts["seen"] += 1
    prompt = extract_prompt(record, args.prompt_key, args.strict_prompt_key)
    if prompt is None:
        return

    counts["with_prompt"] += 1
    if args.prompt_family_regex:
        prompt = re.sub(str(args.prompt_family_regex), "", prompt)

    group_name = group_name_for_record(record, args.filter_scope)
    variants_by_group[group_name].add(prompt)


def build_step_csv_row(aggregator: StepAgg) -> List[Any]:
    """Build a single CSV row of numeric metrics for one step."""
    num_examples = len(aggregator.examples) if aggregator.examples else 0

    acc1_sample = (
        100.0 * aggregator.pass1.sample_correct / aggregator.pass1.n_samples
        if aggregator.pass1.n_samples
        else None
    )
    acc2_sample = (
        100.0 * aggregator.pass2.sample_correct / aggregator.pass2.n_samples
        if aggregator.pass2.n_samples
        else None
    )
    acc1_example = (
        100.0
        * sum(1 for value in aggregator.pass1.correct_by_problem.values() if value)
        / num_examples
        if num_examples
        else None
    )
    acc2_example = (
        100.0
        * sum(1 for value in aggregator.pass2.correct_by_problem.values() if value)
        / num_examples
        if num_examples
        else None
    )
    imp_sample = (
        100.0 * aggregator.pass2.sample_improved / aggregator.pass2.n_samples
        if aggregator.pass2.n_samples
        else None
    )
    imp_example = (
        100.0 * aggregator.ex_improved_p2 / num_examples if num_examples else None
    )
    tag1_pct = (
        100.0 * aggregator.pass1.tag_ok / aggregator.pass1.n_samples
        if aggregator.pass1.n_samples
        else None
    )
    tag2_pct = (
        100.0 * aggregator.pass2.tag_ok / aggregator.pass2.n_samples
        if aggregator.pass2.n_samples
        else None
    )

    soft1_sample = mean_safe(aggregator.pass1.soft_values)
    soft2_sample = mean_safe(aggregator.pass2.soft_values)
    soft1_example = mean_safe(
        [
            max(float(value) for value in values if value is not None)
            for values in aggregator.pass1.soft_by_problem.values()
            if any(value is not None for value in values)
        ],
    )
    soft2_example = mean_safe(
        [
            max(float(value) for value in values if value is not None)
            for values in aggregator.pass2.soft_by_problem.values()
            if any(value is not None for value in values)
        ],
    )

    return [
        aggregator.step,
        aggregator.pass1.n_samples,
        acc1_sample,
        acc1_example,
        mean_safe(aggregator.pass1.entropy_all),
        mean_safe(aggregator.pass1.entropy_think),
        mean_safe(aggregator.pass1.entropy_answer),
        soft1_sample,
        soft1_example,
        aggregator.pass2.n_samples,
        acc2_sample,
        acc2_example,
        mean_safe(aggregator.pass2.entropy_all),
        mean_safe(aggregator.pass2.entropy_think),
        mean_safe(aggregator.pass2.entropy_answer),
        soft2_sample,
        soft2_example,
        imp_sample,
        imp_example,
        tag1_pct,
        tag2_pct,
    ]


__all__ = [
    "StepAgg",
    "PassAggregate",
    "mean_safe",
    "pct",
    "fmt_float",
    "nat_step_from_path",
    "scan_files",
    "extract_prompt",
    "extract_group",
    "should_drop_group",
    "group_name_for_record",
    "accumulate_prompt_variants",
    "build_step_csv_row",
    "maybe_recompute_correctness",
]
