#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate crossword pass-1/pass-2 evaluation JSONLs by step, with optional:
  • Prompt-variant filtering and per-group caps (same semantics as your math tool)
  • Recompute correctness using crossword-friendly canon fields

Assumes crossword inference JSONLs where:
  - rec["problem"] holds the clue text
  - rec["gold_answer_canon"] is the canonical gold
  - rec["pass1"]["pred_answer_canon"], rec["pass2"]["pred_answer_canon"] exist
  - rec["pass1"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
  - rec["pass2"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------ Utilities ------------------------------------

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

# --- Prompt & grouping helpers -----------------------------------------------

DEFAULT_PROMPT_KEYS = [
    "prompt", "prompt_text", "input_prompt", "input",
    "question_prompt", "fmt_prompt", "aug_prompt"
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
        # Friendly fallback if someone asks for 'prompt'
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

# ------------------------------ Correctness -----------------------------------

def _substr(hay: Optional[str], needle: Optional[str]) -> bool:
    if not hay or not needle:
        return False
    return needle in hay


def _exact(left: Optional[str], right: Optional[str]) -> bool:
    """Return True iff both strings are non-None and exactly equal."""
    if left is None or right is None:
        return False
    return left == right


def maybe_recompute_correctness(
    pass_dict: Dict[str, Any],
    gold_canon: Optional[str],
    mode: str,
) -> Optional[bool]:
    """Optionally recompute correctness based on canonical fields; else return None."""
    if mode == "none":
        return None
    pred = pass_dict.get("pred_answer_canon")
    if mode == "substring":
        return _substr(pred, gold_canon)
    if mode == "exact":
        return _exact(pred, gold_canon)
    return None

@dataclass
class PassAggregate:
    """Aggregate metrics for a single pass within a step."""

    correct_by_problem: Dict[str, bool] = field(default_factory=dict)
    counts: Dict[str, int] = field(
        default_factory=lambda: {
            "n_samples": 0,
            "sample_correct": 0,
            "sample_improved": 0,
            "tag_ok": 0,
            "reconsider_numer": 0,
        },
    )
    entropies: Dict[str, List[Optional[float]]] = field(
        default_factory=lambda: {
            "all": [],
            "think": [],
            "answer": [],
        },
    )
    tokens: Dict[str, List[Optional[int]]] = field(
        default_factory=lambda: {
            "think": [],
            "answer": [],
        },
    )
    soft_values: List[Optional[float]] = field(default_factory=list)
    soft_by_problem: Dict[str, List[Optional[float]]] = field(
        default_factory=lambda: defaultdict(list),
    )
    stop_counts: Dict[str, Counter] = field(
        default_factory=lambda: {
            "think": Counter(),
            "answer": Counter(),
        },
    )

    @property
    def n_samples(self) -> int:
        return self.counts["n_samples"]

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        self.counts["n_samples"] = value

    @property
    def sample_correct(self) -> int:
        return self.counts["sample_correct"]

    @sample_correct.setter
    def sample_correct(self, value: int) -> None:
        self.counts["sample_correct"] = value

    @property
    def sample_improved(self) -> int:
        return self.counts["sample_improved"]

    @sample_improved.setter
    def sample_improved(self, value: int) -> None:
        self.counts["sample_improved"] = value

    @property
    def tag_ok(self) -> int:
        return self.counts["tag_ok"]

    @tag_ok.setter
    def tag_ok(self, value: int) -> None:
        self.counts["tag_ok"] = value

    @property
    def reconsider_numer(self) -> int:
        return self.counts["reconsider_numer"]

    @reconsider_numer.setter
    def reconsider_numer(self, value: int) -> None:
        self.counts["reconsider_numer"] = value

    @property
    def entropy_all(self) -> List[Optional[float]]:
        return self.entropies["all"]

    @property
    def entropy_think(self) -> List[Optional[float]]:
        return self.entropies["think"]

    @property
    def entropy_answer(self) -> List[Optional[float]]:
        return self.entropies["answer"]

    @property
    def tokens_think(self) -> List[Optional[int]]:
        return self.tokens["think"]

    @property
    def tokens_answer(self) -> List[Optional[int]]:
        return self.tokens["answer"]

    @property
    def stop_think(self) -> Counter:
        return self.stop_counts["think"]

    @property
    def stop_answer(self) -> Counter:
        return self.stop_counts["answer"]


class StepAgg:
    """Aggregate per-step metrics across all examples and samples."""

    def __init__(self, step: int):
        self.step = step
        self.pass1 = PassAggregate()
        self.pass2 = PassAggregate()
        self.examples: set[str] = set()
        self.ex_improved_p2 = 0

    def add(self, record: Dict[str, Any], recompute_mode: str) -> None:
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

    def _example_level_soft_mean(
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

    def row(self) -> str:
        """Return a single formatted summary row for this step."""
        metrics = self._row_metrics()
        return (
            f"{self.step:6d} "
            f"{self.n_samp_p1:6d} {metrics['acc1_sample']:>6} {metrics['acc1_example']:>6} "
            f"{metrics['ent1']:>8} {metrics['think1']:>7} {metrics['answer1']:>7} "
            f"{metrics['soft1_sample']:>7} {metrics['soft1_example']:>7} "
            f"{self.n_samp_p2:6d} {metrics['acc2_sample']:>6} {metrics['acc2_example']:>6} "
            f"{metrics['ent2']:>8} {metrics['think2']:>7} {metrics['answer2']:>7} "
            f"{metrics['soft2_sample']:>7} {metrics['soft2_example']:>7} "
            f"{metrics['improvement_sample']:>6} {metrics['improvement_example']:>6} "
            f"{metrics['tag1']:>6} {metrics['tag2']:>6}"
        )

    def footer(self) -> str:
        """Return a multi-line human-readable summary for this step."""
        lines: List[str] = [f"   • examples: {len(self.examples)}"]
        self._append_stop_reason_lines(lines)
        self._append_reconsider_lines(lines)
        self._append_token_stats_lines(lines)
        self._append_soft_reward_lines(lines)
        return "\n".join(lines)

    def _row_metrics(self) -> Dict[str, str]:
        """Compute formatted metric strings used in row()."""
        num_examples = len(self.examples) if self.examples else 0
        metrics: Dict[str, str] = {}

        metrics["acc1_sample"] = pct(self.samp_correct_p1, self.n_samp_p1)
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
        metrics["soft1_example"] = fmt_float(
            self._example_level_soft_mean(self.pass1.soft_by_problem),
        )
        metrics["soft2_sample"] = fmt_float(mean_safe(self.pass2.soft_values))
        metrics["soft2_example"] = fmt_float(
            self._example_level_soft_mean(self.pass2.soft_by_problem),
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
        mean_soft1_examples = self._example_level_soft_mean(self.pass1.soft_by_problem)
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
        mean_soft2_examples = self._example_level_soft_mean(self.pass2.soft_by_problem)
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


def _group_name_for_record(
    record: Dict[str, Any],
    filter_scope: str,
) -> str:
    """Return the group key (problem or global) for a record."""
    if filter_scope == "per_problem":
        return str(record.get("problem", ""))
    return "__GLOBAL__"


def _accumulate_prompt_variants_from_line(
    line: str,
    args: argparse.Namespace,
    variants_by_group: Dict[str, set],
    counts: Dict[str, int],
) -> None:
    """Update prompt-variant tracking structures from a single JSONL line."""
    line = line.strip()
    if not line:
        return
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return

    counts["seen"] += 1
    prompt = extract_prompt(record, args.prompt_key, args.strict_prompt_key)
    if prompt is None:
        return

    counts["with_prompt"] += 1
    if args.prompt_family_regex:
        prompt = re.sub(str(args.prompt_family_regex), "", prompt)

    group_name = _group_name_for_record(record, args.filter_scope)
    variants_by_group[group_name].add(prompt)


def _compute_prompt_drop_groups(files: List[str], args: argparse.Namespace) -> set[str]:
    """Return the set of group names to drop based on prompt-variant limits."""
    drop_groups: set[str] = set()
    if args.max_prompt_versions is None:
        return drop_groups

    variants_by_group: Dict[str, set] = defaultdict(set)
    counts = {"seen": 0, "with_prompt": 0}

    for path in files:
        with open(path, "r", encoding="utf-8") as infile:
            for line in infile:
                _accumulate_prompt_variants_from_line(
                    line=line,
                    args=args,
                    variants_by_group=variants_by_group,
                    counts=counts,
                )

    for group_name, variants in variants_by_group.items():
        if len(variants) > args.max_prompt_versions:
            drop_groups.add(group_name)

    scope = "per-problem" if args.filter_scope == "per_problem" else "global"
    print(
        "[filter] "
        f"Scope={scope} | "
        f"threshold={args.max_prompt_versions} | "
        f"records_seen={counts['seen']} | "
        f"with_prompt={counts['with_prompt']} | "
        f"groups_dropped={len(drop_groups)}",
    )
    return drop_groups


def _maybe_write_filtered_files(
    files: List[str],
    args: argparse.Namespace,
    drop_groups: set[str],
) -> Tuple[bool, List[str]]:
    """
    Optionally write filtered/capped copies of the input JSONLs.

    Returns (wrote_any, files_for_agg), where files_for_agg is the list of
    files that should be used for aggregation.
    """
    if not (args.rewrite_filtered or args.write_filtered_to):
        return False, files

    if args.rewrite_filtered and args.write_filtered_to:
        raise SystemExit("Choose only one of --rewrite_filtered or --write_filtered_to.")

    wrote_any = False

    for src_path in files:
        if args.rewrite_filtered:
            out_path = src_path + ".tmp_filter"
        else:
            out_path = os.path.join(
                args.write_filtered_to,
                os.path.relpath(src_path, args.results_root),
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        stats, group_counts, exceeded_groups = _filter_single_file(
            src_path,
            out_path,
            args,
            drop_groups,
        )

        if args.rewrite_filtered:
            os.replace(out_path, src_path)

        wrote_any = True
        print(
            "[filter-write] "
            f"{src_path} -> "
            f"{src_path if args.rewrite_filtered else out_path} | "
            f"kept={stats['kept']} of {stats['total']} "
            f"| unique_groups={len(group_counts)} "
            f"| groups_truncated={len(exceeded_groups)}",
        )

    files_for_agg = files
    if wrote_any and args.write_filtered_to and args.aggregate_from_filtered:
        files_for_agg = [
            os.path.join(
                args.write_filtered_to,
                os.path.relpath(src_path, args.results_root),
            )
            for src_path in files
        ]

    return wrote_any, files_for_agg


def _filter_single_file(
    src_path: str,
    out_path: str,
    args: argparse.Namespace,
    drop_groups: set[str],
) -> Tuple[Dict[str, int], Dict[str, int], set[str]]:
    """Filter one JSONL file according to prompt/group caps."""
    group_counts: Dict[str, int] = defaultdict(int)
    stats = {"total": 0, "kept": 0}
    exceeded_groups: set[str] = set()

    with open(src_path, "r", encoding="utf-8") as src_file, open(
        out_path,
        "w",
        encoding="utf-8",
    ) as dst_file:
        for line in src_file:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if should_drop_group(record, drop_groups, args.filter_scope):
                continue

            if args.max_per_group is not None:
                group_key_value = extract_group(record, args.group_key)
                count = group_counts[group_key_value]
                if count >= args.max_per_group:
                    exceeded_groups.add(group_key_value)
                    continue
                group_counts[group_key_value] = count + 1

            dst_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats, group_counts, exceeded_groups


def _aggregate_steps(
    files_for_agg: List[str],
    args: argparse.Namespace,
    drop_groups: set[str],
    wrote_any: bool,
) -> List[StepAgg]:
    """Aggregate records from JSONL files into per-step StepAgg objects."""
    steps: Dict[int, StepAgg] = {}

    for path in files_for_agg:
        group_counts: Dict[str, int] = defaultdict(int)
        step_from_name = nat_step_from_path(path)
        try:
            infile = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            continue

        with infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if should_drop_group(record, drop_groups, args.filter_scope):
                    continue

                if args.max_per_group is not None and not wrote_any:
                    group_key_value = extract_group(record, args.group_key)
                    count = group_counts[group_key_value]
                    if count >= args.max_per_group:
                        continue
                    group_counts[group_key_value] = count + 1

                step_value = record.get(
                    "step",
                    step_from_name if step_from_name is not None else 0,
                )
                aggregator = steps.setdefault(step_value, StepAgg(step_value))
                aggregator.add(record, args.recompute_correctness)

    for aggregator in steps.values():
        aggregator.finalize()

    return [steps[key] for key in sorted(steps)]


def _print_step_summaries(aggregates: List[StepAgg]) -> None:
    """Print the main table and per-step footer summaries."""
    header = (
        "  step   n1S  acc1S  acc1E    ent1      t1      a1  soft1S soft1E  "
        "n2S  acc2S  acc2E    ent2      t2      a2  soft2S soft2E  "
        "impS  impE  tag1  tag2"
    )
    print(header)
    print("-" * len(header))
    for aggregator in aggregates:
        print(aggregator.row())
    print()

    for aggregator in aggregates:
        print(f"[step {aggregator.step}]")
        print(aggregator.footer())
        print()


def _build_step_csv_row(aggregator: StepAgg) -> List[Any]:
    """Build a single CSV row of numeric metrics for one step."""
    num_examples = len(aggregator.examples) if aggregator.examples else 0

    acc1_sample = (
        100.0 * aggregator.samp_correct_p1 / aggregator.n_samp_p1
        if aggregator.n_samp_p1
        else None
    )
    acc2_sample = (
        100.0 * aggregator.samp_correct_p2 / aggregator.n_samp_p2
        if aggregator.n_samp_p2
        else None
    )
    acc1_example = (
        100.0 * sum(1 for value in aggregator.ex_correct_p1.values() if value) / num_examples
        if num_examples
        else None
    )
    acc2_example = (
        100.0 * sum(1 for value in aggregator.ex_correct_p2.values() if value) / num_examples
        if num_examples
        else None
    )
    imp_sample = (
        100.0 * aggregator.samp_improved_p2 / aggregator.n_samp_p2
        if aggregator.n_samp_p2
        else None
    )
    imp_example = (
        100.0 * aggregator.ex_improved_p2 / num_examples if num_examples else None
    )
    tag1_pct = (
        100.0 * aggregator.tag_ok_p1 / aggregator.n_samp_p1 if aggregator.n_samp_p1 else None
    )
    tag2_pct = (
        100.0 * aggregator.tag_ok_p2 / aggregator.n_samp_p2 if aggregator.n_samp_p2 else None
    )

    soft1_sample = mean_safe(aggregator.soft1_vals)
    soft2_sample = mean_safe(aggregator.soft2_vals)
    soft1_example = mean_safe(
        [
            max(float(value) for value in values if value is not None)
            for _, values in aggregator.soft1_by_problem.items()
            if any(value is not None for value in values)
        ],
    )
    soft2_example = mean_safe(
        [
            max(float(value) for value in values if value is not None)
            for _, values in aggregator.soft2_by_problem.items()
            if any(value is not None for value in values)
        ],
    )

    return [
        aggregator.step,
        aggregator.n_samp_p1,
        acc1_sample,
        acc1_example,
        mean_safe(aggregator.ent_p1_all),
        mean_safe(aggregator.ent_p1_think),
        mean_safe(aggregator.ent_p1_ans),
        soft1_sample,
        soft1_example,
        aggregator.n_samp_p2,
        acc2_sample,
        acc2_example,
        mean_safe(aggregator.ent_p2_all),
        mean_safe(aggregator.ent_p2_think),
        mean_safe(aggregator.ent_p2_ans),
        soft2_sample,
        soft2_example,
        imp_sample,
        imp_example,
        tag1_pct,
        tag2_pct,
    ]


def _write_csv_outputs(args: argparse.Namespace, aggregates: List[StepAgg]) -> None:
    """Write optional CSV outputs summarizing per-step and per-example metrics."""
    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "step",
                    "n1S",
                    "acc1S_pct",
                    "acc1E_pct",
                    "ent1",
                    "t1",
                    "a1",
                    "soft1S",
                    "soft1E",
                    "n2S",
                    "acc2S_pct",
                    "acc2E_pct",
                    "ent2",
                    "t2",
                    "a2",
                    "soft2S",
                    "soft2E",
                    "impS_pct",
                    "impE_pct",
                    "tag1_pct",
                    "tag2_pct",
                ],
            )
            for aggregator in aggregates:
                writer.writerow(_build_step_csv_row(aggregator))

    if args.per_example_csv:
        with open(args.per_example_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "problem", "p1_correct", "p2_correct", "improved"])
            for aggregator in aggregates:
                for problem in sorted(aggregator.examples):
                    p1_ok = aggregator.ex_correct_p1.get(problem, False)
                    p2_ok = aggregator.ex_correct_p2.get(problem, False)
                    improved = p2_ok and not p1_ok
                    writer.writerow(
                        [
                            aggregator.step,
                            problem,
                            int(bool(p1_ok)),
                            int(bool(p2_ok)),
                            int(bool(improved)),
                        ],
                    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        help="Root directory containing step*/.../*.jsonl",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Filter filenames containing this split substring "
            "(e.g., 'test')."
        ),
    )

    # Prompt-variant filtering
    parser.add_argument(
        "--max_prompt_versions",
        type=int,
        default=None,
        help=(
            "Drop any group that has more than this many distinct "
            "prompt variants."
        ),
    )
    parser.add_argument(
        "--prompt_key",
        default="prompt",
        help="Dot-path to the prompt field (default: 'prompt').",
    )
    parser.add_argument(
        "--strict_prompt_key",
        action="store_true",
        help="Only use --prompt_key; do not fall back to alternatives.",
    )
    parser.add_argument(
        "--filter_scope",
        choices=["per_problem", "global"],
        default="per_problem",
        help="Count prompt variants per problem (default) or globally.",
    )
    parser.add_argument(
        "--prompt_family_regex",
        default=None,
        help="Regex to strip version markers before counting variants.",
    )

    # Per-group hard cap
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=None,
        help=(
            "Keep at most this many records per group "
            "(grouping uses --group_key, default 'problem')."
        ),
    )
    parser.add_argument(
        "--group_key",
        default="problem",
        help="Dot-path field to group by for capping (default: 'problem').",
    )

    # Write filtered copies or rewrite in place
    parser.add_argument(
        "--rewrite_filtered",
        action="store_true",
        help="Rewrite input JSONLs IN PLACE (dangerous).",
    )
    parser.add_argument(
        "--write_filtered_to",
        default=None,
        help=(
            "Write filtered JSONLs under this output root, "
            "mirroring the input tree."
        ),
    )
    parser.add_argument(
        "--aggregate_from_filtered",
        action="store_true",
        help="Aggregate from filtered copies instead of original JSONLs.",
    )

    # Crossword-specific: recompute correctness
    parser.add_argument(
        "--recompute_correctness",
        choices=["none", "substring", "exact"],
        default="none",
        help=(
            "Override recorded correctness using pred_answer_canon vs "
            "gold_answer_canon."
        ),
    )

    # CSV outputs
    parser.add_argument(
        "--save_csv",
        default=None,
        help="Optional CSV output path for the step table.",
    )
    parser.add_argument(
        "--per_example_csv",
        default=None,
        help="Optional CSV with per-example correctness (p1, p2, improved).",
    )
    return parser


# ------------------------------ Main -----------------------------------------

def main() -> None:
    """CLI entrypoint that filters and aggregates crossword inference JSONL results."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    files = scan_files(args.results_root, args.split)
    if not files:
        print("No JSONL files found. Check the path or split filter.")
        return

    drop_groups = _compute_prompt_drop_groups(files, args)
    wrote_any, files_for_agg = _maybe_write_filtered_files(files, args, drop_groups)
    aggregates = _aggregate_steps(files_for_agg, args, drop_groups, wrote_any)
    _print_step_summaries(aggregates)
    _write_csv_outputs(args, aggregates)


if __name__ == "__main__":
    main()
