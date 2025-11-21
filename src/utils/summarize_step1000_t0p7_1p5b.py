#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binned summary by pass-1 (think+answer) entropy with equal-count (quantile) bins per domain.

Features
--------
• Sample-level: p(correct_p1), p(correct_p2), optional p_soft@τ(...)
• 'none_right_p1' (aligned to HARD or SOFT depending on conditional setting)
• Example-level conditionals: p(p2|p1_none) and p(p2|p1_any) with counts
• Conditionals can use SOFT≥τ for selected domains (e.g., carpark)

Bin logic
---------
• Binning key: pass-1 entropy_think and entropy_answer combined via --combine {sum|mean}
• --binning {equal_count|equal_width}, default equal_count (quantiles per domain)
• Samples missing either entropy_think/entropy_answer are skipped from binning
"""
from dataclasses import dataclass
import argparse
import json
import os
import re
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Set, Tuple

DOM_MAP = {
    "carpark": "Carpark",
    "xword":   "Crossword",
    "math":    "Math",
}

def nat_step_from_path(path: str) -> Optional[int]:
    """Extract the training step number from a path like .../step1000/...."""
    match = re.search(r"step(\d+)", path)
    return int(match.group(1)) if match else None


def _substr(haystack: Optional[str], needle: Optional[str]) -> bool:
    """Return True if needle appears in haystack, handling None safely."""
    if not haystack or not needle:
        return False
    return needle in haystack


def _exact(left: Optional[str], right: Optional[str]) -> bool:
    """Return True if two strings are exactly equal, handling None safely."""
    if left is None or right is None:
        return False
    return left == right


def compute_correct(
    pass_dict: Dict[str, Any],
    gold_canon: Optional[str],
    mode: str,
) -> Optional[bool]:
    """Compute correctness for a pass-1/2 record given a mode."""
    if pass_dict is None:
        return None
    if mode == "none":
        value = pass_dict.get("is_correct_pred")
        return None if value is None else bool(value)
    pred = pass_dict.get("pred_answer_canon")
    if mode == "substring":
        return _substr(pred, gold_canon)
    if mode == "exact":
        return _exact(pred, gold_canon)
    return None


def get_soft(pass_dict: Dict[str, Any]) -> Optional[float]:
    """Return soft_reward (or 'soft reward') as a float, when available."""
    if pass_dict is None:
        return None
    value = pass_dict.get("soft_reward", pass_dict.get("soft reward"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def domain_from_dirname(dir_name: str) -> Optional[str]:
    """Map a GRPO directory name to a pretty domain label."""
    for key, label in DOM_MAP.items():
        if f"-{key}-" in dir_name:
            return label
    return None


def label_to_token(label: str) -> str:
    """Normalize a human-readable domain label to a token."""
    token = label.strip().lower()
    if token in ("carpark", "rush hour", "rush_hour", "rushhour"):
        return "carpark"
    if token in ("crossword", "xword", "crosswords"):
        return "xword"
    if token in ("math", "math2"):
        return "math"
    return token


def parse_cond_soft_domains(raw: Optional[str]) -> Set[str]:
    """Parse --cond-soft-domains into a normalized set of domain tokens."""
    if not raw:
        return set()
    tokens: Set[str] = set()
    for part in raw.split(","):
        normalized = part.strip().lower()
        if normalized in ("carpark", "rush hour", "rush_hour", "rushhour"):
            tokens.add("carpark")
        elif normalized in ("crossword", "xword", "crosswords"):
            tokens.add("xword")
        elif normalized in ("math", "math2"):
            tokens.add("math")
        else:
            tokens.add(normalized)
    return tokens


def find_model_dirs(results_root: str, target_temp: float) -> List[Tuple[str, str]]:
    """Return (domain_label, dirpath) pairs matching GRPO-1.5B temp folders."""
    candidates: List[Tuple[str, str]] = []
    for name in os.listdir(results_root):
        if not name.startswith("GRPO-1.5B-"):
            continue
        match = re.search(r"-temp-([0-9.]+)$", name)
        if not match:
            continue
        try:
            temp_val = float(match.group(1))
        except ValueError:
            continue
        if abs(temp_val - target_temp) > 1e-12:
            continue
        domain_label = domain_from_dirname(name)
        if domain_label is None:
            continue
        full_path = os.path.join(results_root, name)
        if os.path.isdir(full_path):
            candidates.append((domain_label, full_path))
    chosen: Dict[str, str] = {}
    for domain_label, path in sorted(candidates, key=lambda item: item[1]):
        chosen[domain_label] = path
    return sorted(chosen.items(), key=lambda item: item[0])


def iter_jsonl_files(root: str, split_substr: Optional[str]) -> List[str]:
    """Collect all JSONL files under root whose names contain split_substr."""
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in filename:
                continue
            paths.append(os.path.join(dirpath, filename))
    return paths


def extract_problem_key(record: Dict[str, Any]) -> str:
    """Derive a stable per-example key from a record."""
    for key in (
        "problem",
        "problem_id",
        "question",
        "clue",
        "title",
        "id",
        "uid",
        "example_id",
        "idx",
    ):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"__line_{id(record)}"

def p1_think_answer_value(pass1_record: Dict[str, Any], combine: str) -> Optional[float]:
    """Combine entropy_think and entropy_answer for pass-1 using sum or mean."""
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    think_entropy = _to_float(pass1_record.get("entropy_think"))
    answer_entropy = _to_float(pass1_record.get("entropy_answer"))
    if think_entropy is None or answer_entropy is None:
        return None
    if combine == "mean":
        return 0.5 * (think_entropy + answer_entropy)
    return think_entropy + answer_entropy


def compute_equal_width_edges(vmin: float, vmax: float, bins: int) -> List[float]:
    """Return equally spaced bin edges between vmin and vmax."""
    if bins <= 0:
        return [vmin, vmax]
    if vmax <= vmin:
        return [vmin] * bins + [vmax]
    width = (vmax - vmin) / bins
    edges = [vmin + i * width for i in range(bins)]
    edges.append(vmax)
    edges[-1] = vmax
    return edges


def compute_equal_count_edges(values: List[float], bins: int) -> List[float]:
    """Return quantile-style bin edges with approximately equal counts per bin."""
    sorted_values = sorted(values)
    num_values = len(sorted_values)
    if num_values == 0:
        return [0.0, 0.0]
    bins = max(1, min(bins, num_values))
    boundaries = [int(round(i * num_values / bins)) for i in range(bins + 1)]
    boundaries[0] = 0
    boundaries[-1] = num_values
    edges = [sorted_values[boundaries[i]] for i in range(bins)]
    edges.append(sorted_values[-1])
    return edges


def assign_bin_equal_width(value: float, edges: List[float]) -> int:
    """Assign a value to an equal-width bin defined by edges."""
    num_bins = len(edges) - 1
    if num_bins <= 0:
        return 0
    vmin, vmax = edges[0], edges[-1]
    if vmax <= vmin:
        return 0
    width = (vmax - vmin) / num_bins
    if width == 0:
        return 0
    idx = int((value - vmin) / width)
    idx = max(idx, 0)
    if idx >= num_bins:
        idx = num_bins - 1
    return idx


def assign_bin_equal_count(value: float, edges: List[float]) -> int:
    """Assign a value to an equal-count bin defined by edges."""
    if len(edges) <= 2:  # one bin
        return 0
    cutpoints = edges[1:-1]
    return bisect_right(cutpoints, value)


def bin_label(index: int, edges: List[float]) -> str:
    """Human-readable bin label from edges (inclusive on right for last bin)."""
    num_bins = len(edges) - 1
    low = edges[index]
    high = edges[index + 1]
    if index < num_bins - 1:
        return f"[{low:.3f}, {high:.3f})"
    return f"[{low:.3f}, {high:.3f}]"

# ---------- per-bin aggregator ----------


@dataclass
class BinCountStats:
    """Hard/soft counts for a single pass within a bin."""

    hard_total: int = 0
    hard_correct: int = 0
    soft_total: int = 0
    soft_correct: int = 0


class BinAggregate:
    """Per-bin aggregate statistics for pass-1 and pass-2."""

    __slots__ = (
        "pass1",
        "pass2",
        "ex_seen",
        "ex_has_p1",
        "ex_p1_any_ok_cond",
        "ex_p2_any_ok_cond",
    )

    def __init__(self) -> None:
        self.pass1 = BinCountStats()
        self.pass2 = BinCountStats()
        self.ex_seen: Set[str] = set()
        self.ex_has_p1: Set[str] = set()
        self.ex_p1_any_ok_cond: Dict[str, bool] = {}
        self.ex_p2_any_ok_cond: Dict[str, bool] = {}

    # Convenience properties exposing pass-1/2 hard/soft counts.
    @property
    def n1_s(self) -> int:
        """Total number of pass-1 samples (hard correctness)."""
        return self.pass1.hard_total

    @n1_s.setter
    def n1_s(self, value: int) -> None:
        """Set the total number of pass-1 samples (hard correctness)."""
        self.pass1.hard_total = value

    @property
    def c1_s(self) -> int:
        """Total number of correct pass-1 samples (hard correctness)."""
        return self.pass1.hard_correct

    @c1_s.setter
    def c1_s(self, value: int) -> None:
        """Set the total number of correct pass-1 samples (hard correctness)."""
        self.pass1.hard_correct = value

    @property
    def n2_s(self) -> int:
        """Total number of pass-2 samples (hard correctness)."""
        return self.pass2.hard_total

    @n2_s.setter
    def n2_s(self, value: int) -> None:
        """Set the total number of pass-2 samples (hard correctness)."""
        self.pass2.hard_total = value

    @property
    def c2_s(self) -> int:
        """Total number of correct pass-2 samples (hard correctness)."""
        return self.pass2.hard_correct

    @c2_s.setter
    def c2_s(self, value: int) -> None:
        """Set the total number of correct pass-2 samples (hard correctness)."""
        self.pass2.hard_correct = value

    @property
    def n1_soft(self) -> int:
        """Total number of pass-1 samples with a soft score."""
        return self.pass1.soft_total

    @n1_soft.setter
    def n1_soft(self, value: int) -> None:
        """Set the total number of pass-1 samples with a soft score."""
        self.pass1.soft_total = value

    @property
    def c1_soft(self) -> int:
        """Total number of correct pass-1 samples under the soft metric."""
        return self.pass1.soft_correct

    @c1_soft.setter
    def c1_soft(self, value: int) -> None:
        """Set the total number of correct pass-1 samples under the soft metric."""
        self.pass1.soft_correct = value

    @property
    def n2_soft(self) -> int:
        """Total number of pass-2 samples with a soft score."""
        return self.pass2.soft_total

    @n2_soft.setter
    def n2_soft(self, value: int) -> None:
        """Set the total number of pass-2 samples with a soft score."""
        self.pass2.soft_total = value

    @property
    def c2_soft(self) -> int:
        """Total number of correct pass-2 samples under the soft metric."""
        return self.pass2.soft_correct

    @c2_soft.setter
    def c2_soft(self, value: int) -> None:
        """Set the total number of correct pass-2 samples under the soft metric."""
        self.pass2.soft_correct = value


@dataclass
class ConditionalCounts:
    """Example-level conditional counts for a bin."""

    p1_none_total: int
    p1_none_and_p2: int
    p1_any_total: int
    p1_any_and_p2: int


@dataclass
class BinRowInputs:
    """Inputs required to build a single output row for a bin."""

    domain_label: str
    bin_index: int
    edges: List[float]
    agg: BinAggregate
    conditional: ConditionalCounts
    use_soft_for_conditionals: bool


@dataclass
class ConditionalConfig:
    """Configuration for soft-threshold conditionals."""

    soft_threshold: Optional[float]
    use_soft_for_conditionals: bool


@dataclass
class BinningConfig:
    """Configuration for binning and correctness aggregation."""

    target_step: int
    split_substr: Optional[str]
    recompute_mode: str
    bins: int
    combine_mode: str
    binning_mode: str
    conditional: ConditionalConfig


def _iter_records_for_step(
    files: List[str],
    target_step: int,
):
    """Yield parsed JSON records matching a target training step."""
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step", step_from_name)
                if step != target_step:
                    continue
                yield record


def _collect_pass1_values(
    files: List[str],
    config: BinningConfig,
) -> Tuple[List[float], float, float]:
    """Collect pass-1 entropy values and global min/max for binning."""
    values: List[float] = []
    vmin = float("inf")
    vmax = float("-inf")
    for record in _iter_records_for_step(files, config.target_step):
        pass1_record = record.get("pass1") or {}
        value = p1_think_answer_value(pass1_record, config.combine_mode)
        if value is None:
            continue
        values.append(value)
        vmin = min(vmin, value)
        vmax = max(vmax, value)
    return values, vmin, vmax


def _populate_bin_aggregates(
    files: List[str],
    edges: List[float],
    config: BinningConfig,
) -> Tuple[List[BinAggregate], int]:
    """Populate per-bin aggregates for pass-1 and pass-2."""
    aggs = [BinAggregate() for _ in range(len(edges) - 1)]
    skipped_missing = 0

    for record in _iter_records_for_step(files, config.target_step):
        gold = record.get("gold_answer_canon")
        prob_key = extract_problem_key(record)

        pass1_record = record.get("pass1") or {}
        value = p1_think_answer_value(pass1_record, config.combine_mode)
        if value is None:
            skipped_missing += 1
            continue

        if config.binning_mode == "equal_count":
            bin_index = assign_bin_equal_count(value, edges)
        else:
            bin_index = assign_bin_equal_width(value, edges)

        agg = aggs[bin_index]
        agg.ex_seen.add(prob_key)

        _update_pass1_agg(agg, prob_key, pass1_record, gold, config)

        # Prefer canonical pass2; fall back to multi-cue variants when present.
        pass2_section = (
            record.get("pass2")
            or record.get("pass2c")
            or record.get("pass2b")
            or record.get("pass2a")
            or {}
        )
        _update_pass2_agg(agg, prob_key, pass2_section, gold, config)

    return aggs, skipped_missing


def _update_pass1_agg(
    agg: BinAggregate,
    prob_key: str,
    pass1_record: Dict[str, Any],
    gold: Optional[str],
    config: BinningConfig,
) -> None:
    """Update aggregates for pass-1 metrics."""
    if not pass1_record:
        return

    agg.ex_has_p1.add(prob_key)

    ok1_hard = compute_correct(pass1_record, gold, config.recompute_mode)
    if ok1_hard is not None:
        agg.n1_s += 1
        agg.c1_s += int(bool(ok1_hard))

    sr1 = get_soft(pass1_record)
    if sr1 is not None:
        agg.n1_soft += 1
        if (
            config.conditional.soft_threshold is not None
            and sr1 >= config.conditional.soft_threshold
        ):
            agg.c1_soft += 1

    if (
        config.conditional.use_soft_for_conditionals
        and config.conditional.soft_threshold is not None
    ):
        ok1_cond = (sr1 is not None) and (sr1 >= config.conditional.soft_threshold)
    else:
        ok1_cond = bool(ok1_hard) if ok1_hard is not None else False

    if ok1_cond:
        agg.ex_p1_any_ok_cond[prob_key] = True


def _update_pass2_agg(
    agg: BinAggregate,
    prob_key: str,
    pass2_section: Dict[str, Any],
    gold: Optional[str],
    config: BinningConfig,
) -> None:
    """Update aggregates for pass-2 metrics (bucketed by pass-1 bin)."""
    if not pass2_section:
        return

    ok2_hard = compute_correct(pass2_section, gold, config.recompute_mode)
    if ok2_hard is not None:
        agg.n2_s += 1
        agg.c2_s += int(bool(ok2_hard))

    sr2 = get_soft(pass2_section)
    if sr2 is not None:
        agg.n2_soft += 1
        if (
            config.conditional.soft_threshold is not None
            and sr2 >= config.conditional.soft_threshold
        ):
            agg.c2_soft += 1

    if (
        config.conditional.use_soft_for_conditionals
        and config.conditional.soft_threshold is not None
    ):
        ok2_cond = (sr2 is not None) and (sr2 >= config.conditional.soft_threshold)
    else:
        ok2_cond = bool(ok2_hard) if ok2_hard is not None else False

    if ok2_cond:
        agg.ex_p2_any_ok_cond[prob_key] = True


def summarize_dir_binned(
    dirpath: str,
    config: BinningConfig,
) -> Tuple[List[BinAggregate], List[float], int]:
    """
    Summarize a directory of JSONL files into entropy bins.

    Returns:
      (per-bin BinAggregate list, edges list, skipped_count_missing_entropy)
    """
    files = iter_jsonl_files(dirpath, config.split_substr)

    values, vmin, vmax = _collect_pass1_values(files, config)
    if not values:
        return [BinAggregate()], [0.0, 0.0], 0

    if config.binning_mode == "equal_count":
        edges = compute_equal_count_edges(values, config.bins)
    else:
        edges = compute_equal_width_edges(vmin, vmax, config.bins)

    aggs, skipped_missing = _populate_bin_aggregates(files, edges, config)
    return aggs, edges, skipped_missing


# ---------- pretty helpers ----------
def fmt_prob(num: int, den: int) -> str:
    """Format num/den as a probability, or '-' if undefined."""
    return "-" if den == 0 else f"{num/den:.3f}"


def yes_no_na(num_total: int, num_yes: int) -> str:
    """Return 'Yes'/'No'/'NA' based on counts."""
    if num_total == 0:
        return "NA"
    return "Yes" if num_yes == 0 else "No"


def none_right_label(
    agg: BinAggregate,
    use_soft_for_conditionals: bool,
    tau_set: bool,
) -> str:
    """
    Mirror conditional semantics for the 'none_right_p1' column.

    If conditionals use SOFT>=τ, base the label on soft counts; otherwise,
    align it with the hard correctness counts.
    """
    if use_soft_for_conditionals and tau_set:
        return yes_no_na(agg.n1_soft, agg.c1_soft)
    return yes_no_na(agg.n1_s, agg.c1_s)


def domain_order_key(label: str) -> int:
    """Provide a stable sort key for domain labels (xword, math, carpark)."""
    tok = label_to_token(label)
    order = {"xword": 0, "math": 1, "carpark": 2}
    return order.get(tok, 99)

# ---------- main ----------

def build_arg_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results",
        help="Path containing GRPO-* result folders",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature to select (e.g., 0.05, 0.3, 0.7)",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=1000,
        help="Step to summarize (default: 1000)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Only include files whose names contain this substring "
            "(e.g., 'test')."
        ),
    )
    parser.add_argument(
        "--recompute",
        choices=["none", "substring", "exact"],
        default="none",
        help="Recompute correctness from *_canon fields (useful for crosswords).",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=None,
        help=(
            "If set, also compute soft accuracy columns where "
            "soft_reward >= τ."
        ),
    )
    parser.add_argument(
        "--none-right-col",
        action="store_true",
        help=(
            "Add 'none_right_p1' (within-bin; semantics aligned "
            "with conditionals)."
        ),
    )
    parser.add_argument(
        "--cond-col",
        action="store_true",
        help="Add example-level conditional columns.",
    )
    parser.add_argument(
        "--cond-soft-domains",
        default="",
        help=(
            "Comma-separated domains whose conditionals use "
            "SOFT>=τ (e.g., 'carpark')."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of bins (default: 4).",
    )
    parser.add_argument(
        "--combine",
        choices=["sum", "mean"],
        default="sum",
        help=(
            "Combine pass-1 entropy_think and entropy_answer "
            "via 'sum' or 'mean'."
        ),
    )
    parser.add_argument(
        "--binning",
        choices=["equal_count", "equal_width"],
        default="equal_count",
        help="Bin strategy: equal_count (quantiles) or equal_width (ranges).",
    )
    return parser


def build_header(args: argparse.Namespace) -> List[str]:
    """Build the table header row based on CLI options."""
    header = [
        f"{'Domain':<10}",
        f"{'Bin':<16}",
        f"{'Number of Samples':>16}",
        f"{'p(correct_p1)':>16}",
        f"{'p(correct_p2)':>16}",
    ]
    if args.soft_threshold is not None:
        tau = args.soft_threshold
        header += [
            f"{f'p_soft@{tau:g}(p1)':>16}",
            f"{f'p_soft@{tau:g}(p2)':>16}",
            f"{'n_soft_p1':>16}",
            f"{'n_soft_p2':>16}",
        ]
    if args.none_right_col:
        header.append(f"{'none_right_p1':>16}")
    if args.cond_col:
        header += [
            f"{'p(p2|p1_none)':>16}",
            f"{'nE_p1_none':>16}",
            f"{'nE_p1_none&p2':>16}",
            f"{'p(p2|p1_any)':>16}",
            f"{'nE_p1_any':>16}",
            f"{'nE_p1_any&p2':>16}",
        ]
    return header


def compute_example_conditionals(agg: BinAggregate) -> Tuple[int, int, int, int]:
    """Compute example-level conditional counts within a bin."""
    ex_p1_none_total = ex_p1_none_and_p2 = 0
    ex_p1_any_total = ex_p1_any_and_p2 = 0
    for problem_key in agg.ex_seen:
        if problem_key in agg.ex_has_p1:
            if agg.ex_p1_any_ok_cond.get(problem_key, False):
                ex_p1_any_total += 1
                if agg.ex_p2_any_ok_cond.get(problem_key, False):
                    ex_p1_any_and_p2 += 1
            else:
                ex_p1_none_total += 1
                if agg.ex_p2_any_ok_cond.get(problem_key, False):
                    ex_p1_none_and_p2 += 1
    return (
        ex_p1_none_total,
        ex_p1_none_and_p2,
        ex_p1_any_total,
        ex_p1_any_and_p2,
    )


def build_row_for_bin(
    row_inputs: BinRowInputs,
    args: argparse.Namespace,
) -> List[str]:
    """Format a single output row for a given bin."""
    agg = row_inputs.agg
    sample_count = agg.n2_s if agg.n2_s > 0 else agg.n1_s
    pass1_prob = fmt_prob(agg.c1_s, agg.n1_s)
    pass2_prob = fmt_prob(agg.c2_s, agg.n2_s)
    parts = [
        f"{row_inputs.domain_label:<10}",
        f"{bin_label(row_inputs.bin_index, row_inputs.edges):<16}",
        f"{sample_count:>16d}",
        f"{pass1_prob:>16}",
        f"{pass2_prob:>16}",
    ]
    if args.soft_threshold is not None:
        pass1_soft_prob = fmt_prob(agg.c1_soft, agg.n1_soft)
        pass2_soft_prob = fmt_prob(agg.c2_soft, agg.n2_soft)
        parts += [
            f"{pass1_soft_prob:>16}",
            f"{pass2_soft_prob:>16}",
            f"{agg.n1_soft:>16d}",
            f"{agg.n2_soft:>16d}",
        ]
    if args.none_right_col:
        none_right_value = none_right_label(
            agg,
            row_inputs.use_soft_for_conditionals,
            args.soft_threshold is not None,
        )
        parts.append(f"{none_right_value:>16}")
        if args.cond_col:
            cond = row_inputs.conditional
            parts += [
                f"{fmt_prob(cond.p1_none_and_p2, cond.p1_none_total):>16}",
                f"{cond.p1_none_total:>16d}",
                f"{cond.p1_none_and_p2:>16d}",
                f"{fmt_prob(cond.p1_any_and_p2, cond.p1_any_total):>16}",
                f"{cond.p1_any_total:>16d}",
                f"{cond.p1_any_and_p2:>16d}",
            ]
    return parts


@dataclass
class DomainSummaryInputs:
    """Inputs required to summarize a single domain."""

    domain_label: str
    dirpath: str
    args: argparse.Namespace
    cond_soft_domains: Set[str]


def summarize_domain(summary_inputs: DomainSummaryInputs) -> None:
    """Summarize a single domain and print per-bin rows."""
    token = label_to_token(summary_inputs.domain_label)
    use_soft_for_conditionals = token in summary_inputs.cond_soft_domains

    bin_config = BinningConfig(
        target_step=summary_inputs.args.target_step,
        split_substr=summary_inputs.args.split,
        recompute_mode=summary_inputs.args.recompute,
        bins=summary_inputs.args.bins,
        combine_mode=summary_inputs.args.combine,
        binning_mode=summary_inputs.args.binning,
        conditional=ConditionalConfig(
            soft_threshold=summary_inputs.args.soft_threshold,
            use_soft_for_conditionals=use_soft_for_conditionals,
        ),
    )

    aggs, edges, skipped_missing = summarize_dir_binned(
        summary_inputs.dirpath,
        bin_config,
    )

    for bin_index, agg in enumerate(aggs):
        (
            ex_p1_none_total,
            ex_p1_none_and_p2,
            ex_p1_any_total,
            ex_p1_any_and_p2,
        ) = compute_example_conditionals(agg)

        row_inputs = BinRowInputs(
            domain_label=summary_inputs.domain_label,
            bin_index=bin_index,
            edges=edges,
            agg=agg,
            conditional=ConditionalCounts(
                p1_none_total=ex_p1_none_total,
                p1_none_and_p2=ex_p1_none_and_p2,
                p1_any_total=ex_p1_any_total,
                p1_any_and_p2=ex_p1_any_and_p2,
            ),
            use_soft_for_conditionals=use_soft_for_conditionals,
        )
        parts = build_row_for_bin(row_inputs, summary_inputs.args)
        print("  ".join(parts))

    if skipped_missing > 0:
        print(
            f"[{summary_inputs.domain_label}] skipped {skipped_missing} rows missing "
            "pass-1 entropy_think or entropy_answer for binning.",
        )


def main() -> None:
    """CLI entrypoint for the entropy-binned summarizer."""
    parser = build_arg_parser()
    args = parser.parse_args()

    cond_soft_domains = parse_cond_soft_domains(args.cond_soft_domains)
    dirs = find_model_dirs(args.results_root, args.temperature)
    if not dirs:
        print(
            f"No matching GRPO-1.5B * temp-{args.temperature:g} "
            "result folders found.",
        )
        return

    header = build_header(args)
    print("  ".join(header))
    print("-" * (sum(len(h) for h in header) + 2 * (len(header) - 1)))

    for domain_label, dirpath in sorted(dirs, key=lambda item: domain_order_key(item[0])):
        summary_inputs = DomainSummaryInputs(
            domain_label=domain_label,
            dirpath=dirpath,
            args=args,
            cond_soft_domains=cond_soft_domains,
        )
        summarize_domain(summary_inputs)


if __name__ == "__main__":
    main()
