#!/usr/bin/env python3
"""
Plan resume/fill runs for missing MATH-500 questions in existing result JSONLs.

What it does
------------
- Scans each `step????_test.jsonl` under `artifacts/results/*math-temp-*`.
- Figures out which dataset question indices (0..499) are missing any samples
  (based on the "problem" field + sample_idx coverage).
- Groups missing indices into fixed-size chunks (default: 5 questions per chunk).
- Writes a manifest (one line per chunk) consumable by the Slurm array runner:
    scripts/inference/math-fill-missing-questions-array.slurm
- Prints the `sbatch --array=...%<cap>` command to run the remaining work.

Notes
-----
- This is MATH-specific because the per-row JSONL does not store `example_id`;
  we map `problem` -> dataset index by loading MATH-500 (cached via datasets).
- If you do not have MATH-500 cached, pass `--dataset_path` pointing to a local
  JSON/JSONL file with `problem`+`answer` fields.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class FilePlan:
    family: str
    temp: str
    step: int
    output_dir: str
    jsonl_path: str
    missing_indices: List[int]
    missing_chunks: List[int]
    unknown_problems: int


def _iter_math_result_files(results_root: Path) -> Iterable[Path]:
    patterns = (
        "GRPO-1.5B-math-temp-*",
        "GRPO-7B-math-temp-*",
        "GRPO-Llama8B-math-temp-*",
    )
    for pat in patterns:
        for temp_root in sorted(results_root.glob(pat)):
            if not temp_root.is_dir():
                continue
            # Only scan the canonical per-step outputs:
            #   <temp_root>/<step_dir>/step????_test.jsonl
            # and avoid deeper chunk/probem folders like:
            #   <temp_root>/step-450/problem-0480/step0450_test.jsonl
            yield from sorted(temp_root.glob("*/step????_test.jsonl"))
            yield from sorted(temp_root.glob("step????_test.jsonl"))


def _family_and_temp_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Find a parent directory like GRPO-<family>-math-temp-<temp>
    for parent in [path] + list(path.parents):
        name = parent.name
        if name.startswith("GRPO-1.5B-math-temp-"):
            return "1.5B", name.split("temp-", 1)[-1]
        if name.startswith("GRPO-7B-math-temp-"):
            return "7B", name.split("temp-", 1)[-1]
        if name.startswith("GRPO-Llama8B-math-temp-"):
            return "Llama8B", name.split("temp-", 1)[-1]
    return None, None


def _parse_step_from_filename(name: str) -> Optional[int]:
    # step####_test.jsonl
    if not (name.startswith("step") and "_test.jsonl" in name):
        return None
    step_str = name[4:8]
    if not step_str.isdigit():
        return None
    return int(step_str)


def _require_datasets():
    try:
        import datasets  # type: ignore

        return datasets
    except ImportError as exc:
        raise RuntimeError("This planner requires `datasets` (pip install datasets).") from exc


def _extract_problem(example: dict) -> Optional[str]:
    for key in ("problem", "question", "prompt", "instruction", "query"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_gold(example: dict) -> Optional[str]:
    for key in ("gold_answer", "answer", "solution", "final_answer", "target", "boxed_answer"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _scan_reference_dataset_from_jsonl(reference_jsonl: Path) -> List[dict]:
    """
    Build a local MATH-like dataset list from an existing results JSONL.

    Uses unique `problem` and the first observed `gold_answer` for that problem.
    """
    records: List[dict] = []
    seen: Set[str] = set()
    with reference_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            prob = obj.get("problem")
            if not isinstance(prob, str) or not prob:
                continue
            if prob in seen:
                continue
            gold = _extract_gold(obj) or ""
            records.append({"problem": prob, "answer": gold})
            seen.add(prob)
    return records


def _choose_reference_jsonl(
    *,
    results_root: Path,
    total_examples: int,
    reference_jsonl: Optional[str],
) -> Path:
    if reference_jsonl:
        path = Path(reference_jsonl)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    best_path: Optional[Path] = None
    best_unique = -1
    for candidate in _iter_math_result_files(results_root):
        # Fast scan: count distinct problem strings (stop once we hit target).
        uniq: Set[str] = set()
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get("problem"), str):
                        uniq.add(obj["problem"])
                        if len(uniq) >= total_examples:
                            break
        except OSError:
            continue
        if len(uniq) > best_unique:
            best_unique = len(uniq)
            best_path = candidate
        if best_unique >= total_examples:
            break

    if best_path is None:
        raise RuntimeError("Could not find any math results JSONL to use as a reference dataset.")
    if best_unique < total_examples:
        raise RuntimeError(
            f"Reference search found at most {best_unique} unique problems; need {total_examples}. "
            f"Provide --reference_jsonl pointing to a complete file."
        )
    return best_path


def _scan_seen_samples(path: Path) -> Tuple[DefaultDict[str, Set[int]], int]:
    seen: DefaultDict[str, Set[int]] = defaultdict(set)
    unknown_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            prob = obj.get("problem")
            sidx = obj.get("sample_idx")
            if not isinstance(prob, str):
                unknown_lines += 1
                continue
            try:
                sidx_i = int(sidx)
            except (TypeError, ValueError):
                unknown_lines += 1
                continue
            seen[prob].add(sidx_i)
    return seen, unknown_lines


def _chunks_for_indices(indices: Sequence[int], chunk_size: int) -> List[int]:
    return sorted({idx // chunk_size for idx in indices})


def _write_manifest(out_path: Path, plans: Sequence[FilePlan], chunk_size: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            for chunk_idx in plan.missing_chunks:
                dataset_start = chunk_idx * chunk_size
                num_examples = chunk_size
                handle.write(
                    f"{plan.family}\t{plan.temp}\t{plan.step}\t{plan.output_dir}\t"
                    f"{dataset_start}\t{num_examples}\n"
                )
                n += 1
    return n


def _parse_steps_filter(values: Sequence[str]) -> Optional[Set[int]]:
    if not values:
        return None
    out: Set[int] = set()
    for v in values:
        for part in v.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="artifacts/results")
    parser.add_argument("--split", default="test")
    parser.add_argument("--total_examples", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument(
        "--reference_jsonl",
        default=None,
        help="Optional: a known-complete results JSONL to define the question set/order.",
    )
    parser.add_argument(
        "--dataset_out",
        default="tmp/math_reference_dataset.jsonl",
        help="Where to write the local dataset JSONL derived from reference_jsonl.",
    )
    parser.add_argument("--only_family", default=None, choices=[None, "1.5B", "7B", "Llama8B"])
    parser.add_argument("--only_temp", default=None, help="E.g. 0.3")
    parser.add_argument("--only_steps", action="append", default=[], help="E.g. 0,50,100 or 0-500")
    parser.add_argument("--out_manifest", default="tmp/math_fill_missing_questions_manifest.tsv")
    parser.add_argument("--print_missing", action="store_true", help="Print missing index lists per file.")
    parser.add_argument("--max_print", type=int, default=30, help="Max missing indices to print per file.")
    parser.add_argument("--array_cap", type=int, default=2000, help="Concurrency cap for sbatch --array (default: 2000).")
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(results_root)

    steps_filter = _parse_steps_filter(args.only_steps)

    # Define the question set/order from an existing complete results JSONL,
    # then write a local dataset file so Slurm jobs can run offline.
    ref_jsonl = _choose_reference_jsonl(
        results_root=results_root,
        total_examples=args.total_examples,
        reference_jsonl=args.reference_jsonl,
    )
    ref_records = _scan_reference_dataset_from_jsonl(ref_jsonl)
    if len(ref_records) < args.total_examples:
        raise RuntimeError(
            f"Reference {ref_jsonl} yielded {len(ref_records)} unique problems; need {args.total_examples}."
        )
    ref_records = ref_records[: args.total_examples]

    dataset_out = Path(args.dataset_out)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with dataset_out.open("w", encoding="utf-8") as handle:
        for rec in ref_records:
            json.dump(rec, handle, ensure_ascii=False)
            handle.write("\n")

    problems = [rec["problem"] for rec in ref_records]
    problem_to_index: Dict[str, int] = {p: i for i, p in enumerate(problems)}
    expected_samples = set(range(args.num_samples))

    plans: List[FilePlan] = []
    for jsonl in _iter_math_result_files(results_root):
        family, temp = _family_and_temp_from_path(jsonl)
        if family is None or temp is None:
            continue
        if args.only_family and family != args.only_family:
            continue
        if args.only_temp and temp != args.only_temp:
            continue

        step = _parse_step_from_filename(jsonl.name)
        if step is None:
            continue
        if steps_filter is not None and step not in steps_filter:
            continue

        seen_samples_by_problem, unknown_lines = _scan_seen_samples(jsonl)
        missing_indices: List[int] = []
        unknown_problems = 0
        for prob, idx in problem_to_index.items():
            have = seen_samples_by_problem.get(prob)
            if not have:
                missing_indices.append(idx)
            else:
                if not expected_samples.issubset(have):
                    missing_indices.append(idx)

        # Count rows whose problem isn't in the dataset mapping (helpful debugging).
        for prob in seen_samples_by_problem.keys():
            if prob not in problem_to_index:
                unknown_problems += 1

        missing_chunks = _chunks_for_indices(missing_indices, args.chunk_size)
        if not missing_chunks:
            continue

        plans.append(
            FilePlan(
                family=family,
                temp=temp,
                step=step,
                output_dir=jsonl.parent.as_posix(),
                jsonl_path=jsonl.as_posix(),
                missing_indices=missing_indices,
                missing_chunks=missing_chunks,
                unknown_problems=unknown_problems + unknown_lines,
            )
        )

    plans.sort(key=lambda p: (p.family, float(p.temp), p.step, p.jsonl_path))

    manifest_path = Path(args.out_manifest)
    # Manifest format consumed by the Slurm array runner:
    # family temp step output_dir dataset_start num_examples dataset_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    n_tasks = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            for chunk_idx in plan.missing_chunks:
                dataset_start = chunk_idx * args.chunk_size
                num_examples = args.chunk_size
                handle.write(
                    f"{plan.family}\t{plan.temp}\t{plan.step}\t{plan.output_dir}\t"
                    f"{dataset_start}\t{num_examples}\t{dataset_out.as_posix()}\n"
                )
                n_tasks += 1

    if not plans:
        print("No missing questions found.")
        return

    print(f"Planned {len(plans)} files with missing questions.")
    print(f"Manifest tasks: {n_tasks} (one chunk per task) → {manifest_path}")
    print(f"Reference dataset: {dataset_out} (from {ref_jsonl})")
    print()
    for plan in plans:
        miss_n = len(plan.missing_indices)
        chunks_n = len(plan.missing_chunks)
        extra = f", unknown_problems={plan.unknown_problems}" if plan.unknown_problems else ""
        print(f"- {plan.jsonl_path}: missing_questions={miss_n}, missing_chunks={chunks_n}{extra}")
        if args.print_missing:
            sample = plan.missing_indices[: args.max_print]
            suffix = " ..." if len(plan.missing_indices) > len(sample) else ""
            print(f"  missing_idx: {sample}{suffix}")
    print()

    if n_tasks <= 0:
        print("No array tasks to run (manifest empty).")
        return

    array_max = n_tasks - 1
    cap = max(1, int(args.array_cap))
    print("Run:")
    print(
        f"  sbatch --array=0-{array_max}%{cap} "
        f"scripts/inference/math-fill-missing-questions-array.slurm "
        f"MANIFEST={manifest_path}"
    )


if __name__ == "__main__":
    main()
