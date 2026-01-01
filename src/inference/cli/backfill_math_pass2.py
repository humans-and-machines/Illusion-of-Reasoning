#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill missing pass-2 generations in an existing math JSONL results file.

Why this exists
---------------
The unified math runner writes one JSONL object per (problem, sample_idx).
If a run was done without ``--two_pass`` (or was interrupted before pass-2),
those rows can have ``"pass2": null``. The standard resume logic treats the
row as "present", so it will not regenerate pass 2 automatically.

This script:
  - scans an input JSONL for rows missing pass-2 (default key: ``pass2``),
  - selects a pass-1 trace per problem/group (default sample idx: 0),
  - regenerates pass-2 (and optional pass2a/pass2b/pass2c for multi-cue phrases),
  - rewrites the JSONL with those fields filled (in-place or to a new file).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple


def _parse_kv_filters(items: Sequence[str]) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--filter expects key=value; got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--filter expects key=value; got: {item!r}")
        filters[key] = value
    return filters


def _record_matches_filters(record: Dict[str, Any], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        value = record.get(key)
        if value is None:
            return False
        if str(value) != expected:
            return False
    return True


def _parse_group_by(value: str) -> List[str]:
    fields = [part.strip() for part in (value or "").split(",") if part.strip()]
    if not fields:
        raise ValueError("--group_by must contain at least one field")
    return fields


def _is_missing_pass(record: Dict[str, Any], pass_key: str) -> bool:
    if pass_key not in record:
        return True
    value = record.get(pass_key)
    return value is None or value == {}


def _get_pass1_output(record: Dict[str, Any]) -> Optional[str]:
    pass1_obj = record.get("pass1") or {}
    output = pass1_obj.get("output")
    if isinstance(output, str) and output.strip():
        return output
    return None


def _build_group_key(record: Dict[str, Any], group_fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(record.get(field) for field in group_fields)


def _build_row_key(record: Dict[str, Any], group_fields: Sequence[str]) -> Tuple[Any, ...]:
    group_key = _build_group_key(record, group_fields)
    sample_idx = record.get("sample_idx")
    try:
        sample_idx_i = int(sample_idx) if sample_idx is not None else None
    except (TypeError, ValueError):
        sample_idx_i = None
    return group_key + (sample_idx_i,)


@dataclass(frozen=True)
class _BackfillTask:
    row_key: Tuple[Any, ...]
    group_key: Tuple[Any, ...]
    sample_idx: int
    problem: str
    gold_answer: Any
    pass1_obj: Dict[str, Any]
    prev_output: str


def _select_prev_output(
    pass1_by_group: Dict[Tuple[Any, ...], Dict[int, str]],
    group_key: Tuple[Any, ...],
    preferred_sample_idx: int,
) -> Optional[str]:
    by_sample = pass1_by_group.get(group_key) or {}
    if not by_sample:
        return None
    if preferred_sample_idx in by_sample:
        return by_sample[preferred_sample_idx]
    first_sample = min(by_sample)
    return by_sample[first_sample]


def _merge_generated_passes(
    record: Dict[str, Any],
    updates: Dict[str, Any],
    *,
    force: bool,
) -> Dict[str, Any]:
    out = dict(record)
    for key, value in updates.items():
        if force or _is_missing_pass(out, key):
            out[key] = value
    return out


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True, help="Existing results file to backfill.")
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Where to write the updated JSONL (omit with --inplace).",
    )
    parser.add_argument("--inplace", action="store_true", help="Rewrite input file in place (atomic replace).")
    parser.add_argument("--dry_run", action="store_true", help="Scan and report, but do not generate/write.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing pass2/pass2a/b/c fields.")
    parser.add_argument("--pass_key", default="pass2", help="Pass field to backfill (default: pass2).")
    parser.add_argument(
        "--group_by",
        default="model,suite,temperature,step,split,problem",
        help="Comma-separated fields used to select a shared pass-1 trace for pass-2 conditioning.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Only backfill rows matching key=value (repeatable).",
    )

    # Model load / runtime (mirrors unified runners enough to be usable).
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )

    # Generation knobs.
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--think_cap", type=int, default=750)
    parser.add_argument("--answer_cap", type=int, default=50)
    parser.add_argument("--entropy_mode", choices=["full", "reconsider", "none"], default="reconsider")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--second_pass_phrase", default=None)
    parser.add_argument("--second_pass_use_sample_idx", type=int, default=0)
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help=(
            "Optional: only backfill up to this many distinct `problem` values per run "
            "(useful for chunking long backfills across multiple jobs)."
        ),
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional: only backfill up to this many rows per run (after any --max_problems filtering).",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=None,
        help=(
            "Optional: rewrite the JSONL to disk every N newly-backfilled rows (helps preserve progress on long runs). "
            "Default: only write once at the end."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    from src.inference.backends import HFBackend, _load_torch_and_transformers
    from src.inference.domains.math.math_core import (
        MathInferenceConfig,
        MathInferenceContext,
        MathPassMetaArgs,
        _pack_pass_result,
        chat_base_for_pass2,
        logger,
    )
    from src.inference.utils.common import DEFAULT_SECOND_PASS_PHRASE, build_second_pass_cue_strings, canon_math
    from src.inference.utils.generation import run_generate_batch
    from src.inference.utils.gateway_dataset_utils import iter_jsonl_objects, locked_file
    from src.inference.utils.gateway_utils import configure_tokenizer_and_eos, setup_hf_cache_dir_env

    group_fields = _parse_group_by(args.group_by)
    filters = _parse_kv_filters(args.filter)

    input_path = args.input_jsonl
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if args.flush_every is not None and args.flush_every <= 0:
        raise ValueError("--flush_every must be >= 1 (or omit it).")

    if args.inplace:
        if args.output_jsonl:
            raise ValueError("Use --inplace without --output_jsonl (or omit --inplace).")
        output_path = input_path
    else:
        if not args.output_jsonl:
            raise ValueError("Provide --output_jsonl or use --inplace.")
        output_path = args.output_jsonl

    lock_path = f"{input_path}.backfill_pass2.lock"
    with locked_file(lock_path, "w", lock_type=fcntl.LOCK_EX):
        def _flush_updates_to_disk(*, updates_by_row: Dict[Tuple[Any, ...], Dict[str, Any]]) -> int:
            tmp_path = f"{output_path}.tmp_backfill"
            wrote = 0
            with open(tmp_path, "w", encoding="utf-8") as out_handle:
                for obj in iter_jsonl_objects(input_path):
                    if not isinstance(obj, dict):
                        continue
                    row_key = _build_row_key(obj, group_fields)
                    updates = updates_by_row.get(row_key)
                    if updates:
                        obj = _merge_generated_passes(obj, updates, force=args.force)
                        wrote += 1
                    json.dump(obj, out_handle, ensure_ascii=False)
                    out_handle.write("\n")

            if args.inplace:
                backup_path = f"{input_path}.bak_pass2"
                if not os.path.exists(backup_path):
                    os.replace(input_path, backup_path)
                    os.replace(tmp_path, input_path)
                else:
                    os.replace(tmp_path, input_path)
                logger.info("Flushed %d updated rows → %s (backup: %s)", wrote, input_path, backup_path)
            else:
                os.replace(tmp_path, output_path)
                logger.info("Flushed %d updated rows → %s", wrote, output_path)
            return wrote

        def _gen_pass_for_prefixes(
            *,
            context: MathInferenceContext,
            prefixes: List[str],
            cap: int,
            stop_strings: List[str],
        ) -> Tuple[List[str], List[List[float]], List[str]]:
            torch_mod, _, _, stopping_criteria_list_cls = _load_torch_and_transformers(require_transformers=False)
            decoded, entropies, _input_lengths, _seq, stop_reasons = run_generate_batch(
                prefixes=prefixes,
                cap=cap,
                stop_strings=stop_strings,
                config_like=context.config,
                max_length=4096,
                tokenizer=context.tokenizer,
                model=context.model,
                torch_module=torch_mod,
                stopping_criteria_list_cls=stopping_criteria_list_cls,
            )
            return decoded, entropies, stop_reasons

        def _run_pass2_for_tasks(
            *,
            context: MathInferenceContext,
            tasks: List[_BackfillTask],
            cue_str: str,
        ) -> Tuple[List[str], List[List[float]], List[List[float]], List[str], List[str]]:
            tokenizer = context.tokenizer
            phrase = context.config.second_pass_phrase.strip()
            base_prompts = [
                chat_base_for_pass2(tokenizer, task.problem, task.prev_output, phrase) for task in tasks
            ]
            pre2_think = [base + "<think>\n" + cue_str for base in base_prompts]

            think_new, ent_think, stop_think = _gen_pass_for_prefixes(
                context=context,
                prefixes=pre2_think,
                cap=context.config.think_cap,
                stop_strings=["</think>"],
            )
            pre2_answer = [
                pre + think + "</think>\n<answer>\n" for pre, think in zip(pre2_think, think_new)
            ]
            answer_new, ent_answer, stop_answer = _gen_pass_for_prefixes(
                context=context,
                prefixes=pre2_answer,
                cap=context.config.answer_cap,
                stop_strings=["</answer>"],
            )
            full_texts = [
                f"<think>{cue_str}{think}</think>\n<answer>{answer}</answer>"
                for think, answer in zip(think_new, answer_new)
            ]
            return full_texts, ent_think, ent_answer, stop_think, stop_answer

        def _pack_pass2_results(
            *,
            tasks: List[_BackfillTask],
            cue_str: str,
            full_texts: List[str],
            ent_think: List[List[float]],
            ent_answer: List[List[float]],
            stop_think: List[str],
            stop_answer: List[str],
        ) -> List[Dict[str, Any]]:
            packed: List[Dict[str, Any]] = []
            for idx, task in enumerate(tasks):
                canon_gold = canon_math(task.gold_answer) if isinstance(task.gold_answer, str) else None
                res = _pack_pass_result(
                    full_text=full_texts[idx],
                    ent_think=ent_think[idx],
                    ent_answer=ent_answer[idx],
                    meta_args=MathPassMetaArgs(
                        problem=task.problem,
                        canon_gold=canon_gold,
                        injected_cue=True,
                        prev_output=task.prev_output,
                        cue_prefix_str=cue_str,
                        stop_reason_think=stop_think[idx],
                        stop_reason_answer=stop_answer[idx],
                    ),
                )
                res["improved_over_pass1"] = bool(res.get("is_correct_pred")) and not bool(
                    task.pass1_obj.get("is_correct_pred")
                )
                packed.append(res)
            return packed

        def _backfill_for_rows(
            *,
            context: MathInferenceContext,
            tasks: List[_BackfillTask],
            cue_strs: List[str],
        ) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
            if not cue_strs:
                raise ValueError("No cue strings resolved from --second_pass_phrase")

            extra_by_row: Dict[Tuple[Any, ...], Dict[str, Any]] = {task.row_key: {} for task in tasks}
            packed_by_row_and_cue: List[Tuple[str, List[Dict[str, Any]]]] = []

            for cue_str in cue_strs:
                full_texts, ent_think, ent_answer, stop_think, stop_answer = _run_pass2_for_tasks(
                    context=context,
                    tasks=tasks,
                    cue_str=cue_str,
                )
                packed = _pack_pass2_results(
                    tasks=tasks,
                    cue_str=cue_str,
                    full_texts=full_texts,
                    ent_think=ent_think,
                    ent_answer=ent_answer,
                    stop_think=stop_think,
                    stop_answer=stop_answer,
                )
                packed_by_row_and_cue.append((cue_str, packed))

            last_idx = len(packed_by_row_and_cue) - 1
            extra_names = ("pass2a", "pass2b", "pass2c")
            for cue_idx, (_cue, packed_rows) in enumerate(packed_by_row_and_cue):
                if cue_idx == last_idx:
                    for task, packed in zip(tasks, packed_rows):
                        extra_by_row[task.row_key]["pass2"] = packed
                else:
                    name = extra_names[cue_idx] if cue_idx < len(extra_names) else None
                    if name:
                        for task, packed in zip(tasks, packed_rows):
                            extra_by_row[task.row_key][name] = packed

            if len(cue_strs) >= 3:
                for task in tasks:
                    extra_by_row[task.row_key].setdefault("pass2c", extra_by_row[task.row_key].get("pass2"))

            return extra_by_row

        # 1) Scan: collect pass1 outputs per group/sample and rows needing pass2.
        pass1_by_group: DefaultDict[Tuple[Any, ...], Dict[int, str]] = defaultdict(dict)
        todo_min: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        scanned = 0
        todo = 0
        skipped_no_pass1 = 0

        for obj in iter_jsonl_objects(input_path):
            if not isinstance(obj, dict):
                continue
            scanned += 1

            group_key = _build_group_key(obj, group_fields)
            sample_idx = obj.get("sample_idx")
            try:
                sample_idx_i = int(sample_idx) if sample_idx is not None else None
            except (TypeError, ValueError):
                sample_idx_i = None

            pass1_out = _get_pass1_output(obj)
            if pass1_out is not None and sample_idx_i is not None:
                pass1_by_group[group_key][sample_idx_i] = pass1_out

            if not _record_matches_filters(obj, filters):
                continue
            if not _is_missing_pass(obj, args.pass_key) and not args.force:
                continue

            if sample_idx_i is None or not isinstance(obj.get("problem"), str):
                continue

            if pass1_out is None:
                skipped_no_pass1 += 1
                continue

            row_key = _build_row_key(obj, group_fields)
            todo_min[row_key] = {
                "group_key": group_key,
                "sample_idx": sample_idx_i,
                "problem": obj["problem"],
                "gold_answer": obj.get("gold_answer"),
                "pass1_obj": obj.get("pass1") or {},
            }
            todo += 1

        def _select_row_keys_for_limits() -> List[Tuple[Any, ...]]:
            if args.max_problems is None and args.max_rows is None:
                return list(todo_min.keys())

            selected_problems: Optional[set] = None
            if args.max_problems is not None:
                if args.max_problems <= 0:
                    return []
                selected_problems = set()
                for item in todo_min.values():
                    prob = item.get("problem")
                    if prob not in selected_problems:
                        if len(selected_problems) >= args.max_problems:
                            break
                        selected_problems.add(prob)

            selected: List[Tuple[Any, ...]] = []
            for row_key, item in todo_min.items():
                if selected_problems is not None and item.get("problem") not in selected_problems:
                    continue
                selected.append(row_key)
                if args.max_rows is not None and len(selected) >= args.max_rows:
                    break
            return selected

        selected_row_keys = _select_row_keys_for_limits()
        if todo > 0 and not selected_row_keys:
            print(
                f"No rows selected for backfill in {input_path} "
                f"(todo={todo}, max_problems={args.max_problems}, max_rows={args.max_rows})."
            )
            return
        if args.max_problems is not None or args.max_rows is not None:
            todo_selected = len(selected_row_keys)
            logger.info(
                "Scan complete: scanned=%d todo=%d selected=%d skipped_no_pass1=%d",
                scanned,
                todo,
                todo_selected,
                skipped_no_pass1,
            )
        else:
            logger.info("Scan complete: scanned=%d todo=%d skipped_no_pass1=%d", scanned, todo, skipped_no_pass1)
        if todo == 0:
            if args.inplace:
                print(f"No rows to backfill in {input_path}.")
            else:
                # If user asked for a new file, just copy through unchanged.
                with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
                    for line in src:
                        dst.write(line)
                print(f"No rows to backfill; copied → {output_path}")
            return

        if args.dry_run:
            if args.max_problems is not None or args.max_rows is not None:
                print(
                    f"[dry-run] Would backfill {len(selected_row_keys)}/{todo} rows in {input_path} "
                    f"(max_problems={args.max_problems}, max_rows={args.max_rows})."
                )
            else:
                print(f"[dry-run] Would backfill {todo} rows in {input_path}.")
            return

        # 2) Initialize backend once (assumes the file corresponds to this model).
        hf_cache_dir = setup_hf_cache_dir_env("./.hf_cache")
        backend = HFBackend.from_pretrained(
            args.model_name_or_path,
            revision=args.revision,
            cache_dir=hf_cache_dir,
            dtype=args.dtype,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            tokenizer_path=args.tokenizer_path,
        )
        eos_ids = configure_tokenizer_and_eos(backend.tokenizer, extra_tokens=["<|im_end|>", "<|endoftext|>"])

        second_pass_phrase = args.second_pass_phrase or DEFAULT_SECOND_PASS_PHRASE
        cue_strs = build_second_pass_cue_strings(second_pass_phrase)

        cfg = MathInferenceConfig(
            split_name="backfill",
            output_dir=".",
            step=0,
            batch_size=args.batch_size,
            num_samples=1,
            temperature=args.temperature,
            top_p=args.top_p,
            entropy_mode=args.entropy_mode,
            eos_ids=eos_ids,
            two_pass=True,
            second_pass_phrase=second_pass_phrase,
            second_pass_use_sample_idx=args.second_pass_use_sample_idx,
            think_cap=args.think_cap,
            answer_cap=args.answer_cap,
        )
        context = MathInferenceContext(tokenizer=backend.tokenizer, model=backend.model, config=cfg)

        torch_mod, _, _, _ = _load_torch_and_transformers(require_transformers=False)
        random.seed(args.seed)
        if hasattr(torch_mod, "manual_seed"):
            torch_mod.manual_seed(args.seed)

        # 3) Build tasks with chosen prev_output.
        tasks: List[_BackfillTask] = []
        skipped_no_prev = 0
        for row_key in selected_row_keys:
            item = todo_min[row_key]
            group_key = item["group_key"]
            prev_output = _select_prev_output(
                pass1_by_group=pass1_by_group,
                group_key=group_key,
                preferred_sample_idx=args.second_pass_use_sample_idx,
            )
            if not prev_output:
                skipped_no_prev += 1
                continue
            tasks.append(
                _BackfillTask(
                    row_key=row_key,
                    group_key=group_key,
                    sample_idx=item["sample_idx"],
                    problem=item["problem"],
                    gold_answer=item["gold_answer"],
                    pass1_obj=item["pass1_obj"],
                    prev_output=prev_output,
                )
            )
        if skipped_no_prev:
            logger.warning("Skipping %d rows with no available prev_output in-group.", skipped_no_prev)

        # 4) Generate in batches; store packed results keyed by row_key.
        updates_by_row: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        processed = 0
        last_flushed_processed = 0
        next_flush_at = args.flush_every if args.flush_every is not None else None
        total = len(tasks)
        for start in range(0, total, args.batch_size):
            batch = tasks[start : start + args.batch_size]
            logger.info("Backfill batch %d/%d (%d rows).", (start // args.batch_size) + 1, (total + args.batch_size - 1) // args.batch_size, len(batch))
            updates_by_row.update(
                _backfill_for_rows(
                    context=context,
                    tasks=batch,
                    cue_strs=cue_strs,
                )
            )
            processed += len(batch)

            if next_flush_at is not None and processed >= next_flush_at and processed < total:
                _flush_updates_to_disk(updates_by_row=updates_by_row)
                last_flushed_processed = processed
                next_flush_at += args.flush_every

        # 5) Final rewrite (atomic if --inplace).
        # If we already flushed exactly at the end, avoid an extra rewrite.
        if processed != last_flushed_processed:
            wrote = _flush_updates_to_disk(updates_by_row=updates_by_row)
        else:
            wrote = len(updates_by_row)

        if args.inplace:
            backup_path = f"{input_path}.bak_pass2"
            print(f"Backfilled {wrote} rows → {input_path} (backup: {backup_path})")
        else:
            print(f"Backfilled {wrote} rows → {output_path}")


if __name__ == "__main__":
    main()
