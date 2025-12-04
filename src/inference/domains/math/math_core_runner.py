#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for the two-pass math inference driver.

These utilities are split out from :mod:`src.inference.domains.math.math_core` to keep
that module within linting size limits while preserving the public API.

New code should continue to import :func:`run_inference_on_split` and
:func:`load_math500` from :mod:`src.inference.domains.math.math_core`.
"""

from __future__ import annotations

import fcntl
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.inference.domains.math.math_core import (
    BatchLayout,
    BatchWriteContext,
    ExistingPassState,
    ExtraPassRowContext,
    MathInferenceConfig,
    MathInferenceContext,
    MathPassMetaArgs,
    SecondPassInputs,
    TwoPassBatchOutputs,
    _build_first_pass_choice,
    _build_work_items_for_slice,
    _pack_pass_result,
    _run_pass1_for_batch,
    _run_pass2_for_batch,
    logger,
)
from src.inference.utils.common import PassOutputs, build_second_pass_cue_strings
from src.inference.utils.common import canon_math as _canon_math
from src.inference.utils.common import (
    extract_problem_and_answer,
    load_local_json_dataset,
    locked_file,
    require_datasets,
    scan_existing_pass1_results,
)
from src.inference.utils.generation import build_extra_pass_results_for_cues


def _build_extra_pass_results_for_row(
    *,
    row_index: int,
    row_context: ExtraPassRowContext,
) -> Dict[str, Dict[str, Any]]:
    """
    Pack optional multi-cue reconsideration results for a single row.

    :param row_index: Index of the row within the batch outputs.
    :param row_context: Inputs needed to pack the extra pass outputs.
    :returns: Mapping from cue tags (for example, ``\"pass2a\"``) to packed
        result dictionaries for each extra pass.
    """

    def _pack_extra_result(
        cue_str_extra: str,
        outputs_extra: PassOutputs,
    ) -> Dict[str, Any]:
        extra_res = _pack_pass_result(
            full_text=outputs_extra.full_texts[row_index],
            ent_think=outputs_extra.ent_think[row_index],
            ent_answer=outputs_extra.ent_answer[row_index],
            meta_args=MathPassMetaArgs(
                problem=row_context.prob,
                canon_gold=row_context.canon_gold,
                injected_cue=True,
                prev_output=row_context.context.firstpass_choice_text_per_ex[
                    row_context.layout.row_to_ex_idx[row_index]
                ],
                cue_prefix_str=cue_str_extra,
                stop_reason_think=outputs_extra.stop_reason_think[row_index],
                stop_reason_answer=outputs_extra.stop_reason_answer[row_index],
            ),
        )
        extra_res["improved_over_pass1"] = bool(
            extra_res.get("is_correct_pred"),
        ) and not bool(
            row_context.pass1_result.get("is_correct_pred"),
        )
        return extra_res

    return build_extra_pass_results_for_cues(
        two_pass=row_context.context.config.two_pass,
        extra_passes=row_context.extra_passes,
        pack_result_for_extra=_pack_extra_result,
    )


def _write_results_for_batch(
    *,
    layout: BatchLayout,
    outputs: TwoPassBatchOutputs,
    context: BatchWriteContext,
    extra_passes: Optional[List[Tuple[str, PassOutputs]]] = None,
) -> None:
    """
    Write pass-1 and optional pass-2 results for a batch to JSONL.

    :param layout: Batch layout relating rows to examples and samples.
    :param outputs: Combined pass-1 and pass-2 outputs for the batch.
    :param context: Batch write context describing output path and cues.
    :param extra_passes: Optional extra reconsideration passes to pack and write.
    :returns: ``None``. Results are appended to the JSONL file on disk.
    """
    with locked_file(context.outpath, "a", lock_type=fcntl.LOCK_EX) as outfile:
        for row_index, full_row in enumerate(outputs.pass1.full_texts):
            sample_idx = layout.row_target_sample_idx[row_index]
            item = layout.work_items[layout.row_to_ex_idx[row_index]]

            pass1_result = _pack_pass_result(
                full_text=full_row,
                ent_think=outputs.pass1.ent_think[row_index],
                ent_answer=outputs.pass1.ent_answer[row_index],
                meta_args=MathPassMetaArgs(
                    problem=item["_normalized_problem"],
                    canon_gold=_canon_math(item["_normalized_gold"]),
                    injected_cue=False,
                    prev_output=None,
                    cue_prefix_str="",
                    stop_reason_think=outputs.pass1.stop_reason_think[row_index],
                    stop_reason_answer=outputs.pass1.stop_reason_answer[row_index],
                ),
            )

            pass2_result = None
            if context.config.two_pass and outputs.pass2 is not None:
                pass2_result = _pack_pass_result(
                    full_text=outputs.pass2.full_texts[row_index],
                    ent_think=outputs.pass2.ent_think[row_index],
                    ent_answer=outputs.pass2.ent_answer[row_index],
                    meta_args=MathPassMetaArgs(
                        problem=item["_normalized_problem"],
                        canon_gold=_canon_math(item["_normalized_gold"]),
                        injected_cue=True,
                        prev_output=context.firstpass_choice_text_per_ex[layout.row_to_ex_idx[row_index]],
                        cue_prefix_str=context.cue_strs[-1] if context.cue_strs else "",
                        stop_reason_think=outputs.pass2.stop_reason_think[row_index],
                        stop_reason_answer=outputs.pass2.stop_reason_answer[row_index],
                    ),
                )
                pass2_result["improved_over_pass1"] = bool(
                    pass2_result.get("is_correct_pred"),
                ) and not bool(
                    pass1_result.get("is_correct_pred"),
                )

            extra_pass_results: Dict[str, Dict[str, Any]] = _build_extra_pass_results_for_row(
                row_index=row_index,
                row_context=ExtraPassRowContext(
                    prob=item["_normalized_problem"],
                    canon_gold=_canon_math(item["_normalized_gold"]),
                    layout=layout,
                    context=context,
                    extra_passes=extra_passes,
                    pass1_result=pass1_result,
                ),
            )

            # For convenience, expose the main pass2 as pass2c when we have ≥3 cues.
            if context.config.two_pass and outputs.pass2 is not None and len(context.cue_strs) >= 3:
                extra_pass_results.setdefault("pass2c", pass2_result)

            row_dict: Dict[str, Any] = {
                "problem": item["_normalized_problem"],
                "gold_answer": item["_normalized_gold"],
                "gold_answer_canon": _canon_math(item["_normalized_gold"]),
                "step": context.config.step,
                "split": context.config.split_name,
                "sample_idx": sample_idx,
                "pass1": pass1_result,
                "pass2": pass2_result,
            }
            for extra_key in ("pass2a", "pass2b", "pass2c"):
                if extra_pass_results.get(extra_key):
                    row_dict[extra_key] = extra_pass_results[extra_key]

            json.dump(row_dict, outfile, ensure_ascii=False)
            outfile.write("\n")

            context.existing_state.existing_samples.setdefault(
                item["_normalized_problem"],
                set(),
            ).add(sample_idx)
            context.existing_state.existing_pass1[(item["_normalized_problem"], sample_idx)] = pass1_result["output"]

    filled = sum(len(item["_todo_samples"]) for item in layout.work_items)
    logger.info(
        "Filled %d missing samples across %d problems in this batch.",
        filled,
        len(layout.work_items),
    )


def _select_examples(examples, start: int, end: int):
    """Return a slice of examples even when given a plain list."""
    if hasattr(examples, "select"):
        return examples.select(range(start, end))
    return list(examples[start:end])


def _compute_second_pass_outputs(
    *,
    context: MathInferenceContext,
    layout: BatchLayout,
    pre1_think: List[str],
    firstpass_choice_text_per_ex: List[str],
) -> Tuple[Optional[PassOutputs], Optional[List[Tuple[str, PassOutputs]]], List[str]]:
    """
    Run optional second-pass generations for one or more reconsideration cues.

    The function may produce multiple reconsideration passes based on the
    configured cue strings, designating the last pass as the “main” second
    pass and treating earlier ones as auxiliary.

    :param context: Generation context containing tokenizer, model, and config.
    :param layout: Batch layout relating rows to examples and samples.
    :param pre1_think: First-pass think-prefix prompts for each row.
    :param firstpass_choice_text_per_ex: Chosen pass-1 full text per example.
    :returns: A tuple ``(main_pass2, extra_passes, cue_strs)`` where
        ``main_pass2`` is the final second-pass outputs or ``None``,
        ``extra_passes`` is an optional list of preceding ``(cue_str, PassOutputs)``
        pairs, and ``cue_strs`` is the list of cue strings used.
    """
    cue_strs = build_second_pass_cue_strings(context.config.second_pass_phrase)

    if not (context.config.two_pass and cue_strs):
        return None, None, cue_strs

    extra_passes: List[Tuple[str, PassOutputs]] = []
    main_pass2: Optional[PassOutputs] = None
    for idx, cue_str in enumerate(cue_strs):
        outputs_i = _run_pass2_for_batch(
            context=context,
            second_pass_inputs=SecondPassInputs(
                layout=layout,
                pre1_think=pre1_think,
                firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
                cue_str=cue_str,
            ),
        )
        if idx == len(cue_strs) - 1:
            main_pass2 = outputs_i
        else:
            extra_passes.append((cue_str, outputs_i))

    return main_pass2, extra_passes, cue_strs


def _run_inference_batch(
    *,
    slice_ds,
    context: MathInferenceContext,
    outpath: str,
    existing_state: ExistingPassState,
) -> None:
    """
    Run inference for a single dataset slice and append JSONL rows.

    This function builds work items for the slice, runs first and optional
    second passes, and appends packed results to the JSONL file at
    ``outpath`` while updating ``existing_state``.

    :param slice_ds: Dataset slice object supporting ``select``-style access.
    :param context: Generation context containing tokenizer, model, and config.
    :param outpath: Path to the JSONL file where results are written.
    :param existing_state: Mutable state tracking existing pass-1 samples.
    :returns: ``None``. Results are written to disk and state is updated in place.
    """
    work_items = _build_work_items_for_slice(
        slice_ds,
        existing_state.existing_samples,
        context.config,
    )
    if not work_items:
        return

    pass1_outputs, layout, pre1_think = _run_pass1_for_batch(work_items, context)

    firstpass_choice_text_per_ex = _build_first_pass_choice(
        layout=layout,
        pass1_full_texts=pass1_outputs.full_texts,
        existing_state=existing_state,
        config=context.config,
    )

    main_pass2, extra_passes, cue_strs = _compute_second_pass_outputs(
        context=context,
        layout=layout,
        pre1_think=pre1_think,
        firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
    )

    write_context = BatchWriteContext(
        outpath=outpath,
        config=context.config,
        cue_strs=cue_strs,
        existing_state=existing_state,
        firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
    )
    _write_results_for_batch(
        layout=layout,
        outputs=TwoPassBatchOutputs(
            pass1=pass1_outputs,
            pass2=main_pass2 if context.config.two_pass else None,
        ),
        context=write_context,
        extra_passes=extra_passes if extra_passes else None,
    )


# Public aliases used by thin wrappers in :mod:`math_core`.
build_extra_pass_results_for_row = _build_extra_pass_results_for_row
write_results_for_batch = _write_results_for_batch
compute_second_pass_outputs = _compute_second_pass_outputs
run_inference_batch = _run_inference_batch


def run_inference_on_split(
    examples,  # datasets.Dataset
    tokenizer,
    model,
    config: MathInferenceConfig,
) -> None:
    """
    Run math inference over a dataset split.

    The function respects existing results on disk and only generates new
    samples for problems and sample indices that have not yet been filled,
    up to ``config.num_samples`` per problem.

    :param examples: Dataset object containing math problems and answers.
    :param tokenizer: Tokenizer compatible with the underlying language model.
    :param model: Causal language model used to generate solutions.
    :param config: Inference configuration controlling sampling and second-pass behavior.
    :returns: ``None``. Results are appended to a JSONL file under ``config.output_dir``.
    """
    if (config.temperature is None or float(config.temperature) == 0.0) and config.num_samples > 1:
        logger.warning(
            "temperature=0 with num_samples=%d → all samples will be identical (greedy).",
            config.num_samples,
        )

    outpath = os.path.join(
        config.output_dir,
        f"step{config.step:04d}_{config.split_name}.jsonl",
    )
    existing_samples, existing_pass1 = scan_existing_pass1_results(outpath)
    logger.info("Resume scan: %d problems already present", len(existing_samples))

    total_examples = len(examples)
    logger.info(
        "Starting inference on %d examples (batch_size=%d, num_samples=%d, two_pass=%s).",
        total_examples,
        config.batch_size,
        config.num_samples,
        bool(config.two_pass),
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    existing_state = ExistingPassState(
        existing_samples=existing_samples,
        existing_pass1=existing_pass1,
    )
    context = MathInferenceContext(
        tokenizer=tokenizer,
        model=model,
        config=config,
    )

    total_batches = (total_examples + config.batch_size - 1) // config.batch_size

    for start_idx in range(0, total_examples, config.batch_size):
        end_idx = min(start_idx + config.batch_size, total_examples)
        batch_index = start_idx // config.batch_size
        logger.info(
            "Processing batch %d/%d (examples %d–%d).",
            batch_index + 1,
            total_batches,
            start_idx,
            end_idx - 1,
        )
        slice_ds = _select_examples(examples, start_idx, end_idx)
        _run_inference_batch(
            slice_ds=slice_ds,
            context=context,
            outpath=outpath,
            existing_state=existing_state,
        )


def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    """
    Load MATH-500 (or a competition-math fallback) and normalize fields.

    The function first tries a sequence of MATH-500 repositories. If all of
    them fail, it falls back to a shuffled subset of ``hendrycks/competition_math``.

    :param cache_dir: Directory to use as a datasets cache.
    :param split: Dataset split name (for example, ``\"train\"`` or ``\"test\"``).
    :param seed: Random seed used when sub-sampling the fallback dataset.
    :param dataset_path: Optional local JSON file to load instead of remote datasets.
    :returns: A datasets-like object exposing ``map``/``filter`` and ``select``.
    :raises RuntimeError: If neither MATH-500 nor the competition-math fallback can be loaded.
    """
    _, load_dataset = require_datasets()

    if dataset_path:
        logger.info("Loading MATH-500 from local file: %s", dataset_path)
        return load_local_json_dataset(dataset_path)

    candidates = [
        "HuggingFaceH4/MATH-500",
        "AI-MO/MATH-500",
        "lighteval/MATH-500",
        "openai/math-500",
        "TIGER-Lab/MATH-500",
    ]
    for repo in candidates:
        try:
            logger.info("Trying remote MATH-500 candidate: %s", repo)
            ds_full = load_dataset(repo, split=split, cache_dir=cache_dir)
            # Some test stubs return plain lists; accept them directly.
            if not hasattr(ds_full, "column_names"):
                logger.info("Loaded MATH-500 from %s | N=%d", repo, len(ds_full))
                return ds_full

            colnames = set(ds_full.column_names)

            def _norm(example):
                problem, answer = extract_problem_and_answer(example)
                return {"problem": problem, "answer": answer}

            normalized_ds = ds_full.map(_norm, remove_columns=list(colnames))
            normalized_ds = normalized_ds.filter(
                lambda row: row["problem"] is not None and row["answer"] is not None,
            )
            if len(normalized_ds) == 0:
                raise ValueError(f"{repo} contained no usable (problem,answer) pairs")
            logger.info("Loaded MATH-500 from %s | N=%d", repo, len(normalized_ds))
            return normalized_ds
        except (OSError, ValueError, RuntimeError) as load_exc:
            logger.warning("Skipping %s (%r)", repo, load_exc)

    try:
        ds_full = load_dataset("hendrycks/competition_math", split=split, cache_dir=cache_dir)
        max_examples = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(max_examples))
    except (OSError, RuntimeError) as load_exc:
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {load_exc}") from load_exc


__all__ = ["run_inference_on_split", "load_math500"]
