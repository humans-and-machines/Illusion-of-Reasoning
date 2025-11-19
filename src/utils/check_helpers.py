#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation, stopping, and scoring helpers used by src.utils.check.

These utilities are factored out so that the main crossword evaluation script
stays focused on orchestration while this module handles torch/transformers-
level details. All functions accept a `torch_mod` argument instead of importing
torch directly so that callers control the runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


class StopOnGeneratedSubstring:
    """
    Batch-safe stopper that only scans the generated span for a substring.

    This mirrors the original behaviour in src.utils.check: it receives full
    `input_ids` at each decoding step together with per-row prompt lengths and
    marks rows as finished once the substring appears in the decoded suffix.
    """

    def __init__(self, tok, prompt_lens: List[int], substr: str) -> None:
        self.tok = tok
        self.substr = substr
        self.prompt_lens = list(prompt_lens)
        self.done = [False] * len(self.prompt_lens)

    def __call__(self, input_ids, scores, **_kwargs: Any) -> bool:
        """Return True when all rows have generated the substring at least once."""
        batch_size = input_ids.size(0)
        for row_index in range(batch_size):
            if self.done[row_index]:
                continue
            prompt_length = int(self.prompt_lens[row_index])
            if input_ids.size(1) <= prompt_length:
                continue
            generated_ids = input_ids[row_index, prompt_length:]
            text = self.tok.decode(generated_ids.tolist(), skip_special_tokens=True)
            if self.substr in text:
                self.done[row_index] = True
        return all(self.done)

    def reset(self) -> None:
        """Reset internal done flags to allow reuse for a new batch."""
        self.done = [False] * len(self.prompt_lens)


class BanEosForSteps:
    """Mask EOS ids for the first N steps of generation (batch-wide)."""

    def __init__(self, eos_token_ids: Union[int, List[int]], ban_steps: int) -> None:
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        # Filter out sentinel values such as -1 / None
        self.eos_ids = [e for e in eos_token_ids if e is not None and e >= 0]
        self.ban_steps = max(0, int(ban_steps))
        self.step = 0

    def __call__(self, input_ids, scores):
        """Apply an EOS mask to the scores tensor."""
        if self.step < self.ban_steps and self.eos_ids:
            for eos_id in self.eos_ids:
                if eos_id < scores.size(-1):
                    scores[:, eos_id] = -float("inf")
        self.step += 1
        return scores

    def reset(self) -> None:
        """Reset the internal step counter."""
        self.step = 0


def _sanitize_generation_config(model) -> None:
    """Reset generation config to a conservative, deterministic baseline."""
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    if hasattr(generation_config, "sliding_window"):
        generation_config.sliding_window = None


def _concat_anchor(
    torch_mod,
    tok,
    input_ids,
    attention_mask,
    anchor_text: Optional[str],
) -> Tuple[Any, Any]:
    """Append an assistant anchor as prompt tokens (not generated)."""
    if not anchor_text:
        return input_ids, attention_mask
    anchor_encoding = tok(
        anchor_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    anchor_ids = anchor_encoding["input_ids"].to(input_ids.device)
    batch_size = input_ids.size(0)
    expanded_anchor = anchor_ids.expand(batch_size, -1)
    new_ids = torch_mod.cat([input_ids, expanded_anchor], dim=1)
    new_mask = torch_mod.cat(
        [attention_mask, torch_mod.ones_like(expanded_anchor)],
        dim=1,
    )
    return new_ids, new_mask


@dataclass
class GenerationConfig:
    """Configuration bundle for _generate to avoid many positional args."""

    max_new_tokens: int
    num_beams: int
    min_new_tokens: int
    stop_criteria: Any
    eos_token_ids: Union[int, List[int]]
    num_return_sequences: int = 1
    ban_eos_steps: int = 0
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


def _generate(
    torch_mod,
    model,
    tok,
    input_ids,
    attention_mask,
    config: GenerationConfig,
):
    """Run model.generate with a stable configuration."""
    logits_processors = []
    if config.ban_eos_steps:
        logits_processors.append(BanEosForSteps(config.eos_token_ids, config.ban_eos_steps))

    min_new = (
        config.min_new_tokens
        if config.min_new_tokens and config.min_new_tokens > 0
        else None
    )
    beams = (
        config.num_beams
        if (config.num_beams and config.num_beams > 1 and not config.do_sample)
        else 1
    )
    temperature = config.temperature if config.do_sample else None
    top_p = (
        config.top_p
        if (config.do_sample and config.top_p not in (None, 0))
        else None
    )
    top_k = (
        config.top_k
        if (config.do_sample and config.top_k not in (None, 0))
        else None
    )

    with torch_mod.inference_mode():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            min_new_tokens=min_new,
            num_beams=beams,
            do_sample=config.do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=config.num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=config.eos_token_ids,
            use_cache=True,
            stopping_criteria=config.stop_criteria,
            logits_processor=logits_processors or None,
        )

    # HF duplicates the batch internally for num_return_sequences; mirror that for lens.
    input_lens = attention_mask.sum(dim=-1)
    if config.num_return_sequences and config.num_return_sequences > 1:
        input_lens = input_lens.repeat_interleave(config.num_return_sequences)
    return generation_output.sequences, generation_output.scores, input_lens


def _decode_generated_only(tok, sequences, input_lens) -> List[str]:
    """Decode only the generated span (excluding prompt tokens)."""
    texts: List[str] = []
    for index in range(sequences.size(0)):
        prompt_length = int(input_lens[index].item())
        generated_ids = sequences[index, prompt_length:]
        texts.append(
            tok.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )
    return texts


def _first_eos_pos(tensor_row: Any, eos_ids: Optional[List[int]]) -> Optional[int]:
    """Return index just past the first EOS token in a row, or None if absent."""
    if eos_ids is None or tensor_row.numel() == 0:
        return None

    earliest_position: Optional[int] = None
    for eos_token_id in eos_ids:
        hits = (tensor_row == eos_token_id).nonzero(as_tuple=False)
        if hits.numel() == 0:
            continue
        eos_position = int(hits[0].item()) + 1  # include EOS token
        if earliest_position is None:
            earliest_position = eos_position
        else:
            earliest_position = min(earliest_position, eos_position)
    return earliest_position


def _compute_effective_lengths(
    sequences,
    input_lens,
    num_steps: int,
    eos_ids: Optional[List[int]],
) -> List[int]:
    """Helper to compute effective generated lengths per row."""
    effective_lengths: List[int] = []
    batch_size = sequences.size(0)
    for batch_index in range(batch_size):
        prompt_length = int(input_lens[batch_index].item())
        generated_ids = sequences[batch_index, prompt_length:]
        generated_length = min(generated_ids.size(0), num_steps)
        if generated_length > 0:
            eos_position = _first_eos_pos(
                generated_ids[:generated_length],
                eos_ids,
            )
            if eos_position is not None:
                generated_length = min(generated_length, eos_position)
        effective_lengths.append(generated_length)
    return effective_lengths


def _token_logprobs_stream(
    torch_mod,
    scores,
    sequences,
    input_lens,
    eos_ids: Optional[List[int]] = None,
):
    """Per-sequence average token log-probabilities, truncated at EOS."""
    batch_size = sequences.size(0)
    num_steps = len(scores)
    logprob_sums = [0.0] * batch_size
    token_counts = [0] * batch_size
    if num_steps == 0:
        return logprob_sums, [0.0] * batch_size, token_counts

    effective_lengths = _compute_effective_lengths(
        sequences,
        input_lens,
        num_steps,
        eos_ids,
    )

    max_effective_length = max(effective_lengths) if effective_lengths else 0
    for time_index in range(max_effective_length):
        active_indices = [
            idx
            for idx, length in enumerate(effective_lengths)
            if time_index < length
        ]
        if not active_indices:
            break

        token_ids_tensor = torch_mod.tensor(
            [
                int(
                    sequences[batch_index, int(input_lens[batch_index].item()) + time_index].item(),
                )
                for batch_index in active_indices
            ],
            device=scores[time_index].device,
            dtype=torch_mod.long,
        )
        step_logits = scores[time_index][active_indices].float()
        step_logprobs = torch_mod.log_softmax(step_logits, dim=-1)
        picked_logprobs = step_logprobs.gather(
            1,
            token_ids_tensor.view(-1, 1),
        ).squeeze(1)

        for pos_in_active, batch_index in enumerate(active_indices):
            value = float(picked_logprobs[pos_in_active].item())
            if torch_mod.isfinite(torch_mod.tensor(value)):
                logprob_sums[batch_index] += value
                token_counts[batch_index] += 1

        del token_ids_tensor, step_logits, step_logprobs, picked_logprobs

    averages = [
        (sum_val / count if count > 0 else 0.0)
        for sum_val, count in zip(logprob_sums, token_counts)
    ]
    return logprob_sums, averages, token_counts


def _token_entropy_stream(
    torch_mod,
    scores,
    sequences,
    input_lens,
    eos_ids: Optional[List[int]] = None,
):
    """Average token entropy per sequence, truncated at EOS."""
    batch_size = sequences.size(0)
    num_steps = len(scores)
    entropy_sums = [0.0] * batch_size
    token_counts = [0] * batch_size
    if num_steps == 0:
        return [0.0] * batch_size, token_counts

    effective_lengths = _compute_effective_lengths(
        sequences,
        input_lens,
        num_steps,
        eos_ids,
    )

    max_effective_length = max(effective_lengths) if effective_lengths else 0
    for time_index in range(max_effective_length):
        active_indices = [
            idx
            for idx, length in enumerate(effective_lengths)
            if time_index < length
        ]
        if not active_indices:
            break

        step_logits = scores[time_index][active_indices].float()
        probs = torch_mod.softmax(step_logits, dim=-1)
        entropies = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

        for pos_in_active, batch_index in enumerate(active_indices):
            value = float(entropies[pos_in_active].item())
            if torch_mod.isfinite(torch_mod.tensor(value)):
                entropy_sums[batch_index] += value
                token_counts[batch_index] += 1

        del step_logits, probs, entropies

    averages = [
        (sum_val / count if count > 0 else 0.0)
        for sum_val, count in zip(entropy_sums, token_counts)
    ]
    return averages, token_counts
