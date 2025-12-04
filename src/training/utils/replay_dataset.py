"""Replay-aware dataset and collate helpers used during GRPO training."""

from __future__ import annotations

import copy
from typing import Any, Iterable, List, Mapping


class ReplayMixDataset:
    """Thin wrapper that tracks `is_replay` on items from a base dataset."""

    def __init__(self, base_dataset: Iterable[Mapping[str, Any]], tokenizer: Any):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return a deep-copied item with `is_replay` set to 0."""
        item = copy.deepcopy(self.base_dataset[index])  # type: ignore[index]

        # Quick sanity check only if already tokenised
        if "input_ids" in item:
            last_user = next(message for message in reversed(item["prompt"]) if message["role"] == "user")
            decoded = self.tokenizer.decode(item["input_ids"])
            assert last_user["content"] in decoded, "⛔ clue missing from encoded prompt!"

        item["is_replay"] = 0
        return item


def replay_collate(
    batch: List[Mapping[str, Any]],
    *,
    _replay_buffer: Any,
    _replay_prob: float,
) -> List[Mapping[str, Any]]:
    """
    Identity collate — no sampling here.

    We only want to clear the old `is_replay` flag,
    but we *must not* drop 'accuracy'.
    """
    for example in batch:
        example["is_replay"] = 0
    return batch
