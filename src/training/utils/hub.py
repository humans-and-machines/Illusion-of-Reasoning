#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight helpers for interacting with the Hugging Face Hub."""

import logging
import re
from concurrent.futures import Future
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    from transformers import AutoConfig as _AutoConfig
except ImportError:  # pragma: no cover - type-check / lint env
    _AutoConfig = None

AutoConfig = _AutoConfig

try:  # pragma: no cover - optional dependency
    from huggingface_hub import (
        create_branch as _create_branch,
        create_repo as _create_repo,
        get_safetensors_metadata as _get_safetensors_metadata,
        list_repo_commits as _list_repo_commits,
        list_repo_files as _list_repo_files,
        list_repo_refs as _list_repo_refs,
        repo_exists as _repo_exists,
        upload_folder as _upload_folder,
    )
except ImportError:  # pragma: no cover - type-check / lint env
    def _hub_unavailable(*_args: Any, **_kwargs: Any) -> Any:
        msg = "huggingface_hub is required for Hub utilities."
        raise ImportError(msg)

    _create_branch = _hub_unavailable
    _create_repo = _hub_unavailable
    _get_safetensors_metadata = _hub_unavailable
    _list_repo_commits = _hub_unavailable
    _list_repo_files = _hub_unavailable
    _list_repo_refs = _hub_unavailable
    _repo_exists = _hub_unavailable
    _upload_folder = _hub_unavailable

create_branch = _create_branch
create_repo = _create_repo
get_safetensors_metadata = _get_safetensors_metadata
list_repo_commits = _list_repo_commits
list_repo_files = _list_repo_files
list_repo_refs = _list_repo_refs
repo_exists = _repo_exists
upload_folder = _upload_folder

try:  # pragma: no cover - optional dependency
    from huggingface_hub.utils import HfHubHTTPError as _HfHubHTTPError
except ImportError:  # pragma: no cover - type-check / lint env
    # Fall back to the generic Exception type so callers can still catch
    # ``HfHubHTTPError`` even when huggingface_hub is not installed.
    _HfHubHTTPError = Exception

HfHubHTTPError = _HfHubHTTPError

try:  # pragma: no cover - optional dependency
    from trl import GRPOConfig as _GRPOConfig, SFTConfig as _SFTConfig
except ImportError:  # pragma: no cover - type-check / lint env
    # Re-use our local configuration dataclasses when ``trl`` is unavailable.
    from ..configs import GRPOConfig as _GRPOConfig, SFTConfig as _SFTConfig

GRPOConfig = _GRPOConfig
SFTConfig = _SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(
    training_args: SFTConfig | GRPOConfig,
    extra_ignore_patterns: Optional[list[str]] = None,
) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info("Created target repo at %s", repo_url)
    logger.info(
        "Pushing to the Hub revision %s...",
        training_args.hub_model_revision,
    )
    ignore_patterns = ["checkpoint-*", "*.pth"]
    if extra_ignore_patterns:
        ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(
        "Pushed to %s revision %s successfully!",
        repo_url,
        training_args.hub_model_revision,
    )

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """Checks if a given Hub revision exists."""
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # First check if the revision exists
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            # If the revision exists, we next check it has a README file
            if training_args.hub_model_revision in revisions:
                # Use positional arguments for compatibility with simple test stubs.
                repo_files = list_repo_files(
                    training_args.hub_model_id,
                    training_args.hub_model_revision,
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """Infer parameter count for a model repo.

    Tries safetensors metadata first, then patterns like ``42m``, ``1.5b`` or
    products such as ``8x7b`` in the repo ID when metadata is unavailable.
    """
    try:
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except (ImportError, OSError, ValueError, KeyError, AttributeError, HfHubHTTPError):
        # Pattern to match products (like 8x7b) and single values (like 42m)
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for _full_match, number1, _, _, number2, _, unit in matches:
            if number2:  # If there's a second number, it's a product
                number = float(number1) * float(number2)
            else:  # Otherwise, it's a single value
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000  # Convert to billion
            elif unit == "m":
                number *= 1_000_000  # Convert to million

            param_counts.append(number)

        if param_counts:
            # Return the largest number
            return int(max(param_counts))
        # Return -1 if no match found
        return -1


def get_gpu_count_for_vllm(
    model_name: str,
    revision: str = "main",
    num_gpus: int = 8,
) -> int:
    """Choose a vLLM GPU count compatible with the model.

    vLLM requires both ``num_attention_heads`` and ``64`` to be divisible by
    ``num_gpus``; this routine decreases ``num_gpus`` until that holds.
    """
    if AutoConfig is None:
        msg = (
            "transformers.AutoConfig is required for get_gpu_count_for_vllm; "
            "install transformers."
        )
        raise RuntimeError(msg)

    config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    # Get number of attention heads
    num_heads = config.num_attention_heads
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(
            "Reducing num_gpus from %d to %d to make num_heads divisible by num_gpus",
            num_gpus,
            num_gpus - 1,
        )
        num_gpus -= 1
    return num_gpus
