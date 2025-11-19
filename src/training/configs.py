# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Configuration dataclasses for training scripts (SFT/GRPO)."""

from dataclasses import dataclass, field
from typing import Any, Optional

import importlib

try:
    trl = importlib.import_module("trl")
except ImportError:  # pragma: no cover - optional dependency
    class _ScriptArgumentsBase:  # pylint: disable=too-few-public-methods
        """Fallback ScriptArguments base when `trl` is absent."""

    class _GRPOConfigBase:  # pylint: disable=too-few-public-methods
        """Fallback GRPOConfig base when `trl` is absent."""

    class _SFTConfigBase:  # pylint: disable=too-few-public-methods
        """Fallback SFTConfig base when `trl` is absent."""

    class _TrlNamespace:  # pylint: disable=too-few-public-methods
        """Lightweight stand-in for the `trl` module."""

        ScriptArguments = _ScriptArgumentsBase
        GRPOConfig = _GRPOConfigBase
        SFTConfig = _SFTConfigBase

    trl = _TrlNamespace()


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str  # pylint: disable=invalid-name
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # Override the dataset_name to make it optional
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset name. Can be omitted if using dataset_mixture."
        },
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Configuration for creating dataset mixtures with advanced "
                "options like shuffling."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [
                set(dataset.columns)
                for dataset in datasets_list
                if dataset.columns is not None
            ]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset "
                        "configurations in a mixture. "
                        f"Found different column sets: "
                        f"{[list(cols) for cols in columns_sets]}"
                    )


# Note: shared options could use a mixin to reduce code duplication.
@dataclass
class GRPOConfig(trl.GRPOConfig):  # pylint: disable=too-many-instance-attributes
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(
        default=0,
        metadata={"help": "Number of completions to print."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to push to a Hub revision/branch."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to log the unique prompts to wandb. This will create "
                "a new run for each unique prompt."
            )
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )

@dataclass
class SFTConfig(trl.SFTConfig):  # pylint: disable=too-many-instance-attributes
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to push to a Hub revision/branch."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):  # pylint: disable=too-many-instance-attributes
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values:
            'accuracy', 'format', 'reasoning_steps', 'cosine',
            'repetition_penalty', 'length', 'tag_count'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        max_completion_len (`int`):
            Maximum number of tokens in completion.
        soft_punish_cache (`int`):
            Minimum number of tokens in completion.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": (
                "List of reward functions. Possible values: 'accuracy', "
                "'format', 'reasoning_steps', 'cosine', 'repetition_penalty', "
                "'length', 'tag_count'"
            )
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )

    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column to use as prompts for training."},
    )
    dataset_solution_column: str = field(
        default="answer",
        metadata={"help": "Column to use as the gold solution/answer for training."},
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )

    span_kl_target: float = field(
        default=0.05,
        metadata={"help": "per-token KL target"},
    )
    span_kl_beta0: float = field(
        default=0.12,
        metadata={"help": "initial KL coeff"},
    )
    span_kl_horizon: int = field(
        default=10000,
        metadata={"help": "horizon for KL controller"},
    )
