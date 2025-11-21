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

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

import importlib

try:
    trl = importlib.import_module("trl")
except ImportError:  # pragma: no cover - optional dependency
    class _ConfigBase:
        """Fallback base for TRL-style config/argument classes.

        Provides a minimal dictionary-like interface so that downstream code
        which expects TRL dataclasses can run in a degraded mode when the
        optional dependency is not installed.
        """

        def to_dict(self) -> dict[str, Any]:
            """Return a shallow dict of attributes."""
            return vars(self).copy()

        def update_from_dict(self, values: dict[str, Any]) -> None:
            """Update attributes from a mapping of key/value pairs."""
            for key, value in values.items():
                setattr(self, key, value)

    trl = SimpleNamespace(
        ScriptArguments=_ConfigBase,
        GRPOConfig=_ConfigBase,
        SFTConfig=_ConfigBase,
    )


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    dataset_id: str
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
                            dataset_id=dataset_config.get("id"),
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


@dataclass
class ChatBenchmarkConfig:
    """Shared options for callbacks, benchmarks, and chat-style prompting."""

    benchmarks: list[str] = field(
        default_factory=list,
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=list,
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The optional system prompt to use "
                "(for training and/or benchmarking)."
            )
        },
    )


@dataclass
class HubRevisionConfig:
    """Configuration related to pushing checkpoints to the Hub."""

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


@dataclass
class WandbRunConfig:
    """Configuration for Weights & Biases runs."""

    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The entity to store runs under."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The project to store runs under."},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": "The group to store runs under."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to log unique prompts to W&B. This will create "
                "a new run for each unique prompt."
            )
        },
    )


@dataclass
class GRPOOnlyTrainingConfig:
    """GRPO-specific training extras."""

    num_completions_to_print: int = field(
        default=0,
        metadata={"help": "Number of completions to print."},
    )


class GRPOConfig(trl.GRPOConfig):  # pylint: disable=too-few-public-methods
    """
    Thin wrapper around :class:`trl.GRPOConfig` for type-checking.

    All project-specific training options live in separate dataclasses
    (:class:`ChatBenchmarkConfig`, :class:`HubRevisionConfig`,
    :class:`WandbRunConfig`, :class:`GRPOOnlyTrainingConfig`) which are merged
    into an instance of this class at CLI parse time.
    """


class SFTConfig(trl.SFTConfig):  # pylint: disable=too-few-public-methods
    """
    Thin wrapper around :class:`trl.SFTConfig` for type-checking.

    Project-specific training options are provided by
    :class:`ChatBenchmarkConfig`, :class:`HubRevisionConfig`, and
    :class:`WandbRunConfig` and merged into instances of this class by callers.
    """


@dataclass
class GRPORewardConfig:
    """
    Reward-related script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values:
            'accuracy', 'format', 'reasoning_steps', 'cosine',
            'repetition_penalty', 'length', 'tag_count'.
        repetition_n_grams (`int`):
            Number of n-grams for repetition penalty reward.
        repetition_max_penalty (`float`):
            Maximum (negative) penalty for repetition penalty reward.
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
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Maximum (negative) penalty for the repetition penalty reward."
            )
        },
    )


@dataclass
class GRPOCosineRewardConfig:
    """
    Cosine-scaling parameters for GRPO reward shaping.

    Args:
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
    """

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


@dataclass
class GRPODatasetColumnsConfig:
    """
    Dataset column and generation-length parameters for GRPO.

    Args:
        dataset_prompt_column (`str`):
            Column to use as prompts for training.
        dataset_solution_column (`str`):
            Column to use as the gold solution/answer for training.
        max_completion_len (`int`):
            Maximum number of characters in completion.
        soft_punish_cache (`int`):
            Minimum number of characters in completion.
    """

    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Column to use as prompts for training."},
    )
    dataset_solution_column: str = field(
        default="answer",
        metadata={
            "help": "Column to use as the gold solution/answer for training."
        },
    )
    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )


@dataclass
class GRPOSpanKLConfig:
    """
    Span-wise KL control parameters for GRPO.

    Args:
        span_kl_target (`float`):
            Per-token KL target.
        span_kl_beta0 (`float`):
            Initial KL coefficient.
        span_kl_horizon (`int`):
            Horizon for KL controller.
    """

    span_kl_target: float = field(
        default=0.05,
        metadata={"help": "Per-token KL target."},
    )
    span_kl_beta0: float = field(
        default=0.12,
        metadata={"help": "Initial KL coefficient."},
    )
    span_kl_horizon: int = field(
        default=10000,
        metadata={"help": "Horizon for KL controller."},
    )


class GRPOScriptArguments(ScriptArguments):  # pylint: disable=too-few-public-methods
    """
    Base script arguments for the GRPO training script.

    GRPO-specific options are provided via separate dataclasses
    (:class:`GRPORewardConfig`, :class:`GRPOCosineRewardConfig`,
    :class:`GRPODatasetColumnsConfig`, :class:`GRPOSpanKLConfig`) and merged
    into an instance of this class at CLI parse time.
    """


def merge_dataclass_attributes(target: Any, *configs: Any) -> Any:
    """
    Copy all fields from one or more dataclass instances onto ``target``.

    This is used to keep the configuration objects that external code sees
    (``GRPOConfig``, ``SFTConfig``, ``GRPOScriptArguments``) small and tidy
    while still exposing a flat attribute namespace at runtime.
    """
    for cfg in configs:
        if cfg is None:
            continue
        for key, value in asdict(cfg).items():
            setattr(target, key, value)
    return target
